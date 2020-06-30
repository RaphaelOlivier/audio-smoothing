from importlib import import_module

import json
import logging
logger = logging.getLogger(__name__)

import os, sys

import torch
import torch.nn as nn
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


import eval1.audio.model as cnn_model
import os
logger.debug(os.getcwd())
from art.attacks import ProjectedGradientDescent, FastGradientMethod
from eval1.audio.defenses.wrapper import SmoothedAudioClassifier


from sklearn.metrics import roc_auc_score

def get_auc(y_score, y_true):
    auc = roc_auc_score(y_true.detach().cpu().numpy(), y_score.detach().cpu().numpy())
    return auc

def load_preprocessing_fn(model_config):
    model_module = import_module(model_config["module"])
    preprocessing_fn = getattr(model_module, "preprocessing_fn", None)
    if preprocessing_fn is not None and not callable(preprocessing_fn):
        raise TypeError(f"preprocessing_fn {preprocessing_fn} must be None or callable")
    return  preprocessing_fn

def make_sincnet_smooth_wrapper(noise_type="none",noise_param=-1,noise_raw=-1, niters=-1,**kwargs):
    sys.path.append("SincNet")        
    from armory.baseline_models.pytorch import sincnet

    #wf = kwargs["weights_file"] if "weights_file" in kwargs else None
    classifier=  sincnet.sincnet()
    class SincnetWrapper(nn.Module):
        def __init__(self,model):
            super(SincnetWrapper,self).__init__()
            self.model=model
        def get_feats(self,x,**kwargs):
            return x.unsqueeze(1)
        def get_output(self,x,**kwargs):
            return self.model.forward(x.squeeze(1))
        def forward(self,x,**kwargs):
            return self.model.forward(x)
    
    classifier = SmoothedAudioClassifier(SincnetWrapper(classifier),nclasses=40,noise_type=noise_type,noise_param=noise_param, noise_raw=noise_raw, niters=niters)
    preprocessing_fn = sincnet.preprocessing_fn
    return classifier, preprocessing_fn
    
    
def load(config, model, nb_epochs=100,cnndim=None, **kwargs):
    from armory.utils.config_loading import load_dataset
    model_config = config["model"]
    if model in ["cnn","cnn1d", "cnn2d"]:
        classifier=  cnn_model.make_audio_model(cnndim=cnndim,**kwargs)
        preprocessing_fn = cnn_model.preprocessing_fn
    elif model=="sincnet":
        classifier, preprocessing_fn = make_sincnet_smooth_wrapper(**kwargs)
    
    train_data = load_dataset(
                    config["dataset"],
                    epochs=nb_epochs,
                    split_type="train",
                    preprocessing_fn=preprocessing_fn,
                )

    test_data = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                preprocessing_fn=preprocessing_fn,
            )
    val_data = load_dataset(
                config["dataset"],
                epochs=nb_epochs,
                split_type="validation",
                preprocessing_fn=preprocessing_fn,
            )

    classifier.to(DEVICE)
    return classifier, (train_data,val_data, test_data)

def fit( model ,generator, optimizer, nb_epochs=1, log_every=10, adversarial_attacker = None, adversarial_frac=0.,**kwargs):
        model.train()
        # Train directly in PyTorch
        num_batch = int(np.ceil(generator.size / float(generator.batch_size)))
        logger.info("Number of batches : %r" % num_batch)
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        for _ in range(nb_epochs):
            for m in range(num_batch):
                feats,targets = generator.get_batch()
                if adversarial_attacker is not None:
                    p = np.random.rand(1)
                    if p<adversarial_frac:
                        feats = adversarial_attacker.generate(feats, targets)
                        feats = np.clip(feats,-1,1)
                utterances_batch = torch.tensor(feats).to(DEVICE).float()
                targets_batch = torch.tensor(targets).to(DEVICE)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Apply defenses
                #utterances_batch, targets_batch = self._apply_preprocessing_defences(utterances_batch, targets_batch,fit=True)
                # Adversarial training
                # Actual training
                model_outputs = model(utterances_batch, smooth=False,noise=True,update_smoother_params=True,**kwargs)

                #print(feats[0][0],model_outputs[0][0])
                loss = loss_fct(model_outputs, targets_batch)
                if (m+1) % log_every == 0:
                    logger.info("Loss at batch %r : %r" % (m+1, loss.item()))
                loss.backward()
                optimizer.step()

def eval_benign(model,generator, adversarial_attacker = None, **kwargs):
    model.eval()
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    # Train directly in PyTorch
    num_batch = int(np.ceil(generator.size / float(generator.batch_size)))
    logger.info("Number of batches : %r" % num_batch)
    loss=0
    acc = 0
    adv_acc = 0
    for m in range(num_batch):
        feats,targets = generator.get_batch()
        utterances_batch = torch.tensor(feats).to(DEVICE).float()
        targets_batch = torch.tensor(targets).to(DEVICE)
        model_outputs = model(utterances_batch,smooth=True, **kwargs)
        loss += loss_fct(model_outputs, targets_batch).item()
        acc+=(model_outputs.argmax(dim=-1)==targets_batch).sum().item()
        if adversarial_attacker is not None:
            adv_feats = adversarial_attacker.generate(feats, targets)
            adv_batch = torch.tensor(adv_feats).to(DEVICE).float()
            adv_outputs = model(adv_batch,smooth=True, **kwargs)
            adv_acc+=(adv_outputs.argmax(dim=-1)==targets_batch).sum().item()

    if adversarial_attacker is not None:
        return float(loss)/generator.size,float(acc)/generator.size, float(adv_acc)/generator.size
        
    return float(loss)/generator.size,float(acc)/generator.size, None
            
            

def train_model(model ,train_generator, val_generator, save_path,nb_epochs=20,adv_eps=0.0,adv_frac=0.5, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_acc=0
    attacker=None
    if adv_eps>0:
        classifier = cnn_model.SmoothedPytorchClassifier(model,nb_classes=40,loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),input_shape=(cnn_model.WINDOW_LENGTH,))
        attacker = ProjectedGradientDescent(classifier,eps=adv_eps,eps_step = adv_eps/5,max_iter=10,batch_size=train_generator.batch_size)
        #attacker = FastGradientMethod(classifier,eps=adv_eps,batch_size=train_generator.batch_size)
    for i in range(nb_epochs):
        logger.info("Epoch %d"%i)
        fit(model ,train_generator, optimizer,adversarial_attacker=attacker,adversarial_frac=adv_frac, **kwargs)
        loss, nat_acc, adv_acc = eval_benign(model,val_generator,adversarial_attacker=attacker,niters=1)
        logger.info("Validation loss : %f"%loss)
        logger.info("Validation accuracy : %f"%nat_acc)
        acc=nat_acc
        if adv_eps>0:
            logger.info("Adversarial accuracy : %f"%adv_acc)
            acc=adv_acc
        if acc>best_acc:
            best_acc=acc
            logger.info("Saving model")
            torch.save(model.state_dict(),save_path)

if __name__ == "__main__":
    config_path = "configs/librispeech.json"
    save_path = "../saved_models/simple_cnn.pth"
    with open(config_path, 'r') as f:
        config = json.load(f)
    try:
        logger.info("Loading model weights")
        dic = torch.load(save_path)
        model.load_state_dict(dic)
    except:
        logger.warn("Impossible to load weights")

    model, data = load(config)

    train_model(model ,data[0], data[1],save_path)
    
    loss, acc = eval_benign(model,data[2])
    logger.info("Test loss : %f"%loss)
    logger.info("Test accuracy : %f"%acc)


def eval_verif(model,generator, adversarial_attacker = None, **kwargs):
    model.eval()
    loss_fct = nn.CosineSimilarity(dim=1)
    # Train directly in PyTorch
    num_batch = int(np.ceil(generator.size / float(generator.batch_size)))
    logger.info("Number of batches : %r" % num_batch)
    adv_acc = 0
    scores=[]
    adv_scores=[]
    labels=[]
    if adversarial_attacker is not None:
        def adv_loss(o,y):
            x1,l=y 
            o1 = model(x1,smooth=True,features=True,**kwargs)
            loss = loss_fct(o,o1)
            y1s = 2*y-1
            return loss * (-y1s)
        adversarial_attacker.classifier._loss = adv_loss
    for m in range(num_batch):
        feats1,feats2,lab = generator.get_batch()
        utterances_batch1 = torch.tensor(feats1).to(DEVICE).float()
        utterances_batch2 = torch.tensor(feats2).to(DEVICE).float()
        label_batch = torch.tensor(lab)
        outputs1 = model(utterances_batch1,smooth=True, features=True,**kwargs)
        outputs2 = model(utterances_batch2,smooth=True, features=True,**kwargs)
        loss = loss_fct(outputs1,outputs2)
        
        loss=loss.detach().cpu()
        scores.append(loss)
        labels.append(label_batch)
        auc=get_auc(loss,label_batch)
        logger.info("Batch AUC :%f"%auc)
        if adversarial_attacker is not None:
            adv_feats = adversarial_attacker.generate(feats, targets)
            adv_batch = torch.tensor(adv_feats).to(DEVICE).float()
            adv_outputs = model(adv_batch,smooth=True, **kwargs)
            adv_scores+=(adv_outputs.argmax(dim=-1)==targets_batch).sum().item()

    scores=torch.cat(scores)
    labels=torch.cat(labels)

    auc = get_auc(scores,labels)
    
    if adversarial_attacker is not None:
        return float(loss)/generator.size,float(acc)/generator.size, float(adv_acc)/generator.size
        
    return auc
            