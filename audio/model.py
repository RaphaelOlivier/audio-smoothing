import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append("..")
from eval1.audio.transforms import MFCC, MelSpectrogram, Spectrogram, SpectrogramToDB, Compose, PNCC, LPBiquad

from art.classifiers import PyTorchClassifier
import numpy as np

import logging
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 8000
WINDOW_STEP_SIZE = 2000
WINDOW_LENGTH = int(SAMPLE_RATE * WINDOW_STEP_SIZE / 1000)

from eval1.audio.defenses.wrapper import SmoothedAudioClassifier
from art.classifiers.pytorch import PyTorchClassifier
def preprocessing_fn(batch):
    """
    Standardize, then normalize sound clips
    """
    processed_batch = []
    for clip in batch:

        signal = clip.astype(np.float32)
        # Signal normalization
        signal = signal / np.max(np.abs(signal))

        # get random chunk of fixed length (from SincNet's create_batches_rnd)
        signal_length = len(signal)
        signal_start = np.random.randint(signal_length - WINDOW_LENGTH - 1)
        signal_stop = signal_start + WINDOW_LENGTH
        signal = signal[signal_start:signal_stop]
        processed_batch.append(signal)

    return np.array(processed_batch)

class CNNAudioClassifier(nn.Module):
    # Simple 1dCNN classfier
    def __init__(self,audio_features, transform,nclasses, filters, kernel=5, stride=2, padding=0,kernel_pool=3, stride_pool=1, dropout=0.2):
        super(CNNAudioClassifier, self).__init__()
        
        
        transform, audio_features = self.get_transform_module(transform, audio_features)
        self.transform=transform
        self.audio_features=audio_features
        filters = [audio_features] +filters
        convs=[]
        for i in range(len(filters)-1):
            conv = nn.Conv1d(filters[i],filters[i+1],kernel_size=kernel,stride=stride,padding=padding)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)
        self.pool = nn.AvgPool1d(kernel_size=kernel_pool,stride=stride_pool,padding=0)
        self.act = nn.ReLU()
        self.dr = nn.Dropout(dropout)
        self.proj = nn.Linear(filters[-1],nclasses)

    def get_transform_module(self,transform, audio_features):
        nfeats = audio_features
        if transform is None:
                transform_fct = None 
        elif isinstance(transform,str):
            if transform=="none":
                transform_fct=None 
            elif transform=="mfcc":
                transform_fct = MFCC(n_mfcc=audio_features)
            elif transform =="mel":
                transform_fct = MelSpectrogram(n_fft=audio_features)
            elif transform=="specdb":
                transform_fct = Compose([Spectrogram(n_fft=audio_features,hop=audio_features//2), SpectrogramToDB()])
                nfeats = audio_features//2+1
            elif transform=="pncc":
                transform_fct = PNCC(n_pncc=audio_features)
            elif transform=="filter":
                transform_fct = LPBiquad(sample_rate=SAMPLE_RATE, cutoff_freq=200)
            elif transform=="filterspec":
                transform_fct = Compose([LPBiquad(sample_rate=SAMPLE_RATE, cutoff_freq=3000), Spectrogram(n_fft=audio_features,hop=audio_features//2), SpectrogramToDB()])
                nfeats = audio_features//2+1
            else:
                raise ValueError("Unknown transform %s"%transform)
        else:
            transform_fct=transform 
        return transform_fct, nfeats
    def get_feats(self,x,**kwargs):
        if self.transform is not None:
            x=self.transform(x)
        if x.dim()==2:
            return x.unsqueeze(1)
        else:
            return x.transpose(1,2)

    def forward(self,x, **kwargs):
        assert isinstance(x,torch.Tensor)
        assert x.dim()==2 #NxL
        h = self.get_feats(x,**kwargs) #NxDxL
        #print(x.size(),h.size())
        assert h.dim()==3 #NxDxL
        return self.get_output(h,**kwargs)

    def get_output(self,h,**kwargs):
        #print(x.size(),h.size())
        for layer in self.convs:
            h = layer(h) #NxDxL
            h = self.pool(h) #NxDxL
            h = self.dr(h)
            h = self.act(h) #NxDxL
        h = torch.mean(h,dim=2) #NxD
        out = self.proj(h)

        return out 

class CNN2DAudioClassifier(CNNAudioClassifier):
    def __init__(self,audio_features, transform,nclasses, filters, kernel=5, stride=1, padding=0,kernel_pool=3, stride_pool=2, dropout=0.2):
        super(CNNAudioClassifier, self).__init__()
        
        
        transform, audio_features = self.get_transform_module(transform, audio_features)
        self.transform=transform
        self.audio_features=audio_features
        filters = [1] +filters
        convs=[]
        for i in range(len(filters)-1):
            conv = nn.Conv2d(filters[i],filters[i+1],kernel_size=kernel,stride=stride,padding=padding)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)
        self.pool = nn.AvgPool2d(kernel_size=kernel_pool,stride=stride_pool,padding=0)
        self.act = nn.ReLU()
        self.dr = nn.Dropout(dropout)
        self.proj = nn.Linear(filters[-1],nclasses)

    def get_feats(self,x,**kwargs):
        if self.transform is not None:
            return self.transform(x).transpose(1,2).unsqueeze(1)
        else:
            return x.unsqueeze(1).unsqueeze(1)

    def get_output(self,h,**kwargs):
        #print(x.size(),h.size())
        for layer in self.convs:
            h = layer(h) #NxDxLxH
            h = self.pool(h)  #NxDxLxH
            h = self.dr(h)
            h = self.act(h)  #NxDxLxH
        h = torch.mean(torch.mean(h,dim=3),dim=2) #NxD
        out = self.proj(h)

        return out 

def make_audio_model(cnndim=1,noise_type="none",noise_param=-1,noise_raw=-1, niters=-1,**kwargs):
    model_kwargs = get_default_model_config(kwargs,cnndim=cnndim)
    if cnndim==1:
        model= CNNAudioClassifier(nclasses=40, **model_kwargs)
    else:
        model= CNN2DAudioClassifier(nclasses=40, **model_kwargs)
    return SmoothedAudioClassifier(model,nclasses=40,noise_type=noise_type,noise_param=noise_param, noise_raw=noise_raw, niters=niters)

def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_audio_model(**model_kwargs)
    model.to(DEVICE)

    if weights_file:
        try:
            dic = torch.load("saved_models/"+weights_file)
            model.load_state_dict(dic)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.info(os.listdir(os.getcwd()))
            logger.warning(str(e))


    
    wrapped_model = SmoothedPytorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(
            model.parameters(), lr=0.0001),
        input_shape=(WINDOW_LENGTH,),
        nb_classes=40,
    )
    
    return wrapped_model
    
class SmoothedPytorchClassifier(PyTorchClassifier):
        def predict(self, x, batch_size=128, smooth=True,**kwargs):
            import torch
            # Apply preprocessing
            x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
            #x_preprocessed = x
            # Run prediction with batch processing
            results = np.zeros((x_preprocessed.shape[0], self.nb_classes()), dtype=np.float32)
            num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
            for m in range(num_batch):
                # Batch indexes
                begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])

                model_outputs = self._model._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device), smooth=smooth)
                output = model_outputs
                results[begin:end] = output.detach().cpu().numpy()

            # Apply postprocessing
            predictions = self._apply_postprocessing(preds=results, fit=False)
            #predictions=results
            return predictions


def get_default_model_config(kwargs, cnndim=1):
    new_kwargs={}
    if cnndim==1:
        default_none = {"audio_features":1,"filters":[32,64,128,256]}
        default_mfcc = {"audio_features":40,"filters":[64,128,256]}
        default_pncc = {"audio_features":40,"filters":[64,128,256]}
        default_specdb = {"audio_features":400,"filters":[256,256,256], "padding":2}
        default_dic = {"none":default_none,"mfcc":default_mfcc, "pncc":default_pncc, "specdb":default_specdb,
        "filter":default_none,"filterspec":default_specdb}
    else:
        default_specdb = {"audio_features":400,"filters":[8,16,32,64,128], "padding":3,"kernel":7}
        default_dic = {"specdb":default_specdb}
    
    if "transform" in kwargs:
        assert kwargs["transform"] in default_dic
        new_kwargs["transform"]=kwargs["transform"]
    else:
        new_kwargs["transform"]="none"
    dic = default_dic[new_kwargs["transform"]]
    for k in dic:
        new_kwargs[k] = kwargs[k] if k in kwargs else dic[k]
    for k in kwargs:
        if k not in dic:
            new_kwargs[k]=kwargs[k]
    return new_kwargs

    
