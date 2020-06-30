"""
This scenario, alternative to the default armory one, is used to train models without using ART.
"""


"""
General audio classification scenario
"""
from tqdm import tqdm
import logging
from importlib import import_module

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import json
import torch
import argparse
import os
import sys
sys.path.append("eval1")

from eval1.audio.train import train_model, eval_benign

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_FOLDER = "saved_models"


class AudioClassificationTrain(Scenario):

    def extract_model_file(self,model_config):
        model_kwargs = model_config["model_kwargs"]
        fit_kwargs = model_config["fit_kwargs"]
        noise_raw = model_kwargs["noise_raw"]
        noise_type = model_kwargs["noise_type"]
        noise_val = model_kwargs["noise_param"]
        feats = model_kwargs["transform"]
        adv_eps = fit_kwargs["adv_eps"]
        adv_frac = fit_kwargs["adv_frac"]
        if model_config["module"]=="eval1.audio.model" and model_kwargs["cnndim"]==1:
            model_file = "cnn_feats-"+feats
            if model_kwargs["noise_type"] != "none":
                model_file+="_noise-"+model_kwargs["noise_type"]+("-raw" if noise_raw==1 else ("-spec" if noise_raw==0 else ""))+"-"+str(noise_val)
            if adv_eps >0 :
                model_file+="_adv-"+str(adv_eps)+"-"+str(adv_frac)
            model_file+=".pth"
        elif model_config["module"]=="eval1.audio.model" and model_kwargs["cnndim"]==2:
            model_file = "cnn2d_feats-"+feats
            if noise_type != "none":
                model_file+="_noise-"+noise_type+("-raw" if noise_raw==1 else ("-spec" if noise_raw==0 else ""))+"-"+str(noise_val)
            if adv_eps >0 :
                model_file+="_adv-"+str(adv_eps)+"-"+str(adv_frac)
            model_file+=".pth"
        else:
            assert model_config["module"] == "armory.baseline_models.pytorch.sincnet"
            model_file = "sincnet"
            if feats!="none":
                raise NotImplementedError("Sincnet does not support feature representations at this point")
            if noise_type != "none":
                raise NotImplementedError("Sincnet does not support noise smoothing at this point")
            if adv_eps >0 :
                model_file+="_adv-"+str(adv_eps)+"-"+str(adv_frac)
            model_file+=".pth"

        return model_file

    def load_data(self,config_dataset,nb_epochs,preprocessing_fn):
        train_data = load_dataset(
                    config_dataset,
                    epochs=nb_epochs,
                    split_type="train",
                    preprocessing_fn=preprocessing_fn,
                )

        test_data = load_dataset(
                    config_dataset,
                    epochs=1,
                    split_type="test",
                    preprocessing_fn=preprocessing_fn,
                )
        val_data = load_dataset(
                    config_dataset,
                    epochs=nb_epochs,
                    split_type="validation",
                    preprocessing_fn=preprocessing_fn,
                )

        return train_data, val_data, test_data

    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """

        model_config = config["model"]
        model_module = import_module(model_config["module"])
        model_fn = getattr(model_module, model_config["name"])
        weights_file = model_config.get("weights_file", None)
        classifier = model_fn(
            **model_config["model_kwargs"]
        )
        preprocessing_fn = getattr(model_module, "preprocessing_fn", None)
        classifier = classifier.to(DEVICE)
        
        model_file = model_config.get("weights_file")
        if model_file is None or model_file=="":
            model_file=self.extract_model_file(model_config)

        assert model_config["fit"]
        fit_kwargs = model_config["fit_kwargs"]
        logger.info(
            f"Fitting model {model_config['module']}.{model_config['name']}..."
        )

        save_path = os.path.join(MODELS_FOLDER,model_file)

        try:
            logger.info("Loading model weights")
            dic = torch.load(save_path)
            classifier.load_state_dict(dic)
        except Exception as e:
            logger.warn(str(e))

        niters = model_config["model_kwargs"]["niters"]

        train_data, val_data,test_data = self.load_data(config["dataset"],model_config["fit_kwargs"]["nb_epochs"], preprocessing_fn)

        train_model(classifier, train_data, val_data,save_path,**fit_kwargs)

        loss, acc,_ = eval_benign(model,test_data,niters=niters)
        logger.info("Test loss : %f"%loss)
        logger.info("Test accuracy : %f"%acc)
