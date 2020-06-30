
import logging
logger = logging.getLogger(__name__)
import torch.nn as nn
import torch
from collections import Counter
from art.defences.preprocessor import Preprocessor
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import numpy as np 
from scipy.cluster.hierarchy import linkage,fcluster

from scipy.stats import norm

from eval1.audio.gmm import gmm
#from statsmodels.stats.proportion import proportion_confint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from eval1.audio.defenses.deterministicmasking import *
from eval1.audio.defenses.smoothing import *

from eval1.audio.defenses.base import AudioSmoother, NoSmoother



class SmoothedAudioClassifier(nn.Module):
    def __init__(self,model,nclasses ,noise_type,noise_param,noise_raw, niters=50, **kwargs):
        super(SmoothedAudioClassifier, self).__init__()
        self.model=model 
        self.noise_type=noise_type
        self.nclasses=nclasses
        self.niters=niters
        smoothers = []
        noise_types = noise_type.split("_")
        if isinstance(noise_param,float):
            noise_params = [noise_param for _ in noise_types]
        else:
            noise_params = noise_param.split("_")
        noise_params = [float(p) for p in noise_params]
        for n,p in zip(noise_types,noise_params):
            if n=="none":
                s= NoSmoother(noise_raw)
                assert len(noise_types)==1
                self.niters=-1
            elif n=="gaussian":
                s = GaussianSmoother(sigma=p,noise_raw=noise_raw)
            elif n=="hfgaussian":
                s = HighFreqSmoother(sigma=p,noise_raw=noise_raw)
            elif n=="bwgaussian":
                s = ButterworthSmoother(sigma=p,noise_raw=noise_raw, order=3,cutoff=100)
            elif n=="time":
                s = TimeSmoother(p=p,noise_raw=noise_raw)
            elif n=="freq":
                s = FreqSmoother(p=p,nfeats=model.audio_features, noise_raw=noise_raw)
            elif n=="freq-zero":
                s = FreqSmoother(p=p,nfeats=model.audio_features, fill="zero",noise_raw=noise_raw)
            elif n=="freq-avg":
                s = FreqSmoother(p=p,nfeats=model.audio_features, fill="avg", noise_raw=noise_raw)
            elif n=="freq-gmm":
                s = FreqSmoother(p=p,nfeats=model.audio_features, fill="gmm", noise_raw=noise_raw, gmm_ncomponents=32, gmm_weights_path = "wav/gmm/specdb/32/")
            elif n=="lowpass":
                s = LowPassSmoother(p=p,nfeats=model.audio_features, noise_raw=noise_raw)
            elif n=="highpass":
                s = HighPassSmoother(p=p,nfeats=model.audio_features, noise_raw=noise_raw)
            elif n=="corr":
                s = CorrReconSmoother(p=p,nfeats=model.audio_features, noise_raw=noise_raw)
                self.niters=1 # not random
            else:
                raise NotImplementedError
            smoothers.append(s)
        if len(smoothers)==1:
            self.smoother=smoothers[0]
        else:
            self.smoother=ComposeSmoother(*smoothers,noise_raw=noise_raw)

    
    def forward(self, utterances, *args, smooth=False,noise=True,niters=-1,**kwargs):
        if niters<=0:
            niters=self.niters
        if smooth and (niters>0):
            #utterances=self.smoother(utterances)
            all_preds=[]
            for i in range(niters):
                logits= self.smooth_and_forward(utterances,*args,**kwargs)
                pred = torch.max(logits,dim=-1)[1]
                all_preds.append(pred)
            preds = torch.stack(all_preds,dim=1).view(len(utterances),niters,1)
            idx = torch.arange(self.nclasses).view(1,1,self.nclasses).expand(len(utterances),niters,self.nclasses).float().to(DEVICE)
            counter = (preds==idx).sum(dim=1).float()
            probs = (counter / counter.sum(dim=1,keepdim=True)).max(dim=1)[0]
            prob_min,prob_max, prob_mean = probs.min(),probs.max(), probs.mean()
            if niters>1:
                logger.info("Classes predicted with an average probability of %f (min %f, max %f, %d iters)"% (prob_mean, prob_min,prob_max,niters)) 
            return counter
        else:
            if noise:
                logits=self.smooth_and_forward(utterances,*args,**kwargs)
            else:
                logits = self.model(utterances,*args,**kwargs)
            return logits

    def smooth_and_forward(self,x,*args,feats_only=False,**kwargs):
        h=self.smoother(x, post_feats=False,**kwargs)
        h = self.model.get_feats(h)
        h = self.smoother(h,post_feats=True,**kwargs)
        if feats_only:
            return h
        logits= self.model.get_output(h,*args, **kwargs)
        return logits

    def certify(self,utterances,*args,niters_sampling=10, niters=10):
        counts0=torch.zeros(len(utterances),self.nclasses)
        for i in range(niters_sampling):
            logits= self.smooth_and_forward(utterances,*args, **kwargs)
            pred = torch.max(logits,dim=-1)[1]
            counts0[torch.arange(len(logits)),pred]+=1
        pred = counts0.max(dim=1)[1]
        counts = torch.zeros(len(utterances),self.nclasses, alpha=0.01)
        for i in range(niters):
            logits= self.smooth_and_forward(utterances,*args, **kwargs)
            pred = torch.max(logits,dim=-1)[1]
            counts[torch.arange(len(logits)),pred]+=1

        k = counts[pred]
        radius=-torch.ones(len(utterances)).float()
        for i in range(len(utterances)):
            lowerbound = proportion_confint(k[i], niters, alpha=2*alpha, method="beta")[0]
            if lowerbound>1/2:
                radius[i] = self.noise * norm.ppf(lowerbound)

        return radius

