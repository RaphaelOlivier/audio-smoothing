

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
from scipy.signal import butter, sosfilt

from scipy.stats import norm

from audio.gmm import gmm
#from statsmodels.stats.proportion import proportion_confint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from audio.defenses.base import AudioSmoother

class GaussianSmoother(AudioSmoother):
    def __init__(self, sigma, **kwargs):
        super(GaussianSmoother,self).__init__(**kwargs)
        self.sigma = sigma 
        self.dist = Normal(0,1)
        if self._post_feats is None:
            self._post_feats=False #default

    def apply_noise(self,x,**kwargs):
        perturbation = self.dist.sample(x.size()).to(x.device)*self.sigma
        x = x + perturbation
        return x

class HighFreqSmoother(AudioSmoother):
    def __init__(self, sigma, **kwargs):
        super(HighFreqSmoother,self).__init__(**kwargs)
        self.sigma = sigma 
        self.dist = Normal(0,1)
        if self._post_feats is None:
            self._post_feats=False #default

    def apply_noise(self,x,**kwargs):
        size = list(x.size())
        size[0] = size[0]+1 # one more noise step for hp filter application
        size = tuple(size) 
        noise = self.dist.sample(size).to(x.device)*self.sigma
        perturbation = 0.5*(noise[1:]-noise[:x.size(0)]) # basic high-pass filter
        x = x + perturbation
        return x

class ButterworthSmoother(AudioSmoother):
    def __init__(self, sigma, order,cutoff, **kwargs):
        super(ButterworthSmoother,self).__init__(**kwargs)
        self.sigma = sigma 
        self.order=order 
        self.cutoff=cutoff 
        self.sos = butter(3, 15, 'hp', fs=100, output='sos')
        self.dist = Normal(0,1)
        if self._post_feats is None:
            self._post_feats=False #default

    def apply_noise(self,x,**kwargs):
         # one more noise step for hp filter application
        size = x.size() 
        noise = self.dist.sample(size)*self.sigma
        perturbation = [sosfilt(self.sos, ns.numpy()) for ns in noise]
        perturbation = torch.tensor(np.stack(perturbation,axis=0)).to(x.device).float()
        x = x + perturbation
        return x

class TimeSmoother(AudioSmoother):
    def __init__(self, p, **kwargs):
        super(TimeSmoother,self).__init__(**kwargs)
        self.p = p 
        self.dist = Bernoulli(1-p)
        if self._post_feats is None:
            self._post_feats=True #default

    def apply_noise(self,x,**kwargs):
        assert x.dim()==2+self.post_feats
        # 1 is the time dimension
        perturbation = self.dist.sample((x.size(0),x.size(-1))).to(x.device)
        if self.post_feats:
            perturbation=perturbation.unsqueeze(1)
        x = x * perturbation
        return x
        

class FreqSmoother(AudioSmoother):
    def __init__(self, p, nfeats, fill="avg",running_lambda=0.9,gmm_ncomponents=32, gmm_weights_path = None,**kwargs):
        super(FreqSmoother,self).__init__(**kwargs)
        self.p = p 
        self.dist = Bernoulli(1-p)
        if self._post_feats is None:
            self._post_feats=True
        assert self.post_feats,"Frequency smoothing cannot be applied on raw inputs"
        assert fill in ["avg","zero","gmm"]
        self.use_running_mean = fill=="avg"
        if self.use_running_mean:
            self.register_buffer('running_mean', torch.zeros(nfeats).view(1,-1,1))
            self.running_lambda = running_lambda
        self.use_gmm = fill=="gmm"
        if self.use_gmm:
            mu,var, pi = None,None, None
            if gmm_weights_path is not None:
                mu,var, pi = gmm.GaussianMixture.load_params_from_np(gmm_weights_path)
            self.gmm = gmm.GaussianMixture(n_components=gmm_ncomponents, n_features=nfeats,mu_init=mu,var_init=var, pi_init=pi)
    def apply_noise(self,x,update_smoother_params=False, **kwargs):
        # 2 is the time dimension
        assert x.dim()==3
        perturbation_mask = self.dist.sample((x.size(0),x.size(1))).to(x.device)
        perturbation_mask = perturbation_mask.unsqueeze(2)
        if self.use_running_mean:
            x = x * perturbation_mask +  self.running_mean.to(x.device) * (1.-perturbation_mask)
            if update_smoother_params:
                self.running_mean = self.running_lambda*self.running_mean + (1.-self.running_lambda) * x.mean(dim=-1).mean(dim=0).to(self.device)
        elif self.use_gmm:
            gmm_components = self.gmm.predict_utterance(x.transpose(1,2))
            noise = self.gmm.sample_batch(gmm_components,x.size(2))

            x = x * perturbation_mask + noise * (1-perturbation_mask)
        else:
            x = x * perturbation_mask

        return x
        



        
