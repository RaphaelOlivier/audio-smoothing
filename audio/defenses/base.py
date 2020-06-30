

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

from audio.gmm import gmm
#from statsmodels.stats.proportion import proportion_confint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioSmoother(nn.Module):
    def __init__(self, noise_raw, apply_fit=True, apply_predict=False):
        super(AudioSmoother,self).__init__()
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        if noise_raw==1:
            self._post_feats=False
        elif noise_raw==0:
            self._post_feats=True
        else:
            assert noise_raw==-1
            self._post_feats=None


    @property
    def post_feats(self):
        return self._post_feats
    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def estimate_gradient(self, x, grad):
        raise grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def __call__(self,x, post_feats,*args,**kwargs):
        #logger.debug("Calling smoother, noise %f"%self.sigma)
        if post_feats==self.post_feats:
            if isinstance(x, torch.Tensor):
                x = self.apply_noise(x,**kwargs)
            elif isinstance(x,PackedSequence):
                x,l = pad_packed_sequence(x,)
                x = self.apply_noise(x,**kwargs)
                x = pack_padded_sequence(x,l)
            elif isinstance(x[0], torch.Tensor):
                for i in range(len(x)):
                    x[i] = self.apply_noise(x[i],**kwargs)
            else:
                raise TypeError("AudioSmoother can only take as input tensors or iterables of tensors")
        return x

    def apply_noise(self,x,**kwargs):
        raise NotImplementedError

class NoSmoother(AudioSmoother):
    def apply_noise(self,x,**kwargs):
        return x


class ComposeSmoother(AudioSmoother):
    def __init__(self,*args, **kwargs):
        super(ComposeSmoother,self).__init__(**kwargs)
        self.subsmoothers=[]
        for s in args:
            self.subsmoothers.append(s)
        for i in range(1,len(self.subsmoothers)):
            assert self.subsmoothers[i].post_feats>self.subsmoothers[i-1].post_feats # post-feats are after pre-feats
    def __call__(self,x,*args,**kwargs):
        for s in self.subsmoothers:
            x = s(x,*args,**kwargs)
        return x

        
