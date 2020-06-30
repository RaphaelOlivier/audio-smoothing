"""
The models in this file inherit the smoothing methods, however they do not involve random noise. 
When using them niters should be set to 1.

"""

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

from audio.defenses.base import AudioSmoother



class LowPassSmoother(AudioSmoother): # not random
    def __init__(self, p, nfeats, **kwargs):
        super(LowPassSmoother,self).__init__(**kwargs)
        self.p = p 
        self.mask = torch.zeros(nfeats).float()
        self.mask[:int(nfeats*(1.-p))] = 1
        self.mask = self.mask.view(1,-1,1)
        if self._post_feats is None:
            self._post_feats=True
        assert self.post_feats,"Frequency smoothing cannot be applied on raw inputs"

    def apply_noise(self,x,**kwargs):
        # 1 is the time dimension
        assert x.dim()==3
        perturbation = self.mask.to(x.device)
        x = x * perturbation
        return x
    
class HighPassSmoother(AudioSmoother): # not random
    def __init__(self, p, nfeats, **kwargs):
        super(HighPassSmoother,self).__init__(**kwargs)
        self.p = p 
        self.mask = torch.zeros(nfeats).float()
        self.mask[-int(nfeats*(1.-p)):] = 1
        self.mask = self.mask.view(1,-1,1)
        if self._post_feats is None:
            self._post_feats=True
        assert self.post_feats,"Frequency smoothing cannot be applied on raw inputs"

    def apply_noise(self,x,**kwargs):
        # 1 is the time dimension
        assert x.dim()==3
        perturbation = self.mask.to(x.device)
        x = x * perturbation
        return x
        
class CorrReconSmoother(AudioSmoother):
    # low pass smoother with Correlation-based features reconstruction (Raj&Stern04) 
    def __init__(self, p, nfeats,max_iteration_steps=50,convergence_tol=1e-4, running_lambda = 0.9, **kwargs):
        super(CorrReconSmoother,self).__init__(**kwargs)
        self.nfeats = nfeats
        self.n_u = int(p*nfeats) # reliable features
        self.n_r = self.nfeats - self.n_u
        self.mask = torch.zeros(nfeats).float()
        self.mask[:int(nfeats*(1.-p))] = 1
        self.mask = self.mask.view(1,-1,1)

        self.register_buffer('mu', torch.zeros(nfeats).view(1,-1,1))
        self.register_buffer('corr',torch.zeros(nfeats,nfeats))
        self.inv_corr = None 
        self.prediction_rowvects = None

        self.max_iteration_steps=max_iteration_steps
        self.convergence_tol=convergence_tol

        self.running_lambda=running_lambda
        if self._post_feats is None:
            self._post_feats=True
        assert self.post_feats,"Correlation reconstruction cannot be applied on raw inputs"

    def apply_noise(self,x,update_smoother_params=False, **kwargs):
        if update_smoother_params:
            self.update_mu_corr(x)
            self.inv_corr = None
            self.prediction_rowvects = None
        else:
            if self.inv_corr is None:
                self.inv_corr=[]
                self.prediction_rowvects = []
                logger.info("Inversing correlation matrices")
                for i in range(self.nfeats):
                    c = self.remove_row(self.corr,i)
                    c = c.t()
                    c = self.remove_row(c,i)
                    c = c.t()
                    cinv = torch.inverse(c)
                    self.inv_corr.append(cinv)
                for i in range(self.nfeats):
                    c = self.remove_row(self.corr[i].t(),i).view(1,-1)
                    self.prediction_rowvects.append(torch.matmul(c,self.inv_corr[i]))
                self.prediction_rowvects = torch.stack(self.prediction_rowvects,dim=0)
                logger.info("Done")
            x = self.reconstruct_features(x)
        return x
    
    def remove_row(self,c,i):
        if 0<i and self.nfeats-1>i:
            c = torch.cat([c[:i],c[i+1:]],dim=0)
        elif i==0:
            c = c[i+1:]
        else:
            c = c[:i]
        return c

    def update_mu_corr(self,x):
        mu = x.mean(dim=2).mean(dim=0).view(1,-1,1)
        self.mu = self.running_lambda*self.mu + (1.-self.running_lambda)*mu

        norm_x = x - self.mu
        mat = norm_x.transpose(1,2).view(-1,self.nfeats) # batch*time x nfeats
        corr = torch.bmm(mat.unsqueeze(2),mat.unsqueeze(1)).mean(dim=0) # nfeatsxnfeats
        self.corr = self.running_lambda*self.corr + (1.-self.running_lambda)*corr
        

    def reconstruct_features(self,x):
        rec_x = x
        for i in range(self.max_iteration_steps):
            t_rec_x = rec_x.transpose(0,1) # nfeats x batch x T
            v = torch.cat([self.remove_row(t_rec_x,j) for j in range(self.nfeats)],dim=1) # nfeats-1 x nfeats x batch x T
            v = v.transpose(0,1).reshape(self.nfeats,self.nfeats-1,-1) # nfeats x nfeats-1 x batch * T
            prod = torch.bmm(self.prediction_rowvects,v) # nfeats x 1 x batch*T
            prod = prod.view(self.nfeats,x.size(0),-1).transpose(0,1)
            estimate = self.mu + prod
            new_rec_x = self.mask.to(x.device) * x + (1.-self.mask.to(x.device))*estimate
            if torch.abs(new_rec_x-rec_x).max()<self.convergence_tol:
                break 
            rec_x = new_rec_x
        return new_rec_x
