from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, MiniBatchKMeans
from eval1.audio.transforms import MFCC, MelSpectrogram, Spectrogram, SpectrogramToDB, Compose, PNCC
from scipy.io import wavfile
import torch
import numpy as np
import argparse
import os

class AudioFeatureExtractor(object):
    # Simple 1dCNN classfier
    def __init__(self,audio_features, transform):
        super(AudioFeatureExtractor, self).__init__()
        
        
        if transform is None:
                self.transform = None 
        elif isinstance(transform,str):
            if transform=="none":
                self.transform=None 
            elif transform=="mfcc":
                self.transform = MFCC(n_mfcc=audio_features)
            elif transform =="mel":
                self.transform = MelSpectrogram(n_fft=audio_features)
            elif transform=="specdb":
                self.transform = Compose([Spectrogram(n_fft=audio_features,hop=audio_features//2), SpectrogramToDB()])
                audio_features = audio_features//2+1
            elif transform=="pncc":
                self.transform = PNCC(n_pncc=audio_features)
            else:
                raise ValueError("Unknown transform %s"%transform)
        else:
            self.transform=transform

    def get_batch_feats(self,x):
        if self.transform is not None:
            return self.transform(x).detach().cpu().numpy()
        else:
            return x.detach().cpu().numpy()

    def get_feats(self, x, batch_size=None):
        if batch_size is None:
            batch_size = len(x)
        n_batches = (len(x) + batch_size - 1) // batch_size
        feat_batches = []
        for i in range(n_batches):
            batch = x[i*batch_size:(i+1)*batch_size]
            if isinstance(batch, list):
                batch = torch.Tensor(batch)
            feats = self.get_batch_feats(batch)
            feat_batches.append(feats)
        feats = np.concatenate(feat_batches, axis=0)
        return feats

def read_wav(path):
    data = wavfile.read(path)
    return data

def train_gmm(num_components, feats):
    # resp = np.zeros((len(feats), num_components))
    # kmeans = MiniBatchKMeans(n_clusters=num_components, batch_size=256, verbose=True, n_init=1)
    # labels = kmeans.fit(feats).labels_
    # resp[np.arange(len(feats)), labels] = 1

    gmm = GaussianMixture(n_components=num_components, covariance_type='diag', init_params='kmeans', verbose=True)

    def _initialize_parameters(X, random_state):
        """Initialize the model parameters.
        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if gmm.init_params == 'kmeans':
            resp = np.zeros((n_samples, gmm.n_components))
            label = MiniBatchKMeans(n_clusters=gmm.n_components, batch_size=256, verbose=True, n_init=1,
                                   random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif gmm.init_params == 'random':
            resp = random_state.rand(n_samples, gmm.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % gmm.init_params)

        gmm._initialize(X, resp)
    
    gmm._initialize_parameters = _initialize_parameters
    # gmm._initialize(feats, resp)
    # gmm.converged_ = True
    gmm.fit(feats)
    return gmm    
