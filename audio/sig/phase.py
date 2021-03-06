# Some algorithms for phase estimation
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)

# Change log:
#   09/07/17:
#       * Create this file
from .transform import *
import numpy as np
from numpy.fft import rfft, irfft
from util import white_noise
import pdb

CONFIG_GL = {  # Configuration parameters for Griffin-Lim algorithm
    # STFT Parameters
    'SAMPLE_RATE': 16000,  # in Hz
    'WINDOW_LENGTH': 0.032,  # in seconds
    'FFT_SIZE': 512,  # in number of samples
    'HOP_FRACTION': 0.5,  # 0.25,
    'ZERO_PHASE_STFT': True,  # use new zero-phase STFT
}


def griffin_lim(X_mag, x=None, zero_phase=False, iters=100, verbose=False):
    """
    Implement the Griffin-Lim algorithm for phase reconstruction.
    Arguments:
        X            - Magnitude STFT as a numpy 2-D array or an iterator
        [zero_phase] - Initial phase estimate. Use zero phase if True, or phase
                     of white noise if False
        [iters]      - Number of iterations. Default to 100.

    """
    if x is None:
        xlen = (X_mag.shape[0]-1)*int(CONFIG_GL['WINDOW_LENGTH'] *
                                      CONFIG_GL['SAMPLE_RATE']*CONFIG_GL['HOP_FRACTION'])
    else:
        xlen = len(x)
    # Estimate initial phase
    # Defaut zero-phase
    x_est = istft(X_mag, CONFIG_GL['SAMPLE_RATE'], xlen,
                  window_length=CONFIG_GL['WINDOW_LENGTH'],
                  hop_fraction=CONFIG_GL['HOP_FRACTION'])
    if not zero_phase:  # initial random phase from white noise
        wn = white_noise(x_est, snr=0)
        N_phase = magphase(stft(wn, CONFIG_GL['SAMPLE_RATE'],
                                window_length=CONFIG_GL['WINDOW_LENGTH'],
                                hop_fraction=CONFIG_GL['HOP_FRACTION'],
                                nfft=CONFIG_GL['FFT_SIZE'])[-1])[1]
        x_est = istft(X_mag*N_phase, CONFIG_GL['SAMPLE_RATE'], xlen,
                      window_length=CONFIG_GL['WINDOW_LENGTH'],
                      hop_fraction=CONFIG_GL['HOP_FRACTION'])

    # Now iteratively re-estimate phase
    X_energy = np.sum(X_mag**2)
    err = np.zeros(iters)
    for i in xrange(iters):
        X_mag_est, X_phase_est = magphase(stft(x_est, CONFIG_GL['SAMPLE_RATE'],
                                               window_length=CONFIG_GL['WINDOW_LENGTH'],
                                               hop_fraction=CONFIG_GL['HOP_FRACTION'],
                                               nfft=CONFIG_GL['FFT_SIZE'])[-1])

        x_est = istft(X_mag*X_phase_est, CONFIG_GL['SAMPLE_RATE'], xlen,
                      window_length=CONFIG_GL['WINDOW_LENGTH'],
                      hop_fraction=CONFIG_GL['HOP_FRACTION'])
        err[i] = np.sum((X_mag-X_mag_est)**2)/X_energy
        if verbose:
            print(err[i])

    return x_est


######### Use as standalone application ############
if __name__ == '__main__':
    import argparse
    from audio_io import audioread, audiowrite

    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', help='Iteration number (default to 100)',
                        required=False, type=int, default=100)
    parser.add_argument('-i', help='Input file', required=True)
    parser.add_argument('-o', help='Output file', required=True)
    parser.add_argument('-z', help='Enable zero-phase. Default to phase of white noise.', required=False,
                        action='store_true', default=False)
    parser.add_argument('-v', help='Verbose', required=False,
                        action='store_true', default=False)
    args = parser.parse_args()

    # Processing block
    x, sr = audioread(args.i, sr=CONFIG_GL['SAMPLE_RATE'],
                      force_mono=True, verbose=args.v)
    X_mag = magphase(stft(x, CONFIG_GL['SAMPLE_RATE'],
                          window_length=CONFIG_GL['WINDOW_LENGTH'],
                          hop_fraction=CONFIG_GL['HOP_FRACTION'],
                          nfft=CONFIG_GL['FFT_SIZE'])[-1])[0]
    x_est = griffin_lim(X_mag, x=x, zero_phase=args.z, iters=args.iter,
                        verbose=args.v)
    audiowrite(x_est, CONFIG_GL['SAMPLE_RATE'], args.o,
               normalize=True, verbose=args.v)
