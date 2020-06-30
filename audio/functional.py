"""
Copied from the torchaudio source code, commit d92de5b97
"""
import math
import torch
import numpy as np
from audio.sig.spectemp import pncc

from audio.sig.spectemp import strf
from audio.sig.window import hamming
from audio.sig.transform import stpowspec
from audio.sig.fbanks import Gammatone
from audio.sig.spectemp import pncc
from audio.sig.util import asymfilt

from scipy.signal import hilbert, lfilter
from scipy.fftpack import dct, idct

import time

import torch.nn.functional as F
# from torch_dct import dct as torch_dct
from audio.sig.pytorch_dct import dct as torch_dct

__all__ = [
    'scale',
    'pad_trim',
    'downmix_mono',
    'LC2CL',
    'spectrogram',
    'create_fb_matrix',
    'mel_scale',
    'spectrogram_to_DB',
    'create_dct',
    'MFCC',
    'BLC2CBL',
    'mu_law_encoding',
    'mu_law_expanding'
]

def asymfilt_pt(xin, la, lb, zi=None):
    r"""Asymmetric nonlinear filter in eq.4 of Kim and Stern.

    This implementation largely follows paper by Kim and Stern:
    Kim, C., & Stern, R. M. (2016).
    Power-Normalized Cepstral Coefficients (PNCC) for Robust Speech Recognition.
    IEEE/ACM Transactions on Audio Speech and Language Processing, 24(7),
    1315–1329. https://doi.org/10.1109/TASLP.2016.2545928

    """
    if zi is None:
        zi = torch.zeros(xin.size(1),device=xin.device)
    assert xin.size(1) == len(zi), "Dimension mismatch."
    def filta(qin, qout_tm1): return la * qout_tm1 + (1-la) * qin
    def filtb(qin, qout_tm1): return lb * qout_tm1 + (1-lb) * qin
    xout = []
    mask = (xin[0] >= zi).float()
    xout.append(mask*filta(xin[0], zi) + (1-mask)*filtb(xin[0],zi))

    for tt in range(1, len(xin)):
        mask = (xin[tt] >= xout[tt-1]).float()
        xout.append(mask * filta(xin[tt], xout[tt-1]) + (1-mask)*filtb(xin[tt], xout[tt-1]))
        
    xout = torch.stack(xout)

    return xout

def pncc_tensor(powerspec, medtime=2, medfreq=4, synth=False,
         vad_const=2, lambda_mu=.999, powerlaw=True, cmn=True, ccdim=13,
         tempmask=True, lambda_t=.85, mu_t=.2):
    tr_spec = powerspec.permute((1,2,0))
    """
    qtild = tr_spec.new_full(tr_spec.size(),0)
    for mm in range(len(tr_spec)):
        ms = max(0, mm-medtime)
        me = min(len(tr_spec), mm+medtime+1)
        qtild[mm] = tr_spec[ms:me].mean(dim=0)
    """
    qtild = F.avg_pool1d(powerspec.permute((0,2,1)),kernel_size=2*medtime+1,stride=1, padding=medtime,count_include_pad=False).permute((2,1,0))
    qtild_le = torch.tensor(asymfilt(qtild.detach().cpu().numpy(), .999, .5, zi=.9*qtild[0].detach().cpu().numpy()),device=qtild.device)
    #qtild_le = asymfilt_pt(qtild, .999, .5, zi=.9*qtild[0])
    qtild0 = F.relu(qtild - qtild_le)
    #qtild0[qtild0 < 0] = 0
    qtild_p = []
    qtild_p.append(qtild0[0])
    for tt in range(1, len(qtild0)):
        qtild_p.append(torch.max(lambda_t*qtild_p[-1], qtild0[tt]))
    qtild_p = torch.stack(qtild_p,dim=0)

    if tempmask:
        qtild_tm = []
        qtild_tm.append(qtild0[0].unsqueeze(0))
        #for tt in range(1, len(qtild_p)):
        #    mask = (qtild0[tt] >= (lambda_t * qtild_p[tt-1])).double()
        #    qtild_tm_new = mask * qtild0[tt] + mu_t * (1-mask)*qtild_p[tt-1]
        #    qtild_tm.append(qtild_tm_new)
        mask = (qtild0[1:] >= (lambda_t * qtild_p[:-1])).float()
        qtild_tm_new = mask * qtild0[1:] + mu_t * (1-mask)*qtild_p[:-1]
        qtild_tm.append(qtild_tm_new)
        qtild_tm = torch.cat(qtild_tm,dim=0)
    else:
        qtild_tm = 0
    

    qtild_f = torch.tensor(asymfilt(qtild0.detach().cpu().numpy(), .999, .5, zi=.9*qtild0[0].detach().cpu().numpy()),device=qtild0.device)
    #qtild_f = asymfilt_pt(qtild0, .999, .5, zi=.9*qtild0[0])
    qtild1 = torch.max(qtild_tm, qtild_f)

    excitation = (qtild >= vad_const*qtild_le).float()

    # C-D. Compare noise modeling and temporal masking
    rtild = excitation*qtild1+(1-excitation)*qtild_f
    #rtild[excitation] = qtild1[excitation]
    #rtild[~excitation] = qtild_f[~excitation]

    """
    stild = qtild.new_full(qtild.size(),0)
    for kk in range(stild.size(1)):
        ks, ke = max(0, kk-medfreq), min(stild.size(1), kk+medfreq+1)
        stild[:, kk] = (rtild[:, ks:ke] / qtild[:, ks:ke]).mean(dim=1)
    """
    stild = F.avg_pool1d(rtild.permute((0,2,1))/qtild.permute((0,2,1)),kernel_size=2*medfreq+1,stride=1, padding=medfreq,count_include_pad=False).permute((0,2,1))
    out0 = tr_spec * stild  # this is T[m,l] in eq.14

    meanpower = out0.mean(dim=1)  # T[m]
    mu= torch.tensor(lfilter([1-lambda_mu], [1, -lambda_mu], meanpower.detach().cpu().numpy(),
                    zi=[meanpower.mean(dim=0).detach().cpu().numpy()],axis=0)[0],device=meanpower.device).float()

    out0 = out0 / mu.unsqueeze(1)  # U[m,l] in eq.16, ignoring the k constant
    # G. Rate-level nonlinearity
    if powerlaw:
        out0 = out0 ** (1/15)
    else:
        out0 = (out0 + 1e-8).log()

    """
    out = torch.tensor(dct(out0.detach().cpu().numpy(), norm='ortho',axis=1),device=out0.device)[:, :ccdim]
    if cmn:
        out = out - out.mean(dim=0)
    out = out.permute((2,0,1))
    """

    out = torch_dct(out0.permute(2,0,1),norm="ortho").permute(2,0,1)[:ccdim]
    if cmn:
        out = out - out.mean(dim=2,keepdim=True)
    out = out.permute((1,2,0))
    return out



def pncc_batch(powerspec, medtime=2, medfreq=4, synth=False,
         vad_const=2, lambda_mu=.999, powerlaw=True, cmn=True, ccdim=13,
         tempmask=True, lambda_t=.85, mu_t=.2):
    """Power-Normalized Cepstral Coefficients (PNCC).

    This implementation largely follows paper by Kim and Stern:
    Kim, C., & Stern, R. M. (2016).
    Power-Normalized Cepstral Coefficients (PNCC) for Robust Speech
    Recognition. IEEE/ACM Transactions on Audio Speech and Language Processing,
    24(7), 1315–1329. https://doi.org/10.1109/TASLP.2016.2545928

    Parameters
    ----------

    See Also
    --------
    fbank.Gammatone

    """
    #print(powerspec)
    tr_spec = np.transpose(powerspec,(1,2,0))
    # B. Calculate median-time power
    qtild = np.empty_like(tr_spec)
    for mm in range(len(tr_spec)):
        ms = max(0, mm-medtime)
        me = min(len(tr_spec), mm+medtime+1)
        qtild[mm] = tr_spec[ms:me].mean(axis=0)
    # C. Calculate noise floor
    qtild_le = asymfilt(qtild, .999, .5, zi=.9*qtild[0])
    #print(qtild_le.shape)
    qtild0 = qtild - qtild_le
    qtild0[qtild0 < 0] = 0

    # D. Model temporal masking
    qtild_p = np.empty_like(qtild0)
    qtild_p[0] = qtild0[0]
    for tt in range(1, len(qtild_p)):
        qtild_p[tt] = np.maximum(lambda_t*qtild_p[tt-1], qtild0[tt])

    if tempmask:
        qtild_tm = np.empty_like(qtild0)
        qtild_tm[0] = qtild0[0]
        for tt in range(1, len(qtild_p)):
            mask = qtild0[tt] >= (lambda_t * qtild_p[tt-1])
            qtild_tm[tt, mask] = qtild0[tt, mask]
            qtild_tm[tt, ~mask] = mu_t * qtild_p[tt-1, ~mask]
    else:
        qtild_tm = 0

    # C-D. Track floor of noise floor
    qtild_f = asymfilt(qtild0, .999, .5, zi=.9*qtild0[0])
    qtild1 = np.maximum(qtild_tm, qtild_f)

    # C-D. Excitation segment vs. non-excitation segment
    excitation = qtild >= vad_const*qtild_le

    # C-D. Compare noise modeling and temporal masking
    rtild = np.empty_like(qtild)
    rtild[excitation] = qtild1[excitation]
    rtild[~excitation] = qtild_f[~excitation]

    # E. Spectral weight smoothing
    stild = np.empty_like(qtild)
    for kk in range(stild.shape[1]):
        ks, ke = max(0, kk-medfreq), min(stild.shape[1], kk+medfreq+1)
        stild[:, kk] = (rtild[:, ks:ke] / qtild[:, ks:ke]).mean(axis=1)
    out0 = tr_spec * stild  # this is T[m,l] in eq.14
    #print(tr_spec.shape,stild.shape,out0.shape)
    # F. Mean power normalization
    meanpower = out0.mean(axis=1)  # T[m]
    mu, _ = lfilter([1-lambda_mu], [1, -lambda_mu], meanpower,
                    zi=[meanpower.mean(axis=0)],axis=0)

    out0 /= mu[:, np.newaxis]  # U[m,l] in eq.16, ignoring the k constant
    # G. Rate-level nonlinearity
    if powerlaw:
        out0 = out0 ** (1/15)
    else:
        out0 = np.log(out0 + 1e-8)

    # Finally, apply CMN if needed
    out = dct(out0, norm='ortho',axis=1)[:, :ccdim]
    if cmn:
        out -= out.mean(axis=0)
    out = out.transpose(2,0,1)

    def compare_to_last(a,b, tr=True):
        if tr:
            a = np.transpose(a,(-1,*[i for i in range(len(b.shape))]))
        assert a[-1].shape==b.shape
        print(np.abs(a[-1]).max(),np.abs(b).max(),np.abs(a[-1]-b).max())
    #compare_to_last(qtild,qtild2)
    #compare_to_last(qtild_le,qtild_le2)
    #compare_to_last(qtild0,qtild02)
    #compare_to_last(qtild,qtild2)
    #compare_to_last(qtild_p,qtild_p2)
    #compare_to_last(qtild_tm,qtild_tm2)
    #compare_to_last(qtild_f,qtild_f2)
    #compare_to_last(qtild1,qtild12)
    #compare_to_last(excitation,excitation2)
    #compare_to_last(rtild,rtild2)
    #compare_to_last(stild,stild2)
    #compare_to_last(meanpower,meanpower2)
    #compare_to_last(mu,mu2)
    #compare_to_last(out0,out02)
    #compare_to_last(out,out2,tr=False)
    return out




def scale(tensor, factor):
    # type: (Tensor, int) -> Tensor
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".
    Inputs:
        tensor (Tensor): Tensor of audio of size (Samples x Channels)
        factor (int): Maximum value of input tensor
    Outputs:
        Tensor: Scaled by the scale factor
    """
    if not tensor.dtype.is_floating_point:
        tensor = tensor.to(torch.float32)

    return tensor / factor


def pad_trim(tensor, ch_dim, max_len, len_dim, fill_value):
    # type: (Tensor, int, int, int, float) -> Tensor
    """Pad/Trim a 2d-Tensor (Signal or Labels)
    Inputs:
        tensor (Tensor): Tensor of audio of size (n x c) or (c x n)
        ch_dim (int): Dimension of channel (not size)
        max_len (int): Length to which the tensor will be padded
        len_dim (int): Dimension of length (not size)
        fill_value (float): Value to fill in
    Outputs:
        Tensor: Padded/trimmed tensor
    """
    if max_len > tensor.size(len_dim):
        # tuple of (padding_left, padding_right, padding_top, padding_bottom)
        # so pad similar to append (aka only right/bottom) and do not pad
        # the length dimension. assumes equal sizes of padding.
        padding = [max_len - tensor.size(len_dim)
                   if (i % 2 == 1) and (i // 2 != len_dim)
                   else 0
                   for i in range(4)]
        with torch.no_grad():
            tensor = torch.nn.functional.pad(tensor, padding, "constant", fill_value)
    elif max_len < tensor.size(len_dim):
        tensor = tensor.narrow(len_dim, 0, max_len)
    return tensor


def downmix_mono(tensor, ch_dim):
    # type: (Tensor, int) -> Tensor
    """Downmix any stereo signals to mono.
    Inputs:
        tensor (Tensor): Tensor of audio of size (c x n) or (n x c)
        ch_dim (int): Dimension of channel (not size)
    Outputs:
        Tensor: Mono signal
    """
    if not tensor.dtype.is_floating_point:
        tensor = tensor.to(torch.float32)

    tensor = torch.mean(tensor, ch_dim, True)
    return tensor


def LC2CL(tensor):
    # type: (Tensor) -> Tensor
    """Permute a 2d tensor from samples (n x c) to (c x n)
    Inputs:
        tensor (Tensor): Tensor of audio signal with shape (LxC)
    Outputs:
        Tensor: Tensor of audio signal with shape (CxL)
    """
    return tensor.transpose(0, 1).contiguous()


def spectrogram(sig, pad, window, n_fft, hop, ws, power, normalize):
    # type: (Tensor, int, Tensor, int, int, int, int, bool) -> Tensor
    """Create a spectrogram from a raw audio signal
    Inputs:
        sig (Tensor): Tensor of audio of size (c, n)
        pad (int): two sided padding of signal
        window (Tensor): window_tensor
        n_fft (int): size of fft
        hop (int): length of hop between STFT windows
        ws (int): window size
        power (int > 0 ) : Exponent for the magnitude spectrogram,
                        e.g., 1 for energy, 2 for power, etc.
        normalize (bool) : whether to normalize by magnitude after stft
    Outputs:
        Tensor: channels x hops x n_fft (c, l, f), where channels
            is unchanged, hops is the number of hops, and n_fft is the
            number of fourier bins, which should be the window size divided
            by 2 plus 1.
    """
    assert sig.dim() == 2

    if pad > 0:
        with torch.no_grad():
            sig = torch.nn.functional.pad(sig, (pad, pad), "constant")
    window = window.to(sig.device)

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(sig, n_fft, hop, ws,
                        window, center=True,
                        normalized=False, onesided=True,
                        pad_mode='reflect').transpose(1, 2)
    if normalize:
        spec_f /= window.pow(2).sum().sqrt()
    spec_f = spec_f.pow(power).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
    return spec_f


def create_fb_matrix(n_stft, f_min, f_max, n_mels):
    # type: (int, float, float, int) -> Tensor
    """ Create a frequency bin conversion matrix.
    Inputs:
        n_stft (int): number of filter banks from spectrogram
        f_min (float): minimum frequency
        f_max (float): maximum frequency
        n_mels (int): number of mel bins
    Outputs:
        Tensor: triangular filter banks (fb matrix)
    """
    def _hertz_to_mel(f):
        # type: (float) -> Tensor
        return 2595. * torch.log10(torch.tensor(1.) + (f / 700.))

    def _mel_to_hertz(mel):
        # type: (Tensor) -> Tensor
        return 700. * (10**(mel / 2595.) - 1.)

    # get stft freq bins
    stft_freqs = torch.linspace(f_min, f_max, n_stft)
    # calculate mel freq bins
    m_min = 0. if f_min == 0 else _hertz_to_mel(f_min)
    m_max = _hertz_to_mel(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hertz(m_pts)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)  # (n_stft, n_mels + 2)
    # create overlapping triangles
    z = torch.tensor(0.)
    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_stft, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_stft, n_mels)
    fb = torch.max(z, torch.min(down_slopes, up_slopes))
    return fb


def mel_scale(spec_f, f_min, f_max, n_mels, fb=None):
    # type: (Tensor, float, float, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
    """ This turns a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.
    Inputs:
        spec_f (Tensor): normal STFT
        f_min (float): minimum frequency
        f_max (float): maximum frequency
        n_mels (int): number of mel bins
        fb (Optional[Tensor]): triangular filter banks (fb matrix)
    Outputs:
        Tuple[Tensor, Tensor]: triangular filter banks (fb matrix) and mel frequency STFT
    """
    if fb is None:
        fb = create_fb_matrix(spec_f.size(2), f_min, f_max, n_mels).to(spec_f.device)
    else:
        # need to ensure same device for dot product
        fb = fb.to(spec_f.device)
    spec_m = torch.matmul(spec_f, fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
    return fb, spec_m


def spectrogram_to_DB(spec, multiplier, amin, db_multiplier, top_db=None):
    # type: (Tensor, float, float, float, Optional[float]) -> Tensor
    """Turns a spectrogram from the power/amplitude scale to the decibel scale.
    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.
    Inputs:
        spec (Tensor): normal STFT
        multiplier (float): use 10. for power and 20. for amplitude
        amin (float): number to clamp spec
        db_multiplier (float): log10(max(reference value and amin))
        top_db (Optional[float]): minimum negative cut-off in decibels.  A reasonable number
            is 80.
    Outputs:
        Tensor: spectrogram in DB
    """
    spec_db = multiplier * torch.log10(torch.clamp(spec, min=amin))
    spec_db -= multiplier * db_multiplier

    if top_db is not None:
        spec_db = torch.max(spec_db, spec_db.new_full((1,), spec_db.max().item() - top_db))
    return spec_db


def create_dct(n_mfcc, n_mels, norm):
    # type: (int, int, string) -> Tensor
    """
    Creates a DCT transformation matrix with shape (num_mels, num_mfcc),
    normalized depending on norm
    Inputs:
        n_mfcc (int) : number of mfc coefficients to retain
        n_mels (int): number of MEL bins
        norm (string) : norm to use
    Outputs:
        Tensor: The transformation matrix, to be right-multiplied to row-wise data.
    """
    outdim = n_mfcc
    dim = n_mels
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(dim, dtype=torch.get_default_dtype())
    k = torch.arange(outdim, dtype=torch.get_default_dtype())[:, None]
    dct = torch.cos(math.pi / dim * (n + 0.5) * k)
    if norm == 'ortho':
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / dim)
    else:
        dct *= 2
    return dct.t()


def MFCC(sig, mel_spect, log_mels, s2db, dct_mat):
    # type: (Tensor, MelSpectrogram, bool, SpectrogramToDB, Tensor) -> Tensor
    """Create the Mel-frequency cepstrum coefficients from an audio signal
    By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.
    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.
    Inputs:
        sig (Tensor): Tensor of audio of size (channels [c], samples [n])
        mel_spect (MelSpectrogram): melspectrogram of sig
        log_mels (bool): whether to use log-mel spectrograms instead of db-scaled
        s2db (SpectrogramToDB): a SpectrogramToDB instance
        dct_mat (Tensor): The transformation matrix (dct matrix), to be
            right-multiplied to row-wise data
    Outputs:
        Tensor: Mel-frequency cepstrum coefficients
    """
    if log_mels:
        log_offset = 1e-6
        mel_spect = torch.log(mel_spect + log_offset)
    else:
        mel_spect = s2db(mel_spect)
    mfcc = torch.matmul(mel_spect, dct_mat.to(mel_spect.device))
    return mfcc

def PNCC(sig, log_mels, s2db, n_pncc):
    # type: (Tensor, MelSpectrogram, bool, SpectrogramToDB, Tensor) -> Tensor
    #np_spect = mel_spect.squeeze(0).detach().cpu().numpy()

    #np_pncc = pncc_copy(np_spect, ccdim=n_pncc)

    #pt_pncc = torch.tensor(np_pncc,device = sig.device).unsqueeze(0)
    sr =16000
    #np_sig = sig.detach().cpu().numpy()
    wlen = .025
    hop = .01
    nfft = 400
    wind_np = hamming(int(wlen*sr))
    wind = torch.tensor(wind_np,device=sig.device).float()
    #powerspec2 = torch.tensor(np.stack([stpowspec(utt, sr, wind_np, int(hop*sr), nfft, synth=False)[:-1] for utt in np_sig],axis=0),device = sig.device).float()
    spec = torch.stft(sig.float(), n_fft=nfft, hop_length=int(hop*sr), win_length=int(wlen*sr),window=wind).permute((3,0,2,1))
    powerspec = spec[0]**2 + spec[1]**2
    gtbank = Gammatone(sr, 40)

    wts = torch.tensor(gtbank.gammawgt(nfft, powernorm=True, squared=True),device = sig.device).float()
    gammaspec = powerspec @ wts

    min_pow = 1e-8
    limit_spec = torch.clamp(gammaspec,min = min_pow)

    #coef = pncc_batch(limit_spec, tempmask=True, ccdim = n_pncc)
    #coef_compare = np.stack([pncc(utt, tempmask=True, ccdim = n_pncc) for utt in limit_spec])
    #pt_coeff2 = torch.tensor(coef,device = sig.device).float()
    pt_coeff = pncc_tensor(limit_spec, tempmask=True, ccdim = n_pncc)
    #print(powerspec.size(),powerspec.abs().mean())
    #print(powerspec2.size(),powerspec2.abs().mean())
    #print((powerspec-powerspec2).size(),(powerspec-powerspec2).abs().mean())
    return pt_coeff

def BLC2CBL(tensor):
    # type: (Tensor) -> Tensor
    """Permute a 3d tensor from Bands x Sample length x Channels to Channels x
       Bands x Samples length
    Inputs:
        tensor (Tensor): Tensor of spectrogram with shape (BxLxC)
    Outputs:
        Tensor: Tensor of spectrogram with shape (CxBxL)
    """
    return tensor.permute(2, 0, 1).contiguous()


def mu_law_encoding(x, qc):
    # type: (Tensor, int) -> Tensor
    """Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1
    Inputs:
        x (Tensor): Input tensor
        qc (int): Number of channels (i.e. quantization channels)
    Outputs:
        Tensor: Input after mu-law companding
    """
    assert isinstance(x, torch.Tensor), 'mu_law_encoding expects a Tensor'
    mu = qc - 1.
    if not x.dtype.is_floating_point:
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype)
    x_mu = torch.sign(x) * torch.log1p(mu *
                                       torch.abs(x)) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu


def mu_law_expanding(x_mu, qc):
    # type: (Tensor, int) -> Tensor
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.
    Inputs:
        x_mu (Tensor): Input tensor
        qc (int): Number of channels (i.e. quantization channels)
    Outputs:
        Tensor: Input after decoding
    """
    assert isinstance(x_mu, torch.Tensor), 'mu_law_expanding expects a Tensor'
    mu = qc - 1.
    if not x_mu.dtype.is_floating_point:
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype)
    x = ((x_mu) / mu) * 2 - 1.
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
    return x

def lfilter(waveform,a_coeffs,b_coeffs):
    """Perform an IIR filter by evaluating difference equation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`.  Must be normalized to -1 to 1.
        a_coeffs (Tensor): denominator coefficients of difference equation of dimension of `(n_order + 1)`.
                                Lower delays coefficients are first, e.g. `[a0, a1, a2, ...]`.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Tensor): numerator coefficients of difference equation of dimension of `(n_order + 1)`.
                                 Lower delays coefficients are first, e.g. `[b0, b1, b2, ...]`.
                                 Must be same size as a_coeffs (pad with 0's as necessary).

    Returns:
        Tensor: Waveform with dimension of `(..., time)`.  Output will be clipped to -1 to 1.
    """
    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    assert (a_coeffs.size(0) == b_coeffs.size(0))
    assert (len(waveform.size()) == 2)
    assert (waveform.device == a_coeffs.device)
    assert (b_coeffs.device == a_coeffs.device)

    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(0)
    n_sample_padded = n_sample + n_order - 1
    assert (n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)
    padded_waveform[:, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(0)
    b_coeffs_flipped = b_coeffs.flip(0)

    # calculate windowed_input_signal in parallel
    # create indices of original with shape (n_channel, n_order, n_sample)
    window_idxs = torch.arange(n_sample, device=device).unsqueeze(0) + torch.arange(n_order, device=device).unsqueeze(1)
    window_idxs = window_idxs.repeat(n_channel, 1, 1)
    window_idxs += (torch.arange(n_channel, device=device).unsqueeze(-1).unsqueeze(-1) * n_sample_padded)
    window_idxs = window_idxs.long()
    # (n_order, ) matmul (n_channel, n_order, n_sample) -> (n_channel, n_sample)
    input_signal_windows = torch.matmul(b_coeffs_flipped, torch.take(padded_waveform, window_idxs))

    for i_sample, o0 in enumerate(input_signal_windows.t()):
        windowed_output_signal = padded_output_waveform[:, i_sample:(i_sample + n_order)]
        o0.sub_(torch.mv(windowed_output_signal, a_coeffs_flipped))
        o0.div_(a_coeffs[0])

        padded_output_waveform[:, i_sample + n_order - 1] = o0

    output = torch.clamp(padded_output_waveform[:, (n_order - 1):], min=-1., max=1.)

    # unpack batch
    output = output.view(shape[:-1] + output.shape[-1:])

    return output

def biquad(waveform,b0: float,b1: float,b2: float,a0: float,a1: float,a2: float):
    """Perform a biquad filter of input tensor.  Initial conditions set to 0.
    https://en.wikipedia.org/wiki/Digital_biquad_filter

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        b0 (float): numerator coefficient of current input, x[n]
        b1 (float): numerator coefficient of input one time step ago x[n-1]
        b2 (float): numerator coefficient of input two time steps ago x[n-2]
        a0 (float): denominator coefficient of current output y[n], typically 1
        a1 (float): denominator coefficient of current output y[n-1]
        a2 (float): denominator coefficient of current output y[n-2]

    Returns:
        Tensor: Waveform with dimension of `(..., time)`
    """

    device = waveform.device
    dtype = waveform.dtype

    output_waveform = lfilter(
        waveform,
        torch.tensor([a0, a1, a2], dtype=dtype, device=device),
        torch.tensor([b0, b1, b2], dtype=dtype, device=device)
    )
    return output_waveform

def lowpass_biquad(waveform,sample_rate: int,cutoff_freq: float,Q: float = 0.707):
    """Design biquad lowpass filter and perform filtering.  Similar to SoX implementation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`
    """
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)