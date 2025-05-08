# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   █ █ █ █ █   Invertible-Wavelets Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  filterbank.py                                               ║
# ║  Package      :  invertiblewavelets                                          ║
# ║  Author       :  Dr. Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>            ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-04-30                                                  ║
# ║  Last Updated :  2025-05-07                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  :                                                              ║
# ║      Filter-bank generators (linear & dyadic) for the invertible-wavelets    ║
# ║      framework.  Provides base 'FilterBank' ABC plus concrete                ║
# ║      `LinearFilterBank` and `DyadicFilterBank` implementations.              ║
# ║      This template can also be used to make Matched Filters. Enjoy!          ║
# ║                                                                              ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from abc import ABC, abstractmethod
import numpy as np
from scipy import signal

__all__ = [
    "FilterBank",
    "LinearFilterBank",
    "DyadicFilterBank",
]

class FilterBank(ABC):
    """
    Abstract base for any wavelet filterbank.
    Subclasses must implement:
      - _init_params(): to parse and set parameters
      - _define_channel_indices(): to return channel indices or scales
      - _compute_filters(): to produce Wfreq and channel_freqs
    Provides utility compute_effective_support for default parameter computation.
    """
    def __init__(self, wavelet, fs, N, real=False, **params):
        self.wavelet = wavelet
        self.fs = fs
        self.N = N

        "Make N even for FFT"
        if N % 2 == 1:
            N += 1
            self.N = N

        self.real = real
        self._init_params(**params)
        self.j_channels = self._define_channel_indices()
        self.Wfreq, self.channel_freqs = self._compute_filters()
        self.Wtime = np.fft.ifft(self.Wfreq, axis=1)

    @abstractmethod
    def _init_params(self, **params):
        pass

    @abstractmethod
    def _define_channel_indices(self):
        pass

    @abstractmethod
    def _compute_filters(self):
        pass

class LinearFilterBank(FilterBank):
    """
    Generates a linear scales.

    Parameters
    ----------
    b : float
        Maximum scale.
    q : float or None
        Scale resolution. Must be a factor of b.
    M : float or None
        Number of scales
    """
    def _init_params(self, b=None, q=None, M=None, compensation=False, **_):
        # Initialize and compute defaults for linear-scale parameters (Similar to Holighaus et al., 2023)
        self.b = b
        self.q = q
        self.M = M
        self.compensation = bool(compensation)
        self.Mc = 0
        if self.compensation: self.Mc = 1

        # time vector centered
        t = np.arange(self.N) / self.fs
        t -= np.mean(t)

        # Largest scale b
        if self.b is None:                   # user did not specify
            T_sig   = self.N / self.fs
            margin  = 0.25 * T_sig           # keep 10 % head-room
            T_wave  = self.wavelet.effective_half_width()
            self.b  = max(1.0, (T_sig - margin) / (2*T_wave))

        # Scale resolution q = n*b
        if self.q is None:
            if self.M is None:
                self.q = 5 * self.b
            else: 
                nyq_gap = self.fs / 2 - 1 / self.b
                self.q = max((self.M - 1) / nyq_gap, b)

        # Number of scales M
        if self.M is None:
            self.M = int(self.q * (self.fs / 2 - 1 / self.b)) + 1
            if self.M < 1:
                self.M = 4

    def _define_channel_indices(self):
        return np.arange(-self.Mc, self.M)

    def _compute_filters(self):
        real_freqs = np.fft.fftfreq(self.N, 1/self.fs)
        W = np.zeros((len(self.j_channels), self.N), dtype=complex)
        ch_freqs = np.zeros(len(self.j_channels))
        t = np.arange(self.N) / self.fs
        t -= np.mean(t)

        for i, j in enumerate(self.j_channels):
            if j >= 0:
                alpha = (1.0 / self.b) + (j / self.q)
                wtime = np.sqrt(alpha) * self.wavelet.eval_analysis(alpha * t)
            else:
                continue

            if self.real:
                wtime = wtime.real

            W[i, :] = np.fft.fft(wtime)
            ch_freqs[i] = real_freqs[np.argmax(np.abs(W[i, :self.N//2]))]

        if self.compensation:
            psum = np.sum(np.abs(W[1:])**2, axis=0)
            comp_mag = np.sqrt(np.clip(1.0 - psum, 0.0, None))
            comp_t = np.fft.ifft(comp_mag)
            comp_t = np.fft.fftshift(comp_t)
            comp_t = signal.hilbert(comp_t.real)
            if self.real:
                comp_t = comp_t.real
            W[0] = np.fft.fft(comp_t)

        # Normalize by energy in scale
        energy = np.sum(np.abs(W)**2, axis=1)    
        W = W/np.sqrt(energy)[:,None]

        self.Wfreq = W

        return self.Wfreq, ch_freqs

class DyadicFilterBank(FilterBank):
    """
    Generates a dyadic (base-2) scales.

    Parameters
    ----------
    dj : float
        Spacing between adjacent scales in octaves (normally 1/4).
    s_max : float or None
        Largest scale.  If None, it is set so that the widest wavelet
        still fits into the effective signal duration with 10 % margin.
    """

    def _init_params(self, dj=1/4, s_max=None, compensation=False, **_):
        self.dj = float(dj)
        self.compensation = bool(compensation)
        self.s_max = s_max

        if self.s_max is None:                   # user did not specify
            T_sig   = self.N / self.fs
            margin  = 0.5 * T_sig           # keep 10 % head-room
            T_wave  = self.wavelet.effective_half_width()
            self.s_max  = max(1.0, (T_sig - margin) / (2*T_wave))

    # Dyadic: channels are just the integer indices of the scale list
    def _define_channel_indices(self):
        return None  # handled implicitly


    # --------------------------------------------------------------
    def _compute_filters(self):
        # smallest scale so peak is at Nyquist
        fc = self.wavelet.fc
        s_min = 2.0 * fc / self.fs
        if s_min <= 0:
            s_min = 1e-6

        # dyadic scale vector high→low frequency
        if self.s_max <= s_min:
            self.s_max = s_min * 2
        J = int(np.floor(np.log2(self.s_max / s_min) / self.dj))
        scales = s_min * 2.0 ** (self.dj * np.arange(J + 1))
        scales = scales[::-1]                       # large→small  (low→high f)
        
        # + compensation channel?
        if self.compensation:
            scales = np.concatenate(([scales[0] * 2], scales))  # dummy slot
        
        W = np.zeros((len(scales), self.N), complex)
        ch_freqs = np.zeros(len(scales))
        freqs = np.fft.fftfreq(self.N, d=1 / self.fs)
        t = (np.arange(self.N) / self.fs) - (self.N / (2 * self.fs))

        for i, s in enumerate(scales):
            if self.compensation and i == 0:
                continue
            else:
                w_t = np.sqrt(s) * self.wavelet.eval_analysis(t / s)

            if self.real:
                w_t = w_t.real
                
            W[i] = np.fft.fft(w_t)
            ch_freqs[i] = freqs[np.argmax(np.abs(W[i, : self.N // 2]))]

        if self.compensation:
            psum = np.sum(np.abs(W[1:])**2, axis=0)
            comp_mag = np.sqrt(np.clip(1.0 - psum, 0.0, None))
            comp_t = np.fft.ifft(comp_mag)
            comp_t = np.fft.fftshift(comp_t)
            comp_t = signal.hilbert(comp_t.real)
            if self.real:
                comp_t = comp_t.real
            W[0] = np.fft.fft(comp_t)

        # Normalize by energy in scale
        energy = np.sum(np.abs(W)**2, axis=1)    
        W = W/np.sqrt(energy)[:,None]

        self.Wfreq = W
        return self.Wfreq, ch_freqs

