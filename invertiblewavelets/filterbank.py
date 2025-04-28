"""
Author: Alex Hoffmann
Last Update: 01/27/2025
Description: This module provides the `Transform` class for performing non-decimated wavelet transforms on time-series data. 
             It supports both forward and inverse transformations using customizable wavelet functions (e.g., Morlet, Cauchy). 
             The class allows configuration of various parameters such as scaling methods (linear or dyadic), padding strategies,
             and scale resolution. Additionally, it includes functionality to visualize the power scalogram of the wavelet coefficients.

Usage Example:
    from invertiblewavelets import Transform
    transform = Transform(data, fs, wavelet=Cauchy(), scales='linear')
    coeffs = transform.forward()
    reconstructed_data = transform.inverse(coeffs)
    transform.power_scalogram(coeffs)
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy import signal


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
    def _init_params(self, b=None, q=None, M=None, compensation=True, **_):
        # Initialize and compute defaults for linear-scale parameters (Holighaus et al.)
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

        # Scale resolution q
        if self.q is None:
            self.q = 5 * self.b

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
                wtime = (1.0 / np.sqrt(self.b)) * np.sinc(t / self.b * 2) * \
                        signal.windows.tukey(self.N, alpha=0.3)
                wtime = signal.hilbert(wtime)
            if self.real:
                wtime = wtime.real
            W[i, :] = np.fft.fft(wtime)
            ch_freqs[i] = real_freqs[np.argmax(np.abs(W[i, :self.N//2]))]

        # Normalize first channel to match second
        W[0] *= np.max(np.abs(W[1])) / np.max(np.abs(W[0]))
        self.Wfreq = W
        return self.Wfreq, ch_freqs


class DyadicFilterBank(FilterBank):
    """
    Generates a dyadic (base-2) scale pyramid robustly.

    Parameters
    ----------
    dj : float
        Spacing between adjacent scales in octaves (normally 1/4).
    s_max : float or None
        Largest scale.  If None, it is set so that the widest wavelet
        still fits into the effective signal duration with 10 % margin.
    """

    def _init_params(self, dj=1/4, s_max=None, compensation=True, **_):
        self.dj = float(dj)
        self.compensation = bool(compensation)
        self.s_max = s_max

        if self.s_max is None:                   # user did not specify
            T_sig   = self.N / self.fs
            margin  = 0.25 * T_sig           # keep 10 % head-room
            T_wave  = self.wavelet.effective_half_width()
            self.s_max  = max(1.0, (T_sig - margin) / (2*T_wave))

    # Dyadic: channels are just the integer indices of the scale list
    def _define_channel_indices(self):
        return None  # handled implicitly

    # ------------------------------------------------------------------
    #  robust helper to get wavelet center frequency
    # ------------------------------------------------------------------
    def _guess_central_freq(self):
        """Return positive center frequency (Hz) of the prototype wavelet."""
        # If the wavelet object exposes it, use that first.
        if hasattr(self.wavelet, "central_frequency"):
            cf = float(self.wavelet.central_frequency)
            if cf > 0:
                return cf

        # Otherwise do a small FFT on an oversampled wavelet.
        oversamp = 8
        L = 1024
        t = np.arange(L * oversamp) / (self.fs * oversamp)
        w = self.wavelet.eval_analysis(t).real
        W = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(W.size * 2 - 2, d=1 / (self.fs * oversamp))

        mag = np.abs(W)
        if mag.max() == 0:
            return self.fs / 4  # fallback (quarter Nyquist)

        peak = np.argmax(mag)
        return abs(freqs[peak])

    # --------------------------------------------------------------
    def _compute_filters(self):
        # smallest scale so peak is at Nyquist
        fc = self._guess_central_freq()
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
                # sinc compensation below lowest dyadic channel
                w_t = (1 / np.sqrt(scales[1])) * np.sinc(t / scales[1] * 2)
                w_t *= signal.windows.tukey(self.N, alpha=0.3)
                w_t = signal.hilbert(w_t)

            else:
                w_t = np.sqrt(s) * self.wavelet.eval_analysis(t / s)

            if self.real:
                w_t = w_t.real
            W[i] = np.fft.fft(w_t)
            ch_freqs[i] = freqs[np.argmax(np.abs(W[i, : self.N // 2]))]

        self.Wfreq = W
        return self.Wfreq, ch_freqs

