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


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .wavelets import Morlet, Cauchy

__all__ = ["Transform"]
class Transform:
    """Non‑decimated wavelet transform using a pre‑built FilterBank."""

    # ------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------
    def __init__(self, data, fs, filterbank, pad_method="symmetric"):
        # ---------------- raw signal & padding ----------------
        self.fs = fs
        self.x_raw = np.asarray(data, float)
        self.N_orig = self.x_raw.size
        self.pad_method = pad_method

        # ---------------- filterbank --------------------------
        self.fb = filterbank
        self.Wfreq = self.fb.Wfreq                # (channels, fb_len)
        self.channel_freqs = self.fb.channel_freqs
        self.fb_len = self.Wfreq.shape[-1]
        self.n_channels = self.Wfreq.shape[0]

        # ---------------- global FFT length -------------------
        pow2_len = 2 ** int(np.ceil(np.log2(self.N_orig * 2)))
        self.N_fft = max(pow2_len, self.fb_len)
        self.pad_left = (self.N_fft - self.N_orig) // 2   # << we need this early

        # ---------------- build sym‑padded bank ---------------
        if self.fb_len < self.N_fft:
            pad = self.N_fft - self.fb_len
            lp, rp = pad // 2, pad - pad // 2
            Wc = np.fft.ifftshift(self.Wfreq, axes=1)      # DC → centre
            Wc = np.pad(Wc, ((0, 0), (lp, rp)))            # symmetric zeros
            self.Wfreq_full = np.fft.fftshift(Wc, axes=1)  # back to FFT order
        else:
            self.Wfreq_full = self.Wfreq.copy()

        # ---------------- phase‑align by pad_left -------------
        freqs_full = np.fft.fftfreq(self.N_fft, d=1 / self.fs)
        phase = np.exp(-1j * 2 * np.pi * freqs_full * self.pad_left)
        self.Wfreq_full *= phase[np.newaxis, :]
        self.Wfreq *= phase[: self.fb_len][np.newaxis, :]

        # ---------------- frame operator ----------------------
        self.Sfreq = np.sum(np.abs(self.Wfreq_full) ** 2, axis=0)
        self.Sfreq[self.Sfreq < 1e-12] = 1e-12

        # ---------------- pad the signal ----------------------
        self.x_padded = self._pad_and_taper(self.x_raw)

    # ------------------------------------------------------------
    # INTERNAL UTILITIES
    # ------------------------------------------------------------
    def _pad_and_taper(self, x):
        """Symmetric‑pad *x* to *N_fft* and Tukey‑taper only new edges."""
        pad_total = self.N_fft - x.size
        if pad_total <= 0:
            return x[: self.N_fft]
        left = pad_total // 2
        right = pad_total - left
        if(self.pad_method == 'constant'): 
            padded = np.pad(x, (left, right), mode=self.pad_method, constant_values=(0,0))
        else: 
            padded = np.pad(x, (left, right), mode=self.pad_method,)

        if left == 0:
            return padded
        # edge‑only taper
        window = np.ones(self.N_fft)
        taper = signal.windows.tukey(2 * left, alpha=1.0)
        window[:left] = taper[:left]
        window[-right:] = taper[-left:]
        return padded * window

    # ------------------------------------------------------------
    # FORWARD / INVERSE
    # ------------------------------------------------------------
    def forward(self, new_data=None):
        if new_data is not None:
            new_data = np.asarray(new_data, float)
            if new_data.size != self.N_orig:
                raise ValueError(
                    f"new_data length {new_data.size} ≠ original {self.N_orig}"
                )
            self.x_padded = self._pad_and_taper(new_data)

        F = np.fft.fft(self.x_padded, n=self.N_fft)
        # multiply by the full‑length bank for every channel
        coeffs = np.fft.ifft(self.Wfreq_full * F, axis=1)
        return coeffs

    def inverse(self, coeffs):
        Cf = np.fft.fft(coeffs, n=self.N_fft, axis=1)
        Xf = np.sum(np.conj(self.Wfreq_full) * Cf, axis=0) / self.Sfreq
        x_full = np.fft.ifft(Xf).real
        return x_full[self.pad_left : self.pad_left + self.N_orig]


    # ------------------------------------------------------------
    # ORTHOGONALITY SUPPORT
    # ------------------------------------------------------------
    def _apply_channel_subset(self, idx):
        self.Wfreq = self.Wfreq[idx, :]
        self.Wfreq_full = self.Wfreq_full[idx, :]
        self.channel_freqs = self.channel_freqs[idx]
        self.n_channels = len(idx)
        self.Sfreq = np.sum(np.abs(self.Wfreq_full) ** 2, axis=0)
        self.Sfreq[self.Sfreq < 1e-12] = 1e-12

    # legacy eps‑based method
    def enforce_orthogonality(self, eps=1e-5):
        """Time‑domain orthogonality pruning (original fast dot‑product method).

        Iteratively keeps the first channel, then looks for the next channel whose
        time‑domain inner product with *all kept channels* is below *eps* (per sample).
        Uses the full‑length IFFT of the filters, so it works regardless of
        zero‑padding length.
        """
        # full‑length time‑domain wavelets (complex)
        X_td = np.fft.ifft(self.Wfreq_full, axis=1)
        N = self.N_fft
        # decide if there are compensation channels
        Mc = getattr(self.fb, "Mc", 0)
        selected = [0] if Mc == 0 else [0, 1]
        current = selected[-1]
        while current < self.n_channels - 1:
            # dot product of current with all remaining channels in one go
            dots = np.dot(X_td[current], X_td[current + 1 :].conj().T) / N  # vector
            valid = np.where(np.abs(dots) < eps)[0]
            if valid.size == 0:
                break  # no further orthogonal channel
            next_idx = current + 1 + valid[0]
            selected.append(next_idx)
            current = next_idx
        self._apply_channel_subset(np.array(selected, int))

    # ------------------------------------------------------------
    # SCALOGRAM PLOT
    # ------------------------------------------------------------
    def scalogram(
        self,
        coeffs,
        *,
        cmap="viridis",
        y_tick_steps=5,
        figsize=(10, 6),
        title="Wavelet Coefficient Log‑Power",
        vmin=None,
        vmax=None,
        interpolation=None,
        align_zero=True,
    ):
        """Plot log‑power scalogram.

        Parameters
        ----------
        align_zero : bool, default True
            If *True* the coefficient matrix is circularly rolled so that
            sample index 0 of the **unpadded** signal is displayed at the
            left edge.  This compensates for the `‑N/2` phase‑shift applied
            to every filter when the bank was built.
        """
        # optional circular shift so that time‑zero is column 0
        if align_zero:
            coeffs = np.roll(coeffs, -self.pad_left, axis=1)

        # window corresponding to genuine (unpadded) data
        coeffs_view = coeffs[:, : self.N_orig]
        t_axis = np.linspace(0, self.N_orig / self.fs, coeffs_view.shape[1])

        power = np.abs(coeffs_view) ** 2

        plt.figure(figsize=figsize)
        plt.imshow(
            np.log(power),
            origin="lower",
            aspect="auto",
            extent=[t_axis[0], t_axis[-1], self.channel_freqs[0], self.channel_freqs[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        yt = np.linspace(self.channel_freqs[0], self.channel_freqs[-1], y_tick_steps)
        plt.yticks(yt, [f"{y:.2f}" for y in yt])
        plt.colorbar(label="Log Power")
        plt.tight_layout()


    def phasegram(
        self,
        coeffs,
        *,
        cmap="twilight",
        y_tick_steps=5,
        figsize=(10, 6),
        title="Unwrapped instantaneous phase",
        interpolation=None,
        align_zero=True,
    ):
        """
        Plot an unwrapped-phase scalogram (phase vs time vs scale).

        Parameters
        ----------
        coeffs : ndarray  (n_scales, n_samples_pad)
            Complex wavelet coefficients returned by Transform.forward().
        cmap : str
            Cyclic colormap (default 'twilight' shows phase nicely).
        align_zero : bool, default True
            Roll coeffs left by `pad_left` so that sample-0 of the *unpadded*
            signal appears at x = 0 s.
        """
        # 1) optional circular roll so padded left edge lines up with t=0
        if align_zero:
            coeffs = np.roll(coeffs, -self.pad_left, axis=1)

        # 2) discard the padded margins in time
        coeffs = coeffs[:, : self.N_orig]          # shape (n_scales, N_orig)
        t_axis  = np.linspace(0, self.N_orig / self.fs, coeffs.shape[1])

        # 3) unwrap instantaneous phase for each scale
        phase = np.unwrap(np.angle(coeffs), axis=1)

        # 4) subtract phase at t=0 so every row starts at 0 rad
        phase -= phase[:, [0]]

        # 5) plot
        plt.figure(figsize=figsize)
        plt.imshow(
            phase,
            origin="lower",
            aspect="auto",
            extent=[t_axis[0], t_axis[-1],
                    self.channel_freqs[0], self.channel_freqs[-1]],
            cmap=cmap,
            interpolation=interpolation,
        )
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        yt = np.linspace(self.channel_freqs[0],
                        self.channel_freqs[-1],
                        y_tick_steps)
        plt.yticks(yt, [f"{y:.2f}" for y in yt])
        cbar = plt.colorbar(label="Phase [rad]")
        plt.tight_layout()