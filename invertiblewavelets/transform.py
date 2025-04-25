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

    # ------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------
    def __init__(self, data, fs, filterbank, pad_method="symmetric"):
        # ---------------- Input Data --------------------------
        self.fs = fs
        self.data = np.asarray(data, float)
        self.N_orig = self.data.size
        self.pad_method = pad_method

        # ---------------- filterbank --------------------------
        self.fb = filterbank
        self.Wfreq = self.fb.Wfreq                # (channels, fb_len)
        self.channel_freqs = self.fb.channel_freqs
        self.fb_len = self.Wfreq.shape[-1]
        self.n_channels = self.Wfreq.shape[0]

        # ---------------- phase‑align by pad_left -------------
        self.freqs = np.fft.fftfreq(self.fb_len)
        self.phase_shift = np.exp(-1j * 2 * np.pi * self.freqs * (self.fb_len / 2))
        #self.Wfreq *= self.phase_shift[np.newaxis, :]

        # ---------------- frame operator ----------------------
        self.Sfreq = np.sum(np.abs(self.Wfreq) ** 2, axis=0)
        self.Sfreq[self.Sfreq < 1e-12] = 1e-12

        # ---------------- pad the signal ----------------------
        self.pad_width = 0
        self.N_padded = self.N_orig         
        self.data = self._pad_and_taper(self.data)

        # ---------------- Save the result ----------------------
        self.coefficients = None

    # ------------------------------------------------------------
    # INTERNAL UTILITIES
    # ------------------------------------------------------------
    def _pad_and_taper(self, data):
        """Symmetric-pad *x* to *N_fft* and Tukey-taper only new edges."""
        self.N_orig = data.shape[-1]
        if self.pad_method is not None:
            target_length = int(2 ** np.ceil(np.log2(self.N_orig*2)))
            initial_pad = (target_length - self.N_orig) // 2
            data = np.pad(data, initial_pad, mode=self.pad_method)
            data *= signal.windows.tukey(data.shape[-1], alpha=0.3)
            self.N_padded = data.shape[-1]
            self.pad_width = initial_pad
        return data
    # ------------------------------------------------------------
    # FORWARD / INVERSE
    # ------------------------------------------------------------
    def forward(self, new_data=None, trim = True):
        if new_data is not None:
            new_data = np.asarray(new_data, float)
            if new_data.size != self.N_orig:
                raise ValueError(
                    f"new_data length {new_data.size} ≠ original {self.N_orig}"
                )
            self.data = self._pad_and_taper(new_data)

        Lx  = self.N_padded                      # padded length of input
        Lh  = self.Wfreq.shape[1]                # length of each FIR channel

        # ========  short-signal path (fits one FFT)  ==================
        if Lx <= Lh:
            shifted = self.Wfreq * self.phase_shift[np.newaxis, :]
            N_fft = Lh                           # same rule you had before
            F = np.fft.fft(self.data, n=N_fft)
            out = np.fft.ifft(shifted * F, n = Lh, axis=1)
            self.coefficients = out

            if trim:
                out = out[:,:Lx][:, self.pad_width:-self.pad_width]

            return out

        # ========  long-signal path (overlap-add)  ============================
        N_fft = int(2 ** np.ceil(np.log2(Lh - 1)))   # single grid for both operands
        hop   = N_fft - Lh + 1
        Wlong = self._filters_on_grid(N_fft)              # <- use the helper
        n_sc  = Wlong.shape[0]

        out_len = Lx + Lh - 1
        out     = np.zeros((n_sc, out_len), dtype=complex)

        for k0 in range(0, Lx, hop):
            blk   = self.data[k0:k0 + hop]
            blk   = np.pad(blk, (0, N_fft - blk.size))
            X_blk = np.fft.fft(blk, n=N_fft)
            Y_blk = np.fft.ifft(Wlong * X_blk, axis=1)

            end = k0 + N_fft
            if end > out_len:               # trim **only** the final block tail
                Y_blk = Y_blk[:, : out_len - k0]
                end   = out_len
            out[:, k0:end] += Y_blk

        self.coefficients = out

        if trim:
            out = np.roll(out, self.N_orig//2)[:,self.pad_width:self.pad_width+self.N_orig]

        return(out)

    def inverse(self, coeffs = None):
        if(coeffs is None):
            coeffs = self.coefficients
            
        Lx  = self.N_padded
        Lh  = self.Wfreq.shape[1]

        # ========  short-signal path  =================================
        if Lx <= Lh:
            shifted = self.Wfreq * self.phase_shift[np.newaxis, :]
            N_fft = Lh
            Cf = np.fft.fft(coeffs, n = Lh, axis=1)
            Xf = np.sum(np.conj(shifted) * Cf, axis=0) / self.Sfreq
            x_full = np.fft.ifft(Xf)
            return x_full[self.pad_width : self.pad_width + self.N_orig]

        # ========  long-signal path (overlap-add)  ====================
        Lx_full = coeffs.shape[1]  # Now Lx + Lh - 1
        N_fft = int(2 ** np.ceil(np.log2(Lx_full + Lh - 1)))
        hop = N_fft - Lh + 1

        Wlong  = self._filters_on_grid(N_fft)             # <- helper again
        n_sc   = Wlong.shape[0]

        Sf     = (np.abs(Wlong) ** 2).sum(axis=0).real
        eps    = 1e-12 * Sf.max()
        Sf_inv = np.where(Sf > eps, 1.0 / Sf, 0.0)        # numerical floor

        out_len = Lx + Lh - 1
        out     = np.zeros(out_len, dtype=complex)

        for k0 in range(0, Lx, hop):
            k1 = min(k0 + hop, Lx_full)
            seg = coeffs[:, k0:k1]

            seg_pad = np.zeros((n_sc, N_fft), dtype=complex)
            seg_pad[:, :seg.shape[1]] = seg
            Cf_blk  = np.fft.fft(seg_pad, axis=1)

            X_blk = (np.conj(Wlong) * Cf_blk).sum(axis=0) * Sf_inv
            x_blk = np.fft.ifft(X_blk, n=N_fft)

            end = k0 + N_fft
            if end > out_len:
                x_blk = x_blk[: out_len - k0]
                end   = out_len
            out[k0:end] += x_blk

        return out[self.pad_width : self.pad_width + self.N_orig].real

    def _filters_on_grid(self, N_fft: int):
        """
        Return the filter-bank spectra resampled onto an FFT grid of length N_fft.
        If they already live on that grid the original array is returned.
        """
        if self.Wfreq.shape[1] == N_fft:
            return self.Wfreq                  # nothing to do

        Lh  = self.Wfreq.shape[1]
        Hf  = np.zeros((self.Wfreq.shape[0], N_fft), dtype=complex)
        for i in range(self.Wfreq.shape[0]):
            h  = np.fft.ifft(self.Wfreq[i], n=Lh)      # impulse response
            h  = np.pad(h, (0, N_fft - Lh))            # zero-pad in **time**
            Hf[i] = np.fft.fft(h, n=N_fft)             # back to frequency
        return Hf

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
        cmap="viridis",
        y_tick_steps=5,
        figsize=(10, 6),
        title="Wavelet Coefficient Log‑Power",
        vmin=None,
        vmax=None,
        interpolation='none',
    ):


        n_sc, n_t = coeffs.shape
        t_axis = np.linspace(0, n_sc / self.fs, coeffs.shape[1])

        power = np.abs(coeffs) ** 2

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

