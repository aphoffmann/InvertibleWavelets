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
    def __init__(self, data, fs, filterbank):
        # ---------------- Input Data --------------------------
        self.fs = fs
        self.data = np.asarray(data, float)
        self.N = self.data.size


        # ---------------- filterbank --------------------------
        self.fb = filterbank
        self.Wfreq = self.fb.Wfreq                # (channels, fb_len)
        self.channel_freqs = self.fb.channel_freqs
        self.fb_len = self.Wfreq.shape[-1]
        self.n_channels = self.Wfreq.shape[0]

        # ---------------- phase‑align --------------------------
        self.freqs = np.fft.fftfreq(self.fb_len)
        self.phase_shift = np.exp(-1j * 2 * np.pi * self.freqs * (self.fb_len / 2))

        # ---------------- frame operator ----------------------
        self.Sfreq = np.sum(np.abs(self.Wfreq) ** 2, axis=0)
        self.Sfreq[self.Sfreq < 1e-12] = 1e-12

        # ---------------- Save the result ----------------------
        self.coefficients = None

    # ------------------------------------------------------------
    # INTERNAL FUNCTIONS
    # ------------------------------------------------------------
    def _trim_coeffs(self, full):
        """
        Return an (n_ch, Lx) matrix that can be passed straight back to inverse().
        • short-signal branch : keep the *front* Lx samples
        • long -signal branch : keep the centre Lx samples
        """
        Lx, Lh = self.N, self.Wfreq.shape[1]
        full_len = full.shape[1]

        if full_len == Lx:           # already 'same'
            return full

        # ---- short-signal path ----
        if Lx < Lh:                 
            return full[:, :Lx]      # keep leading segment

        # ---- long-signal path ----
        start = (full_len - Lx) // 2
        end   = start + Lx
        return full[:, start:end]
    
    def _untrim_coeffs(self, short):
        """
        Restore the coefficient block to the length expected by the
        inverse routine:
            • Lh           when Lx ≤ Lh   (short-signal path)
            • Lx + Lh - 1  otherwise      (overlap-add path)
        """
        n_ch, Ltrim = short.shape
        Lx, Lh = self.N, self.Wfreq.shape[1]
        

        # ---- short-signal path ----
        if Lx < Lh:                        
            full = np.zeros((n_ch, Lh), dtype=short.dtype)
            full[:, :Ltrim] = short           # place at the front
            full[:, Ltrim:] = self.coefficients[:, Ltrim:]
            return full

        # ---- long-signal path ----
        full_len  = Lx + Lh - 1
        pad_left  = (full_len - Ltrim) // 2
        full = np.zeros((n_ch, full_len), dtype=short.dtype)
        full[:, pad_left:pad_left + Ltrim] = short
        print(self.coefficients.shape, full.shape)
        return full
    
    def _filters_on_grid(self, N_fft: int):
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
    # FORWARD / INVERSE
    # ------------------------------------------------------------
    def forward(self, new_data=None, mode="same"):
        if new_data is not None:
            self.data = np.asarray(new_data, float)
            self.N = self.data.size

        Lx, Lh = self.N, self.Wfreq.shape[1]

        # ========  short-signal path (fits one FFT)  ==================
        if Lx < Lh:
            shifted = self.Wfreq * self.phase_shift[np.newaxis, :]
            N_fft = Lh                           # same rule you had before
            F = np.fft.fft(self.data, n=N_fft)
            out = np.fft.ifft(shifted * F, n = Lh, axis=1)
            self.coefficients = out

        # ========  long-signal path (overlap-add)  ============================
        else:
            N_fft = int(2 ** np.ceil(np.log2(Lh - 1)))   # single grid for both operands
            hop   = N_fft - Lh + 1
            Wlong = self._filters_on_grid(N_fft)              # <- use the helper
            n_sc  = Wlong.shape[0]

            out_len = Lx + Lh - 1
            out     = np.zeros((n_sc, out_len), dtype=complex)

            for k0 in range(0, Lx, hop):
                blk   = self.data[k0:k0 + hop]
                try:
                    blk   = np.pad(blk, (0, N_fft - blk.size))
                except ValueError:
                    print("Error: Padding failed. Check the input data size.", N_fft, blk.size)
                    print("blk:", blk.size, "N_fft:", N_fft)
                    print("hop:", hop, "Lh:", Lh, "Lx:", Lx)
                    raise(Exception("Padding failed."))
                X_blk = np.fft.fft(blk, n=N_fft)
                Y_blk = np.fft.ifft(Wlong * X_blk, axis=1)

                end = k0 + N_fft
                if end > out_len:               # trim **only** the final block tail
                    Y_blk = Y_blk[:, : out_len - k0]
                    end   = out_len
                out[:, k0:end] += Y_blk

            self.coefficients = out

        if mode == "full":
            return out
        
        return self._trim_coeffs(out)

    def inverse(self, coeffs = None):
        if(coeffs is None):
            coeffs = self.coefficients
        
        Lx, Lh = self.N, self.Wfreq.shape[1]

        if coeffs.shape[1] >= Lx:            # user passed the short form
            coeffs = self._untrim_coeffs(coeffs)

        # ========  short-signal path  =================================
        if Lx < Lh:
            shifted = self.Wfreq * self.phase_shift[np.newaxis, :]
            N_fft = Lh
            Cf = np.fft.fft(coeffs, n = Lh, axis=1)
            Xf = np.sum(np.conj(shifted) * Cf, axis=0) / self.Sfreq
            x_full = np.fft.ifft(Xf)
            return x_full[:Lx].real

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

        return out[:Lx].real



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
        t_axis = np.linspace(0, n_t / self.fs, coeffs.shape[1])

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

