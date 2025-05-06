# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   █ █ █ █ █   Invertible-Wavelets Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  transform.py                                                ║
# ║  Package      :  invertiblewavelets                                          ║
# ║  Author       :  Dr. Alex P. Hoffmann  <alex.hoffmann@nasa.gov>              ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-04-30                                                  ║
# ║  Last Updated :  2025-04-30                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  :                                                              ║
# ║      Wavelet transform coefficient class.  Given an arbitrary                ║
# ║      :class:`FilterBank`, `Transform` provides FFT-based forward             ║
# ║      and inverse wavelet transforms—including overlap-add                    ║
# ║      handling, phase alignment, and a NumPy-friendly scalogram plot.         ║
# ║                                                                              ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["Transform"]
class Transform:

    # ------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------
    def __init__(self, filterbank):
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


    # ------------------------------------------------------------
    # INTERNAL FUNCTIONS
    # ------------------------------------------------------------
    def _trim_coeffs(self, full, N):
        """
        Return an (n_ch, Lx) matrix that can be passed straight back to inverse().
        • long -signal branch : keep the centre Lx samples
        """
        Lx, Lh = N, self.Wfreq.shape[1]
        full_len = full.shape[1]
        start = (full_len - Lx) // 2
        end   = start + Lx
        return full[:, start:end]
    
    def _untrim_coeffs(self, short):
        """
        Restore the coefficient block to the length expected by the
        inverse routine:
            • Lx + Lh - 1  otherwise  
        """
        n_ch, Lx = short.shape
        Lh = self.Wfreq.shape[1]

        full_len  = Lx + Lh - 1
        pad_left  = (full_len - Lx) // 2
        pad_right = full_len - pad_left - Lx
        full = np.zeros((n_ch, full_len), dtype=complex)
        full[:, pad_left:pad_left + Lx] = short


        if pad_left:
            left_core = short[:, :pad_left][:, ::-1]          # reflect
            # cosine ramp: 0 → 1 over pad_left samples
            w_left = np.sin(0.5 * np.pi *
                            np.linspace(0.0, 1.0, pad_left, endpoint=False))
            full[:, :pad_left] = left_core * w_left

        # -------------------- build & window RIGHT pad ----------------
        if pad_right:
            right_core = short[:, -pad_right:][:, ::-1]       # reflect
            # cosine ramp: 1 → 0 over pad_right samples
            w_right = np.sin(0.5 * np.pi *
                             np.linspace(1.0, 0.0, pad_right, endpoint=False))
            full[:, -pad_right:] = right_core * w_right

        return full
    
    def _forward_filter_freq(self, N_fft: int):
        if self.Wfreq.shape[1] == N_fft:
            return self.Wfreq                  # nothing to do

        Lh  = self.Wfreq.shape[1]
        Hf  = np.zeros((self.Wfreq.shape[0], N_fft), dtype=complex)
        for i in range(self.Wfreq.shape[0]):
            h  = np.fft.ifft(self.Wfreq[i], n=Lh)      # impulse response
            h  = np.pad(h, (0, N_fft - Lh))            # zero-pad in **time**
            Hf[i] = np.fft.fft(h, n=N_fft)             # back to frequency

        return Hf
    
    def _inverse_filter_freq(self, N_fft: int):
        if self.Wfreq.shape[1] == N_fft:
            return self.Wfreq                  # nothing to do

        Lh  = self.Wfreq.shape[1]
        Hf  = np.zeros((self.Wfreq.shape[0], N_fft), dtype=complex)
        for i in range(self.Wfreq.shape[0]):
            h  = np.fft.ifft(self.Wfreq[i], n=Lh)      # impulse response
            h  = np.pad(h, N_fft - Lh)            # zero-pad in **time**
            Hf[i] = np.fft.fft(h.real, n=N_fft)             # back to frequency

        return Hf
    
    # ------------------------------------------------------------
    # FORWARD / INVERSE
    # ------------------------------------------------------------
    def forward(self, data, mode="same"):
        data = np.asarray(data, float)
        N = data.size

        Lx, Lh = N, self.Wfreq.shape[1]

        # ========  short-signal path (fits one FFT)  ==================
        if Lx <= Lh:
            #shifted = self.Wfreq * self.phase_shift[np.newaxis, :]
            N_fft = Lh + Lx - 1                          # same rule you had before
            shifted = self._forward_filter_freq(N_fft)
            F = np.fft.fft(data, n=N_fft)
            coefficients = np.fft.ifft(shifted * F, n = N_fft, axis=1)

        # ========  long-signal path (overlap-add)  ============================
        else:
            N_fft = int(2 ** (np.ceil(np.log2(Lh))+1))   # single grid for both operands
            hop   = N_fft - Lh + 1
            Wlong = self._forward_filter_freq(N_fft)              # <- use the helper
            n_sc  = Wlong.shape[0]

            out_len = Lx + Lh - 1
            coefficients = np.zeros((n_sc, out_len), dtype=complex)

            for k0 in range(0, Lx, hop):
                blk   = data[k0:k0 + hop]
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
                coefficients[:, k0:end] += Y_blk

        if mode == "full":
            return coefficients
        
        return self._trim_coeffs(coefficients, N)

    def inverse(self, coeffs, mode='same', Lx = None):
        Lh = self.Wfreq.shape[1]
        N = coeffs.shape[1]

        if(mode == 'same'):
            Lx = coeffs.shape[1]             # user passed the short form
            coeffs = self._untrim_coeffs(coeffs)
        else:
            if(N > Lh):
                Lx = N - Lh + 1
            else:
                if Lx is None:
                    raise ValueError("Lx must be specified when mode is not 'same'")
                Lx = Lx

        # ========  short-signal path  =================================
        if Lx <= Lh:
            N_fft = Lx + Lh 
            shifted = self._inverse_filter_freq(N_fft)
            Sfreq = np.sum(np.abs(shifted) ** 2, axis=0)
            #Sfreq = np.where(np.abs(Sfreq) < 1, 1, Sfreq) 

            Cf = np.fft.fft(coeffs, n = N_fft, axis=1)
            Xf = np.sum(np.conj(shifted) * Cf, axis=0) / Sfreq
            x_full = np.fft.ifft(Xf)
            return x_full[Lh:Lh+Lx].real

        # ========  long-signal path (overlap-add)  ====================
        "Instead of doing OLA in the inverse, we do the inverse per scale and then sum"


        N_fft = int(2 ** (np.ceil(np.log2(Lh))+1))   # single grid for both operands
        hop   = N_fft - Lh + 1
        Wlong = self._inverse_filter_freq(N_fft)              # <- use the helper
        n_sc  = Wlong.shape[0]

        # Compute normalization factor
        Sf     = (np.abs(Wlong) ** 2).sum(axis=0).real
        eps    = 1e-12 * Sf.max()
        Sf_inv = np.where(Sf > eps, 1.0 / Sf, 0.0)        # numerical floor

        # Output length matches original signal length Lx, but compute full length first
        out_len = Lx + Lh + hop - 1
        out = np.zeros(out_len, dtype=complex)

        k0s = np.arange(0, out_len, hop)
        for k0 in k0s:
            k1 = k0 + hop
            seg = coeffs[:, k0:k1]
            seg_pad = np.zeros((n_sc, N_fft), dtype=complex)
            seg_pad[:, :seg.shape[1]] = seg
            Cf_blk  = np.fft.fft(seg_pad, axis=1)
            X_blk = (np.conj(Wlong) * Cf_blk).sum(axis=0) * Sf_inv
            x_blk = np.fft.ifft(X_blk, n=N_fft)

            end = min(out_len, k0 + N_fft)
            out[k0:end] += x_blk[:end - k0]


        return out[Lh:Lh + Lx].real

    # ------------------------------------------------------------
    # SCALOGRAM PLOT
    # ------------------------------------------------------------
    def scalogram(
        self,
        coeffs,
        fs = 1.0,
        cmap="viridis",
        y_tick_steps=5,
        figsize=(10, 6),
        title="Wavelet Coefficient Log‑Power",
        vmin=None,
        vmax=None,
        interpolation='none',
    ):


        _, n_t = coeffs.shape
        t_axis = np.linspace(0, n_t / fs, coeffs.shape[1])

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

