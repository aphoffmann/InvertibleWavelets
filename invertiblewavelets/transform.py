import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .wavelet_classes import Morlet, Cauchy

__all__ = ["Transform"]

class Transform:
    """
    Implements a non-decimated wavelet transform:

    Forward transform: 
       c2D[j, :] = ifft( fft(data) * fft(wavelet_j) )

    Inverse transform via "frame operator" S(w):
       X_hat(w) = (1 / S(w)) * sum_j conj(W_j(w)) * fft( c2D[j, :] )
       x_hat(n) = ifft( X_hat(w) )

    Where S(w) = sum_j |W_j(w)|^2,  the sum of wavelet magnitude-squares across channels.
    """

    def __init__(self, data, fs, wavelet=Cauchy(),
                b=None, q = None, M=None, Mc=None, xi_1 = None,
                pad_method='symmetric'):
        
        
        self.data = np.asarray(data, dtype=float)
        self.N = self.data.shape[-1] # Number of samples
        self.fs = fs             # Sampling frequency
        self.wavelet = wavelet   # Cauchy wavelet


        # Pad data
        self.pad_width = 0
        if(pad_method is not None):
            self.pad_width =  (int(2 ** np.ceil(np.log2(self.data.shape[-1]))) - self.data.shape[-1]) // 2
            self.data = np.pad(self.data, self.pad_width, mode=pad_method)
            self.N = self.data.shape[-1]
            self.data *= signal.windows.tukey(self.N, alpha=.3)

         # fill default parameters
        self._init_params(b, q, M, Mc, xi_1)

        # create time vector, centered so wavelet is around t=0
        self.time = np.arange(self.N) / self.fs
        self.time -= np.mean(self.time)

        # define wavelet channels
        self.j_channels = np.arange(-self.Mc, self.M)

        # Precompute the wavelets in frequency domain, Wfreq[j, :], shape = (#channels, N).
        self.freqs = np.fft.fftfreq(self.N)
        self.channel_freqs = np.zeros(len(self.j_channels))
        self.Wfreq = self._build_wavelets_FD()

        # Compute the phase shift for time correction  
        self.phase_shift = np.exp(-1j * 2 * np.pi * self.freqs * (self.N / 2))
        self.Wfreq *= self.phase_shift[np.newaxis, :]

        # Frame operator S(w) = sum_j |W_j(w)|^2
        # shape = (N,)
        self.Sfreq = np.sum(np.abs(self.Wfreq)**2, axis=0)

        # Avoid dividing by zero in case some frequency bins are extremely small:
        eps = 1e-12
        self.Sfreq[self.Sfreq < eps] = eps
    
    def _init_params(self, b, q, M, Mc, xi_1):
        self.b = b
        self.q = q
        self.M = M
        self.Mc = Mc
        self.xi_1 = xi_1

        if self.b is None:
            self.b = self.N / (2 * self.fs)
        if self.q is None:
            self.q = self.b
        if self.M is None:
            self.M = int(self.q*(self.fs/2 - 1/self.b))
            if self.M < 1:
                self.M = 4

        if self.Mc is None:
            self.Mc = int(self.q/self.b)
            if self.Mc < 1:
                self.Mc = 1

        if self.xi_1 is None:
            self.xi_1 = (self.fs * self.q) / max(self.M,1) / 2
    
    def _build_wavelets_FD(self):
        """
        Build the frequency-domain wavelets W_j(w). For each channel j:
            wavelet_j(t) = eq3_analysis or eq4_analysis with shift 'delays[j]'.
            Then W_j(w) = fft( wavelet_j(t) ).
        Returns Wfreq of shape (#channels, N).
        """
        jvals = self.j_channels
        Wfreq = np.zeros((len(jvals), self.N), dtype=complex)
        for i, j in enumerate(jvals):
            if (j/self.q + 1/self.b) > 0:
                # eq3
                wtime = self.wavelet.eq3_analysis(self.time, j, self.b, self.q)
            else:
                # eq4
                wtime = self.wavelet.eq4_analysis(self.time, j, self.b, self.q, self.xi_1)

            Wfreq[i,:] = np.fft.fft(wtime)
            self.channel_freqs[i] = self.freqs[np.argmax(np.abs(Wfreq[i,:]))]

        return Wfreq
    
    def forward(self):
        """
        For each channel j:
          coeffs[j, :] = ifft( fft(data) * Wfreq[j, :] )
        shape: (#channels, N)
        """
        Fdata = np.fft.fft(self.data)
        J = self.Wfreq.shape[0]
        coeffs = np.zeros((J, self.N), dtype=complex)
        coeffs = np.fft.ifft(Fdata * self.Wfreq, axis=1)
        for j in range(J):
            coeffs[j, :] = np.fft.ifft(Fdata * self.Wfreq[j, :])
        return coeffs
    
    def inverse(self, coeffs):
        """
        coeffs is (#channels, N).
        1) Convert each row to frequency domain: c2Dfreq[j, :] = fft( c2D[j, :] ).
        2) Sum_j [ conj(W_j(w)) * c2Dfreq_j(w ) ] / Sfreq(w).
        3) ifft -> xhat(n).
        """
        J = coeffs.shape[0]
        c2Dfreq = np.zeros_like(coeffs, dtype=complex)  # same shape
        for j in range(J):
            c2Dfreq[j, :] = np.fft.fft(coeffs[j, :])

        # Weighted sum in freq: XhatFreq = [1/Sfreq] * sum_j conj(Wfreq[j,:]) * c2Dfreq[j,:]
        numerator = np.zeros(self.N, dtype=complex)
        for j in range(J):
            numerator += np.conjugate(self.Wfreq[j,:]) * c2Dfreq[j,:]

        XhatFreq = numerator / self.Sfreq
        xhat_time = np.fft.ifft(XhatFreq).real

        # Remove padding
        if self.pad_width > 0:
            xhat_time = xhat_time[self.pad_width:-self.pad_width]

        return xhat_time
    

    def plot_coeff_power(self, coeffs, cmap='viridis', vmin=None, vmax=None, 
                        y_tick_steps=5, figsize=(10, 6)):
        """
        Plot the power of the wavelet coefficients.

        Parameters:
        - coeffs: 2D numpy array of shape (#channels, N) containing the wavelet coefficients.
        - cmap: Colormap for the plot.
        - vmin, vmax: Minimum and maximum values for the colormap scaling.
        - y_tick_steps: Number of frequency ticks to display on the y-axis.
        - figsize: Size of the figure.
        """
        power = np.abs(coeffs) ** 2  # Power of coefficients

        # Calculate frequency for each channel
        alpha_j = (1.0 / self.b) + (self.j_channels / self.q)
        frequency_j = alpha_j / (2 * np.pi)  # Convert scale to frequency in Hz

        # Sort frequencies and corresponding power for better visualization
        sorted_indices = np.argsort(frequency_j)
        frequency_j_sorted = frequency_j[sorted_indices]
        power_sorted = power[sorted_indices, :]

        plt.figure(figsize=figsize)
        extent = [self.time[0], self.time[-1], frequency_j_sorted[0], frequency_j_sorted[-1]]

        im = plt.imshow(np.log(power_sorted), aspect='auto', origin='lower', extent=extent, 
                        cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, label='Power')

        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')

        # Set y-axis ticks
        y_min, y_max = frequency_j_sorted[0], frequency_j_sorted[-1]
        y_ticks = np.linspace(y_min, y_max, y_tick_steps)
        plt.yticks(y_ticks, [f"{freq:.2f}" for freq in y_ticks])

        plt.title('Wavelet Coefficients Power')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()  