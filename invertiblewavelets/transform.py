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
from .wavelet_classes import Morlet, Cauchy

__all__ = ["Transform"]

class Transform:
    """
    Implements a FFT-based wavelet transform:

    Forward transform: 
       c2D[j, :] = ifft( fft(data) * fft(wavelet_j) )

    Inverse transform via "frame operator" S(w):
       X_hat(w) = (1 / S(w)) * sum_j conj(W_j(w)) * fft( c2D[j, :] )
       x_hat(n) = ifft( X_hat(w) )

    Where S(w) = sum_j |W_j(w)|^2,  the sum of wavelet magnitude-squares across channels.
    """

    def __init__(self, data, fs, wavelet=Cauchy(),
                b=None, q = None, M=None, Mc=None, xi_1 = None,
                pad_method='symmetric', scales='linear', dj = 1/4):
        
        
        self.data = np.asarray(data, dtype=float)
        self.N = None                                   # Number of samples
        self.N_orig = self.data.shape[-1]
        self.fs = fs                                    # Sampling frequency
        self.wavelet = wavelet                          # Wavelet class that includes wavelet.eval_analysis(t)
        self.scales = scales                            # Frequency scale type: 'linear' or 'dyadic'
        self.dj = dj                                    # Dyadic scale spacing                       
        self.pad_method = pad_method                    # Padding method (See numpy.pad)


        # Pad data with Tukey window
        self.pad_width = 0
        if(pad_method is not None):
            self._pad_data(self.data, mode=pad_method)

        # create time vector, centered so wavelet is around t=0
        self.time = np.arange(self.N) / self.fs
        self.time -= np.mean(self.time)

        # fill default parameters for linear scales
        self._init_params(b, q, M, Mc, xi_1)

  

        # define wavelet channels. Mc is compensation channels for low frequences (See Holighaus et al. 2023)
        self.j_channels = np.arange(-self.Mc, self.M)

        # Precompute the analysis filter in frequency domain, Wfreq[j, :], shape = (# scale channels, #N fft frequency channels).
        self.freqs = np.fft.fftfreq(self.N)
        self.channel_freqs = np.zeros(len(self.j_channels))
        self.Wfreq = self._build_analysis_filter()
        
        # Compute the phase shift for time correction  
        self.phase_shift = np.exp(-1j * 2 * np.pi * self.freqs * (self.N / 2))
        self.Wfreq *= self.phase_shift[np.newaxis, :]

        # Compute Frame operator S(w) = sum_j |W_j(w)|^2 for synthesis filter
        # shape = (N,) from fftfreq
        self.Sfreq = np.sum(np.abs(self.Wfreq)**2, axis=0)

        # Avoid dividing by zero in case some frequency bins are extremely small:
        eps = 1e-12
        self.Sfreq[self.Sfreq < eps] = eps
    
    def _init_params(self, b, q, M, Mc, xi_1):
        """
        Initialize parameters for scaling linear wavelets via Holighaus et al. 2023.
        """
        self.b = b          # Largest Scale of interest
        self.q = q          # Scale resolution step (Often set to b)
        self.M = M          # Number of scales
        self.Mc = Mc        # Compensation channels for low frequencies
        self.xi_1 = xi_1    

        if self.b is None:
            T_eff, _, _ = self.compute_effective_support(self.time)
            T_signal = self.N_orig / self.fs
            margin = 0.1 * T_signal 
            self.b =  (T_signal - margin) / T_eff

        self.s_max = self.b

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
    
    def _build_analysis_filter(self):
        """
        Build the frequency-domain wavelets W_j(w). 
        Returns Wfreq of shape (#channels, N).
        """
        jvals = self.j_channels
        real_freqs = np.fft.fftfreq(self.N, 1/self.fs)

        if self.scales == 'linear':
            self.channel_freqs = np.zeros(len(self.j_channels))
            Wfreq = np.zeros((len(jvals), self.N), dtype=complex)
            for i, j in enumerate(jvals): 
                if (j >= 0):
                    # Normal Channels
                    alpha_j = (1.0 / self.b) + (j / self.q)
                    wtime = np.sqrt(alpha_j) * self.wavelet.eval_analysis(alpha_j * self.time)
                else: 
                    # Compensation Channels
                    #phase = np.exp(2j * np.pi * self.xi_1 * j * self.time / self.q)
                    #wtime = (1.0 / np.sqrt(self.b)) * self.wavelet.eval_analysis(self.time / self.b) * phase
                    wtime = 100*(1.0 / np.sqrt(2*self.b)) * np.sinc(self.time / self.b/2) * signal.windows.tukey(self.N, alpha=0.3)
                    # TODO: Remove hardcoding of 100 and change Mc to Boolean?


                # Calculate FFT of wavelet filter
                Wfreq[i,:] = np.fft.fft(wtime)
                self.channel_freqs[i] = real_freqs[np.argmax(np.abs(Wfreq[i,:self.N//2]))]

        elif self.scales == 'dyadic':
            # 1) Evaluate unscaled wavelet in time domain
            wavelet_unscaled = self.wavelet.eval_analysis(self.time)
            
            # 2) FFT and find the index of the peak magnitude
            wavelet_unscaled_fft = np.fft.fft(wavelet_unscaled)
            freqs = np.fft.fftfreq(len(wavelet_unscaled_fft), d=1.0/self.fs)
            peak_idx = np.argmax(np.abs(wavelet_unscaled_fft))
            center_freq = abs(freqs[peak_idx])  # take absolute value in case the bin is negative

            # 3) Define the smallest scale from the center frequency
            #    (For a Morlet wavelet at scale=1 => wavelet center frequency ≈ center_freq.)
            s0 = 1.0 / center_freq
            
            # 4) Define the number of scales J
            dj = self.dj
            # For the largest scale, use e.g. the signal duration in seconds
            J = int(np.floor(np.log2(self.s_max / s0) / dj))

            # 5) Generate scales (flip if you prefer largest->smallest)
            scales = s0 * 2.0**(dj * np.arange(J+1))
            scales = np.flip(scales)
            
            # 6) Wavelet transform across scales
            self.channel_freqs = np.zeros(J+1)
            Wfreq = np.zeros((len(scales), self.N), dtype=complex)
            real_freqs = np.fft.fftfreq(self.N, d=1.0/self.fs)
            
            for i, scale in enumerate(scales):
                wtime = np.sqrt(scale) * self.wavelet.eval_analysis(self.time / scale)
                Wfreq[i, :] = np.fft.fft(wtime)
                # Grab channel frequency by looking at the peak in the FFT
                self.channel_freqs[i] = real_freqs[np.argmax(np.abs(Wfreq[i, :]))]

           
        return Wfreq
    
    def _pad_data(self, data, mode = 'symmetric'):
        """
        Pad the data to the nearest power of 2 for FFT.
        """
        if mode is not None:
            target_length = int(2 ** np.ceil(np.log2(self.N_orig*4)))
            initial_pad = (target_length - self.N_orig) // 2
            data = np.pad(data, initial_pad, mode=mode)
            data *= signal.windows.tukey(data.shape[-1], alpha=0.3)
            self.data = data
            self.pad_width = initial_pad
            self.N = self.data.shape[-1]
    
    def forward(self, new_data=None):
        """
        For each channel j:
          coeffs[j, :] = ifft( fft(data) * Wfreq[j, :] )
        shape: (#channels, N)
        """
        if new_data is not None and new_data.shape[-1] != self.N_orig:
            raise ValueError(f"New data length {new_data.shape[-1]} does not match initialized data length {self.N}. Create new Transform object to reinitialize filterbanks.")
        
        elif new_data is not None:
            self.data = new_data
            if self.pad_method is not None:
                self._pad_data(new_data)

        Fdata = np.fft.fft(self.data)
        coeffs = np.fft.ifft(Fdata * self.Wfreq, axis=1)

        return coeffs
    
    def inverse(self, coeffs_t):
        """
        coeffs is (#channels, N).
        1) Convert each wavelet channel to frequency domain: coeffs_f[j, :] = fft( coeffs_t[j, :] ).
        2) Sum_j [ conj(W_j(w)) * c2Dfreq_j(w ) ] / Sfreq(w).
        3) ifft -> xhat(n).
        """
        coeffs_f = np.fft.fft(coeffs_t, axis=1)
        
        # Weighted sum in freq: XhatFreq = [1/Sfreq] * sum_j conj(Wfreq[j,:]) * c2Dfreq[j,:]
        # Normalizes frequency overlap between channels
        numerator =  np.sum(np.conj(self.Wfreq) * coeffs_f, axis=0)
        XhatFreq = numerator / self.Sfreq

        # Inverse FFT to get xhat(t)
        xhat_time = np.fft.ifft(XhatFreq).real

        # Remove padding
        if self.pad_width > 0:
            xhat_time = xhat_time[self.pad_width:-self.pad_width]

        return xhat_time
    
    def compute_effective_support(self, t, energy_fraction=0.99):
        """
        Computes the effective support of a wavelet as the time interval containing
        a specified fraction of its total energy.
        
        Parameters:
            wavelet         : An instance of a wavelet class that has an eval_analysis(t) method.
            energy_fraction : The fraction of total energy to be contained within the support (default 0.99).
            time_range      : Tuple (t_min, t_max) defining the time interval over which to evaluate the wavelet.
            num_points      : Number of time points to use in the evaluation.
            
        Returns:
            effective_support : The length of the time interval that contains the specified energy fraction.
            t_low             : Lower time bound of the effective support.
            t_high            : Upper time bound of the effective support.
        """
               
        # Evaluate the wavelet over the time vector.
        psi = self.wavelet.eval_analysis(t)
        
        # Compute the energy density |psi(t)|^2.
        energy = np.abs(psi)**2
        dt = t[1] - t[0]
        
        # Total energy (approximated as the integral using the Riemann sum).
        total_energy = np.sum(energy) * dt
        
        # Compute the cumulative energy over time.
        cumulative_energy = np.cumsum(energy) * dt
        
        # Find the indices where the cumulative energy crosses the lower and upper bounds.
        lower_bound_energy = (1 - energy_fraction) / 2 * total_energy
        upper_bound_energy = (1 + energy_fraction) / 2 * total_energy
        
        lower_index = np.searchsorted(cumulative_energy, lower_bound_energy)
        upper_index = np.searchsorted(cumulative_energy, upper_bound_energy)
        
        # Convert these indices back to time values.
        t_low = t[lower_index]
        t_high = t[upper_index]
        
        effective_support = t_high - t_low
        return effective_support, t_low, t_high
    

    def enforce_orthagonality(self, eps=1e-5):
        """
        Optimize enforcement of orthogonality by precomputing the time-domain
        wavelets and vectorizing the inner loop.
        """
        # Precompute inverse FFT of all wavelet scales.
        X = np.fft.ifft(self.Wfreq, axis=1)
        N = self.N  # For brevity in division later.

        # Initialize selected indices based on Mc.
        if self.Mc == 0:
            selected_indices = [0]
        else:
            selected_indices = [0, 1]

        current_index = selected_indices[-1]
        total_scales = self.M + self.Mc

        # Loop until we've processed all scales.
        while current_index < total_scales - 1:
            # Compute dot products with all remaining scales in one go.
            # X[current_index] is the current scale.
            # Compare against scales from current_index+1 to end.
            dots = np.dot(X[current_index], X[current_index+1:].conj().T) / N

            # Find the first index where the orthogonality condition holds.
            valid_indices = np.where(np.abs(dots) < eps)[0]
            if valid_indices.size == 0:
                # No further scale satisfies the condition.
                break

            # The actual next index in the original array.
            next_index = current_index + 1 + valid_indices[0]
            selected_indices.append(next_index)
            current_index = next_index

        # Update the frequency-domain wavelets using the selected indices.
        self.Wfreq = np.fft.fft(X[selected_indices, :], axis=1)
        self.Sfreq = np.sum(np.abs(self.Wfreq)**2, axis=0)
    
    def scalogram(self, coeffs, cmap='viridis', vmin=None, vmax=None, 
                        y_tick_steps=5, figsize=(10, 6), title = 'Wavelet Coefficients Power'):
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

        plt.figure(figsize=figsize)
        extent = [0, self.N/self.fs, self.channel_freqs[0], self.channel_freqs[-1]]

        if self.pad_width !=0:
            im = plt.imshow(np.log(power[:,self.pad_width:-self.pad_width]), aspect='auto', origin='lower', extent=extent, 
                            cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = plt.imshow(np.log(power), aspect='auto', origin='lower', extent=extent, 
                            cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, label='Power')

        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')

        # Set y-axis ticks
        y_min, y_max = self.channel_freqs[0], self.channel_freqs[-1]
        y_ticks = np.linspace(y_min, y_max, y_tick_steps)
        plt.yticks(y_ticks, [f"{freq:.2f}" for freq in y_ticks])

        plt.title(title)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()  