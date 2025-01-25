import numpy as np
from scipy.interpolate import interp1d
import pywt


__all__ = ["Morlet", "Cauchy", "PyWaveletWrapper"]

class Morlet:
    def __init__(self, w0=6):
        """
        TODO
        """
        self.w0 = w0
        self.constant_term = np.exp(-0.5 * self.w0**2)

    def eval_analysis(self, t):
        gaussian = np.exp(-0.5 * t**2) * np.pi**(-0.25)
        wavelet = (np.exp(1j * self.w0 * t) - self.constant_term) * gaussian
        return wavelet

class Cauchy:
    def __init__(self, alpha=300):
        """
        TODO
        """
        self.alpha = float(alpha)

    def eval_analysis(self, t):
        """Continuous-time 'analysis' mother wavelet."""
        # factor^(-1 - alpha)
        factor = 1 - 2j * np.pi * t / self.alpha
        wavelet = factor ** (-1 - self.alpha)
        return wavelet

class PyWaveletWrapper:
    def __init__(self, wavelet_name='cmor1.5-1.0'):
        """
        Initialize the PyWaveletWrapper with a specified wavelet and scale.

        Parameters:
        - wavelet_name (str): Name of the PyWavelet wavelet (e.g., 'morl', 'cmor', 'gaus1', etc.).
        - scale (float): Scaling factor for the wavelet.
        """
        self.wavelet_name = wavelet_name
        try:
            self.wavelet = pywt.ContinuousWavelet(wavelet_name)
        except:
            raise ValueError("Invalid wavelet name. Please check the PyWavelet documentation.\n", pywt.wavelist(kind='continuous'))
        
        # Precompute the wavelet function
        self.psi, self.x_psi = self.wavelet.wavefun()
        
        # Create interpolation function
        self.interp_func = interp1d(self.x_psi, self.psi, kind='cubic', 
                                    fill_value=0, bounds_error=False)
        
    def eval_analysis(self, t):
        """
        Evaluate the scaled wavelet at specified time points.

        Parameters:
        - t (np.ndarray): Time array where the wavelet should be evaluated.

        Returns:
        - np.ndarray: Wavelet function evaluated at time points t.
        """
        # Interpolate the scaled wavelet to the desired time points
        wavelet_at_t = self.interp_func(t)
        return wavelet_at_t