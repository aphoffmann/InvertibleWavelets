import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import windows
from scipy import signal
import pywt


__all__ = ["Morlet", "Cauchy", "PyWaveletWrapper", "Shannon", "Testlet" ]


class Morlet:
    def __init__(self, fc=1, fb=1):
        """
        Initializes the Morlet wavelet with given bandwidth.

        Parameters:
        - fc (float): The bandwidth parameter of the Morlet wavelet.
        """
        self.fc = fc
        self.fb = fb

    def eval_analysis(self, t):
        """
        Evaluates the Morlet wavelet at given time points using the C++ formula.

        Parameters:
        - t (np.ndarray): Time points at which to evaluate the wavelet.
        - scale (float): The scale of the wavelet.

        Returns:
        - np.ndarray: Evaluated complex Morlet wavelet values.
        """
        wavelet = 1/ np.sqrt(2*self.fb) *  np.exp(2j*np.pi*self.fc*t)*np.exp(-(t ** 2) / self.fb)

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
        t = t
        factor = 1 - 2j * np.pi * t / self.alpha
        wavelet = factor ** (-1 - self.alpha)
        return wavelet

class PyWaveletWrapper:
    def __init__(self, wavelet_name='cmor1.5-1.0'):
        """
        Initialize the PyWaveletWrapper with a specified wavelet and scale.

        Parameters:
        - wavelet_name (str): Name of the PyWavelet wavelet (e.g., 'morl', 'cmor', 'gaus1', etc.).
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
    
class Shannon:
    def __init__(self, B=1, C = 1):
        """
        TODO
        """
        self.B = B
        self.C = C

    def eval_analysis(self, t):
        wavelet = np.sqrt(self.B) * np.sin(np.pi*t*self.B)/(np.pi*t*.5) * np.exp(2j*self.C *np.pi*t)
        return wavelet
        

class Testlet:
    """
    Template for testing any filter
    """
    def __init__(self, alpha = 1):
        """
        TODO
        """
        self.alpha = alpha

    def eval_analysis(self, t):
        #wavelet =np.exp(2j*np.pi*t)*windows.gaussian(t.shape[-1], self.alpha) ## Constant window
        wavelet = signal.square(2 * np.pi * t) * windows.gaussian(t.shape[-1], self.alpha)
        return wavelet