import numpy as np

__all__ = ["Morlet", "Cauchy",]

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
