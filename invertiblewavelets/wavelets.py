# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   █ █ █ █ █   Invertible-Wavelets Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  wavelets.py                                                 ║
# ║  Package      :  invertiblewavelets                                          ║
# ║  Author       :  Dr. Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>            ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-04-30                                                  ║
# ║  Last Updated :  2025-05-07                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  :                                                              ║
# ║      Canonical continuous-time mother wavelets for analysis & synthesis.     ║
# ║      Exports:                                                                ║
# ║          • Morlet       — complex analytic, tunable (fc, fb)                 ║
# ║          • Cauchy       — analytic with algebraic tails (α)                  ║
# ║          • MexicanHat   — real Ricker (2nd-deriv. Gaussian)                  ║
# ║          • DoG          — generic n-th derivative of Gaussian                ║
# ║                                                                              ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import numpy as np
from scipy.special import erfcinv

__all__ = [
    "Morlet",
    "Cauchy",
    "MexicanHat",
    "DoG",
]

# ------------------------------------------------------------------
# Consants
# ------------------------------------------------------------------
PI = np.pi


# ------------------------------------------------------------------
# Morlet Wavelet
# ------------------------------------------------------------------
class Morlet:
    def __init__(self, fc=1, fb=1):
        """
        Initializes the Morlet wavelet with given bandwidth.

        Parameters:
        - fc (float): The bandwidth parameter of the Morlet wavelet.
        """
        self.fc = fc
        self.fb = fb

    def effective_half_width(self, eps=1e-6, criterion="energy"):
        if criterion == "ampl":
            return np.sqrt(self.fb * np.log(1/eps))
        
        return np.sqrt(self.fb/2) * erfcinv(2*eps)

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

# ------------------------------------------------------------------
# Cauchy Wavelet
# ------------------------------------------------------------------
class Cauchy:
    def __init__(self, alpha=300):
        self.alpha = float(alpha)
        self.fc = 1

    def effective_half_width(self, eps=1e-3):
        return (self.alpha/(2*np.pi))*np.sqrt(eps**(-2/(1+self.alpha)) - 1)

    def eval_analysis(self, t):
        """Continuous-time 'analysis' mother wavelet."""
        t = t
        factor = 1 - 2j * np.pi * t / self.alpha
        wavelet = factor ** (-1 - self.alpha)
        return wavelet

# ------------------------------------------------------------------
# Who named this?
# ------------------------------------------------------------------
class MexicanHat:
    def __init__(self):
        self.norm = 2 / (np.sqrt(3) * np.pi**0.25)
        self.fc = np.sqrt(2) / 2 / np.pi

    def effective_half_width(self, eps=1e-8):
        return np.sqrt(np.log(1/eps)) 

    def eval_analysis(self, t: np.ndarray):
        return self.norm * (1 - t**2) * np.exp(-t**2 / 2)

# ------------------------------------------------------------------
# DoG Wavelet
# ------------------------------------------------------------------
class DoG:
    def __init__(self, n: int = 1, sigma: float = 1.0):
        if n < 1:
            raise ValueError("Derivative order n must be >=1")
        self.n = int(n)
        self.sigma = float(sigma)
        self.fc = np.sqrt(1.0 * n)/ (2*np.pi*sigma)

    def _hermite(self, n: int, x: np.ndarray):
        """Probabilist Hermite polynomial H_n (recursive)."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        Hnm1 = x
        Hnm2 = np.ones_like(x)
        for k in range(2, n + 1):
            Hn = x * Hnm1 - (k - 1) * Hnm2
            Hnm2, Hnm1 = Hnm1, Hn
        return Hn

    def effective_half_width(self, eps=1e-3):
        return(self.sigma * np.sqrt(2 * np.log(1/eps) + self.n))

    def eval_analysis(self, t: np.ndarray):
        x = t / self.sigma
        gauss = np.exp(-x**2 / 2)
        Hn = self._hermite(self.n, x)
        coef = ((-1) ** self.n) / (self.sigma ** (self.n + 0.5))
        psi = coef * Hn * gauss
        
        return psi
