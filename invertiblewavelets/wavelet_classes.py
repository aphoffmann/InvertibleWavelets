import numpy as np

__all__ = ["Morlet", "Cauchy"]

class Morlet:
    def __init__(self, w0=6):
        """w0 is the nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok; Terrence and Compo set it to 6.
        """
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

    def eval_analysis(self, t):
        w = self.w0

        x = t
        output = np.exp(1j * w * x)
        output -= np.exp(-0.5 * (w ** 2))
        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output
    
    def eq3_analysis(self, t, j, b, q,):
        """
        Compute Equation 4: ψ_{l,j}(t) for α_j > 0.
        
        Equation 4:
        ψ_{l,j}(t) = sqrt(1/b + j/q) * ψ((1/b + j/q) * (t - d(l + δ_j))).
        """
        alpha_j = (1.0 / b) + (j / q)
        arg = alpha_j * t
        return np.sqrt(alpha_j) * self.eval_analysis(arg)
    
    def eq4_analysis(self, t, j, b, q, xi_1):
        """
        Compute Equation 5: ψ_{l,j}^{comp}(t) for α_j ≤ 0.
        
        Equation 5:
        ψ_{l,j}^{comp}(t) = (1/√b) ψ((t - d(l+δ_j)) / b)
                          * exp(2πi ξ₁ * j (t - d(l+δ_j)) / q).
        """
        tau = t / b 
        base = self.eval_analysis(tau) # TODO
        phase = np.exp(2j * np.pi * xi_1 * j * (t) / q)
        return (1.0 / np.sqrt(b)) * base * phase

class Cauchy:
    def __init__(self, alpha=300, epsilon=1e-2):
        """
        Defines the "analysis" wavelet family. We omit the 'synthesis' methods
        since we will perform inversion by the frame-operator formula.
        """
        self.alpha = float(alpha)

    def eval_analysis(self, t):
        """Continuous-time 'analysis' mother wavelet."""
        # factor^(-1 - alpha)
        factor = 1 - 2j * np.pi * t / self.alpha
        return factor ** (-1 - self.alpha)

    def eq3_analysis(self, t, j, b, q):
        """
        eq3 wavelet, alpha_j > 0
        psi_j(t) = sqrt(alpha_j) * mother(alpha_j * t).
        """
        alpha_j = (1.0 / b) + (j / q)
        return np.sqrt(alpha_j) * self.eval_analysis(alpha_j * t)

    def eq4_analysis(self, t, j, b, q, xi_1):
        """
        eq4 wavelet, alpha_j <= 0
        psi_j(t) = (1/sqrt(b)) * mother(t/b) * exp(+2 pi i xi_1 j t / q).
        """
        tau = t / b
        base = self.eval_analysis(tau)
        phase = np.exp(+2j * np.pi * xi_1 * j * t / q)
        return (1.0 / np.sqrt(b)) * base * phase

    # ----------------------------------------------------------------
    # Optional helpers for time/freq cutoff estimates:
    # ----------------------------------------------------------------
    def cutoff_time(self):
        """
        Estimate a time cutoff for wavelet support (where amplitude < epsilon).
        """
        return self.alpha * np.sqrt(self.epsilon ** (-2/(self.alpha+1)) - 1) / (2*np.pi)

    def cutoff_freq(self):
        """
        Estimate a frequency cutoff for wavelet support (where amplitude < epsilon).
        """
        denominator = ((self.alpha ** 2) *
                       (self.epsilon ** (-2/(self.alpha+1)) - 1) /
                       ((2*np.pi) ** 2))
        return 1 + 1/denominator
