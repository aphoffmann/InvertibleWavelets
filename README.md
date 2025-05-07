# InvertibleWavelets
 
A lightweight, fully invertible wavelet transform toolkit for Python — implement time-frequency filterbanks, run forward/inverse transforms, and visualize scalograms with ease.


## Features

- **Abstract FilterBank** base class with pluggable implementations:
  - `LinearFilterBank` (linear-scale center frequencies)
  - `DyadicFilterBank` (dyadic/power-of-two scales)
- FFT-based **forward** and **inverse** transforms with overlap-add handling
- **frame operator** normalization built in to assist the dual-frame inverse
- **Scalogram** plotting (log-power over time–frequency)
- Four canonical mother wavelets:
  - `Morlet`, `Cauchy`, `MexicanHat`, `DoG`
- Pure Python, minimal dependencies (`numpy`, `scipy`, `matplotlib`)

---

## Installation

Install directly from GitHub (no PyPI publish required):

```bash
pip install git+https://github.com/aphoffmann/invertiblewavelets.git
```
## Quickstart
```
import numpy as np
import matplotlib.pyplot as plt

from invertiblewavelets.filterbank import LinearFilterBank
from invertiblewavelets.transform import Transform
from invertiblewavelets.wavelets import Morlet

# 1. Create a filter bank
fs = 1000                       # sampling rate [Hz]
data = np.sin(2*np.pi*50*np.linspace(0,1,fs,endpoint=False))
fb  = LinearFilterBank(
    wavelet=Morlet(fc=1, fb=1), fs=fs, N=fs, real=False
)

# 2. Build the Transform
xfm = Transform(data, fs, fb)

# 3. Forward & inverse
coeffs = xfm.forward()
reconstructed = xfm.inverse(coeffs)

# 4. Plot scalogram
xfm.scalogram(coeffs, title="Demo Scalogram")
plt.show()
```

