import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load kernels
# ------------------------------------------------------------
K  = np.load("kernel_correction_full.npy")
LF = np.load("kernel_correction_low_sigma8.npy")
HF = np.load("kernel_correction_high_sigma8.npy")

ny, nx = K.shape
cy, cx = ny // 2, nx // 2

# ------------------------------------------------------------
# Zoom for 2D views
# ------------------------------------------------------------
R = 55


Kz  = K [cy-R:cy+R+1, cx-R:cx+R+1]
LFz = LF[cy-R:cy+R+1, cx-R:cx+R+1]
HFz = HF[cy-R:cy+R+1, cx-R:cx+R+1]

extent = [-R, R, R, -R]

# ------------------------------------------------------------
# 1D vertical profiles through the kernel centre
# ------------------------------------------------------------
pK  = K[:, cx]
pLF = LF[:, cx]
pHF = HF[:, cx]

distance = np.arange(ny) - cy
mask = np.abs(distance) <= R

# Independent normalization:
# compare spatial extent, not absolute amplitude
def normalize(p):
    m = np.max(np.abs(p))
    return p / m if m != 0 else p

pK  = normalize(pK)
pLF = normalize(pLF)
pHF = normalize(pHF)

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
PYt.show()



    
