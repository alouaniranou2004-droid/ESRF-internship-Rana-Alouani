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
R = 150
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



upK  = pK*40
upLF = pLF*40
upHF = pHF*10


pK  = normalize(pK)
pLF = normalize(pLF)
pHF = normalize(pHF)
# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 8.5),
    gridspec_kw={"height_ratios": [1.2]}
)
arrays = [Kz, LFz, HFz]
titles = [
    "Full correction kernel",
    "Low-frequency component",
    "High-frequency component"
]
# # ---------------- TOP: 2D ----------------
# for ax, A, title in zip(axes[0], arrays, titles):
#     vmax = np.percentile(np.abs(A), 99.5)
#     ax.imshow(
#         A,
#         cmap="seismic",
#         vmin=-vmax,
#         vmax=vmax,
#         extent=extent
#     )
#     ax.set_title(title, fontsize=14)
#     ax.set_xlabel("x [pixels]")
#     ax.set_ylabel("y [pixels]")


# # ---------------- BOTTOM: 1D ----------------
# profiles = [pK, pLF, pHF]
# for ax, p, title in zip(axes[1], profiles, titles):
#     ax.plot(
#         distance[mask],
#         p[mask],
#         linewidth=2.2
#     )
#     ax.axhline(0, linewidth=0.8)
#     ax.set_xlim(-R, R)
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_xlabel("Vertical distance from center [pixels]")
#     ax.set_ylabel("Normalized amplitude")
#     ax.set_title(title + " -- vertical profile", fontsize=12)
#     ax.grid(alpha=0.25)

# ---------------- BOTTOM: 1D ----------------
profiles = [upK, upLF, upHF]
for ax, p, title in zip( axes,  profiles, titles):
    ax.plot(
        distance[mask],
        p[mask],
        linewidth=2.2
    )
    ax.axhline(0, linewidth=0.8)
    ax.set_xlim(-R, R)
    ax.set_ylim(-0.001, 0.001)
    ax.set_xlabel("Vertical distance from center [pixels]")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(title + " -- vertical profile", fontsize=12)
    ax.grid(alpha=0.25)


fig.suptitle(
    "Correction kernel separation -- Gaussian filter, sigma = 8",
    fontsize=17
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    "kernel_correction_low_high_sigma8_2D_1D_FINAL.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
