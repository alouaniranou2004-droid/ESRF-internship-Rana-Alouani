import h5py
import numpy as np
import matplotlib.pyplot as plt
path = "23_full_correction_spatial4_angular4_local_high.h5"
margin = 64
with h5py.File(path, "r") as f:
    low_true_full = f["low_mid_full"][:]
    low_pred_band = f["low_pred_band"][:]
    g = f[f"margin_{margin}"]
    high_local = g["high_band_local"][:]
    reconstructed = g["reconstructed_band"][:]
    error = g["error_reconstruction"][:]
# --------------------------------------------------
# Recover exact vertical position of the target band
# --------------------------------------------------
band_h = low_pred_band.shape[0]
full_h = low_true_full.shape[0]
band_y0 = (full_h - band_h) // 2
band_y1 = band_y0 + band_h
low_true_band = low_true_full[band_y0:band_y1, :]
print("True LF shape :", low_true_band.shape)
print("Pred LF shape :", low_pred_band.shape)
# --------------------------------------------------
# Common scale for TRUE LF / PREDICTED LF
# --------------------------------------------------
vmin = min(
    np.percentile(low_true_band, 1),
    np.percentile(low_pred_band, 1)
)
vmax = max(
    np.percentile(low_true_band, 99),
    np.percentile(low_pred_band, 99)
)
# Signed quantities: symmetric scales
hf_lim = np.percentile(np.abs(high_local), 99)
err_lim = np.percentile(np.abs(error), 99)
# --------------------------------------------------
# Figure
# --------------------------------------------------
fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 6.5),
    constrained_layout=True
)
axes[0,0].imshow(
    low_true_band,
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
    aspect="auto"
)
axes[0,0].set_title(
    "True low-frequency correction\nProjection 102"
)
axes[0,1].imshow(
    low_pred_band,
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
    aspect="auto"
)
axes[0,1].set_title(
    "Predicted low-frequency correction\n"
    "spatial bin = 4, angular bin = 4"
)
im2 = axes[1,0].imshow(
    high_local,
    cmap="seismic",
    vmin=-hf_lim,
    vmax=hf_lim,
    aspect="auto"
)
axes[1,0].set_title(
    "Local high-frequency correction\n"
    "margin = 64 px"
)
im3 = axes[1,1].imshow(
    error,
    cmap="seismic",
    vmin=-err_lim,
    vmax=err_lim,
    aspect="auto"
)
axes[1,1].set_title("Final reconstruction residual")
for ax in axes.flat:
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
fig.colorbar(
    im2,
    ax=axes[1,0],
    fraction=0.025,
    pad=0.02,
    label="HF correction"
)
fig.colorbar(
    im3,
    ax=axes[1,1],
    fraction=0.025,
    pad=0.02,
    label="Residual"
)
fig.suptitle(
    "Multi-scale correction: global low frequency + local high frequency",
    fontsize=15
)
plt.savefig(
    "FINAL_pipeline_multiscale_margin64.png",
    dpi=250,
    bbox_inches="tight"
)
plt.show()

