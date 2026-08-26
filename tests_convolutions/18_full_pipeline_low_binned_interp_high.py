import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from diffusion_digest_copy import deconvolve_simple

# ============================================================
# 1. Parameters
# ============================================================

scan_dir = Path(
    "/data/scisofttmp/mirone/Rana/reco/"
    "automaticSession_ConcatenateToHeliScanScript_concatenated.etf"
)

path_proj = scan_dir / "projections.h5"
path_dark = scan_dir / "dark.h5"
path_params = scan_dir / "diffusion_parameters.json"

ia = 100
i_mid = 105
ib = 110

sigmas = [4, 8, 16]
bin_factors = [2, 4]

band_height = 60

# ============================================================
# 2. Read diffusion parameters
# ============================================================

with open(path_params, "r") as f:
    params = json.load(f)

scale_l = np.asarray(params["scale_l"], dtype=np.float32)
fraction = np.asarray(params["fraction"], dtype=np.float32)

replica_shift_x = int(params["replica_shift_x"])
replica_shift_y = int(params["replica_shift_y"])
replica_factor = float(params["replica_factor"])

mask_border = int(params["mask_border"])
mask_border_v = int(params["mask_border_v"])

# ============================================================
# 3. Read dark
# ============================================================

with h5py.File(path_dark, "r") as f_dark:
    dark = f_dark["data"][:].astype(np.float32)

# ============================================================
# 4. Helper functions
# ============================================================

def compute_corrected(iproj):
    """
    Full-resolution reference computation.
    This is expensive, but here we need it to validate the method.
    """

    with h5py.File(path_proj, "r") as f_proj:
        radio = f_proj["data"][iproj].astype(np.float32)
        current = float(f_proj["framewise/control"][iproj])

    signal = radio - dark

    corrected = deconvolve_simple(
        signal,
        scale_l=scale_l,
        fraction=fraction,
        replica_shift_x=replica_shift_x,
        replica_shift_y=replica_shift_y,
        replica_factor=replica_factor,
        mask_border=mask_border,
        mask_border_v=mask_border_v,
    )

    return signal, corrected, current

def bin_image_mean(image, bin_factor):
    """
    Spatial binning by block average.
    Example bin_factor=2:
    each 2x2 block becomes one pixel.
    """

    h, w = image.shape

    pad_y = (-h) % bin_factor
    pad_x = (-w) % bin_factor

    image_padded = np.pad(
        image,
        ((0, pad_y), (0, pad_x)),
        mode="constant"
    )

    hp, wp = image_padded.shape

    image_binned = image_padded.reshape(
        hp // bin_factor, bin_factor,
        wp // bin_factor, bin_factor
    ).mean(axis=(1, 3))

    return image_binned.astype(np.float32)

def upsample_centered_interpolation(image_binned, original_shape, bin_factor):
    """
    Upsampling with correct pixel-center coordinates.

    After binning by 2, binned pixel centers are:
    0.5, 2.5, 4.5, ...
    not 0, 2, 4, ...
    """

    h, w = original_shape
    hb, wb = image_binned.shape

    y_binned = bin_factor * np.arange(hb) + (bin_factor - 1) / 2.0
    x_binned = bin_factor * np.arange(wb) + (bin_factor - 1) / 2.0

    y_original = np.arange(h)
    x_original = np.arange(w)

    interpolator = RegularGridInterpolator(
        (y_binned, x_binned),
        image_binned,
        method="linear",
        bounds_error=False,
        fill_value=None
    )

    Y, X = np.meshgrid(y_original, x_original, indexing="ij")
    points = np.stack([Y.ravel(), X.ravel()], axis=-1)

    image_up = interpolator(points).reshape(h, w)

    return image_up.astype(np.float32)

def metrics(error):
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    return rmse, mae, max_error

# ============================================================
# 5. Compute full-resolution corrected images
# ============================================================

signal_a, corr_a, current_a = compute_corrected(ia)
signal_mid, corr_mid_true, current_mid = compute_corrected(i_mid)
signal_b, corr_b, current_b = compute_corrected(ib)

t = (i_mid - ia) / (ib - ia)

h, w = corr_mid_true.shape
cy = h // 2
y0 = cy - band_height // 2
y1 = cy + band_height // 2

print("Full-resolution corrected images computed")
print("shape:", corr_mid_true.shape)
print("ia, i_mid, ib:", ia, i_mid, ib)
print("t:", t)
print("currents:", current_a, current_mid, current_b)

# ============================================================
# 6. Full pipeline:
#    corrected -> low/high separation
#    low -> binning -> projection interpolation -> upsampling
#    high -> reinjection
# ============================================================

results = []

with h5py.File("18_full_pipeline_low_binned_interp_high.h5", "w") as f_out:

    f_out.create_dataset("corrected_mid_true", data=corr_mid_true.astype(np.float32))

    for sigma in sigmas:

        low_a_full = gaussian_filter(
            corr_a,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )

        low_mid_true = gaussian_filter(
            corr_mid_true,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )

        low_b_full = gaussian_filter(
            corr_b,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )

        high_mid_true = corr_mid_true - low_mid_true

        sigma_group = f_out.create_group(f"sigma_{sigma}")
        sigma_group.create_dataset("low_mid_true", data=low_mid_true.astype(np.float32))
        sigma_group.create_dataset("high_mid_true", data=high_mid_true.astype(np.float32))

        for bin_factor in bin_factors:

            print()
            print("sigma =", sigma, "bin =", bin_factor)

            # Low frequency is stored/calculated at lower spatial resolution
            low_a_bin = bin_image_mean(low_a_full, bin_factor)
            low_b_bin = bin_image_mean(low_b_full, bin_factor)

            # Projection interpolation in binned space
            low_pred_bin_naive = (1.0 - t) * low_a_bin + t * low_b_bin

            low_pred_bin_current = (
                (1.0 - t) * (low_a_bin / current_a)
                + t * (low_b_bin / current_b)
            ) * current_mid

            # Spatial upsampling back to full size
            low_pred_up = upsample_centered_interpolation(
                low_pred_bin_current,
                original_shape=corr_mid_true.shape,
                bin_factor=bin_factor
            )

            # Reinject high frequency from the target projection
            # This is an oracle/local placeholder for now.
            reconstructed = low_pred_up + high_mid_true

            # Compare with full-resolution truth
            error_low = low_mid_true - low_pred_up
            error_recon = corr_mid_true - reconstructed

            band_error_low = error_low[y0:y1, :]
            band_error_recon = error_recon[y0:y1, :]

            rmse_low, mae_low, max_low = metrics(band_error_low)
            rmse_recon, mae_recon, max_recon = metrics(band_error_recon)

            print("Low-frequency prediction error:")
            print("  RMSE =", rmse_low)
            print("  MAE  =", mae_low)
            print("  Max  =", max_low)

            print("Final reconstruction error:")
            print("  RMSE =", rmse_recon)
            print("  MAE  =", mae_recon)
            print("  Max  =", max_recon)

            results.append(
                (
                    sigma,
                    bin_factor,
                    rmse_low,
                    mae_low,
                    max_low,
                    rmse_recon,
                    mae_recon,
                    max_recon,
                )
            )

            group = sigma_group.create_group(f"bin_{bin_factor}")

            group.create_dataset("low_a_bin", data=low_a_bin.astype(np.float32))
            group.create_dataset("low_b_bin", data=low_b_bin.astype(np.float32))
            group.create_dataset("low_pred_bin_current", data=low_pred_bin_current.astype(np.float32))
            group.create_dataset("low_pred_up", data=low_pred_up.astype(np.float32))

            group.create_dataset("reconstructed", data=reconstructed.astype(np.float32))
            group.create_dataset("error_low", data=error_low.astype(np.float32))
            group.create_dataset("error_reconstruction", data=error_recon.astype(np.float32))

            group.attrs["sigma"] = sigma
            group.attrs["bin_factor"] = bin_factor
            group.attrs["rmse_low"] = float(rmse_low)
            group.attrs["rmse_reconstruction"] = float(rmse_recon)

# ============================================================
# 7. Save metrics
# ============================================================

with open("18_full_pipeline_low_binned_interp_high_metrics.txt", "w") as f:
    f.write("sigma bin_factor rmse_low mae_low max_low rmse_recon mae_recon max_recon\n")
    for row in results:
        f.write(" ".join(str(x) for x in row) + "\n")

# ============================================================
# 8. Plot metrics
# ============================================================

plt.figure(figsize=(8, 5))

for bin_factor in bin_factors:
    sigmas_plot = []
    rmse_plot = []

    for row in results:
        sigma, b, rmse_low, mae_low, max_low, rmse_recon, mae_recon, max_recon = row
        if b == bin_factor:
            sigmas_plot.append(sigma)
            rmse_plot.append(rmse_recon)

    plt.plot(sigmas_plot, rmse_plot, marker="o", label=f"bin {bin_factor}")

plt.xlabel("Gaussian sigma")
plt.ylabel("Final reconstruction RMSE on central band")
plt.title("Low-frequency binned interpolation + high-frequency reinjection")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("18_full_pipeline_low_binned_interp_high_metrics.png", dpi=150)

# ============================================================
# 9. Summary image
# ============================================================

selected_sigma = 8
selected_bin = 2

with h5py.File("18_full_pipeline_low_binned_interp_high.h5", "r") as f:
    low_true = f[f"sigma_{selected_sigma}/low_mid_true"][:]
    high_true = f[f"sigma_{selected_sigma}/high_mid_true"][:]
    low_pred = f[f"sigma_{selected_sigma}/bin_{selected_bin}/low_pred_up"][:]
    recon = f[f"sigma_{selected_sigma}/bin_{selected_bin}/reconstructed"][:]
    err = f[f"sigma_{selected_sigma}/bin_{selected_bin}/error_reconstruction"][:]

vmin = np.percentile(corr_mid_true[y0:y1, :], 1)
vmax = np.percentile(corr_mid_true[y0:y1, :], 99)

err_vmin = np.percentile(err[y0:y1, :], 1)
err_vmax = np.percentile(err[y0:y1, :], 99)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(low_true[y0:y1, :], cmap="gray")
plt.title("True low frequency")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(low_pred[y0:y1, :], cmap="gray")
plt.title("Predicted low frequency")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(recon[y0:y1, :], cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Reconstructed = predicted low + true high")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(err[y0:y1, :], cmap="gray", vmin=err_vmin, vmax=err_vmax)
plt.title("Reconstruction error")
plt.colorbar()

plt.tight_layout()
plt.savefig("18_full_pipeline_low_binned_interp_high_images.png", dpi=150)

print()
print("Saved:")
print("18_full_pipeline_low_binned_interp_high.h5")
print("18_full_pipeline_low_binned_interp_high_metrics.txt")
print("18_full_pipeline_low_binned_interp_high_metrics.png")
print("18_full_pipeline_low_binned_interp_high_images.png")
