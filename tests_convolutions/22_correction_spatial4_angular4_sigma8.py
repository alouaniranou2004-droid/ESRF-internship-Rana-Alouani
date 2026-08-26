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


# 1. Paramètres 

scan_dir = Path(
    "/data/scisofttmp/mirone/Rana/reco/"
    "automaticSession_ConcatenateToHeliScanScript_concatenated.etf"
)

path_proj = scan_dir / "projections.h5"
path_dark = scan_dir / "dark.h5"
path_params = scan_dir / "diffusion_parameters.json"

sigma = 8

spatial_bin = 4      # binning spatial 4x4
angular_bin = 4      # correction connue toutes les 4 projections

ia = 100
ib = ia + angular_bin
i_mid = ia + angular_bin // 2

band_height = 60
local_margins = [16, 32, 64, 128, 192]

# ============================================================
# 2. Lire les paramètres de diffusion
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
# 3. Lire le dark
# ============================================================

with h5py.File(path_dark, "r") as f_dark:
    dark = f_dark["data"][:].astype(np.float32)

# ============================================================
# 4. Fonctions utiles
# ============================================================

def read_signal(iproj):
    with h5py.File(path_proj, "r") as f_proj:
        radio = f_proj["data"][iproj].astype(np.float32)
        current = float(f_proj["framewise/control"][iproj])

    signal = radio - dark
    return signal, current

def deconvolve_signal(signal):
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

    return corrected

def compute_correction_from_signal(signal):
    corrected = deconvolve_signal(signal)
    correction = signal - corrected
    return correction

def bin_image_mean(image, bin_factor):
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
        hp // bin_factor,
        bin_factor,
        wp // bin_factor,
        bin_factor
    ).mean(axis=(1, 3))

    return image_binned.astype(np.float32)

def upsample_centered_interpolation(image_binned, original_shape, bin_factor):
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
        fill_value=None,
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
# 5. Lire les vraies projections
# ============================================================

signal_a, current_a = read_signal(ia)
signal_mid, current_mid = read_signal(i_mid)
signal_b, current_b = read_signal(ib)

h, w = signal_mid.shape
cy = h // 2

band_y0 = cy - band_height // 2
band_y1 = cy + band_height // 2

t = (i_mid - ia) / (ib - ia)

print("Full correction pipeline on correction = signal - corrected")
print("image shape:", signal_mid.shape)
print("band:", band_y0, band_y1)
print("ia, i_mid, ib:", ia, i_mid, ib)
print("angular_bin:", angular_bin)
print("spatial_bin:", spatial_bin)
print("sigma:", sigma)
print("t:", t)
print("currents:", current_a, current_mid, current_b)

# ============================================================
# 6. Référence full resolution de la correction 102
# ============================================================
# Cette correction full est calculée uniquement pour comparer.
# Dans le vrai code optimisé, on voudrait éviter de la calculer partout.

correction_mid_full = compute_correction_from_signal(signal_mid)
true_band = correction_mid_full[band_y0:band_y1, :]

low_mid_full = gaussian_filter(
    correction_mid_full,
    sigma=sigma,
    mode="nearest",
    truncate=4.0
)

high_mid_full = correction_mid_full - low_mid_full
high_band_full = high_mid_full[band_y0:band_y1, :]

# ============================================================
# 7. Basse fréquence prédite depuis corrections 100 et 104
# ============================================================

correction_a = compute_correction_from_signal(signal_a)
correction_b = compute_correction_from_signal(signal_b)

low_a = gaussian_filter(
    correction_a,
    sigma=sigma,
    mode="nearest",
    truncate=4.0
)

low_b = gaussian_filter(
    correction_b,
    sigma=sigma,
    mode="nearest",
    truncate=4.0
)

low_a_bin = bin_image_mean(low_a, spatial_bin)
low_b_bin = bin_image_mean(low_b, spatial_bin)

low_pred_bin = (
    (1.0 - t) * (low_a_bin / current_a)
    + t * (low_b_bin / current_b)
) * current_mid

low_pred_up = upsample_centered_interpolation(
    low_pred_bin,
    original_shape=signal_mid.shape,
    bin_factor=spatial_bin,
)

low_pred_band = low_pred_up[band_y0:band_y1, :]

# ============================================================
# 8. Haute fréquence locale sur la projection 102
# ============================================================

results = []

with h5py.File("23_full_correction_spatial4_angular4_local_high.h5", "w") as f_out:
    f_out.create_dataset("correction_mid_full", data=correction_mid_full.astype(np.float32))
    f_out.create_dataset("true_band", data=true_band.astype(np.float32))
    f_out.create_dataset("low_mid_full", data=low_mid_full.astype(np.float32))
    f_out.create_dataset("high_mid_full", data=high_mid_full.astype(np.float32))
    f_out.create_dataset("high_band_full_reference", data=high_band_full.astype(np.float32))

    f_out.create_dataset("low_a_bin", data=low_a_bin.astype(np.float32))
    f_out.create_dataset("low_b_bin", data=low_b_bin.astype(np.float32))
    f_out.create_dataset("low_pred_bin", data=low_pred_bin.astype(np.float32))
    f_out.create_dataset("low_pred_up", data=low_pred_up.astype(np.float32))
    f_out.create_dataset("low_pred_band", data=low_pred_band.astype(np.float32))

    for margin in local_margins:
        print()
        print("margin =", margin)

        crop_y0 = max(0, band_y0 - margin)
        crop_y1 = min(h, band_y1 + margin)

        local_band_y0 = band_y0 - crop_y0
        local_band_y1 = band_y1 - crop_y0

        signal_local = signal_mid[crop_y0:crop_y1, :]

        correction_local = compute_correction_from_signal(signal_local)

        low_local = gaussian_filter(
            correction_local,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )

        high_local = correction_local - low_local

        high_band_local = high_local[local_band_y0:local_band_y1, :]

        reconstructed_band = low_pred_band + high_band_local

        error_reconstruction = true_band - reconstructed_band
        error_low = low_mid_full[band_y0:band_y1, :] - low_pred_band
        error_high = high_band_full - high_band_local

        rmse_low, mae_low, max_low = metrics(error_low)
        rmse_high, mae_high, max_high = metrics(error_high)
        rmse_recon, mae_recon, max_recon = metrics(error_reconstruction)

        print("local crop:", crop_y0, crop_y1, "shape:", signal_local.shape)

        print("Low-frequency prediction error:")
        print("  RMSE =", rmse_low)
        print("  MAE  =", mae_low)
        print("  Max  =", max_low)

        print("High-frequency local error:")
        print("  RMSE =", rmse_high)
        print("  MAE  =", mae_high)
        print("  Max  =", max_high)

        print("Final reconstruction error:")
        print("  RMSE =", rmse_recon)
        print("  MAE  =", mae_recon)
        print("  Max  =", max_recon)

        results.append(
            (
                margin,
                crop_y0,
                crop_y1,
                rmse_low,
                mae_low,
                max_low,
                rmse_high,
                mae_high,
                max_high,
                rmse_recon,
                mae_recon,
                max_recon,
            )
        )

        group = f_out.create_group(f"margin_{margin}")

        group.create_dataset("signal_local", data=signal_local.astype(np.float32))
        group.create_dataset("correction_local", data=correction_local.astype(np.float32))
        group.create_dataset("low_local", data=low_local.astype(np.float32))
        group.create_dataset("high_local", data=high_local.astype(np.float32))

        group.create_dataset("high_band_local", data=high_band_local.astype(np.float32))
        group.create_dataset("reconstructed_band", data=reconstructed_band.astype(np.float32))

        group.create_dataset("error_low", data=error_low.astype(np.float32))
        group.create_dataset("error_high", data=error_high.astype(np.float32))
        group.create_dataset("error_reconstruction", data=error_reconstruction.astype(np.float32))

        group.attrs["margin"] = margin
        group.attrs["crop_y0"] = crop_y0
        group.attrs["crop_y1"] = crop_y1
        group.attrs["rmse_low"] = float(rmse_low)
        group.attrs["rmse_high"] = float(rmse_high)
        group.attrs["rmse_reconstruction"] = float(rmse_recon)

# ============================================================
# 9. Sauvegarder les métriques texte
# ============================================================

with open("23_full_correction_spatial4_angular4_local_high_metrics.txt", "w") as f:
    f.write(
        "margin crop_y0 crop_y1 "
        "rmse_low mae_low max_low "
        "rmse_high mae_high max_high "
        "rmse_recon mae_recon max_recon\n"
    )

    for row in results:
        f.write(" ".join(str(x) for x in row) + "\n")

# ============================================================
# 10. Graphe des erreurs
# ============================================================

margins_plot = np.array([r[0] for r in results])
rmse_low_plot = np.array([r[3] for r in results])
rmse_high_plot = np.array([r[6] for r in results])
rmse_recon_plot = np.array([r[9] for r in results])

plt.figure(figsize=(8, 5))

plt.plot(margins_plot, rmse_low_plot, marker="o", label="low prediction error")
plt.plot(margins_plot, rmse_high_plot, marker="o", label="local high error")
plt.plot(margins_plot, rmse_recon_plot, marker="o", label="final reconstruction error")

plt.xlabel("Local margin around band")
plt.ylabel("RMSE on target band")
plt.title("Correction pipeline: sigma=8, spatial bin=4, angular bin=4")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("23_full_correction_spatial4_angular4_local_high_metrics.png", dpi=150)

# ============================================================
# 11. Image résumé
# ============================================================

selected_margin = local_margins[-1]

with h5py.File("23_full_correction_spatial4_angular4_local_high.h5", "r") as f:
    recon = f[f"margin_{selected_margin}/reconstructed_band"][:]
    err = f[f"margin_{selected_margin}/error_reconstruction"][:]
    high_local_band = f[f"margin_{selected_margin}/high_band_local"][:]

vmin = np.percentile(true_band, 1)
vmax = np.percentile(true_band, 99)

err_vmin = np.percentile(err, 1)
err_vmax = np.percentile(err, 99)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(true_band, cmap="gray", vmin=vmin, vmax=vmax)
plt.title("True correction band")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(recon, cmap="gray", vmin=vmin, vmax=vmax)
plt.title(f"Reconstructed band, margin={selected_margin}")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(high_local_band, cmap="gray")
plt.title("Local high frequency")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(err, cmap="gray", vmin=err_vmin, vmax=err_vmax)
plt.title("Final reconstruction error")
plt.colorbar()

plt.tight_layout()
plt.savefig("23_full_correction_spatial4_angular4_local_high_images.png", dpi=150)

print()
print("Saved:")
print("23_full_correction_spatial4_angular4_local_high.h5")
print("23_full_correction_spatial4_angular4_local_high_metrics.txt")
print("23_full_correction_spatial4_angular4_local_high_metrics.png")
print("23_full_correction_spatial4_angular4_local_high_images.png")
