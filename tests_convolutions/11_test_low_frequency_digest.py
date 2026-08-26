import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

input_h5 = "test_binning2_centered_interpolation.h5"
sigmas = [2, 4, 8, 16, 32]
bin_factor = 2

def bin_image_mean(image, bin_factor):
    h, w = image.shape
    pad_y = (-h) % bin_factor
    pad_x = (-w) % bin_factor

    image_padded = np.pad(image, ((0, pad_y), (0, pad_x)), mode="constant")
    hp, wp = image_padded.shape

    image_binned = image_padded.reshape(
        hp // bin_factor, bin_factor,
        wp // bin_factor, bin_factor
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
        fill_value=None
    )

    Y, X = np.meshgrid(y_original, x_original, indexing="ij")
    points = np.stack([Y.ravel(), X.ravel()], axis=-1)

    image_up = interpolator(points).reshape(h, w)

    return image_up.astype(np.float32)

def compute_metrics(error, border=10):
    if border > 0:
        error_used = error[border:-border, border:-border]
    else:
        error_used = error

    rmse = np.sqrt(np.mean(error_used**2))
    mae = np.mean(np.abs(error_used))
    max_error = np.max(np.abs(error_used))

    return rmse, mae, max_error

with h5py.File(input_h5, "r") as f:
    difference_full = f["difference_full"][:].astype(np.float32)

results = []

with h5py.File("11_test_low_frequency_digest.h5", "w") as f_out:
    f_out.create_dataset("difference_full", data=difference_full)

    for sigma in sigmas:
        print()
        print("sigma =", sigma)

        low_full = gaussian_filter(
            difference_full,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )

        low_bin2 = bin_image_mean(low_full, bin_factor)

        low_bin2_up = upsample_centered_interpolation(
            low_bin2,
            original_shape=difference_full.shape,
            bin_factor=bin_factor
        )

        error = low_full - low_bin2_up

        rmse, mae, max_error = compute_metrics(error, border=10)
        results.append((sigma, rmse, mae, max_error))

        print("RMSE =", rmse)
        print("MAE  =", mae)
        print("Max  =", max_error)

        group = f_out.create_group(f"sigma_{sigma}")
        group.create_dataset("low_full", data=low_full.astype(np.float32))
        group.create_dataset("low_bin2", data=low_bin2.astype(np.float32))
        group.create_dataset("low_bin2_up_centered", data=low_bin2_up.astype(np.float32))
        group.create_dataset("error", data=error.astype(np.float32))

        group.attrs["sigma"] = sigma
        group.attrs["rmse"] = float(rmse)
        group.attrs["mae"] = float(mae)
        group.attrs["max_error"] = float(max_error)

sigmas_plot = np.array([r[0] for r in results])
rmse_plot = np.array([r[1] for r in results])
mae_plot = np.array([r[2] for r in results])

plt.figure(figsize=(8, 5))
plt.plot(sigmas_plot, rmse_plot, marker="o", label="RMSE")
plt.plot(sigmas_plot, mae_plot, marker="o", label="MAE")
plt.xlabel("Gaussian sigma applied to correction")
plt.ylabel("Error after binning 2 and centered interpolation")
plt.title("Low-frequency correction: binning 2 test")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("11_test_low_frequency_digest_metrics.png", dpi=150)

selected_sigma = 8

with h5py.File("11_test_low_frequency_digest.h5", "r") as f:
    low_full = f[f"sigma_{selected_sigma}/low_full"][:]
    low_bin2_up = f[f"sigma_{selected_sigma}/low_bin2_up_centered"][:]
    error = f[f"sigma_{selected_sigma}/error"][:]

vmin = np.percentile(low_full, 1)
vmax = np.percentile(low_full, 99)
err_vmin = np.percentile(error, 1)
err_vmax = np.percentile(error, 99)

plt.figure(figsize=(14, 8))

plt.subplot(1, 3, 1)
plt.imshow(low_full, cmap="gray", vmin=vmin, vmax=vmax)
plt.title(f"Low full, sigma={selected_sigma}")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(low_bin2_up, cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Low bin2 up centered")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(error, cmap="gray", vmin=err_vmin, vmax=err_vmax)
plt.title("Error")
plt.colorbar()

plt.tight_layout()
plt.savefig("11_test_low_frequency_digest_images.png", dpi=150)

print()
print("Saved:")
print("11_test_low_frequency_digest.h5")
print("11_test_low_frequency_digest_metrics.png")
print("11_test_low_frequency_digest_images.png")
