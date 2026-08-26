import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

input_h5 = "test_binning2_centered_interpolation.h5"

sigmas = [4, 8, 16, 32]
bin_factors = [2, 4, 8]

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

    return interpolator(points).reshape(h, w).astype(np.float32)

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

with h5py.File("12_test_low_frequency_multiple_binnings.h5", "w") as f_out:
    f_out.create_dataset("difference_full", data=difference_full)

    for sigma in sigmas:
        low_full = gaussian_filter(
            difference_full,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )

        sigma_group = f_out.create_group(f"sigma_{sigma}")
        sigma_group.create_dataset("low_full", data=low_full.astype(np.float32))

        for bin_factor in bin_factors:
            print()
            print("sigma =", sigma, "bin =", bin_factor)

            low_binned = bin_image_mean(low_full, bin_factor)

            low_up = upsample_centered_interpolation(
                low_binned,
                original_shape=difference_full.shape,
                bin_factor=bin_factor
            )

            error = low_full - low_up

            rmse, mae, max_error = compute_metrics(error, border=10)

            results.append((sigma, bin_factor, rmse, mae, max_error))

            print("RMSE =", rmse)
            print("MAE  =", mae)
            print("Max  =", max_error)

            group = sigma_group.create_group(f"bin_{bin_factor}")
            group.create_dataset("low_binned", data=low_binned.astype(np.float32))
            group.create_dataset("low_up_centered", data=low_up.astype(np.float32))
            group.create_dataset("error", data=error.astype(np.float32))

            group.attrs["sigma"] = sigma
            group.attrs["bin_factor"] = bin_factor
            group.attrs["rmse"] = float(rmse)
            group.attrs["mae"] = float(mae)
            group.attrs["max_error"] = float(max_error)

with open("12_test_low_frequency_multiple_binnings_metrics.txt", "w") as f:
    f.write("sigma bin_factor rmse mae max_error\n")
    for sigma, bin_factor, rmse, mae, max_error in results:
        f.write(f"{sigma} {bin_factor} {rmse} {mae} {max_error}\n")

plt.figure(figsize=(8, 5))

for bin_factor in bin_factors:
    sigmas_plot = []
    rmse_plot = []

    for sigma, b, rmse, mae, max_error in results:
        if b == bin_factor:
            sigmas_plot.append(sigma)
            rmse_plot.append(rmse)

    plt.plot(sigmas_plot, rmse_plot, marker="o", label=f"bin {bin_factor}")

plt.xlabel("Gaussian sigma applied to correction")
plt.ylabel("RMSE after binning and centered interpolation")
plt.title("Low-frequency correction vs binning factor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("12_test_low_frequency_multiple_binnings_metrics.png", dpi=150)

selected_sigma = 8

plt.figure(figsize=(14, 8))

with h5py.File("12_test_low_frequency_multiple_binnings.h5", "r") as f:
    low_full = f[f"sigma_{selected_sigma}/low_full"][:]

    vmin = np.percentile(low_full, 1)
    vmax = np.percentile(low_full, 99)

    plt.subplot(2, 2, 1)
    plt.imshow(low_full, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(f"Low full sigma={selected_sigma}")
    plt.colorbar()

    for i, bin_factor in enumerate(bin_factors):
        error = f[f"sigma_{selected_sigma}/bin_{bin_factor}/error"][:]
        err_vmin = np.percentile(error, 1)
        err_vmax = np.percentile(error, 99)

        plt.subplot(2, 2, i + 2)
        plt.imshow(error, cmap="gray", vmin=err_vmin, vmax=err_vmax)
        plt.title(f"Error bin {bin_factor}")
        plt.colorbar()

plt.tight_layout()
plt.savefig("12_test_low_frequency_multiple_binnings_images.png", dpi=150)

print()
print("Saved:")
print("12_test_low_frequency_multiple_binnings.h5")
print("12_test_low_frequency_multiple_binnings_metrics.txt")
print("12_test_low_frequency_multiple_binnings_metrics.png")
print("12_test_low_frequency_multiple_binnings_images.png")
