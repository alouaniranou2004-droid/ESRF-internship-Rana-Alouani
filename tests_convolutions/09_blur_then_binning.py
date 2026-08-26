import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

projection_index = 1000
sigmas = [0, 1, 2, 4, 8, 16, 32]

scan_dir = (
    "/data/scisofttmp/mirone/Rana/reco/"
    "automaticSession_ConcatenateToHeliScanScript_concatenated.etf"
)

path_proj = scan_dir + "/projections.h5"
path_dark = scan_dir + "/dark.h5"

def bin_image_mean(image, bin_factor):
    height, width = image.shape

    pad_y = (-height) % bin_factor
    pad_x = (-width) % bin_factor

    image_padded = np.pad(
        image,
        ((0, pad_y), (0, pad_x)),
        mode="constant"
    )

    height_padded, width_padded = image_padded.shape

    image_binned = image_padded.reshape(
        height_padded // bin_factor,
        bin_factor,
        width_padded // bin_factor,
        bin_factor
    ).mean(axis=(1, 3))

    return image_binned.astype(np.float32)

def upsample_centered_interpolation(image_binned, original_shape, bin_factor):
    original_height, original_width = original_shape
    binned_height, binned_width = image_binned.shape

    y_binned = bin_factor * np.arange(binned_height) + (bin_factor - 1) / 2.0
    x_binned = bin_factor * np.arange(binned_width) + (bin_factor - 1) / 2.0

    y_original = np.arange(original_height)
    x_original = np.arange(original_width)

    interpolator = RegularGridInterpolator(
        (y_binned, x_binned),
        image_binned,
        method="linear",
        bounds_error=False,
        fill_value=None
    )

    Y, X = np.meshgrid(y_original, x_original, indexing="ij")
    points = np.stack([Y.ravel(), X.ravel()], axis=-1)

    image_upsampled = interpolator(points).reshape(
        original_height,
        original_width
    )

    return image_upsampled.astype(np.float32)

def compute_metrics(error, border=10):
    if border > 0:
        error_used = error[border:-border, border:-border]
    else:
        error_used = error

    rmse = np.sqrt(np.mean(error_used**2))
    mae = np.mean(np.abs(error_used))
    max_error = np.max(np.abs(error_used))

    return rmse, mae, max_error

with h5py.File(path_proj, "r") as f:
    radio = f["data"][projection_index].astype(np.float32)

with h5py.File(path_dark, "r") as f:
    dark = f["data"][:].astype(np.float32)

signal = radio - dark
print("Original shape:", signal.shape)

bin_factor = 2
results = []

with h5py.File("09_blur_then_binning.h5", "w") as f_out:
    f_out.create_dataset("signal", data=signal)

    for sigma in sigmas:
        print()
        print("sigma =", sigma)

        if sigma == 0:
            blurred = signal.copy()
        else:
            blurred = gaussian_filter(
                signal,
                sigma=sigma,
                mode="nearest",
                truncate=4.0
            )

        binned = bin_image_mean(blurred, bin_factor)

        upsampled = upsample_centered_interpolation(
            binned,
            original_shape=signal.shape,
            bin_factor=bin_factor
        )

        error = blurred - upsampled

        rmse, mae, max_error = compute_metrics(error, border=10)

        results.append((sigma, rmse, mae, max_error))

        print("RMSE =", rmse)
        print("MAE  =", mae)
        print("Max  =", max_error)

        group = f_out.create_group(f"sigma_{sigma}")
        group.create_dataset("blurred", data=blurred.astype(np.float32))
        group.create_dataset("binned", data=binned.astype(np.float32))
        group.create_dataset("upsampled_centered", data=upsampled.astype(np.float32))
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
plt.xlabel("Gaussian sigma")
plt.ylabel("Error")
plt.title("Blur -> binning 2 -> centered interpolation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("09_blur_then_binning_metrics.png", dpi=150)

print()
print("Saved:")
print("09_blur_then_binning.h5")
print("09_blur_then_binning_metrics.png")






