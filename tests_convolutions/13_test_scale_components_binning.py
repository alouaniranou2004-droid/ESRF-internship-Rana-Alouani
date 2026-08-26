import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

input_h5 = "test_binning2_centered_interpolation.h5"

scale_l = np.array([115.0, 34.63, 7.414, 3.472, 1.016], dtype=np.float32)
bin_factors = [2, 4, 8]

def make_exponential_component(shape, lamb):
    h, w = shape

    y = np.arange(h) - h // 2
    x = np.arange(w) - w // 2
    Y, X = np.meshgrid(y, x, indexing="ij")

    r = np.sqrt(X**2 + Y**2)

    eps = 1e-12
    component = np.exp(-r / lamb) / np.maximum(r, eps)

    component[h // 2, w // 2] = 0.0
    component = component / component.sum()

    return component.astype(np.float32)

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
    reference_shape = f["difference_full"].shape

print("Reference shape:", reference_shape)

results = []

with h5py.File("13_test_scale_components_binning.h5", "w") as f_out:

    for lamb in scale_l:
        print()
        print("scale_l =", lamb)

        component = make_exponential_component(reference_shape, lamb)

        scale_group = f_out.create_group(f"scale_l_{lamb:.3f}")
        scale_group.create_dataset("component_full", data=component)

        for bin_factor in bin_factors:
            component_bin = bin_image_mean(component, bin_factor)

            component_up = upsample_centered_interpolation(
                component_bin,
                original_shape=reference_shape,
                bin_factor=bin_factor
            )

            error = component - component_up

            rmse, mae, max_error = compute_metrics(error, border=10)
            results.append((lamb, bin_factor, rmse, mae, max_error))

            print("  bin =", bin_factor)
            print("    RMSE =", rmse)
            print("    MAE  =", mae)
            print("    Max  =", max_error)

            group = scale_group.create_group(f"bin_{bin_factor}")
            group.create_dataset("component_bin", data=component_bin)
            group.create_dataset("component_up_centered", data=component_up)
            group.create_dataset("error", data=error)

            group.attrs["scale_l"] = float(lamb)
            group.attrs["bin_factor"] = int(bin_factor)
            group.attrs["rmse"] = float(rmse)
            group.attrs["mae"] = float(mae)
            group.attrs["max_error"] = float(max_error)

with open("13_test_scale_components_binning_metrics.txt", "w") as f:
    f.write("scale_l bin_factor rmse mae max_error\n")
    for lamb, bin_factor, rmse, mae, max_error in results:
        f.write(f"{lamb} {bin_factor} {rmse} {mae} {max_error}\n")

plt.figure(figsize=(8, 5))

for bin_factor in bin_factors:
    lambs = []
    rmses = []

    for lamb, b, rmse, mae, max_error in results:
        if b == bin_factor:
            lambs.append(lamb)
            rmses.append(rmse)

    plt.plot(lambs, rmses, marker="o", label=f"bin {bin_factor}")

plt.xlabel("scale_l")
plt.ylabel("RMSE after binning and centered interpolation")
plt.title("Binning error for individual diffusion components")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("13_test_scale_components_binning_metrics.png", dpi=150)

print()
print("Saved:")
print("13_test_scale_components_binning.h5")
print("13_test_scale_components_binning_metrics.txt")
print("13_test_scale_components_binning_metrics.png")
