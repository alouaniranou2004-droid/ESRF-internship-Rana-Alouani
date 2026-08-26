import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_digest_copy import deconvolve_simple

scan_dir = Path(
    "/data/scisofttmp/mirone/Rana/reco/"
    "automaticSession_ConcatenateToHeliScanScript_concatenated.etf"
)

path_proj = scan_dir / "projections.h5"
path_dark = scan_dir / "dark.h5"
path_params = scan_dir / "diffusion_parameters.json"

center_projection = 100
distances = [2, 4, 10, 20, 40]
band_height = 60

with open(path_params, "r") as f:
    params = json.load(f)

scale_l = np.asarray(params["scale_l"], dtype=np.float32)
fraction = np.asarray(params["fraction"], dtype=np.float32)

replica_shift_x = int(params["replica_shift_x"])
replica_shift_y = int(params["replica_shift_y"])
replica_factor = float(params["replica_factor"])

mask_border = int(params["mask_border"])
mask_border_v = int(params["mask_border_v"])

with h5py.File(path_dark, "r") as f_dark:
    dark = f_dark["data"][:].astype(np.float32)

def compute_corrected(iproj):
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

def metrics(error):
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    return rmse, mae, max_error

results = []

with h5py.File("16_interpolate_corrected_image.h5", "w") as f_out:

    for distance in distances:
        ia = center_projection
        ib = center_projection + distance
        i_mid = center_projection + distance // 2

        print()
        print("Distance test")
        print("ia =", ia, "i_mid =", i_mid, "ib =", ib)

        signal_a, corr_a, current_a = compute_corrected(ia)
        signal_mid, corr_mid_true, current_mid = compute_corrected(i_mid)
        signal_b, corr_b, current_b = compute_corrected(ib)

        t = (i_mid - ia) / (ib - ia)

        corr_pred_naive = (1.0 - t) * corr_a + t * corr_b

        corr_pred_current = (
            (1.0 - t) * (corr_a / current_a)
            + t * (corr_b / current_b)
        ) * current_mid

        error_naive = corr_mid_true - corr_pred_naive
        error_current = corr_mid_true - corr_pred_current

        h, w = corr_mid_true.shape
        cy = h // 2
        y0 = cy - band_height // 2
        y1 = cy + band_height // 2

        band_error_naive = error_naive[y0:y1, :]
        band_error_current = error_current[y0:y1, :]

        rmse_naive, mae_naive, max_naive = metrics(band_error_naive)
        rmse_current, mae_current, max_current = metrics(band_error_current)

        print("Naive RMSE:", rmse_naive)
        print("Current RMSE:", rmse_current)

        results.append(
            (
                distance,
                ia,
                i_mid,
                ib,
                rmse_naive,
                mae_naive,
                max_naive,
                rmse_current,
                mae_current,
                max_current,
            )
        )

        group = f_out.create_group(f"distance_{distance}")

        group.create_dataset("corrected_mid_true", data=corr_mid_true.astype(np.float32))
        group.create_dataset("corrected_pred_naive", data=corr_pred_naive.astype(np.float32))
        group.create_dataset("corrected_pred_current", data=corr_pred_current.astype(np.float32))

        group.create_dataset("error_naive", data=error_naive.astype(np.float32))
        group.create_dataset("error_current", data=error_current.astype(np.float32))

        group.create_dataset("band_error_naive", data=band_error_naive.astype(np.float32))
        group.create_dataset("band_error_current", data=band_error_current.astype(np.float32))

        group.attrs["ia"] = ia
        group.attrs["i_mid"] = i_mid
        group.attrs["ib"] = ib
        group.attrs["t"] = t
        group.attrs["current_a"] = current_a
        group.attrs["current_mid"] = current_mid
        group.attrs["current_b"] = current_b
        group.attrs["rmse_naive"] = float(rmse_naive)
        group.attrs["rmse_current"] = float(rmse_current)

with open("16_interpolate_corrected_image_metrics.txt", "w") as f:
    f.write("distance ia i_mid ib rmse_naive mae_naive max_naive rmse_current mae_current max_current\n")
    for row in results:
        f.write(" ".join(str(x) for x in row) + "\n")

distances_plot = np.array([r[0] for r in results])
rmse_naive_plot = np.array([r[4] for r in results])
rmse_current_plot = np.array([r[7] for r in results])

plt.figure(figsize=(8, 5))
plt.plot(distances_plot, rmse_naive_plot, marker="o", label="naive")
plt.plot(distances_plot, rmse_current_plot, marker="o", label="current-normalized")
plt.xlabel("Distance between known projections")
plt.ylabel("RMSE on central band")
plt.title("Corrected image interpolation error vs distance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("16_interpolate_corrected_image_metrics.png", dpi=150)

print()
print("Saved:")
print("16_interpolate_corrected_image.h5")
print("16_interpolate_corrected_image_metrics.txt")
print("16_interpolate_corrected_image_metrics.png")
