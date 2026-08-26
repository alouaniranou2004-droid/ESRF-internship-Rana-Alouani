import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_digest_copy import deconvolve_simple

# --------------------------------------------------
# 1. Parameters
# --------------------------------------------------

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

band_height = 60

# --------------------------------------------------
# 2. Read diffusion parameters
# --------------------------------------------------

with open(path_params, "r") as f:
    params = json.load(f)

scale_l = np.asarray(params["scale_l"], dtype=np.float32)
fraction = np.asarray(params["fraction"], dtype=np.float32)

replica_shift_x = int(params["replica_shift_x"])
replica_shift_y = int(params["replica_shift_y"])
replica_factor = float(params["replica_factor"])

mask_border = int(params["mask_border"])
mask_border_v = int(params["mask_border_v"])

# --------------------------------------------------
# 3. Read dark
# --------------------------------------------------

with h5py.File(path_dark, "r") as f_dark:
    dark = f_dark["data"][:].astype(np.float32)

# --------------------------------------------------
# 4. Function: compute full correction for one projection
# --------------------------------------------------

def compute_difference_for_projection(iproj):
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

    difference = signal - corrected

    return signal, corrected, difference, current

# --------------------------------------------------
# 5. Compute full corrections for ia, i_mid, ib
# --------------------------------------------------

signal_a, corrected_a, diff_a, current_a = compute_difference_for_projection(ia)
signal_mid, corrected_mid, diff_mid, current_mid = compute_difference_for_projection(i_mid)
signal_b, corrected_b, diff_b, current_b = compute_difference_for_projection(ib)

# --------------------------------------------------
# 6. Interpolate correction in projection index
# --------------------------------------------------

t = (i_mid - ia) / (ib - ia)

diff_pred_naive = (1.0 - t) * diff_a + t * diff_b

# --------------------------------------------------
# 7. Interpolate with current normalization
# --------------------------------------------------

diff_a_norm = diff_a / current_a
diff_b_norm = diff_b / current_b

diff_pred_norm = (1.0 - t) * diff_a_norm + t * diff_b_norm
diff_pred_current = diff_pred_norm * current_mid

# --------------------------------------------------
# 8. Compare with true full correction of i_mid
# --------------------------------------------------

error_naive = diff_mid - diff_pred_naive
error_current = diff_mid - diff_pred_current

# --------------------------------------------------
# 9. Extract horizontal band around the center
# --------------------------------------------------

height, width = diff_mid.shape
center_y = height // 2

y0 = center_y - band_height // 2
y1 = center_y + band_height // 2

band_true = diff_mid[y0:y1, :]
band_pred_naive = diff_pred_naive[y0:y1, :]
band_pred_current = diff_pred_current[y0:y1, :]

band_error_naive = error_naive[y0:y1, :]
band_error_current = error_current[y0:y1, :]

# --------------------------------------------------
# 10. Metrics
# --------------------------------------------------

def metrics(error):
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    return rmse, mae, max_error

rmse_naive, mae_naive, max_naive = metrics(band_error_naive)
rmse_current, mae_current, max_current = metrics(band_error_current)

print("Projection interpolation test")
print("ia =", ia)
print("i_mid =", i_mid)
print("ib =", ib)
print("t =", t)

print()
print("Currents:")
print("current_a =", current_a)
print("current_mid =", current_mid)
print("current_b =", current_b)

print()
print("Band:")
print("y0 =", y0)
print("y1 =", y1)
print("band shape =", band_true.shape)

print()
print("Naive interpolation error on band:")
print("RMSE =", rmse_naive)
print("MAE  =", mae_naive)
print("Max  =", max_naive)

print()
print("Current-normalized interpolation error on band:")
print("RMSE =", rmse_current)
print("MAE  =", mae_current)
print("Max  =", max_current)

# --------------------------------------------------
# 11. Save HDF5
# --------------------------------------------------

with h5py.File("14_projection_interpolation_band.h5", "w") as f:
    f.create_dataset("diff_a", data=diff_a.astype(np.float32))
    f.create_dataset("diff_mid_true", data=diff_mid.astype(np.float32))
    f.create_dataset("diff_b", data=diff_b.astype(np.float32))

    f.create_dataset("diff_pred_naive", data=diff_pred_naive.astype(np.float32))
    f.create_dataset("diff_pred_current", data=diff_pred_current.astype(np.float32))

    f.create_dataset("error_naive", data=error_naive.astype(np.float32))
    f.create_dataset("error_current", data=error_current.astype(np.float32))

    f.create_dataset("band_true", data=band_true.astype(np.float32))
    f.create_dataset("band_pred_naive", data=band_pred_naive.astype(np.float32))
    f.create_dataset("band_pred_current", data=band_pred_current.astype(np.float32))
    f.create_dataset("band_error_naive", data=band_error_naive.astype(np.float32))
    f.create_dataset("band_error_current", data=band_error_current.astype(np.float32))

    f.attrs["ia"] = ia
    f.attrs["i_mid"] = i_mid
    f.attrs["ib"] = ib
    f.attrs["t"] = t
    f.attrs["current_a"] = current_a
    f.attrs["current_mid"] = current_mid
    f.attrs["current_b"] = current_b
    f.attrs["band_y0"] = y0
    f.attrs["band_y1"] = y1

# --------------------------------------------------
# 12. Save summary figure
# --------------------------------------------------

vmin = np.percentile(band_true, 1)
vmax = np.percentile(band_true, 99)

err_vmin = np.percentile(band_error_current, 1)
err_vmax = np.percentile(band_error_current, 99)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(band_true, cmap="gray", vmin=vmin, vmax=vmax)
plt.title("True correction band, projection 105")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(band_pred_current, cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Predicted band from 100 and 110")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(band_error_current, cmap="gray", vmin=err_vmin, vmax=err_vmax)
plt.title("Error with current normalization")
plt.colorbar()

plt.subplot(2, 2, 4)
center_band_y = band_height // 2
plt.plot(band_true[center_band_y, :], label="true")
plt.plot(band_pred_current[center_band_y, :], label="predicted")
plt.plot(band_error_current[center_band_y, :], label="error")
plt.title("Horizontal profile in band")
plt.legend()

plt.tight_layout()
plt.savefig("14_projection_interpolation_band.png", dpi=150)

print()
print("Saved:")
print("14_projection_interpolation_band.h5")
print("14_projection_interpolation_band.png")
