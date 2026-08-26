#en gros ce que je vais essayer de faire : 
# 1 - prendre une vraie radio comme j ai fait avec test vraie radio
# 2 - soustraire le dark comme j ai fait avant 
# 3 - calculer la correction chill comme j ai fait  
# 4 - mtn calculer la correction sur image binned 
# 6 - remonter la correction binned a la taille normale de base 
# 7 - comparer les deux corrections : correction pleine resolution VS correction binned interpolee



import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

from diffusion_digest_copy import deconvolve_simple

# ------------------------------------------------------------
# 1. Chemins vers les vrais fichiers ESRF
# ------------------------------------------------------------

scan_dir = Path(
    "/data/scisofttmp/mirone/Rana/reco/"
    "automaticSession_ConcatenateToHeliScanScript_concatenated.etf"
)

path_proj = scan_dir / "projections.h5"
path_dark = scan_dir / "dark.h5"
path_params = scan_dir / "diffusion_parameters.json"

# ------------------------------------------------------------
# 2. Lire les paramètres de diffusion déjà utilisés
# ------------------------------------------------------------

with open(path_params, "r") as f:
    params = json.load(f)

scale_l = np.asarray(params["scale_l"], dtype=np.float32)
fraction = np.asarray(params["fraction"], dtype=np.float32)

replica_shift_x = int(params["replica_shift_x"])
replica_shift_y = int(params["replica_shift_y"])
replica_factor = float(params["replica_factor"])

mask_border = int(params["mask_border"])
mask_border_v = int(params["mask_border_v"])

# ------------------------------------------------------------
# 3. Choisir une projection réelle
# ------------------------------------------------------------

iproj = 1000

# ------------------------------------------------------------
# 4. Fonctions utiles
# ------------------------------------------------------------

def bin_image_mean(image: np.ndarray, bin_factor: int):
    """
    Binning spatial par moyenne, comme dans le code original.

    Exemple pour bin_factor = 2 :
    un bloc 2 x 2 pixels devient un seul pixel égal à la moyenne du bloc.

    La fonction ajoute éventuellement du padding en bas et à droite
    pour que la taille soit divisible par bin_factor.
    """

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

    return image_binned.astype(np.float32), pad_y, pad_x

def upsample_centered_interpolation(
    image_binned: np.ndarray,
    original_shape: tuple[int, int],
    bin_factor: int,
):
    """
    Remonte une image binned vers la taille originale en tenant compte
    de la position réelle des centres des pixels binned.

    Après un binning 2 :
    - le premier pixel binned correspond au bloc original [0,1] x [0,1]
    - son centre est donc en (0.5, 0.5), pas en (0,0)

    Plus généralement, pour bin_factor = b :
    centres binned = b/2 - 0.5, puis + b à chaque pixel.

    Pour bin_factor = 2 :
    centres = 0.5, 2.5, 4.5, ...
    """

    original_height, original_width = original_shape
    binned_height, binned_width = image_binned.shape

    # Coordonnées des centres des pixels binned dans la grille originale.
    y_binned = bin_factor * np.arange(binned_height) + (bin_factor - 1) / 2.0
    x_binned = bin_factor * np.arange(binned_width) + (bin_factor - 1) / 2.0

    # Coordonnées des pixels originaux.
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

    points = np.stack(
        [Y.ravel(), X.ravel()],
        axis=-1
    )

    image_upsampled = interpolator(points).reshape(
        original_height,
        original_width
    )

    return image_upsampled.astype(np.float32)

def compute_metrics(error: np.ndarray, border: int = 0):
    """
    Calcule des métriques simples sur l'erreur.

    Si border > 0, on ignore les bords, parce que l'interpolation
    et la déconvolution peuvent être moins fiables près des limites.
    """

    if border > 0:
        error_used = error[border:-border, border:-border]
    else:
        error_used = error

    rmse = np.sqrt(np.mean(error_used**2))
    mean_abs_error = np.mean(np.abs(error_used))
    max_abs_error = np.max(np.abs(error_used))

    return rmse, mean_abs_error, max_abs_error

# ------------------------------------------------------------
# 5. Lire la radio et le dark
# ------------------------------------------------------------

with h5py.File(path_proj, "r") as f_proj:
    radio = f_proj["data"][iproj].astype(np.float32)

with h5py.File(path_dark, "r") as f_dark:
    dark = f_dark["data"][:].astype(np.float32)

# ------------------------------------------------------------
# 6. Signal original = radio - dark
# ------------------------------------------------------------

signal = radio - dark

height, width = signal.shape

# ------------------------------------------------------------
# 7. Correction pleine résolution : référence
# ------------------------------------------------------------

corrected_full = deconvolve_simple(
    signal,
    scale_l=scale_l,
    fraction=fraction,
    replica_shift_x=replica_shift_x,
    replica_shift_y=replica_shift_y,
    replica_factor=replica_factor,
    mask_border=mask_border,
    mask_border_v=mask_border_v,
)

difference_full = signal - corrected_full

# ------------------------------------------------------------
# 8. Binning spatial facteur 2
# ------------------------------------------------------------

bin_factor = 2

signal_bin2, pad_y, pad_x = bin_image_mean(signal, bin_factor)

# ------------------------------------------------------------
# 9. Adapter les paramètres au binning 2
# ------------------------------------------------------------

scale_l_bin2 = scale_l / float(bin_factor)

replica_shift_x_bin2 = int(round(replica_shift_x / float(bin_factor)))
replica_shift_y_bin2 = int(round(replica_shift_y / float(bin_factor)))

mask_border_bin2 = mask_border // bin_factor
mask_border_v_bin2 = mask_border_v // bin_factor

# ------------------------------------------------------------
# 10. Correction sur l'image binned
# ------------------------------------------------------------

corrected_bin2 = deconvolve_simple(
    signal_bin2,
    scale_l=scale_l_bin2,
    fraction=fraction,
    replica_shift_x=replica_shift_x_bin2,
    replica_shift_y=replica_shift_y_bin2,
    replica_factor=replica_factor,
    mask_border=mask_border_bin2,
    mask_border_v=mask_border_v_bin2,
)

difference_bin2 = signal_bin2 - corrected_bin2

# ------------------------------------------------------------
# 11. Remonter la correction binned avec interpolation centrée
# ------------------------------------------------------------

difference_bin2_up_centered = upsample_centered_interpolation(
    difference_bin2,
    original_shape=signal.shape,
    bin_factor=bin_factor,
)

# ------------------------------------------------------------
# 12. Comparer correction pleine résolution et correction bin2 interpolée
# ------------------------------------------------------------

error = difference_full - difference_bin2_up_centered

rmse_all, mae_all, max_all = compute_metrics(error, border=0)
rmse_crop, mae_crop, max_crop = compute_metrics(error, border=10)

# ------------------------------------------------------------
# 13. Afficher les informations numériques
# ------------------------------------------------------------

print("Projection index:", iproj)

print("Original signal shape:", signal.shape)
print("Binned signal shape:", signal_bin2.shape)
print("Upsampled correction shape:", difference_bin2_up_centered.shape)

print("Padding added before binning:")
print("  pad_y:", pad_y)
print("  pad_x:", pad_x)

print("Original parameters:")
print("  scale_l:", scale_l)
print("  fraction:", fraction)
print("  replica_shift_x:", replica_shift_x)
print("  replica_shift_y:", replica_shift_y)
print("  replica_factor:", replica_factor)

print("Binned parameters:")
print("  scale_l_bin2:", scale_l_bin2)
print("  replica_shift_x_bin2:", replica_shift_x_bin2)
print("  replica_shift_y_bin2:", replica_shift_y_bin2)

print("Difference full min/max:")
print(" ", float(difference_full.min()), float(difference_full.max()))

print("Difference bin2 up centered min/max:")
print(" ", float(difference_bin2_up_centered.min()), float(difference_bin2_up_centered.max()))

print("Error min/max:")
print(" ", float(error.min()), float(error.max()))

print("Metrics on full image:")
print("  RMSE:", float(rmse_all))
print("  mean abs error:", float(mae_all))
print("  max abs error:", float(max_all))

print("Metrics ignoring 10-pixel border:")
print("  RMSE:", float(rmse_crop))
print("  mean abs error:", float(mae_crop))
print("  max abs error:", float(max_crop))

# ------------------------------------------------------------
# 14. Sauvegarder en HDF5 pour Silx
# ------------------------------------------------------------

with h5py.File("test_binning2_centered_interpolation.h5", "w") as f:
    f.create_dataset("signal", data=signal)
    f.create_dataset("corrected_full", data=corrected_full)
    f.create_dataset("difference_full", data=difference_full)

    f.create_dataset("signal_bin2", data=signal_bin2)
    f.create_dataset("corrected_bin2", data=corrected_bin2)
    f.create_dataset("difference_bin2", data=difference_bin2)

    f.create_dataset("difference_bin2_up_centered", data=difference_bin2_up_centered)
    f.create_dataset("error_full_minus_bin2_up_centered", data=error)

    f.attrs["iproj"] = iproj
    f.attrs["bin_factor"] = bin_factor
    f.attrs["note"] = "Binning 2 with centered interpolation: binned pixel centers at 0.5, 2.5, 4.5, ..."

# ------------------------------------------------------------
# 15. Sauvegarder une figure PNG de résumé
# ------------------------------------------------------------

center_y = height // 2

vmin = np.percentile(difference_full, 1)
vmax = np.percentile(difference_full, 99)

err_vmin = np.percentile(error, 1)
err_vmax = np.percentile(error, 99)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(difference_full, cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Difference full resolution")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(difference_bin2_up_centered, cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Difference bin2 interpolated centered")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(error, cmap="gray", vmin=err_vmin, vmax=err_vmax)
plt.title("Error = full - bin2 centered")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.plot(difference_full[center_y, :], label="full")
plt.plot(difference_bin2_up_centered[center_y, :], label="bin2 centered")
plt.plot(error[center_y, :], label="error")
plt.title("Horizontal profile at center")
plt.legend()

plt.tight_layout()
plt.savefig("test_binning2_centered_interpolation.png", dpi=150)

print("Saved HDF5: test_binning2_centered_interpolation.h5")
print("Saved PNG: test_binning2_centered_interpolation.png")
