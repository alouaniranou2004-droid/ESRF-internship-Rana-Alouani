import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_digest_copy import deconvolve_simple

# ------------------------------------------------------------
# 1. Chemins vers les vrais fichiers ESRF
# ------------------------------------------------------------

scan_dir = Path("/data/scisofttmp/mirone/Rana/reco/automaticSession_ConcatenateToHeliScanScript_concatenated.etf")

path_proj = scan_dir / "projections.h5"
path_dark = scan_dir / "dark.h5"
path_params = scan_dir / "diffusion_parameters.json"

# ------------------------------------------------------------
# 2. Lire les paramètres de diffusion déjà utilisés
# ------------------------------------------------------------

with open(path_params, "r") as f:
    params = json.load(f)

scale_l = params["scale_l"]
fraction = params["fraction"]

replica_shift_x = params["replica_shift_x"]
replica_shift_y = params["replica_shift_y"]
replica_factor = params["replica_factor"]

mask_border = params["mask_border"]
mask_border_v = params["mask_border_v"]

bin_x = params["bin_x"]

# ------------------------------------------------------------
# 3. Choisir une vraie radio
# ------------------------------------------------------------

iproj = 1000

# ------------------------------------------------------------
# 4. Lire la radio et le dark
# ------------------------------------------------------------

with h5py.File(path_proj, "r") as f_proj:
    radio = f_proj["data"][iproj].astype("float32")

with h5py.File(path_dark, "r") as f_dark:
    dark = f_dark["data"][:].astype("float32")

# ------------------------------------------------------------
# 5. Soustraire le dark
# ------------------------------------------------------------

signal = radio - dark

# ------------------------------------------------------------
# 6. Si bin_x > 1, faire le même binning que le vrai code
# ------------------------------------------------------------

if bin_x > 1:
    dz, dx = signal.shape

    pad_z = (-dz) % bin_x
    pad_x = (-dx) % bin_x

    signal_padded = np.pad(signal, ((0, pad_z), (0, pad_x)), mode="constant")

    dim_z_binned = signal_padded.shape[0] // bin_x
    dim_x_binned = signal_padded.shape[1] // bin_x

    signal_binned = signal_padded.reshape(
        dim_z_binned, bin_x,
        dim_x_binned, bin_x
    ).mean(axis=(1, 3))

    scale_l_used = np.asarray(scale_l, dtype="float32") / float(bin_x)
    replica_shift_x_used = int(round(replica_shift_x / bin_x))
    replica_shift_y_used = int(round(replica_shift_y / bin_x))
    mask_border_used = mask_border // bin_x
    mask_border_v_used = mask_border_v // bin_x

else:
    signal_binned = signal

    scale_l_used = scale_l
    replica_shift_x_used = replica_shift_x
    replica_shift_y_used = replica_shift_y
    mask_border_used = mask_border
    mask_border_v_used = mask_border_v

# ------------------------------------------------------------
# 7. Appliquer la déconvolution du vrai code
# ------------------------------------------------------------

corrected = deconvolve_simple(
    signal_binned,
    scale_l=scale_l_used,
    fraction=fraction,
    replica_shift_x=replica_shift_x_used,
    replica_shift_y=replica_shift_y_used,
    replica_factor=replica_factor,
    mask_border=mask_border_used,
    mask_border_v=mask_border_v_used,
)

# ------------------------------------------------------------
# 8. Calculer la différence comme dans le big code
# ------------------------------------------------------------

difference = signal_binned - corrected

# ------------------------------------------------------------
# 9. Afficher quelques informations numériques
# ------------------------------------------------------------

print("Radio index:", iproj)
print("Signal shape:", signal.shape)
print("Signal binned shape:", signal_binned.shape)
print("Corrected shape:", corrected.shape)
print("Difference shape:", difference.shape)

print("Signal min/max:", float(signal_binned.min()), float(signal_binned.max()))
print("Corrected min/max:", float(corrected.min()), float(corrected.max()))
print("Difference min/max:", float(difference.min()), float(difference.max()))

# ------------------------------------------------------------
# 10. Sauvegarder une figure de comparaison
# ------------------------------------------------------------

center_y = signal_binned.shape[0] // 2

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(signal_binned, cmap="gray")
plt.title("Signal = radio - dark")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(corrected, cmap="gray")
plt.title("Corrected = apres deconvolution")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(difference, cmap="gray")
plt.title("Difference = signal - corrected")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.plot(signal_binned[center_y, :], label="signal")
plt.plot(corrected[center_y, :], label="corrected")
plt.plot(difference[center_y, :], label="difference")
plt.title("Profil horizontal au centre")
plt.legend()

plt.tight_layout()
plt.savefig("test_real_radio_result.png", dpi=150)

print("Figure saved as test_real_radio_result.png")
