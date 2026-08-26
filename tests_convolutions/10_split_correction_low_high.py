import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

# --------------------------------------------------
# Paramètres
# --------------------------------------------------

input_h5 = "test_binning2_centered_interpolation.h5"

sigmas = [1, 2, 4, 8, 16, 32]

# --------------------------------------------------
# Lire la correction pleine résolution
# --------------------------------------------------

with h5py.File(input_h5, "r") as f:
    difference_full = f["difference_full"][:].astype(np.float32)

# --------------------------------------------------
# Fichier de sortie
# --------------------------------------------------

with h5py.File("10_split_correction_low_high.h5", "w") as f_out:

    f_out.create_dataset("difference_full", data=difference_full)

    for sigma in sigmas:

        print("sigma =", sigma)

        # Basse fréquence = correction floutée
        low_frequency = gaussian_filter(
            difference_full,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )

        # Haute fréquence = ce qui reste après avoir enlevé la basse fréquence
        high_frequency = difference_full - low_frequency

        # Vérification : low + high doit redonner difference_full
        reconstruction = low_frequency + high_frequency
        check_error = difference_full - reconstruction

        rmse_high = np.sqrt(np.mean(high_frequency**2))
        rmse_check = np.sqrt(np.mean(check_error**2))

        print("  high frequency RMSE =", rmse_high)
        print("  reconstruction check RMSE =", rmse_check)

        group = f_out.create_group(f"sigma_{sigma}")

        group.create_dataset("low_frequency", data=low_frequency.astype(np.float32))
        group.create_dataset("high_frequency", data=high_frequency.astype(np.float32))
        group.create_dataset("reconstruction", data=reconstruction.astype(np.float32))
        group.create_dataset("check_error", data=check_error.astype(np.float32))

        group.attrs["sigma"] = sigma
        group.attrs["rmse_high_frequency"] = float(rmse_high)
        group.attrs["rmse_reconstruction_check"] = float(rmse_check)

# --------------------------------------------------
# Figure résumé
# --------------------------------------------------

selected_sigma = 8

low = gaussian_filter(
    difference_full,
    sigma=selected_sigma,
    mode="nearest",
    truncate=4.0
)

high = difference_full - low

vmin = np.percentile(difference_full, 1)
vmax = np.percentile(difference_full, 99)

plt.figure(figsize=(14, 8))

plt.subplot(1, 3, 1)
plt.imshow(difference_full, cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Correction totale")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(low, cmap="gray", vmin=vmin, vmax=vmax)
plt.title(f"Basse fréquence, sigma={selected_sigma}")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(high, cmap="gray")
plt.title(f"Haute fréquence, sigma={selected_sigma}")
plt.colorbar()

plt.tight_layout()
plt.savefig("10_split_correction_low_high.png", dpi=150)

print()
print("Saved:")
print("10_split_correction_low_high.h5")
print("10_split_correction_low_high.png")
