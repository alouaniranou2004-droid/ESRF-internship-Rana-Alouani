#je vais juste predre les resultats du script precedent et comparer full resolution diff et bin2 centered diff
# apres je vais appliquer des flous gaussian de plus en plus grands pour voir quand est ce que l erreur elle disparait quand on ne garde que les basses frequences>


import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------
# 1. Fichier d'entrée : résultat du test binning 2
# ------------------------------------------------------------

input_h5 = "test_binning2_centered_interpolation.h5"

# ------------------------------------------------------------
# 2. Lire les corrections à comparer
# ------------------------------------------------------------

with h5py.File(input_h5, "r") as f:
    difference_full = f["difference_full"][:].astype(np.float32)
    difference_bin2_up = f["difference_bin2_up_centered"][:].astype(np.float32)
    error_original = f["error_full_minus_bin2_up_centered"][:].astype(np.float32)

# ------------------------------------------------------------
# 3. Fonction pour calculer les métriques
# ------------------------------------------------------------

def compute_metrics(error: np.ndarray, border: int = 10):
    """
    Calcule des métriques simples sur l'erreur.

    On ignore une bordure de quelques pixels parce que les bords peuvent
    être influencés par l'interpolation, le padding ou le flou.
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
# 4. Tester plusieurs flous gaussiens
# ------------------------------------------------------------

sigmas = [0, 1, 2, 4, 8, 16, 32]

results = []

with h5py.File("test_gaussian_blur_error.h5", "w") as f_out:

    f_out.create_dataset("difference_full_original", data=difference_full)
    f_out.create_dataset("difference_bin2_up_original", data=difference_bin2_up)
    f_out.create_dataset("error_original", data=error_original)

    for sigma in sigmas:

        if sigma == 0:
            full_blur = difference_full.copy()
            bin2_blur = difference_bin2_up.copy()
        else:
            full_blur = gaussian_filter(
                difference_full,
                sigma=sigma,
                mode="nearest"
            )

            bin2_blur = gaussian_filter(
                difference_bin2_up,
                sigma=sigma,
                mode="nearest"
            )

        error_blur = full_blur - bin2_blur

        rmse, mae, max_abs = compute_metrics(error_blur, border=10)

        results.append((sigma, rmse, mae, max_abs))

        group = f_out.create_group(f"sigma_{sigma}")

        group.create_dataset("difference_full_blur", data=full_blur.astype(np.float32))
        group.create_dataset("difference_bin2_up_blur", data=bin2_blur.astype(np.float32))
        group.create_dataset("error_blur", data=error_blur.astype(np.float32))

        group.attrs["sigma"] = sigma
        group.attrs["rmse"] = float(rmse)
        group.attrs["mean_abs_error"] = float(mae)
        group.attrs["max_abs_error"] = float(max_abs)

# ------------------------------------------------------------
# 5. Afficher les résultats numériques
# ------------------------------------------------------------

print("Gaussian blur test")
print("Input file:", input_h5)
print()
print("sigma    RMSE        mean_abs_error    max_abs_error")
print("----------------------------------------------------")

for sigma, rmse, mae, max_abs in results:
    print(f"{sigma:>5}  {rmse:>10.4f}  {mae:>16.4f}  {max_abs:>14.4f}")

# ------------------------------------------------------------
# 6. Sauvegarder un fichier texte avec les résultats
# ------------------------------------------------------------

with open("test_gaussian_blur_metrics.txt", "w") as f:
    f.write("sigma RMSE mean_abs_error max_abs_error\n")

    for sigma, rmse, mae, max_abs in results:
        f.write(f"{sigma} {rmse} {mae} {max_abs}\n")

# ------------------------------------------------------------
# 7. Faire une figure : erreur en fonction du flou
# ------------------------------------------------------------

sigmas_array = np.array([r[0] for r in results], dtype=float)
rmse_array = np.array([r[1] for r in results], dtype=float)
mae_array = np.array([r[2] for r in results], dtype=float)
max_array = np.array([r[3] for r in results], dtype=float)

plt.figure(figsize=(8, 6))

plt.plot(sigmas_array, rmse_array, marker="o", label="RMSE")
plt.plot(sigmas_array, mae_array, marker="o", label="mean abs error")
plt.plot(sigmas_array, max_array, marker="o", label="max abs error")

plt.xlabel("Gaussian blur sigma in original pixels")
plt.ylabel("Error amplitude")
plt.title("Error after Gaussian low-pass filtering")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("test_gaussian_blur_metrics.png", dpi=150)

# ------------------------------------------------------------
# 8. Faire une figure visuelle pour quelques sigmas
# ------------------------------------------------------------

selected_sigmas = [0, 2, 8, 32]

plt.figure(figsize=(14, 10))

for i, sigma in enumerate(selected_sigmas):

    if sigma == 0:
        error_to_show = error_original
    else:
        error_to_show = gaussian_filter(
            error_original,
            sigma=sigma,
            mode="nearest"
        )

    vmin = np.percentile(error_to_show, 1)
    vmax = np.percentile(error_to_show, 99)

    plt.subplot(2, 2, i + 1)
    plt.imshow(error_to_show, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(f"Blurred error, sigma = {sigma}")
    plt.colorbar()

plt.tight_layout()
plt.savefig("test_gaussian_blur_error_images.png", dpi=150)

print()
print("Saved HDF5: test_gaussian_blur_error.h5")
print("Saved metrics text: test_gaussian_blur_metrics.txt")
print("Saved metrics plot: test_gaussian_blur_metrics.png")
print("Saved error images: test_gaussian_blur_error_images.png")
