import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from diffusion_digest_copy import deconvolve_simple

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

sigma = 8   #grand flou, comme ca j enleve bcp de details donc separation forte des frqs , avec le test j ai eu RMSE = 28 pour sigma = 8.

bin_factor = 2   #je reduis basse frequences avec binning x2 

band_height = 60  #je teste pls  marges autour de la bande pour calculer haute frequence locale

local_margins = [16, 32, 64, 128, 192]

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

def read_signal(iproj):
    with h5py.File(path_proj, "r") as f_proj:
        radio = f_proj["data"][iproj].astype(np.float32)
        current = float(f_proj["framewise/control"][iproj])
    return radio - dark, current

def deconvolve_signal(signal):
    return deconvolve_simple(
        signal,
        scale_l=scale_l,
        fraction=fraction,
        replica_shift_x=replica_shift_x,
        replica_shift_y=replica_shift_y,
        replica_factor=replica_factor,
        mask_border=mask_border,
        mask_border_v=mask_border_v,
    )    #on a les versions corrigees 

def bin_image_mean(image, bin_factor):
    h, w = image.shape

    pad_y = (-h) % bin_factor
    pad_x = (-w) % bin_factor

    padded = np.pad(image, ((0, pad_y), (0, pad_x)), mode="constant")
    hp, wp = padded.shape

    binned = padded.reshape(
        hp // bin_factor,
        bin_factor,
        wp // bin_factor,
        bin_factor
    ).mean(axis=(1, 3))

    return binned.astype(np.float32)
    # fait le binning et prend en compte le barycentring 

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
    #fonction va remettre mage binnee a sa taille normale : definit grille originale et definit vraie centres des pixels verticalement et horizontalement = renvoie l image remise en full size 
    
    

def metrics(error): # calcul des mesires d erreur 
    rmse = np.sqrt(np.mean(error**2))    #erreur quadratique moyenne 
    mae = np.mean(np.abs(error))     #erreur absolue moyenne 
    max_error = np.max(np.abs(error))      # erreur max 
    return rmse, mae, max_error


#on lit les projections 
signal_a, current_a = read_signal(ia)
signal_mid, current_mid = read_signal(i_mid)
signal_b, current_b = read_signal(ib)


 #on recupere taille et centre vertical 
h, w = signal_mid.shape  
cy = h // 2



#on definit bande centrale de 60 lignes
band_y0 = cy - band_height // 2
band_y1 = cy + band_height // 2



#calcul oposition de 105 entre 100 et 110 , ici t = 0.5
t = (i_mid - ia) / (ib - ia)

print("image shape:", signal_mid.shape)
print("band:", band_y0, band_y1)
print("indices:", ia, i_mid, ib)
print("t:", t)
print("currents:", current_a, current_mid, current_b)

# --------------------------------------------------
# Reference full correction of target projection
# Only for validation.
# This would NOT be computed in the optimized final code.
corr_mid_full = deconvolve_signal(signal_mid)
true_band = corr_mid_full[band_y0:band_y1, :]


#on extrait basse frq et high fr vraies de 105 sur la bande 
low_mid_full = gaussian_filter(
    corr_mid_full,
    sigma=sigma,
    mode="nearest",
    truncate=4.0
)

high_mid_full = corr_mid_full - low_mid_full
high_band_full = high_mid_full[band_y0:band_y1, :]

# --------------------------------------------------
# Low frequency from neighboring projections : on calcule correction 100 et 110 et on garde seulement lwur basse freq avec gaussien, puis on fait le binning x2 des basses frqs 
# --------------------------------------------------

corr_a = deconvolve_signal(signal_a)
corr_b = deconvolve_signal(signal_b)

low_a = gaussian_filter(corr_a, sigma=sigma, mode="nearest", truncate=4.0)
low_b = gaussian_filter(corr_b, sigma=sigma, mode="nearest", truncate=4.0)

low_a_bin = bin_image_mean(low_a, bin_factor)
low_b_bin = bin_image_mean(low_b, bin_factor)




# on predit basse frq de 105 en interpolant entre 100 et 110 : j ai chois de normaliser par le courant meme si j ai check que les courant sont egaux 
low_pred_bin = (
    (1.0 - t) * (low_a_bin / current_a)
    + t * (low_b_bin / current_b)
) * current_mid


# on remet basse frequence predite a taille originale 
low_pred_up = upsample_centered_interpolation(
    low_pred_bin,
    original_shape=signal_mid.shape,
    bin_factor=bin_factor
)


# on extrait juste la bande centrale 
low_pred_band = low_pred_up[band_y0:band_y1, :]

# --------------------------------------------------
# Local high frequency from the target projection
# --------------------------------------------------

results = []

with h5py.File("21_optimized_local_high_margin_test.h5", "w") as f_out:
    f_out.create_dataset("true_band", data=true_band.astype(np.float32))
    f_out.create_dataset("low_pred_band", data=low_pred_band.astype(np.float32))
    f_out.create_dataset("high_band_full_reference", data=high_band_full.astype(np.float32))

    for margin in local_margins:   # on check avec pls marges 
        print()
        print("margin =", margin)


        #on definit zone locale autour de la marge
        crop_y0 = max(0, band_y0 - margin)
        crop_y1 = min(h, band_y1 + margin)

        local_band_y0 = band_y0 - crop_y0
        local_band_y1 = band_y1 - crop_y0

        signal_local = signal_mid[crop_y0:crop_y1, :]   #on coupe seulement cette partie de la proj 105 

        corr_local = deconvolve_signal(signal_local)   #on fait la deconvolution locale pas sur toute proj

        low_local = gaussian_filter(   #on extrait la basse frequence 
            corr_local,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )

        high_local = corr_local - low_local   #on calcule haute freq locale 

        high_band_local = high_local[local_band_y0:local_band_y1, :]  #on garde juste les HF correspondant a la bande utile
        
        

        reconstructed_band = low_pred_band + high_band_local   

        error_reconstruction = true_band - reconstructed_band  #erreur finale on fait la true coriction - la reconstruction 
        error_high = high_band_full - high_band_local

        rmse_recon, mae_recon, max_recon = metrics(error_reconstruction)
        rmse_high, mae_high, max_high = metrics(error_high)

        print("local crop:", crop_y0, crop_y1, "shape:", signal_local.shape)
        print("high local error RMSE:", rmse_high)
        print("final reconstruction RMSE:", rmse_recon)

        results.append(
            (
                margin,
                crop_y0,
                crop_y1,
                rmse_high,
                mae_high,
                max_high,
                rmse_recon,
                mae_recon,
                max_recon,
            )
        )

        group = f_out.create_group(f"margin_{margin}")

        group.create_dataset("signal_local", data=signal_local.astype(np.float32))
        group.create_dataset("corr_local", data=corr_local.astype(np.float32))
        group.create_dataset("low_local", data=low_local.astype(np.float32))
        group.create_dataset("high_local", data=high_local.astype(np.float32))

        group.create_dataset("high_band_local", data=high_band_local.astype(np.float32))
        group.create_dataset("reconstructed_band", data=reconstructed_band.astype(np.float32))

        group.create_dataset("error_high", data=error_high.astype(np.float32))
        group.create_dataset("error_reconstruction", data=error_reconstruction.astype(np.float32))

        group.attrs["margin"] = margin
        group.attrs["crop_y0"] = crop_y0
        group.attrs["crop_y1"] = crop_y1
        group.attrs["rmse_high"] = float(rmse_high)
        group.attrs["rmse_reconstruction"] = float(rmse_recon)

with open("21_optimized_local_high_margin_test_metrics.txt", "w") as f:
    f.write("margin crop_y0 crop_y1 rmse_high mae_high max_high rmse_recon mae_recon max_recon\n")
    for row in results:
        f.write(" ".join(str(x) for x in row) + "\n")

margins_plot = np.array([r[0] for r in results])
rmse_high_plot = np.array([r[3] for r in results])
rmse_recon_plot = np.array([r[6] for r in results])

plt.figure(figsize=(8, 5))
plt.plot(margins_plot, rmse_high_plot, marker="o", label="high-frequency local error")
plt.plot(margins_plot, rmse_recon_plot, marker="o", label="final reconstruction error")
plt.xlabel("Local margin around band")
plt.ylabel("RMSE on target band")
plt.title(f"Local high-frequency recovery, sigma={sigma}, bin={bin_factor}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("21_optimized_local_high_margin_test_metrics.png", dpi=150)

selected_margin = local_margins[-1]

with h5py.File("21_optimized_local_high_margin_test.h5", "r") as f:
    recon = f[f"margin_{selected_margin}/reconstructed_band"][:]
    err = f[f"margin_{selected_margin}/error_reconstruction"][:]
    high_local_band = f[f"margin_{selected_margin}/high_band_local"][:]

vmin = np.percentile(true_band, 1)
vmax = np.percentile(true_band, 99)

err_vmin = np.percentile(err, 1)
err_vmax = np.percentile(err, 99)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(true_band, cmap="gray", vmin=vmin, vmax=vmax)
plt.title("True corrected band")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(recon, cmap="gray", vmin=vmin, vmax=vmax)
plt.title(f"Reconstructed band, margin={selected_margin}")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(high_local_band, cmap="gray")
plt.title("Local high frequency")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(err, cmap="gray", vmin=err_vmin, vmax=err_vmax)
plt.title("Final reconstruction error")
plt.colorbar()

plt.tight_layout()
plt.savefig("21_optimized_local_high_margin_test_images.png", dpi=150)

print()
print("Saved:")
print("21_optimized_local_high_margin_test.h5")
print("21_optimized_local_high_margin_test_metrics.txt")
print("21_optimized_local_high_margin_test_metrics.png")
print("21_optimized_local_high_margin_test_images.png")
