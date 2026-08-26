import json
import math
from pathlib import Path

import h5py
import numpy as np

from diffusion_digest_copy import get_kernel_fft

# ============================================================
# 1. Fichiers
# ============================================================

scan_dir = Path(
    "/data/scisofttmp/mirone/Rana/reco/"
    "automaticSession_ConcatenateToHeliScanScript_concatenated.etf"
)

path_proj = scan_dir / "projections.h5"
path_params = scan_dir / "diffusion_parameters.json"

# ============================================================
# 2. Construction explicite du noyau dans l'espace réel
# ============================================================

def shift_with_zeros(image, dy, dx):
    """
    Décale une image sans repliement périodique.
    Les zones nouvellement créées sont remplies avec des zéros.
    """
    ny, nx = image.shape
    shifted = np.zeros_like(image)

    src_y0 = max(0, -dy)
    src_y1 = min(ny, ny - dy) if dy >= 0 else ny

    dst_y0 = max(0, dy)
    dst_y1 = min(ny, ny + dy) if dy <= 0 else ny

    src_x0 = max(0, -dx)
    src_x1 = min(nx, nx - dx) if dx >= 0 else nx

    dst_x0 = max(0, dx)
    dst_x1 = min(nx, nx + dx) if dx <= 0 else nx

    if (
        src_y1 > src_y0
        and src_x1 > src_x0
        and dst_y1 > dst_y0
        and dst_x1 > dst_x0
    ):
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = image[
            src_y0:src_y1,
            src_x0:src_x1
        ]

    return shifted

def build_real_space_kernel(
    dims,
    scale_l,
    fraction,
    replica_shift_x=0,
    replica_shift_y=0,
    replica_factor=0.0,
):
    """
    Reproduit le noyau construit par get_kernel_fft(),
    mais le conserve dans l'espace réel.
    """
    scale_l = np.asarray(scale_l, dtype=np.float32)
    fraction = np.asarray(fraction, dtype=np.float32)

    if scale_l.shape != fraction.shape:
        raise ValueError("scale_l et fraction doivent avoir la même taille")

    nz, nx = dims

    # Origine FFT placée en [0, 0].
    coords_z, coords_x = np.indices(dims, dtype=np.float32)

    # Distance périodique au point [0, 0].
    distance = np.full(
        dims,
        np.hypot(nz, nx),
        dtype=np.float32,
    )

    for off_z in (0, nz):
        for off_x in (0, nx):
            candidate = np.hypot(
                coords_z - off_z,
                coords_x - off_x,
            )
            distance = np.minimum(distance, candidate)

    tail = np.zeros(dims, dtype=np.float32)
    eps = 1.0e-5

    for s, f in zip(scale_l, fraction):
        component = np.exp(-distance / s) / (distance + eps)

        # L'impulsion centrale est traitée séparément.
        component[0, 0] = 0.0

        component_sum = component.sum(dtype=np.float64)

        if not np.isfinite(component_sum) or component_sum <= 0.0:
            raise ValueError(
                f"Somme invalide pour la composante s={s}, f={f}"
            )

        # La masse totale de cette composante devient exactement f.
        component *= f / component_sum
        tail += component.astype(np.float32)

    base_response = tail.copy()

    # Réponse directe, non diffusée.
    base_response[0, 0] = 1.0

    if replica_factor != 0.0:
        # On centre temporairement le noyau pour appliquer le décalage.
        centered = np.fft.fftshift(base_response)

        shifted = shift_with_zeros(
            centered,
            dy=int(replica_shift_y),
            dx=int(replica_shift_x),
        )

        centered = centered + np.float32(replica_factor) * shifted
        kernel = np.fft.ifftshift(centered)
    else:
        kernel = base_response

    mass = kernel.sum(dtype=np.float64)

    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Masse totale invalide : {mass}")

    kernel = kernel / mass

    return kernel.astype(np.float32), distance

# ============================================================
# 3. Lire les paramètres
# ============================================================

with open(path_params, "r") as f:
    params = json.load(f)

scale_l = np.asarray(params["scale_l"], dtype=np.float32)
fraction = np.asarray(params["fraction"], dtype=np.float32)

replica_shift_x = int(params.get("replica_shift_x", 0))
replica_shift_y = int(params.get("replica_shift_y", 0))
replica_factor = float(params.get("replica_factor", 0.0))

mask_border = int(params.get("mask_border", 0))
mask_border_v = int(params.get("mask_border_v", mask_border))

# ============================================================
# 4. Reproduire les dimensions utilisées par deconvolve_simple
# ============================================================

with h5py.File(path_proj, "r") as f:
    h, w = f["data"].shape[-2:]

pad_edge_v = mask_border_v
pad_edge_x = mask_border

pad_blur = int(float(np.max(scale_l)) * 2.0)

pad_replica_x = int(math.ceil(abs(replica_shift_x)))
pad_replica_y = int(math.ceil(abs(replica_shift_y)))

pad_const_v = pad_blur + pad_replica_y
pad_const_x = pad_blur + pad_replica_x

pad_total_v = pad_edge_v + pad_const_v
pad_total_x = pad_edge_x + pad_const_x

dims = (
    h + 2 * pad_total_v,
    w + 2 * pad_total_x,
)

# ============================================================
# 5. Construire le noyau réel et comparer les FFT
# ============================================================

kernel_real, distance = build_real_space_kernel(
    dims=dims,
    scale_l=scale_l,
    fraction=fraction,
    replica_shift_x=replica_shift_x,
    replica_shift_y=replica_shift_y,
    replica_factor=replica_factor,
)

kernel_fft_manual = np.fft.fftn(kernel_real)

kernel_fft_reference = get_kernel_fft(
    dims,
    scale_l,
    fraction,
    replica_shift_x=replica_shift_x,
    replica_shift_y=replica_shift_y,
    replica_factor=replica_factor,
)

difference = kernel_fft_manual - kernel_fft_reference

max_abs_error = np.max(np.abs(difference))
mean_abs_error = np.mean(np.abs(difference))

reference_max = np.max(np.abs(kernel_fft_reference))
relative_error = max_abs_error / reference_max

# ============================================================
# 6. Vérifications
# ============================================================

print("Image shape:", (h, w))
print("Padded dimensions:", dims)
print("scale_l:", scale_l)
print("fraction:", fraction)

print()
print("Real-space kernel:")
print("  shape =", kernel_real.shape)
print("  sum   =", kernel_real.sum(dtype=np.float64))
print("  min   =", kernel_real.min())
print("  max   =", kernel_real.max())

print()
print("FFT comparison:")
print("  max absolute error  =", max_abs_error)
print("  mean absolute error =", mean_abs_error)
print("  relative error      =", relative_error)

np.save("24_kernel_real.npy", kernel_real)

if relative_error < 1.0e-5:
    print()
    print("VALIDATION OK: the real-space kernel reproduces get_kernel_fft().")
else:
    print()
    print("VALIDATION FAILED: the two kernels are different.")
   
   
   

# ============================================================
# 7. Tester plusieurs rayons de coupure
# ============================================================

scale_max = float(np.max(scale_l))

for factor in [1.0, 2.0, 3.0, 4.0]:
    radius = factor * scale_max

    kernel_cut = kernel_real.copy()
    kernel_cut[distance > radius] = 0.0

    # Renormalisation après coupure
    kernel_cut /= kernel_cut.sum(dtype=np.float64)

    fft_cut = np.fft.fftn(kernel_cut)

    fft_error = np.abs(fft_cut - kernel_fft_manual)

    max_error = np.max(fft_error)
    mean_error = np.mean(fft_error)

    lost_mass = kernel_real[distance > radius].sum(dtype=np.float64)

    print()
    print(f"Cut radius = {radius:.1f} pixels ({factor:.1f} x scale_max)")
    print("  lost kernel mass =", lost_mass)
    print("  max FFT error    =", max_error)
    print("  mean FFT error   =", mean_error) 
    
# ============================================================
# 8. Comparer l'effet sur une vraie projection
# ============================================================

with h5py.File(path_proj, "r") as f:
    projection = f["data"][0].astype(np.float32)

# Même padding que dans deconvolve_simple
projection_padded = np.pad(
    projection,
    ((pad_edge_v, pad_edge_v), (pad_edge_x, pad_edge_x)),
    mode="edge",
)

projection_padded = np.pad(
    projection_padded,
    ((pad_const_v, pad_const_v), (pad_const_x, pad_const_x)),
    mode="constant",
)

projection_fft = np.fft.fftn(projection_padded)

# Convolution avec le noyau complet
blur_full_padded = np.fft.ifftn(
    projection_fft * kernel_fft_manual
).real

blur_full = blur_full_padded[
    pad_total_v:-pad_total_v,
    pad_total_x:-pad_total_x
]

print()
print("Comparison on projection 0:")

for factor in [1.0, 2.0, 3.0, 4.0]:
    radius = factor * scale_max

    kernel_cut = kernel_real.copy()
    kernel_cut[distance > radius] = 0.0
    kernel_cut /= kernel_cut.sum(dtype=np.float64)

    kernel_cut_fft = np.fft.fftn(kernel_cut)

    blur_cut_padded = np.fft.ifftn(
        projection_fft * kernel_cut_fft
    ).real

    blur_cut = blur_cut_padded[
        pad_total_v:-pad_total_v,
        pad_total_x:-pad_total_x
    ]

    difference_image = blur_cut - blur_full

    max_abs_image_error = np.max(np.abs(difference_image))
    mean_abs_image_error = np.mean(np.abs(difference_image))
    rms_image_error = np.sqrt(np.mean(difference_image ** 2))

    reference_range = blur_full.max() - blur_full.min()

    if reference_range > 0:
        relative_max_image_error = max_abs_image_error / reference_range
        relative_rms_image_error = rms_image_error / reference_range
    else:
        relative_max_image_error = np.nan
        relative_rms_image_error = np.nan

    print()
    print(f"Cut radius = {radius:.1f} pixels ({factor:.1f} x scale_max)")
    print("  max absolute image error =", max_abs_image_error)
    print("  mean absolute image error =", mean_abs_image_error)
    print("  RMS image error          =", rms_image_error)
    print("  relative max error       =", relative_max_image_error)
    print("  relative RMS error       =", relative_rms_image_error)
    
    
    
    

