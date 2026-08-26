# -*- coding: utf-8 -*-
"""diffusion_correction.py

Spatial‑diffusion correction for tomography projections using block binning
and simple deconvolution – batching 32 slices per worker.

Each output directory gets two files:

* diffusion_correction.h5   – diffusion difference data & metadata
* diffusion_parameters.json – parameters that produced that result

If both exist and the parameters match the current command‑line options,
that directory is skipped with a clear message.  To force a rerun you must
either delete the two files or change at least one parameter.

Changes in this revision
------------------------
* Batch size = 32: every worker receives up to 32 projection indices and
  opens the HDF5 projection file only once for the whole batch.
* Worker progress print now uses *plain ASCII* ("->") to avoid mojibake when
  the terminal is not UTF‑8.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import math
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple, Dict, Any

# Register HDF5 filters before h5py is used.  This matters when the optional
# HTJ2K output path writes Blosc2/GROK chunks directly into the HDF5 file.
import hdf5plugin
hdf5plugin.register(filters=("blosc2",), force=True)

import h5py
import numpy as np
from filelock import FileLock

import pyfftw

import pyfftw.interfaces.numpy_fft 


import multiprocessing

# ----------------------------------------------------------------------------
# Globals initialised by the pool "initializer"
# ----------------------------------------------------------------------------
_dark_cache: np.ndarray | None = None  # loaded dark‑field per process
_dark_shape: Tuple[int, int] | None = None

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------


pyfftw.config.NUM_THREADS = 1
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1000)

_kernel_mem = {}
_fft2_complex_mem = {}

def _as_tuple(x):
    """Return a hashable tuple: scalar → (scalar,), sequence → tuple(x)."""
    if np.isscalar(x):
        return (float(x),)
    return tuple(float(v) for v in x)


def _normalize_diffusion_terms(scale_l, fraction):
    scale_l = np.atleast_1d(scale_l).astype("f4")
    fraction = np.atleast_1d(fraction).astype("f4")

    if scale_l.ndim != 1 or fraction.ndim != 1:
        raise ValueError("`scale_l` and `fraction` must be 1D sequences.")
    if len(scale_l) < 1:
        raise ValueError("At least one diffusion term is required.")
    if scale_l.shape != fraction.shape:
        raise ValueError(
            f"`scale_l` and `fraction` must have the same length "
            f"(got {len(scale_l)} vs {len(fraction)})."
        )
    if np.any(scale_l <= 0):
        raise ValueError("All `scale_l` values must be strictly positive.")
    if np.any(fraction < 0):
        raise ValueError("All `fraction` values must be non-negative.")

    return scale_l, fraction


def _param_sequence(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _normalize_param_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data)
    if "fraction" in normalized:
        normalized["fraction"] = _param_sequence(normalized["fraction"])
    if "scale_l" in normalized:
        normalized["scale_l"] = _param_sequence(normalized["scale_l"])
    if "replica_shift_x" in normalized:
        normalized["replica_shift_x"] = int(normalized["replica_shift_x"])
    if "replica_shift_y" in normalized:
        normalized["replica_shift_y"] = int(normalized["replica_shift_y"])
    if "replica_factor" in normalized:
        normalized["replica_factor"] = float(normalized["replica_factor"])
    return normalized


def _get_fft2_complex_ctx(dims: Tuple[int, int]) -> Dict[str, Any]:
    dims = (int(dims[0]), int(dims[1]))
    ctx = _fft2_complex_mem.get(dims)
    if ctx is not None:
        return ctx

    in_buf = pyfftw.empty_aligned(dims, dtype="complex64")
    out_buf = pyfftw.empty_aligned(dims, dtype="complex64")
    flags = (pyfftw.config.PLANNER_EFFORT,)
    plan_fwd = pyfftw.FFTW(
        in_buf,
        out_buf,
        axes=(-2, -1),
        direction="FFTW_FORWARD",
        flags=flags,
        threads=1,
    )
    plan_inv = pyfftw.FFTW(
        out_buf,
        in_buf,
        axes=(-2, -1),
        direction="FFTW_BACKWARD",
        flags=flags,
        threads=1,
    )
    ctx = {
        "in_buf": in_buf,
        "out_buf": out_buf,
        "plan_fwd": plan_fwd,
        "plan_inv": plan_inv,
    }
    _fft2_complex_mem[dims] = ctx
    return ctx


def _normalize_replica_user_params(replica_shift_x, replica_shift_y, replica_factor):
    replica_shift_x = int(replica_shift_x)
    replica_shift_y = int(replica_shift_y)
    replica_factor = float(replica_factor)
    if not np.isfinite(replica_factor):
        raise ValueError("`replica_factor` must be finite.")
    if replica_factor < 0:
        raise ValueError("`replica_factor` must be non-negative.")
    return replica_shift_x, replica_shift_y, replica_factor


def _normalize_replica_kernel_params(replica_shift_x, replica_shift_y, replica_factor):
    replica_shift_x = int(replica_shift_x)
    replica_shift_y = int(replica_shift_y)
    replica_factor = float(replica_factor)
    if not np.isfinite(replica_factor):
        raise ValueError("`replica_factor` must be finite.")
    if replica_factor < 0:
        raise ValueError("`replica_factor` must be non-negative.")
    return replica_shift_x, replica_shift_y, replica_factor


def _replica_is_active(replica_shift_x: float, replica_shift_y: float, replica_factor: float) -> bool:
    return replica_factor > 0.0 and (abs(replica_shift_x) > 0.0 or abs(replica_shift_y) > 0.0)


def _add_shifted_replica_fft_layout(
    base_response_fft: np.ndarray,
    *,
    dy: int,
    dx: int,
    factor: float,
) -> np.ndarray:
    if base_response_fft.ndim != 2:
        raise ValueError(f"Expected a 2D kernel, got shape {base_response_fft.shape}")
    if factor == 0.0 or (dx == 0 and dy == 0):
        return base_response_fft.copy()

    centered = np.fft.fftshift(base_response_fft).astype(np.float32, copy=False)
    ny, nx = centered.shape
    shifted = np.zeros_like(centered)

    src_y0 = max(0, -dy)
    src_y1 = min(ny, ny - dy) if dy >= 0 else ny
    dst_y0 = max(0, dy)
    dst_y1 = min(ny, ny + dy) if dy <= 0 else ny

    src_x0 = max(0, -dx)
    src_x1 = min(nx, nx - dx) if dx >= 0 else nx
    dst_x0 = max(0, dx)
    dst_x1 = min(nx, nx + dx) if dx <= 0 else nx

    if src_y1 > src_y0 and src_x1 > src_x0 and dst_y1 > dst_y0 and dst_x1 > dst_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = centered[src_y0:src_y1, src_x0:src_x1]

    combined = centered + np.float32(factor) * shifted
    return np.fft.ifftshift(combined).astype(np.float32, copy=False)

def get_kernel_fft(dims: tuple[int, int],
                   scale_l,        # float | Sequence[float]
                   fraction,       # float | Sequence[float]
                   replica_shift_x: float = 0.0,
                   replica_shift_y: float = 0.0,
                   replica_factor: float = 0.0):
    """
    Returns the FFT of the diffusion kernel. Now `scale_l` and `fraction`
    can be sequences (lists, tuples, ndarrays) of the same length.
    Each pair (s, f) adds an exponential tail with weight `f`.

    - The impulse at the origin remains amplitude 1.0 in the base response.
    - The total mass is 1 + sum(fraction) before the optional replica term.
    """
    scale_l, fraction = _normalize_diffusion_terms(scale_l, fraction)
    replica_shift_x, replica_shift_y, replica_factor = _normalize_replica_kernel_params(
        replica_shift_x, replica_shift_y, replica_factor
    )

    # Hashable key for cache
    key = (
        dims,
        _as_tuple(scale_l),
        _as_tuple(fraction),
        float(replica_shift_x),
        float(replica_shift_y),
        float(replica_factor),
    )
    if key in _kernel_mem:
        return _kernel_mem[key]

    # --- coordinates and minimum distance to border/replicated edge -------
    coords_z, coords_x = np.indices(dims, dtype="f4")
    distance = np.full(dims, np.hypot(*dims), dtype="f4")  # start large

    for off_z in (0, dims[0]):
        for off_x in (0, dims[1]):
            distance = np.minimum(
                distance,
                np.hypot(coords_z - off_z, coords_x - off_x)
            )

    # --- build the tail (sum of components) -------------------------------
    tail = np.zeros_like(distance)
    eps = 1e-5  # avoid division by zero

    for s, f in zip(scale_l, fraction):
        component = np.exp(-distance / s) / (distance + eps)
        component[0, 0] = 0.0
        component *= f / component.sum(dtype="f8")
        tail += component

    base_response = tail.astype("f4")
    base_response[0, 0] = 1.0

    if _replica_is_active(replica_shift_x, replica_shift_y, replica_factor):
        kernel = _add_shifted_replica_fft_layout(
            base_response,
            dy=int(replica_shift_y),
            dx=int(replica_shift_x),
            factor=float(replica_factor),
        )
    else:
        kernel = base_response

    mass = float(kernel.sum(dtype=np.float64))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Invalid kernel mass: {mass}")
    kernel = (kernel / mass).astype("f4", copy=False)

    kernel_fft = pyfftw.interfaces.numpy_fft.fftn(kernel, threads=1)
    _kernel_mem[key] = kernel_fft
    return kernel_fft


def deconvolve_simple(
    data: np.ndarray,
    *,
    safe: int = 0,
    scale_l,
    fraction,
    replica_shift_x: float = 0.0,
    replica_shift_y: float = 0.0,
    replica_factor: float = 0.0,
    mask_border: int = 100,
    mask_border_v: int | None = None,
) -> np.ndarray:
    scale_l, fraction = _normalize_diffusion_terms(scale_l, fraction)
    replica_shift_x, replica_shift_y, replica_factor = _normalize_replica_kernel_params(
        replica_shift_x, replica_shift_y, replica_factor
    )
    # If mask_border_v is not given, use mask_border
    if mask_border_v is None:
        mask_border_v = mask_border
    pad_edge_v = int(round(mask_border_v))
    pad_edge_x = int(round(mask_border))
    pad_blur = int(float(np.max(scale_l)) * 2.0)
    pad_replica_x = int(math.ceil(abs(replica_shift_x)))
    pad_replica_y = int(math.ceil(abs(replica_shift_y)))
    pad_const_v = pad_blur + pad_replica_y
    pad_const_x = pad_blur + pad_replica_x
    pad_total_v = pad_edge_v + pad_const_v
    pad_total_x = pad_edge_x + pad_const_x
    # Padding: ((vertical), (horizontal))
    data = np.pad(data, ((pad_edge_v, pad_edge_v), (pad_edge_x, pad_edge_x)), mode="edge")
    data = np.pad(data, ((pad_const_v, pad_const_v), (pad_const_x, pad_const_x)), mode="constant")
    dims = data.shape
    kernel_fft = get_kernel_fft(
        dims,
        scale_l,
        fraction,
        replica_shift_x=replica_shift_x,
        replica_shift_y=replica_shift_y,
        replica_factor=replica_factor,
    )
    ctx = _get_fft2_complex_ctx(dims)
    in_buf = ctx["in_buf"]
    out_buf = ctx["out_buf"]
    in_buf.real[:] = data
    in_buf.imag[:] = 0.0
    ctx["plan_fwd"]()
    out_buf[:] /= kernel_fft
    ctx["plan_inv"](normalise_idft=True)
    res = in_buf.real[pad_total_v:-pad_total_v, pad_total_x:-pad_total_x]
    return np.maximum(res, 0.0)


def deconvolve_simple_pair(
    data_a: np.ndarray,
    data_b: np.ndarray,
    *,
    scale_l,
    fraction,
    replica_shift_x: float = 0.0,
    replica_shift_y: float = 0.0,
    replica_factor: float = 0.0,
    mask_border: int = 100,
    mask_border_v: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    scale_l, fraction = _normalize_diffusion_terms(scale_l, fraction)
    replica_shift_x, replica_shift_y, replica_factor = _normalize_replica_kernel_params(
        replica_shift_x, replica_shift_y, replica_factor
    )

    data_a = np.asarray(data_a, dtype=np.float32)
    data_b = np.asarray(data_b, dtype=np.float32)
    if data_a.shape != data_b.shape:
        raise ValueError(f"Packed deconvolution expects equal shapes, got {data_a.shape} vs {data_b.shape}")

    if mask_border_v is None:
        mask_border_v = mask_border
    pad_edge_v = int(round(mask_border_v))
    pad_edge_x = int(round(mask_border))
    pad_blur = int(float(np.max(scale_l)) * 2.0)
    pad_replica_x = int(math.ceil(abs(replica_shift_x)))
    pad_replica_y = int(math.ceil(abs(replica_shift_y)))
    pad_const_v = pad_blur + pad_replica_y
    pad_const_x = pad_blur + pad_replica_x
    pad_total_v = pad_edge_v + pad_const_v
    pad_total_x = pad_edge_x + pad_const_x

    packed = data_a.astype(np.complex64, copy=False) + np.complex64(1j) * data_b.astype(np.complex64, copy=False)
    packed = np.pad(packed, ((pad_edge_v, pad_edge_v), (pad_edge_x, pad_edge_x)), mode="edge")
    packed = np.pad(packed, ((pad_const_v, pad_const_v), (pad_const_x, pad_const_x)), mode="constant")
    dims = packed.shape

    kernel_fft = get_kernel_fft(
        dims,
        scale_l,
        fraction,
        replica_shift_x=replica_shift_x,
        replica_shift_y=replica_shift_y,
        replica_factor=replica_factor,
    )
    ctx = _get_fft2_complex_ctx(dims)
    in_buf = ctx["in_buf"]
    out_buf = ctx["out_buf"]
    in_buf[:] = packed
    ctx["plan_fwd"]()
    out_buf[:] /= kernel_fft
    packed_corr = ctx["plan_inv"](normalise_idft=True)
    packed_corr = packed_corr[pad_total_v:-pad_total_v, pad_total_x:-pad_total_x]

    corr_a = np.maximum(packed_corr.real, 0.0).astype(np.float32, copy=False)
    corr_b = np.maximum(packed_corr.imag, 0.0).astype(np.float32, copy=False)
    return corr_a, corr_b

# ----------------------------------------------------------------------------
# Pool helpers
# ----------------------------------------------------------------------------

def _init_worker(dark_path: str) -> None:
    """Load the dark‑field once per worker process."""
    global _dark_cache, _dark_shape, _fft2_complex_mem
    _fft2_complex_mem = {}
    with h5py.File(dark_path, "r") as f_dark:
        _dark_cache = f_dark["data"][:].astype("f4", copy=False)
        _dark_shape = _dark_cache.shape

def _check_padding(h: int, w: int, blocks_z: int, blocks_x: int, bin_x: int) -> None:
    if h != blocks_z * bin_x or w != blocks_x * bin_x:
        raise RuntimeError("Padding mismatch: reshape would be invalid.")

# ----------------------------------------------------------------------------
# Worker – processes BATCH_SIZE projections at once
# ----------------------------------------------------------------------------

def process_batch(args: Tuple) -> None:
    (
        iproj_bins,           # list[int] indices in binned volume
        iprojs,               # list[int] original projection indices
        path_proj,
        dim_z_binned,
        dim_x_binned,
        bin_x,
        scale_l,
        fraction,
        replica_shift_x,
        replica_shift_y,
        replica_factor,
        mask_border,
        mask_border_v,        # new parameter
        path_out,
        current_vals,         # list[float]
        lock_path,
    ) = args

    if _dark_cache is None:
        raise RuntimeError("Dark cache not initialised in worker.")

    diff_list: List[np.ndarray] = []

    n_max_reuse = 200  # to avoid having too many open file by the dataset object
    reuse_count = 0

    BUFFER_SIZE = 200
    buffer_currents = []
    buffer_diff = []
    buffer_ipb = []
    lock_path = path_out + ".lock"
    scale_l, fraction = _normalize_diffusion_terms(scale_l, fraction)
    replica_shift_x, replica_shift_y, replica_factor = _normalize_replica_user_params(
        replica_shift_x, replica_shift_y, replica_factor
    )


    
    with h5py.File(path_proj, "r") as f_proj:
        dataset = f_proj["data"]
        shape_out = (len(iproj_bins), dim_z_binned, dim_x_binned)

        pending = None

        for idx, (ipb, ip, cur) in enumerate(zip(iproj_bins, iprojs, current_vals)):
            radio = dataset[ip].astype("f4", copy=True)
            radio -= _dark_cache
            dz, dx = radio.shape
            pad_z = (-dz) % bin_x
            pad_x = (-dx) % bin_x
            radio_padded = np.pad(radio, ((0, pad_z), (0, pad_x)), mode="constant")

            _check_padding(*radio_padded.shape, dim_z_binned, dim_x_binned, bin_x)
            radio_binned = radio_padded.reshape(dim_z_binned, bin_x, dim_x_binned, bin_x).mean(axis=(1, 3))

            sc_binned = scale_l / float(bin_x)
            replica_shift_x_binned = int(round(float(replica_shift_x) / float(bin_x)))
            replica_shift_y_binned = int(round(float(replica_shift_y) / float(bin_x)))
            # Note: use mask_border_v for vertical, mask_border for horizontal
            mb_v = mask_border_v // bin_x if mask_border_v is not None else mask_border // bin_x
            item = (
                ipb,
                cur,
                radio_binned.astype("f4", copy=False),
                sc_binned,
                replica_shift_x_binned,
                replica_shift_y_binned,
                mb_v,
            )

            if pending is None:
                pending = item
                continue

            ipb0, cur0, radio0, sc_binned0, repx0, repy0, mb_v0 = pending
            if (
                np.array_equal(sc_binned0, sc_binned)
                and repx0 == replica_shift_x_binned
                and repy0 == replica_shift_y_binned
                and mb_v0 == mb_v
            ):
                corr0, corr1 = deconvolve_simple_pair(
                    radio0,
                    radio_binned,
                    scale_l=sc_binned,
                    fraction=fraction,
                    replica_shift_x=replica_shift_x_binned,
                    replica_shift_y=replica_shift_y_binned,
                    replica_factor=replica_factor,
                    mask_border=mask_border // bin_x,
                    mask_border_v=mb_v,
                )
            else:
                corr0 = deconvolve_simple(
                    radio0,
                    scale_l=sc_binned0,
                    fraction=fraction,
                    replica_shift_x=repx0,
                    replica_shift_y=repy0,
                    replica_factor=replica_factor,
                    mask_border=mask_border // bin_x,
                    mask_border_v=mb_v0,
                )
                corr1 = deconvolve_simple(
                    radio_binned,
                    scale_l=sc_binned,
                    fraction=fraction,
                    replica_shift_x=replica_shift_x_binned,
                    replica_shift_y=replica_shift_y_binned,
                    replica_factor=replica_factor,
                    mask_border=mask_border // bin_x,
                    mask_border_v=mb_v,
                )

            diff0 = (radio0 - corr0).astype("f4", copy=False)
            diff1 = (radio_binned - corr1).astype("f4", copy=False)

            buffer_diff.append(diff0)
            buffer_ipb.append(ipb0)
            buffer_currents.append(cur0)

            buffer_diff.append(diff1)
            buffer_ipb.append(ipb)
            buffer_currents.append(cur)

            pending = None
            
            if len(buffer_diff) >= BUFFER_SIZE:
                slab_ipb = np.array(buffer_ipb)
                # Ensure contiguity
                sorted_ipb = np.sort(slab_ipb)
                if not np.all(np.diff(sorted_ipb) == 1):
                    raise RuntimeError(
                        f"ipb indices in buffer are not contiguous: {slab_ipb}"
                    )
                start_ipb = sorted_ipb[0]
                end_ipb = sorted_ipb[-1] + 1
                slab_diff = np.stack(buffer_diff)
                # If order in buffer is not increasing, reorder to match target
                
                print(f" Writing contiguous slab from {start_ipb:>10} to {end_ipb:>10}")
                with FileLock(lock_path):
                    with h5py.File(path_out, "a") as f_out:
                        ds_diff = f_out["difference"]
                        ds_current = f_out["current"]
                        
                        ds_diff[start_ipb:end_ipb] = slab_diff
                        ds_current[start_ipb:end_ipb] = buffer_currents
                        
                buffer_diff.clear()
                buffer_ipb.clear()
                buffer_currents.clear()

        if pending is not None:
            ipb0, cur0, radio0, sc_binned0, repx0, repy0, mb_v0 = pending
            corr0 = deconvolve_simple(
                radio0,
                scale_l=sc_binned0,
                fraction=fraction,
                replica_shift_x=repx0,
                replica_shift_y=repy0,
                replica_factor=replica_factor,
                mask_border=mask_border // bin_x,
                mask_border_v=mb_v0,
            )
            diff0 = (radio0 - corr0).astype("f4", copy=False)
            buffer_diff.append(diff0)
            buffer_ipb.append(ipb0)
            buffer_currents.append(cur0)

        # Write remaining buffer at the end
        if buffer_diff:
            slab_ipb = np.array(buffer_ipb)
            sorted_ipb = np.sort(slab_ipb)
            if not np.all(np.diff(sorted_ipb) == 1):
                raise RuntimeError(
                    f"ipb indices in buffer are not contiguous: {slab_ipb}"
                )
            start_ipb = sorted_ipb[0]
            end_ipb = sorted_ipb[-1] + 1
            slab_diff = np.stack(buffer_diff)
            
            print(f" Writing contiguous slab from {start_ipb:>10} to {end_ipb:>10}")
            with FileLock(lock_path):
                with h5py.File(path_out, "a") as f_out:
                    ds_diff = f_out["difference"]
                    ds_current = f_out["current"]
                    
                    ds_diff[start_ipb:end_ipb] = slab_diff
                    ds_current[start_ipb:end_ipb] = buffer_currents
                
# ----------------------------------------------------------------------------
# Parameter-file helpers
# ----------------------------------------------------------------------------

def build_param_dict(args: argparse.Namespace, bin_proj: int, bin_x: int) -> Dict[str, Any]:
    # Save mask_border_v only if specified, otherwise None
    scale_l, fraction = _normalize_diffusion_terms(args.scale_l, args.fraction)
    replica_shift_x, replica_shift_y, replica_factor = _normalize_replica_user_params(
        args.replica_shift_x,
        args.replica_shift_y,
        args.replica_factor,
    )
    return {
        "fraction": [float(v) for v in fraction],
        "scale_l": [float(v) for v in scale_l],
        "replica_shift_x": int(replica_shift_x),
        "replica_shift_y": int(replica_shift_y),
        "replica_factor": float(replica_factor),
        "mask_border": int(args.diffusion_mask_border),
        "mask_border_v": int(args.diffusion_mask_border_v) if args.diffusion_mask_border_v is not None else int(args.diffusion_mask_border),
        "bin_proj": int(bin_proj),
        "bin_x": int(bin_x),
        "hdf5_compression": str(args.hdf5_compression),
        "htj2k_cratio": float(args.htj2k_cratio) if args.hdf5_compression == "htj2k" else None,
        "htj2k_backend": str(args.htj2k_backend) if args.hdf5_compression == "htj2k" else None,
    }

def read_param_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fp:
            return _normalize_param_dict(json.load(fp))
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def write_param_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)




def _configure_htj2k_backend(backend: str) -> None:
    """Select the HTJ2K replacement backend for transparent Blosc2/GROK writes."""

    backend = (backend or "auto").lower()
    if backend == "auto":
        plugin_dir = os.environ.get("BLOSC2_GROK_HTJ2K_REPLACEMENT_DIR")
        if not plugin_dir:
            raise RuntimeError(
                "HTJ2K diffusion digest output requires BLOSC2_GROK_HTJ2K_REPLACEMENT_DIR "
                "or --htj2k_backend kakadu/openhtj2k"
            )
        if not Path(plugin_dir).is_dir():
            raise RuntimeError(f"HTJ2K backend plugin not found: {plugin_dir}")
        return
    if backend not in ("kakadu", "openhtj2k"):
        raise ValueError(f"Unsupported HTJ2K backend: {backend}")

    import importlib.util

    spec = importlib.util.find_spec("blosc2_grok")
    if spec is None or spec.origin is None:
        raise RuntimeError("blosc2_grok is required for HTJ2K diffusion digest output")
    plugin_dir = Path(spec.origin).parent / "plugins" / "htj2k" / backend
    if not plugin_dir.is_dir():
        raise RuntimeError(f"HTJ2K backend plugin not found: {plugin_dir}")
    os.environ["BLOSC2_GROK_HTJ2K_REPLACEMENT_DIR"] = str(plugin_dir)


def _htj2k_cparams(cratio: float) -> Dict[str, Any]:
    """Configure blosc2_grok for lossy HTJ2K and return Blosc2 cparams."""

    import blosc2
    import blosc2_grok

    cratio = float(cratio)
    if cratio < 1.0:
        raise ValueError("--htj2k_cratio must be >= 1.0 for lossy HTJ2K digest output")

    blosc2_grok.set_params_defaults(
        cod_format=blosc2_grok.GrkFileFmt.GRK_FMT_JP2,
        mode=blosc2_grok.GrkMode.HT,
        num_threads=1,
    )
    return {
        "codec": blosc2.Codec.GROK,
        "filters": [],
        "splitmode": blosc2.SplitMode.NEVER_SPLIT,
        "nthreads": 1,
        "codec_meta": max(1, min(255, int(round(cratio * 10.0)))),
    }


def _linear_uint16_quantization(data_min: float, data_max: float) -> tuple[float, float]:
    data_min = float(data_min)
    data_max = float(data_max)
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        raise RuntimeError("Cannot quantize diffusion digest with non-finite min/max")
    if data_max < data_min:
        raise RuntimeError("Cannot quantize diffusion digest with max smaller than min")
    return data_min, data_max


def _quantize_uint16(data: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
    if data_max == data_min:
        return np.zeros(data.shape, dtype="uint16")
    scale = 65535.0 / (data_max - data_min)
    quantized = np.rint((data.astype("f4", copy=False) - data_min) * scale)
    return np.clip(quantized, 0, 65535).astype("uint16", copy=False)


def _dataset_min_max(src: h5py.Dataset) -> tuple[float, float]:
    data_min = math.inf
    data_max = -math.inf
    for z in range(src.shape[0]):
        plane = src[z]
        data_min = min(data_min, float(np.nanmin(plane)))
        data_max = max(data_max, float(np.nanmax(plane)))
    return _linear_uint16_quantization(data_min, data_max)


def _write_htj2k_quantized_dataset(
    src: h5py.Dataset,
    dst_file: h5py.File,
    name: str,
    cratio: float,
    cparams: Dict[str, Any],
) -> h5py.Dataset:
    """Store a float32 3D dataset as linear-quantized uint16 HTJ2K chunks."""

    import blosc2

    data_min, data_max = _dataset_min_max(src)
    shape = src.shape
    chunk_h = min(128, shape[1])
    chunks = (1, chunk_h, shape[2])
    dst = dst_file.create_dataset(
        name,
        shape=shape,
        dtype="uint16",
        chunks=chunks,
        compression=32026,
    )
    dst.attrs["night_rail_quantization"] = "linear_uint16"
    dst.attrs["quantization_min"] = data_min
    dst.attrs["quantization_max"] = data_max
    dst.attrs["original_dtype"] = "float32"
    dst.attrs["compression"] = "blosc2_grok_htj2k"
    dst.attrs["cratio"] = float(cratio)

    for z in range(shape[0]):
        for y0 in range(0, shape[1], chunk_h):
            y1 = min(y0 + chunk_h, shape[1])
            chunk = np.zeros(chunks, dtype="uint16")
            valid_h = y1 - y0
            chunk[:, :valid_h, :] = _quantize_uint16(src[z:z + 1, y0:y1, :], data_min, data_max)
            b2im = blosc2.asarray(
                chunk,
                chunks=chunks,
                blocks=chunks,
                cparams=cparams,
            )
            dst.id.write_direct_chunk((z, y0, 0), b2im.schunk.to_cframe(), filter_mask=0)
    return dst


def _finalize_htj2k_output(raw_path: Path, path_out: Path, cratio: float, backend: str) -> None:
    """Convert the temporary float32 digest to the final transparent HTJ2K HDF5."""

    _configure_htj2k_backend(backend)
    cparams = _htj2k_cparams(cratio)
    tmp_final = path_out.with_suffix(path_out.suffix + ".tmp")
    if tmp_final.exists():
        tmp_final.unlink()

    with h5py.File(raw_path, "r") as src, h5py.File(tmp_final, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["diffusion_digest_hdf5_compression"] = "htj2k"
        dst.attrs["diffusion_digest_htj2k_backend"] = os.environ.get(
            "BLOSC2_GROK_HTJ2K_REPLACEMENT_DIR", ""
        )

        for dataset_name in ("corrected", "difference"):
            if dataset_name in src:
                _write_htj2k_quantized_dataset(src[dataset_name], dst, dataset_name, cratio, cparams)
        dst.create_dataset("binnings", data=src["binnings"][()])
        dst.create_dataset("current", data=src["current"][()])

    os.replace(tmp_final, path_out)
    raw_path.unlink(missing_ok=True)


def _compute_padded_shape(
    shape_yx: Tuple[int, int],
    *,
    scale_l,
    replica_shift_x: int,
    replica_shift_y: int,
    mask_border: int,
    mask_border_v: int | None,
) -> Tuple[int, int]:
    scale_l, _ = _normalize_diffusion_terms(scale_l, np.ones_like(np.atleast_1d(scale_l), dtype=np.float32))
    replica_shift_x, replica_shift_y, _ = _normalize_replica_kernel_params(replica_shift_x, replica_shift_y, 0.0)
    if mask_border_v is None:
        mask_border_v = mask_border

    pad_edge_v = int(round(mask_border_v))
    pad_edge_x = int(round(mask_border))
    pad_blur = int(float(np.max(scale_l)) * 2.0)
    pad_replica_x = int(math.ceil(abs(replica_shift_x)))
    pad_replica_y = int(math.ceil(abs(replica_shift_y)))
    pad_const_v = pad_blur + pad_replica_y
    pad_const_x = pad_blur + pad_replica_x
    return (
        int(shape_yx[0]) + 2 * (pad_edge_v + pad_const_v),
        int(shape_yx[1]) + 2 * (pad_edge_x + pad_const_x),
    )


def _prewarm_fft_plans_for_scan(
    *,
    scan_dir: str,
    dim_z_raw: int,
    dim_x_raw: int,
    dim_z_binned: int,
    dim_x_binned: int,
    bin_x: int,
    scale_l,
    fraction,
    replica_shift_x: int,
    replica_shift_y: int,
    replica_factor: float,
    mask_border: int,
    mask_border_v: int | None,
) -> None:
    try:
        start_method = multiprocessing.get_start_method()
    except RuntimeError:
        start_method = None

    if start_method != "fork":
        print(f"[prewarm] multiprocessing start method is {start_method!r}; skipping FFTW prewarm.")
        return

    print(f"[prewarm] Warming FFTW plans for {scan_dir}")

    sc_binned = np.asarray(scale_l, dtype=np.float32) / float(bin_x)
    replica_shift_x_binned = int(round(float(replica_shift_x) / float(bin_x)))
    replica_shift_y_binned = int(round(float(replica_shift_y) / float(bin_x)))
    mb_v_binned = mask_border_v // bin_x if mask_border_v is not None else mask_border // bin_x

    proj_shape = _compute_padded_shape(
        (dim_z_binned, dim_x_binned),
        scale_l=sc_binned,
        replica_shift_x=replica_shift_x_binned,
        replica_shift_y=replica_shift_y_binned,
        mask_border=mask_border // bin_x,
        mask_border_v=mb_v_binned,
    )
    flat_shape = _compute_padded_shape(
        (dim_z_raw, dim_x_raw),
        scale_l=scale_l,
        replica_shift_x=replica_shift_x,
        replica_shift_y=replica_shift_y,
        mask_border=mask_border,
        mask_border_v=mask_border_v,
    )

    # Kernel FFT plans and wisdom for projection and flat paths
    get_kernel_fft(
        proj_shape,
        sc_binned,
        fraction,
        replica_shift_x=replica_shift_x_binned,
        replica_shift_y=replica_shift_y_binned,
        replica_factor=replica_factor,
    )
    get_kernel_fft(
        flat_shape,
        scale_l,
        fraction,
        replica_shift_x=replica_shift_x,
        replica_shift_y=replica_shift_y,
        replica_factor=replica_factor,
    )

    for dims in (proj_shape, flat_shape):
        ctx = _get_fft2_complex_ctx(dims)
        ctx["in_buf"][:] = 0.0
        ctx["plan_fwd"]()
        ctx["plan_inv"](normalise_idft=True)

# ----------------------------------------------------------------------------
# Directory helpers
# ----------------------------------------------------------------------------

def build_directory_list(root: str) -> List[str]:
    dirs = [root]
    for name in os.listdir(root):
        if re.fullmatch(r"companion(?:_?\d+)?\.etf", name):
            full = os.path.join(root, name)
            if os.path.isdir(full):
                dirs.append(full)
    return dirs

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "Generate diffusion_correction.h5 and diffusion_parameters.json with "
            "diffusion difference data (batched, 32 slices per worker)."
        ),
    )
    parser.add_argument("--directory", required=True)
    parser.add_argument("--diffusion_bin_proj", type=int, required=True)
    parser.add_argument("--diffusion_bin_x", type=int, required=True)
    parser.add_argument("--scale_l", type=float, nargs="+", required=True)
    parser.add_argument("--fraction", type=float, nargs="+", required=True)
    parser.add_argument("--replica_shift_x", type=int, default=0)
    parser.add_argument("--replica_shift_y", type=int, default=0)
    parser.add_argument("--replica_factor", type=float, default=0.0)
    parser.add_argument("--n_processes", type=int, default=32)
    parser.add_argument("--diffusion_mask_border", type=int, default=100)
    parser.add_argument("--diffusion_mask_border_v", type=int, default=None)  # <--- added here
    parser.add_argument(
        "--hdf5_compression",
        choices=("none", "htj2k"),
        default="none",
        help="Store diffusion_correction.h5 as normal float32 HDF5 or transparent HTJ2K HDF5",
    )
    parser.add_argument(
        "--htj2k_cratio",
        type=float,
        default=10.0,
        help="Lossy target ratio for --hdf5_compression htj2k",
    )
    parser.add_argument(
        "--htj2k_backend",
        choices=("auto", "kakadu", "openhtj2k"),
        default="auto",
        help="HTJ2K backend plugin for --hdf5_compression htj2k",
    )

    args = parser.parse_args()
    args.scale_l, args.fraction = _normalize_diffusion_terms(args.scale_l, args.fraction)
    args.replica_shift_x, args.replica_shift_y, args.replica_factor = _normalize_replica_user_params(
        args.replica_shift_x,
        args.replica_shift_y,
        args.replica_factor,
    )

    for scan_dir in build_directory_list(args.directory):
        print("\n========== Processing:", scan_dir, "==========")
        path_proj = Path(scan_dir) / "projections.h5"
        path_dark = Path(scan_dir) / "dark.h5"
        path_out = Path(scan_dir) / "diffusion_correction.h5"
        path_work = path_out if args.hdf5_compression == "none" else Path(scan_dir) / "diffusion_correction.raw_float32.h5"
        path_json = Path(scan_dir) / "diffusion_parameters.json"
        lock_path = str(path_work) + ".lock"

        with h5py.File(path_proj, "r") as f_proj:
            nprojs, dim_z, dim_x = f_proj["data"].shape
            currents = f_proj["framewise/control"][:]

        with h5py.File(path_dark, "r") as f_dark:
            if f_dark["data"].shape != (dim_z, dim_x):
                raise RuntimeError("Dark-field shape mismatch.")

        bin_proj = args.diffusion_bin_proj
        bin_x = args.diffusion_bin_x
        nprojs_binned = (nprojs + bin_proj - 1) // bin_proj
        dim_z_binned = (dim_z + bin_x - 1) // bin_x
        dim_x_binned = (dim_x + bin_x - 1) // bin_x

        _prewarm_fft_plans_for_scan(
            scan_dir=scan_dir,
            dim_z_raw=dim_z,
            dim_x_raw=dim_x,
            dim_z_binned=dim_z_binned,
            dim_x_binned=dim_x_binned,
            bin_x=bin_x,
            scale_l=args.scale_l,
            fraction=args.fraction,
            replica_shift_x=args.replica_shift_x,
            replica_shift_y=args.replica_shift_y,
            replica_factor=args.replica_factor,
            mask_border=args.diffusion_mask_border,
            mask_border_v=args.diffusion_mask_border_v,
        )

        param_dict = build_param_dict(args, bin_proj, bin_x)
        existing = read_param_json(path_json)
        projections_up_to_date = path_out.exists() and existing == _normalize_param_dict(param_dict)
        if projections_up_to_date:
            print("Output already up-to-date for projections — skipping digest recomputation.")
        else:
            if path_out.exists():
                print("Warning: existing HDF5 found with different parameters — overwriting…")
            else:
                print("Creating new HDF5 output …")

            if path_work != path_out and path_work.exists():
                path_work.unlink()

            expected_shape = (nprojs_binned, dim_z_binned, dim_x_binned)
            with h5py.File(path_work, "w") as f_out:
                f_out.create_dataset("difference", expected_shape, dtype="f4")
                f_out.create_dataset("binnings", data=np.asarray([bin_proj, bin_x], dtype="i4"))
                f_out.create_dataset("current", shape=(nprojs_binned,), dtype="f4")

            # Build batched task list ---------------------------------------------
            task_batches = []

            BATCH_SIZE = int(math.ceil(nprojs_binned / args.n_processes))

            for batch_start in range(0, nprojs_binned, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, nprojs_binned)
                iproj_bins = list(range(batch_start, batch_end))
                iprojs = [min(b * bin_proj, nprojs - 1) for b in iproj_bins]
                current_vals = [float(currents[i]) for i in iprojs]
                task_batches.append(
                    (
                        iproj_bins,
                        iprojs,
                        str(path_proj),
                        dim_z_binned,
                        dim_x_binned,
                        bin_x,
                        args.scale_l,
                        args.fraction,
                        args.replica_shift_x,
                        args.replica_shift_y,
                        args.replica_factor,
                        args.diffusion_mask_border,
                        args.diffusion_mask_border_v ,   
                        str(path_work),
                        current_vals,
                        lock_path,
                    )
                )

            print(f"Launching {len(task_batches)} batches ({BATCH_SIZE} slices each) on {args.n_processes} processes …")
            with Pool(
                processes=args.n_processes,
                initializer=_init_worker,
                initargs=(str(path_dark),),
            ) as pool:
                pool.map(process_batch, task_batches)

            if args.hdf5_compression == "htj2k":
                print("Compressing diffusion digest to transparent HTJ2K HDF5 ...")
                _finalize_htj2k_output(path_work, path_out, args.htj2k_cratio, args.htj2k_backend)

            write_param_json(path_json, param_dict)
            print("Finished directory", scan_dir)


        # ------------------------------------------------------------------
        # Apply diffusion correction to FLATS (robust against symlinks)
        # ------------------------------------------------------------------

        path_flats = Path(scan_dir) / "flats.h5"
        path_orig_flats = Path(scan_dir) / "original_flats.h5"

        if not path_flats.exists():
            print("No flats.h5 found, skipping flat correction.")
            continue

        import shutil

        # 1) One-time true backup (mv)
        if not path_orig_flats.exists():
            print("Moving flats.h5 -> original_flats.h5")
            os.replace(path_flats, path_orig_flats)
        else:
            print("Using existing original_flats.h5")

        # 2) Recreate flats.h5 via simple copy
        print("Recreating flats.h5 from original_flats.h5")
        shutil.copy2(path_orig_flats, path_flats)

        # 3) Materialize ONLY /data to kill symlinks / external refs
        with h5py.File(path_flats, "r+") as f:
            data = f["data"][:].astype("f4", copy=False)
            del f["data"]
            f.create_dataset("data", data=data, dtype="f4")

        # Load dark
        with h5py.File(path_dark, "r") as f_dark:
            dark = f_dark["data"][:].astype("f4", copy=False)

        # Use SAME parameters as projections
        scale_l = args.scale_l
        fraction = args.fraction
        replica_shift_x = args.replica_shift_x
        replica_shift_y = args.replica_shift_y
        replica_factor = args.replica_factor
        mask_border = args.diffusion_mask_border
        mask_border_v = args.diffusion_mask_border_v

        # 4) Apply flat diffusion correction frame-wise
        with h5py.File(path_flats, "r+") as f:
            flats = f["data"]
            n_flats = flats.shape[0]

            print(f"Applying diffusion correction to {n_flats} flat frames")

            pending = None
            for i in range(n_flats):
                signal = flats[i] - dark
                if pending is None:
                    pending = (i, signal.astype("f4", copy=False))
                    continue

                i0, signal0 = pending
                corr0, corr1 = deconvolve_simple_pair(
                    signal0,
                    signal.astype("f4", copy=False),
                    scale_l=scale_l,
                    fraction=fraction,
                    replica_shift_x=replica_shift_x,
                    replica_shift_y=replica_shift_y,
                    replica_factor=replica_factor,
                    mask_border=mask_border,
                    mask_border_v=mask_border_v,
                )
                flats[i0] = corr0 + dark
                flats[i] = corr1 + dark
                pending = None

            if pending is not None:
                i0, signal0 = pending
                corr0 = deconvolve_simple(
                    signal0,
                    scale_l=scale_l,
                    fraction=fraction,
                    replica_shift_x=replica_shift_x,
                    replica_shift_y=replica_shift_y,
                    replica_factor=replica_factor,
                    mask_border=mask_border,
                    mask_border_v=mask_border_v
                )
                flats[i0] = corr0 + dark

        print("Flat-field diffusion correction completed.")
