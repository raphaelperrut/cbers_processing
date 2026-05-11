from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
from rasterio.enums import ColorInterp
from rasterio.windows import Window

SENTINEL_TH = -1e20
EPS = 1e-6

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
    from scipy.ndimage import uniform_filter as _scipy_uniform_filter

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    _scipy_gaussian_filter = None
    _scipy_uniform_filter = None


# ----------------------------
# CUDA perf helpers
# ----------------------------
def _cuda_is_ampere_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def _log_vram(prefix: str = "[VRAM] ", reset_peak: bool = False) -> None:
    if not torch.cuda.is_available():
        return
    if reset_peak:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    alloc = int(torch.cuda.memory_allocated() / (1024 * 1024))
    rsv = int(torch.cuda.memory_reserved() / (1024 * 1024))
    peak_alloc = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    peak_rsv = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    print(
        f"{prefix}alloc={alloc}MB reserved={rsv}MB "
        f"peak_alloc={peak_alloc}MB peak_reserved={peak_rsv}MB"
    )


def _set_cuda_perf_flags(device: str, *, verbose: bool) -> tuple[bool, bool]:
    if device != "cuda":
        return False, False

    torch.backends.cudnn.benchmark = True

    tf32 = _cuda_is_ampere_or_newer()
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = tf32
    except Exception:
        pass

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if verbose:
        print(f"[CUDA] tf32={tf32} cudnn_benchmark={torch.backends.cudnn.benchmark}")
        _log_vram("[CUDA] ", reset_peak=True)

    return tf32, torch.backends.cudnn.benchmark


# ----------------------------
# Utils
# ----------------------------
def sanitize_array(a: np.ndarray, name: str = "", verbose: bool = False) -> tuple[np.ndarray, bool]:
    bad = ~np.isfinite(a)
    had_bad = bool(bad.any())
    if had_bad:
        if verbose:
            cnt = int(bad.sum())
            print(f"[WARN] {name}: encontrado NaN/Inf = {cnt} -> substituindo por 0")
        a = a.copy()
        a[bad] = 0.0
    return a, had_bad


def pad_reflect(tile: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    _, h, w = tile.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return tile
    return np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")


def pad_reflect_2d(tile: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = tile.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return tile
    return np.pad(tile, ((0, pad_h), (0, pad_w)), mode="reflect")


def make_positions(total: int, tile: int, stride: int) -> list[int]:
    if total <= tile:
        return [0]
    pos = []
    p = 0
    while True:
        if p + tile >= total:
            pos.append(total - tile)
            break
        pos.append(p)
        p += stride
    return sorted(set(pos))


def make_weight(h: int, w: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)

    def ramp(n: int) -> np.ndarray:
        t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
        return np.sin(t)

    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    r = ramp(overlap)
    wy[:overlap] = r
    wy[-overlap:] = r[::-1]
    wx[:overlap] = r
    wx[-overlap:] = r[::-1]

    eps = 1e-3
    wy = np.clip(wy, eps, 1.0)
    wx = np.clip(wx, eps, 1.0)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _grad_mag2d(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    gx = np.zeros_like(a, dtype=np.float32)
    gy = np.zeros_like(a, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
    gy[1:-1, :] = 0.5 * (a[2:, :] - a[:-2, :])
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _kernel_size_from_sigma(sigma: float) -> int:
    if sigma <= 0:
        return 1
    return max(3, int(round(6 * sigma)) | 1)


def _local_std2d(a: np.ndarray, sigma: float) -> np.ndarray:
    ksize = _kernel_size_from_sigma(sigma)
    mu = gaussian_blur2d(a, sigma=sigma, ksize=ksize)
    mu2 = gaussian_blur2d(a * a, sigma=sigma, ksize=ksize)
    var = np.maximum(mu2 - mu * mu, 0.0)
    return np.sqrt(var).astype(np.float32)


# ----------------------------
# Norm / Output
# ----------------------------
def _sample_windows(W: int, H: int, win_w: int = 1024, win_h: int = 1024) -> list[Window]:
    win_w = min(win_w, W)
    win_h = min(win_h, H)
    pts = [
        (0, 0),
        (max(0, W - win_w), 0),
        (0, max(0, H - win_h)),
        (max(0, W - win_w), max(0, H - win_h)),
        (max(0, (W - win_w) // 2), max(0, (H - win_h) // 2)),
        (max(0, (W - win_w) // 4), max(0, (H - win_h) // 4)),
        (max(0, (3 * (W - win_w)) // 4), max(0, (H - win_h) // 4)),
        (max(0, (W - win_w) // 4), max(0, (3 * (H - win_h)) // 4)),
        (max(0, (3 * (W - win_w)) // 4), max(0, (3 * (H - win_h)) // 4)),
    ]
    seen = set()
    out = []
    for (x, y) in pts:
        key = (int(x), int(y), int(win_w), int(win_h))
        if key in seen:
            continue
        seen.add(key)
        out.append(Window(x, y, win_w, win_h))
    return out


def _estimate_max(src: rasterio.DatasetReader, band_indexes: list[int], verbose: bool) -> float:
    W, H = src.width, src.height
    wins = _sample_windows(W, H, win_w=1024, win_h=1024)

    m = -np.inf
    for w in wins:
        arr = src.read(indexes=band_indexes, window=w).astype(np.float32)
        arr = np.where(arr < SENTINEL_TH, 0.0, arr)
        v = float(np.nanmax(arr))
        m = max(m, v)

    if verbose:
        print(f"[INFO] max estimado por amostragem: {m:.6f}")
    if not np.isfinite(m):
        return 0.0
    return m


def resolve_norm_mode(mode: str, est_max: float) -> str:
    if mode != "auto":
        return mode
    return "0_255" if est_max > 1.5 else "0_1"


def apply_norm_tile(tile: np.ndarray, chosen: str) -> np.ndarray:
    out = tile.astype(np.float32)
    if chosen == "none":
        return out
    if chosen == "0_255":
        out = out / 255.0
        return np.clip(out, 0.0, 1.0)
    if chosen == "0_1":
        return np.clip(out, 0.0, 1.0)
    if chosen == "minus1_1":
        out = np.clip(out, 0.0, 1.0)
        return out * 2.0 - 1.0
    raise ValueError(f"Modo de normalização desconhecido: {chosen}")


def apply_out_range(out: np.ndarray, out_range: str) -> np.ndarray:
    y = out.astype(np.float32)
    if out_range == "none":
        return y
    if out_range == "0_1":
        return np.clip(y, 0.0, 1.0)
    if out_range == "minus1_1":
        return np.clip(y, -1.0, 1.0)
    raise ValueError(f"out_range desconhecido: {out_range}")


# ----------------------------
# Color helpers
# ----------------------------
def rgb_to_ycbcr(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb01[0], rgb01[1], rgb01[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def ycbcr_to_rgb_noclip(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb2 = cb - 0.5
    cr2 = cr - 0.5
    r = y + 1.402 * cr2
    g = y - 0.344136 * cb2 - 0.714136 * cr2
    b = y + 1.772 * cb2
    return np.stack([r, g, b], axis=0).astype(np.float32)


def _compress_rgb_overflow(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y, cb, cr = rgb_to_ycbcr(np.clip(rgb, -10.0, 10.0).astype(np.float32, copy=False))
    y2 = y.astype(np.float32, copy=False)

    hi_mask = y2 > 1.0
    if np.any(hi_mask):
        z = y2[hi_mask] - 1.0
        y2 = y2.copy()
        y2[hi_mask] = 1.0 + z / (1.0 + z)

    lo_mask = y2 < 0.0
    if np.any(lo_mask):
        z = -y2[lo_mask]
        y2 = y2.copy()
        y2[lo_mask] = -z / (1.0 + z)

    rgb2 = ycbcr_to_rgb_noclip(y2, cb, cr)

    hi = rgb2.max(axis=0)
    scale_hi = np.ones_like(hi, dtype=np.float32)
    m_hi = hi > 1.0
    scale_hi[m_hi] = 1.0 / (hi[m_hi] + eps)
    rgb3 = rgb2 * scale_hi[None, :, :]

    lo2 = rgb3.min(axis=0)
    m_lo = lo2 < 0.0
    if np.any(m_lo):
        shift = (-lo2[m_lo]).astype(np.float32)
        rgb3[:, m_lo] += shift[None, :]
        hi3 = rgb3.max(axis=0)
        scale2 = np.ones_like(hi3, dtype=np.float32)
        m_hi2 = hi3 > 1.0
        scale2[m_hi2] = 1.0 / (hi3[m_hi2] + eps)
        rgb3 = rgb3 * scale2[None, :, :]

    return np.clip(rgb3, 0.0, 1.0).astype(np.float32)


def apply_sat_ycbcr(cb: np.ndarray, cr: np.ndarray, sat: float) -> tuple[np.ndarray, np.ndarray]:
    cb2 = (cb - 0.5) * sat + 0.5
    cr2 = (cr - 0.5) * sat + 0.5
    return np.clip(cb2, 0.0, 1.0), np.clip(cr2, 0.0, 1.0)


def _gauss_kernel1d(sigma: float, ksize: int) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if ksize % 2 == 0:
        ksize += 1
    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= (k.sum() + EPS)
    return k.astype(np.float32)


def gaussian_blur2d(img: np.ndarray, sigma: float, ksize: int) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float32, copy=False)

    a = img.astype(np.float32, copy=False)

    if _HAS_SCIPY:
        return _scipy_gaussian_filter(a, sigma=float(sigma), mode="reflect").astype(np.float32, copy=False)

    k = _gauss_kernel1d(sigma, ksize)
    r = len(k) // 2

    a_pad = np.pad(a, ((0, 0), (r, r)), mode="reflect")
    tmp = np.empty_like(a, dtype=np.float32)
    for j in range(a.shape[1]):
        tmp[:, j] = (a_pad[:, j : j + 2 * r + 1] * k[None, :]).sum(axis=1)

    t_pad = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.empty_like(a, dtype=np.float32)
    for i in range(a.shape[0]):
        out[i, :] = (t_pad[i : i + 2 * r + 1, :] * k[:, None]).sum(axis=0)

    return out.astype(np.float32, copy=False)


def _chroma_mag(cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb0 = cb - 0.5
    cr0 = cr - 0.5
    return np.sqrt(cb0 * cb0 + cr0 * cr0).astype(np.float32)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge0 == edge1:
        return (x >= edge1).astype(np.float32)
    t = (x - edge0) / (edge1 - edge0)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _highlight_desat(
    cb: np.ndarray, cr: np.ndarray, y: np.ndarray, hi_y: float, hi_desat: float
) -> tuple[np.ndarray, np.ndarray]:
    t = _smoothstep(float(hi_y), 1.0, y)
    des = (1.0 - float(hi_desat) * t).astype(np.float32)
    cb2 = 0.5 + (cb - 0.5) * des
    cr2 = 0.5 + (cr - 0.5) * des
    return np.clip(cb2, 0.0, 1.0).astype(np.float32), np.clip(cr2, 0.0, 1.0).astype(np.float32)


def _gamut_map_reduce_chroma(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    gain: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_raw = ycbcr_to_rgb_noclip(y, cb, cr)
    hi = rgb_raw.max(axis=0)
    lo = rgb_raw.min(axis=0)
    over = np.maximum(hi - 1.0, 0.0)
    under = np.maximum(-lo, 0.0)
    exc = np.maximum(over, under)

    f = (1.0 / (1.0 + float(gain) * exc)).astype(np.float32)
    cb2 = 0.5 + (cb - 0.5) * f
    cr2 = 0.5 + (cr - 0.5) * f
    return np.clip(cb2, 0.0, 1.0).astype(np.float32), np.clip(cr2, 0.0, 1.0).astype(np.float32)


def _apply_luma_soft_rolloff(y: np.ndarray, knee: float, strength: float, gamma: float) -> np.ndarray:
    y = y.astype(np.float32, copy=False)

    if gamma != 1.0:
        y = np.clip(y, 0.0, None)
        y = np.power(y, float(max(gamma, 1e-3)), dtype=np.float32)

    knee = float(np.clip(knee, 0.0, 0.995))
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or knee >= 0.999:
        return y.astype(np.float32, copy=False)

    out = y.copy()
    m = out > knee
    if np.any(m):
        z = (out[m] - knee) / max(1.0 - knee, 1e-6)
        alpha = 1.0 + 7.0 * strength
        z2 = np.log1p(alpha * z) / np.log1p(alpha)
        zmix = (1.0 - strength) * z + strength * z2
        out[m] = knee + (1.0 - knee) * zmix
    return out.astype(np.float32, copy=False)


def _box_filter_2d(a: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return a.astype(np.float32, copy=False)

    a = a.astype(np.float32, copy=False)

    if _HAS_SCIPY:
        k = 2 * r + 1
        return _scipy_uniform_filter(a, size=k, mode="reflect").astype(np.float32, copy=False)

    H, W = a.shape
    pad = np.pad(a, ((r, r), (r, r)), mode="reflect")
    integ = np.pad(np.cumsum(np.cumsum(pad, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    k = 2 * r + 1

    out = (integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k]) / float(k * k)
    return out.astype(np.float32, copy=False)


def _guided_filter_gray(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    if r <= 0:
        return p.astype(np.float32, copy=False)

    I = I.astype(np.float32, copy=False)
    p = p.astype(np.float32, copy=False)

    mean_I = _box_filter_2d(I, r)
    mean_p = _box_filter_2d(p, r)
    mean_II = _box_filter_2d(I * I, r)
    mean_Ip = _box_filter_2d(I * p, r)

    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = _box_filter_2d(a, r)
    mean_b = _box_filter_2d(b, r)

    q = mean_a * I + mean_b
    return q.astype(np.float32, copy=False)


def _multiscale_detail(
    src01: np.ndarray,
    sigma1: float,
    sigma2: float,
    sigma3: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b1 = gaussian_blur2d(src01, sigma=sigma1, ksize=_kernel_size_from_sigma(sigma1))
    b2 = gaussian_blur2d(src01, sigma=sigma2, ksize=_kernel_size_from_sigma(sigma2))
    b3 = gaussian_blur2d(src01, sigma=sigma3, ksize=_kernel_size_from_sigma(sigma3))

    d1 = (src01 - b1).astype(np.float32, copy=False)
    d2 = (b1 - b2).astype(np.float32, copy=False)
    d3 = (b2 - b3).astype(np.float32, copy=False)
    return d1, d2, d3


def _safe_percentile(a: np.ndarray, q: float, default: float) -> float:
    v = a[np.isfinite(a)]
    if v.size < 16:
        return float(default)
    return float(np.quantile(v, q))


def _robust_affine_match(
    src: np.ndarray,
    ref: np.ndarray,
    p_lo: float,
    p_hi: float,
) -> tuple[float, float]:
    s_lo = _safe_percentile(src, p_lo, 0.0)
    s_hi = _safe_percentile(src, p_hi, 1.0)
    r_lo = _safe_percentile(ref, p_lo, 0.0)
    r_hi = _safe_percentile(ref, p_hi, 1.0)

    if not np.isfinite(s_lo):
        s_lo = 0.0
    if not np.isfinite(s_hi):
        s_hi = s_lo + 1.0
    if not np.isfinite(r_lo):
        r_lo = 0.0
    if not np.isfinite(r_hi):
        r_hi = r_lo + 1.0

    if s_hi <= s_lo:
        s_hi = s_lo + 1e-3
    if r_hi <= r_lo:
        r_hi = r_lo + 1e-3

    a = (r_hi - r_lo) / (s_hi - s_lo + 1e-6)
    b = r_lo - a * s_lo
    return float(a), float(b)


def _soft_clip01(x: np.ndarray, margin: float) -> np.ndarray:
    m = float(np.clip(margin, 0.0, 0.25))
    if m <= 0.0:
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    y = x.astype(np.float32, copy=False).copy()

    lo_mask = y < 0.0
    if np.any(lo_mask):
        z = (-y[lo_mask]) / max(m, 1e-6)
        y[lo_mask] = -m * (z / (1.0 + z))

    hi_mask = y > 1.0
    if np.any(hi_mask):
        z = (y[hi_mask] - 1.0) / max(m, 1e-6)
        y[hi_mask] = 1.0 + m * (z / (1.0 + z))

    y = (y + m) / (1.0 + 2.0 * m)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


# ----------------------------
# Fusion params
# ----------------------------
@dataclass
class FusionParams:
    mode: str
    chroma_strength: float
    sat: float
    max_gain: float
    chroma_blur_sigma: float
    chroma_blur_ksize: int

    veg_exg_th: float
    veg_sat: float
    veg_chroma: float
    cb_bias: float
    cr_bias: float
    veg_cb_bias: float
    veg_cr_bias: float

    neutral_mag_thr: float
    neutral_strength: float
    neutral_cb_bias: float
    neutral_cr_bias: float

    shadow_y_lo: float
    shadow_y_hi: float
    shadow_strength: float
    shadow_cb_bias: float
    shadow_cr_bias: float
    shadow_chroma: float

    hi_y: float
    hi_desat: float
    gamut_gain: float

    luma_rolloff_knee: float
    luma_rolloff_strength: float
    luma_gamma: float

    detail_strength: float
    detail_strength_urban: float
    detail_strength_veg: float
    detail_strength_shadow: float
    detail_sigma1: float
    detail_sigma2: float
    detail_sigma3: float
    detail_alpha1: float
    detail_alpha2: float
    detail_alpha3: float

    pan_hp_sigma: float
    pan_hp_gain: float
    pan_hp_gain_urban: float
    pan_hp_gain_veg: float
    pan_hp_gain_shadow: float
    highlight_pan_damp_knee: float
    highlight_pan_damp_strength: float

    guided_radius: int
    guided_eps: float
    guided_chroma_mix: float

    ndvi_veg_lo: float
    ndvi_veg_hi: float
    nir_detail_boost: float
    urban_detail_from_nir: float

    use_residual: bool
    pan_syn_base_mix: float
    residual_guided_radius: int
    residual_guided_eps: float
    residual_guided_mix: float

    veg_luma_lift: float
    veg_luma_from_guide: float
    urban_hot_y: float
    urban_hot_strength: float
    final_softclip_margin: float

    simple_y_mix: float


def _apply_common_chroma_controls(
    y_base: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    guide01: np.ndarray,
    fp: FusionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.clip(guide01, 0.0, 1.0).astype(np.float32, copy=False)
    y_g, _cb0, _cr0 = rgb_to_ycbcr(g)

    cb = cb.astype(np.float32, copy=False)
    cr = cr.astype(np.float32, copy=False)
    y = y_base.astype(np.float32, copy=False)

    if abs(fp.cb_bias) > 1e-8:
        cb = np.clip(cb + float(fp.cb_bias), 0.0, 1.0).astype(np.float32, copy=False)
    if abs(fp.cr_bias) > 1e-8:
        cr = np.clip(cr + float(fp.cr_bias), 0.0, 1.0).astype(np.float32, copy=False)

    cb, cr = apply_sat_ycbcr(cb, cr, float(max(fp.sat, 0.0)))

    if fp.chroma_blur_sigma > 0:
        cb = gaussian_blur2d(cb, fp.chroma_blur_sigma, fp.chroma_blur_ksize)
        cr = gaussian_blur2d(cr, fp.chroma_blur_sigma, fp.chroma_blur_ksize)
        cb = np.clip(cb, 0.0, 1.0).astype(np.float32, copy=False)
        cr = np.clip(cr, 0.0, 1.0).astype(np.float32, copy=False)

    exg = 2.0 * g[1] - g[0] - g[2]
    veg_exg = np.clip((exg - float(fp.veg_exg_th)) / 0.5, 0.0, 1.0).astype(np.float32, copy=False)
    veg_mask = veg_exg

    if abs(fp.veg_cb_bias) > 1e-8:
        cb = np.clip(cb + veg_mask * float(fp.veg_cb_bias), 0.0, 1.0).astype(np.float32, copy=False)
    if abs(fp.veg_cr_bias) > 1e-8:
        cr = np.clip(cr + veg_mask * float(fp.veg_cr_bias), 0.0, 1.0).astype(np.float32, copy=False)

    sat_local = (1.0 - veg_mask) + veg_mask * float(np.clip(fp.veg_sat, 0.0, 2.0))
    cb = ((cb - 0.5) * sat_local + 0.5).astype(np.float32, copy=False)
    cr = ((cr - 0.5) * sat_local + 0.5).astype(np.float32, copy=False)

    if float(fp.veg_luma_lift) != 0.0:
        vg = np.clip(y_g, 0.0, 1.0)
        y += veg_mask * float(fp.veg_luma_lift) * (1.0 - vg)

    if float(fp.veg_luma_from_guide) > 0.0:
        mix = float(np.clip(fp.veg_luma_from_guide, 0.0, 1.0))
        y = y * (1.0 - veg_mask * mix) + y_g * (veg_mask * mix)

    mag_now = _chroma_mag(cb, cr)
    if fp.neutral_mag_thr > 0:
        thr = float(fp.neutral_mag_thr)
        u = (mag_now < thr).astype(np.float32) * (1.0 - veg_mask)

        if float(np.mean(u > 0)) > 0.01:
            mu_cb = float(np.mean(cb[u > 0]))
            mu_cr = float(np.mean(cr[u > 0]))
        else:
            mu_cb = float(np.mean(cb))
            mu_cr = float(np.mean(cr))

        ns = float(np.clip(fp.neutral_strength, 0.0, 1.0))
        cb = cb - u * (ns * (mu_cb - 0.5))
        cr = cr - u * (ns * (mu_cr - 0.5))

        if abs(fp.neutral_cb_bias) > 1e-8:
            cb = cb + u * float(fp.neutral_cb_bias)
        if abs(fp.neutral_cr_bias) > 1e-8:
            cr = cr + u * float(fp.neutral_cr_bias)

        cb = np.clip(cb, 0.0, 1.0).astype(np.float32, copy=False)
        cr = np.clip(cr, 0.0, 1.0).astype(np.float32, copy=False)

    if fp.shadow_strength > 0 and fp.shadow_y_hi > fp.shadow_y_lo:
        t_bright = _smoothstep(float(fp.shadow_y_lo), float(fp.shadow_y_hi), y)
        s = (1.0 - t_bright) * float(np.clip(fp.shadow_strength, 0.0, 1.0))
        s = s.astype(np.float32, copy=False)

        if abs(fp.shadow_cb_bias) > 1e-8:
            cb = np.clip(cb + s * float(fp.shadow_cb_bias), 0.0, 1.0).astype(np.float32, copy=False)
        if abs(fp.shadow_cr_bias) > 1e-8:
            cr = np.clip(cr + s * float(fp.shadow_cr_bias), 0.0, 1.0).astype(np.float32, copy=False)

        sc = float(max(fp.shadow_chroma, 0.0))
        if sc != 1.0:
            cb = 0.5 + (cb - 0.5) * (1.0 + s * (sc - 1.0))
            cr = 0.5 + (cr - 0.5) * (1.0 + s * (sc - 1.0))
            cb = np.clip(cb, 0.0, 1.0).astype(np.float32, copy=False)
            cr = np.clip(cr, 0.0, 1.0).astype(np.float32, copy=False)

    if (fp.hi_y > 0.0) and (fp.hi_desat > 0.0):
        cb, cr = _highlight_desat(cb, cr, y, hi_y=float(fp.hi_y), hi_desat=float(fp.hi_desat))
    if fp.gamut_gain > 0.0:
        cb, cr = _gamut_map_reduce_chroma(y, cb, cr, gain=float(fp.gamut_gain))

    return y.astype(np.float32, copy=False), cb.astype(np.float32, copy=False), cr.astype(np.float32, copy=False)


def _apply_final_hot_softclip(rgb: np.ndarray, fp: FusionParams) -> np.ndarray:
    rgb = rgb.astype(np.float32, copy=False)
    y, cb, cr = rgb_to_ycbcr(np.clip(rgb, 0.0, 1.0))
    hot = _smoothstep(float(fp.urban_hot_y), 1.0, y)
    if float(fp.urban_hot_strength) > 0.0:
        des = 1.0 - float(np.clip(fp.urban_hot_strength, 0.0, 1.0)) * hot
        cb = 0.5 + (cb - 0.5) * des
        cr = 0.5 + (cr - 0.5) * des
        rgb = ycbcr_to_rgb_noclip(y, cb, cr)

    rgb = _soft_clip01(rgb, margin=float(fp.final_softclip_margin))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def color_transfer_pan_rgbnir(
    pan01: np.ndarray,
    guide01: np.ndarray,
    nir01: Optional[np.ndarray],
    pan_syn01: Optional[np.ndarray],
    fp: FusionParams,
) -> np.ndarray:
    valid_modes = (
        "multiscale_ycbcr",
        "ratio",
        "guide_chroma_only",
        "pan_y_guide_cbcr",
        "blend_pan_guide_y",
    )
    if fp.mode not in valid_modes:
        raise ValueError(f"fusion_mode inválido: {fp.mode}")

    g = np.clip(guide01, 0.0, 1.0).astype(np.float32, copy=False)
    pan = np.clip(pan01, 0.0, 1.0).astype(np.float32, copy=False)
    nir = None if nir01 is None else np.clip(nir01, 0.0, 1.0).astype(np.float32, copy=False)
    pan_syn = None if pan_syn01 is None else np.clip(pan_syn01, 0.0, 1.0).astype(np.float32, copy=False)

    chroma_strength = float(np.clip(fp.chroma_strength, 0.0, 1.25))
    sat = float(max(fp.sat, 0.0))
    max_gain = float(max(fp.max_gain, 1e-3))

    if fp.mode == "ratio":
        I = (g[0] + g[1] + g[2]) / 3.0
        ratio = pan / (I + EPS)
        ratio = np.clip(ratio, 0.0, max_gain)
        out = g * ratio[None, :, :]
        out = _apply_final_hot_softclip(out, fp)
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    y_g, cb, cr = rgb_to_ycbcr(g)

    # -------------------------
    # Modos diagnósticos
    # -------------------------
    if fp.mode == "pan_y_guide_cbcr":
        y = pan.copy()
        y, cb2, cr2 = _apply_common_chroma_controls(y, cb, cr, g, fp)
        cb2 = 0.5 + (cb2 - 0.5) * chroma_strength
        cr2 = 0.5 + (cr2 - 0.5) * chroma_strength
        rgb = ycbcr_to_rgb_noclip(y, cb2, cr2)
        rgb = _compress_rgb_overflow(rgb)
        rgb = _apply_final_hot_softclip(rgb, fp)
        return rgb

    if fp.mode == "guide_chroma_only":
        y = pan.copy()
        y, cb2, cr2 = _apply_common_chroma_controls(y, cb, cr, g, fp)
        cb2 = 0.5 + (cb2 - 0.5) * chroma_strength
        cr2 = 0.5 + (cr2 - 0.5) * chroma_strength
        rgb = ycbcr_to_rgb_noclip(y, cb2, cr2)
        rgb = _compress_rgb_overflow(rgb)
        rgb = _apply_final_hot_softclip(rgb, fp)
        return rgb

    if fp.mode == "blend_pan_guide_y":
        mix = float(np.clip(fp.simple_y_mix, 0.0, 1.0))
        y = mix * pan + (1.0 - mix) * y_g
        y = np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)
        y, cb2, cr2 = _apply_common_chroma_controls(y, cb, cr, g, fp)
        cb2 = 0.5 + (cb2 - 0.5) * chroma_strength
        cr2 = 0.5 + (cr2 - 0.5) * chroma_strength
        rgb = ycbcr_to_rgb_noclip(y, cb2, cr2)
        rgb = _compress_rgb_overflow(rgb)
        rgb = _apply_final_hot_softclip(rgb, fp)
        return rgb

    # -------------------------
    # Modo principal atual
    # -------------------------
    cb0 = cb.copy()
    cr0 = cr.copy()

    y_base0, cb, cr = _apply_common_chroma_controls(y_g, cb0, cr0, g, fp)

    if fp.guided_radius > 0 and fp.guided_chroma_mix > 0.0:
        cb_ref = _guided_filter_gray(pan, cb, r=int(fp.guided_radius), eps=float(fp.guided_eps))
        cr_ref = _guided_filter_gray(pan, cr, r=int(fp.guided_radius), eps=float(fp.guided_eps))
        mix = float(np.clip(fp.guided_chroma_mix, 0.0, 1.0))
        cb = ((1.0 - mix) * cb + mix * cb_ref).astype(np.float32, copy=False)
        cr = ((1.0 - mix) * cr + mix * cr_ref).astype(np.float32, copy=False)
        cb = np.clip(cb, 0.0, 1.0).astype(np.float32, copy=False)
        cr = np.clip(cr, 0.0, 1.0).astype(np.float32, copy=False)

    if fp.use_residual and pan_syn is not None:
        detail_src = (pan - pan_syn).astype(np.float32, copy=False)
    else:
        detail_src = pan.astype(np.float32, copy=False)

    if fp.residual_guided_radius > 0 and fp.residual_guided_mix > 0.0:
        detail_guided = _guided_filter_gray(
            pan,
            detail_src,
            r=int(fp.residual_guided_radius),
            eps=float(fp.residual_guided_eps),
        )
        rmix = float(np.clip(fp.residual_guided_mix, 0.0, 1.0))
        detail_src = ((1.0 - rmix) * detail_src + rmix * detail_guided).astype(np.float32, copy=False)

    d1, d2, d3 = _multiscale_detail(
        detail_src,
        sigma1=float(fp.detail_sigma1),
        sigma2=float(fp.detail_sigma2),
        sigma3=float(fp.detail_sigma3),
    )

    exg = 2.0 * g[1] - g[0] - g[2]
    veg_exg = np.clip((exg - float(fp.veg_exg_th)) / 0.5, 0.0, 1.0).astype(np.float32, copy=False)

    if nir is not None:
        ndvi = (nir - g[0]) / (nir + g[0] + EPS)
        veg_ndvi = _smoothstep(float(fp.ndvi_veg_lo), float(fp.ndvi_veg_hi), ndvi)
    else:
        ndvi = np.zeros_like(y_g, dtype=np.float32)
        veg_ndvi = np.zeros_like(y_g, dtype=np.float32)

    veg_mask = np.clip(np.maximum(veg_exg, veg_ndvi), 0.0, 1.0).astype(np.float32, copy=False)

    grad = _grad_mag2d(pan)
    grad_q = _safe_percentile(grad, 0.90, 1.0)
    grad_n = np.clip(grad / max(grad_q, 1e-6), 0.0, 1.5).astype(np.float32, copy=False)

    tex = _local_std2d(pan, sigma=1.6)
    tex_q = _safe_percentile(tex, 0.90, 1.0)
    tex_n = np.clip(tex / max(tex_q, 1e-6), 0.0, 1.5).astype(np.float32, copy=False)

    urban_mask = np.clip((1.0 - veg_mask) * (0.55 * grad_n + 0.45 * tex_n), 0.0, 1.0).astype(np.float32, copy=False)
    shadow_mask = (1.0 - _smoothstep(float(fp.shadow_y_lo), float(fp.shadow_y_hi), pan)).astype(np.float32, copy=False)

    nir_urban = np.zeros_like(pan, dtype=np.float32)
    if nir is not None:
        nir_urban = np.clip((1.0 - veg_mask) * np.maximum(0.0, 0.55 - ndvi), 0.0, 1.0).astype(np.float32, copy=False)

    if pan_syn is not None:
        pan_syn_safe = np.clip(pan_syn, 0.0, 1.0).astype(np.float32, copy=False)
        conf_diff = np.abs(pan - pan_syn_safe)
        conf_q = _safe_percentile(conf_diff, 0.95, 1.0)
        conf = 1.0 - np.clip(conf_diff / max(conf_q, 1e-6), 0.0, 1.0)
        conf = np.clip(0.25 + 0.75 * conf, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        conf = np.ones_like(pan, dtype=np.float32)

    detail_gain_map = np.full_like(pan, float(fp.detail_strength), dtype=np.float32)
    detail_gain_map *= (1.0 - veg_mask) + veg_mask * float(fp.detail_strength_veg)
    detail_gain_map *= (1.0 - shadow_mask) + shadow_mask * float(fp.detail_strength_shadow)
    detail_gain_map *= 1.0 + urban_mask * max(0.0, float(fp.detail_strength_urban) - 1.0)
    detail_gain_map *= 1.0 + veg_mask * float(fp.nir_detail_boost) * np.clip(ndvi, 0.0, 1.0)
    detail_gain_map *= 1.0 + nir_urban * float(fp.urban_detail_from_nir)
    detail_gain_map *= conf
    detail_gain_map = np.clip(detail_gain_map, 0.0, 2.5).astype(np.float32, copy=False)

    y_detail = (
        float(fp.detail_alpha1) * d1
        + float(fp.detail_alpha2) * d2
        + float(fp.detail_alpha3) * d3
    ).astype(np.float32, copy=False)

    hp_src = detail_src if fp.use_residual and pan_syn is not None else pan
    pan_lp = gaussian_blur2d(hp_src, sigma=float(fp.pan_hp_sigma), ksize=_kernel_size_from_sigma(float(fp.pan_hp_sigma)))
    pan_hp = (hp_src - pan_lp).astype(np.float32, copy=False)
    pan_hp_scale = _safe_percentile(np.abs(pan_hp), 0.95, 1.0)
    pan_hp_n = np.clip(pan_hp / max(pan_hp_scale, 1e-6), -1.5, 1.5).astype(np.float32, copy=False)

    pan_gain_map = np.full_like(pan, float(fp.pan_hp_gain), dtype=np.float32)
    pan_gain_map *= (1.0 - veg_mask) + veg_mask * float(fp.pan_hp_gain_veg)
    pan_gain_map *= (1.0 - shadow_mask) + shadow_mask * float(fp.pan_hp_gain_shadow)
    pan_gain_map *= 1.0 + urban_mask * max(0.0, float(fp.pan_hp_gain_urban) - 1.0)
    pan_gain_map *= conf

    hi_damp = 1.0 - float(fp.highlight_pan_damp_strength) * _smoothstep(
        float(fp.highlight_pan_damp_knee), 1.0, y_g
    )
    hi_damp = np.clip(hi_damp, 0.0, 1.0).astype(np.float32, copy=False)
    pan_gain_map *= hi_damp
    pan_gain_map = np.clip(pan_gain_map, 0.0, 2.5).astype(np.float32, copy=False)

    if pan_syn is not None:
        y_base = ((1.0 - float(fp.pan_syn_base_mix)) * y_base0 + float(fp.pan_syn_base_mix) * pan_syn).astype(np.float32, copy=False)
    else:
        y_base = y_base0.astype(np.float32, copy=False)

    y_fused = (y_base + detail_gain_map * y_detail + pan_gain_map * pan_hp_n).astype(np.float32, copy=False)

    y_fused = _apply_luma_soft_rolloff(
        y_fused,
        knee=float(fp.luma_rolloff_knee),
        strength=float(fp.luma_rolloff_strength),
        gamma=float(fp.luma_gamma),
    )

    chroma_mix_map = ((1.0 - veg_mask) + veg_mask * float(fp.veg_chroma)).astype(np.float32, copy=False)
    cb_f = 0.5 + (cb - 0.5) * chroma_strength * chroma_mix_map
    cr_f = 0.5 + (cr - 0.5) * chroma_strength * chroma_mix_map
    cb_f = np.clip(cb_f, 0.0, 1.0).astype(np.float32, copy=False)
    cr_f = np.clip(cr_f, 0.0, 1.0).astype(np.float32, copy=False)

    if (fp.hi_y > 0.0) and (fp.hi_desat > 0.0):
        cb_f, cr_f = _highlight_desat(cb_f, cr_f, y_fused, hi_y=float(fp.hi_y), hi_desat=float(fp.hi_desat))
    if fp.gamut_gain > 0.0:
        cb_f, cr_f = _gamut_map_reduce_chroma(y_fused, cb_f, cr_f, gain=float(fp.gamut_gain))

    rgb_raw = ycbcr_to_rgb_noclip(y_fused, cb_f, cr_f)
    out = _compress_rgb_overflow(rgb_raw)
    out = _apply_final_hot_softclip(out, fp)
    return out.astype(np.float32, copy=False)


# ----------------------------
# IO helpers
# ----------------------------
def _tiff_profile_like(
    profile_template: dict,
    *,
    count: int,
    dtype: str,
    height: int,
    width: int,
    block: int,
    compress: str,
) -> dict:
    prof = profile_template.copy()
    prof.pop("nodata", None)

    comp = None if (compress or "").upper() == "NONE" else (compress or "LZW").upper()
    predictor = 3 if str(dtype).lower().startswith("float") else 2

    prof.update(
        driver="GTiff",
        count=int(count),
        dtype=dtype,
        nodata=None,
        height=int(height),
        width=int(width),
        tiled=True,
        blockxsize=int(block),
        blockysize=int(block),
        compress=comp,
        predictor=predictor if comp in ("LZW", "DEFLATE", "ZSTD") else None,
        bigtiff="YES",
        interleave="pixel" if count > 1 else "band",
        num_threads="ALL_CPUS",
    )
    prof.pop("photometric", None)
    return prof


def _write_embedded_stats(path: Path, *, verbose: bool) -> None:
    try:
        with rasterio.open(path, "r+") as ds:
            W, H = ds.width, ds.height
            wins = _sample_windows(W, H, win_w=2048, win_h=2048)

            mins = [np.inf, np.inf, np.inf]
            maxs = [-np.inf, -np.inf, -np.inf]
            sums = [0.0, 0.0, 0.0]
            sums2 = [0.0, 0.0, 0.0]
            counts = [0, 0, 0]

            for w in wins:
                arr = ds.read([1, 2, 3], window=w).astype(np.float32, copy=False)
                arr = np.clip(arr, 0.0, 1.0)
                arr = arr[:, ::4, ::4]

                for b in range(3):
                    v = arr[b].ravel()
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        continue
                    mins[b] = float(min(mins[b], float(v.min())))
                    maxs[b] = float(max(maxs[b], float(v.max())))
                    sums[b] += float(v.sum())
                    sums2[b] += float((v * v).sum())
                    counts[b] += int(v.size)

            for b in range(3):
                if counts[b] <= 0 or not np.isfinite(mins[b]) or not np.isfinite(maxs[b]):
                    continue
                mean = sums[b] / max(counts[b], 1)
                var = max(sums2[b] / max(counts[b], 1) - mean * mean, 0.0)
                std = float(np.sqrt(var))

                ds.update_tags(
                    b + 1,
                    STATISTICS_MINIMUM=f"{mins[b]:.9f}",
                    STATISTICS_MAXIMUM=f"{maxs[b]:.9f}",
                    STATISTICS_MEAN=f"{mean:.9f}",
                    STATISTICS_STDDEV=f"{std:.9f}",
                    STATISTICS_APPROXIMATE="YES",
                )

        if verbose:
            print("[INFO] embedded GeoTIFF stats written (STATISTICS_*).")
    except Exception as e:
        if verbose:
            print("[WARN] failed to write embedded stats:", repr(e))


def _global_radiometric_finish(
    out_path: Path,
    guide_rgb_path: Path,
    *,
    p_lo: float,
    p_hi: float,
    soft_margin: float,
    verbose: bool,
) -> None:
    tmp_path = out_path.with_name(out_path.stem + "_radfix_tmp.tif")

    with rasterio.open(out_path) as src, rasterio.open(guide_rgb_path) as gsrc:
        if src.width != gsrc.width or src.height != gsrc.height:
            raise ValueError("Radiometric finish requer saída e guide RGB alinhados.")

        sample_out = [[], [], []]
        sample_ref = [[], [], []]
        total = 0
        sample_max = 8_000_000
        rng = np.random.default_rng(0)

        for _, win in src.block_windows(1):
            o = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)
            r = gsrc.read([1, 2, 3], window=win).astype(np.float32, copy=False)

            m = np.isfinite(o).all(axis=0) & np.isfinite(r).all(axis=0)
            m &= (r[0] > 0.0) & (r[1] > 0.0) & (r[2] > 0.0)

            idx = np.flatnonzero(m.ravel())
            if idx.size == 0:
                continue

            remaining = sample_max - total
            if remaining <= 0:
                break

            take = min(idx.size, max(4096, remaining // 64))
            sel = rng.choice(idx, size=take, replace=False) if take < idx.size else idx

            for c in range(3):
                sample_out[c].append(o[c].ravel()[sel].astype(np.float32, copy=False))
                sample_ref[c].append(r[c].ravel()[sel].astype(np.float32, copy=False))
            total += int(take)

        coeffs: list[tuple[float, float]] = []
        for c in range(3):
            if sample_out[c] and sample_ref[c]:
                so = np.concatenate(sample_out[c]).astype(np.float32, copy=False)
                sr = np.concatenate(sample_ref[c]).astype(np.float32, copy=False)
                a, b = _robust_affine_match(so, sr, p_lo=float(p_lo), p_hi=float(p_hi))
            else:
                a, b = 1.0, 0.0
            coeffs.append((a, b))

        if verbose:
            print("[RAD] global robust affine coefficients:")
            print("      R: a={:.6f} b={:.6f}".format(coeffs[0][0], coeffs[0][1]))
            print("      G: a={:.6f} b={:.6f}".format(coeffs[1][0], coeffs[1][1]))
            print("      B: a={:.6f} b={:.6f}".format(coeffs[2][0], coeffs[2][1]))

        prof = src.profile.copy()
        prof.update(
            driver="GTiff",
            dtype="float32",
            count=3,
            nodata=None,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="ZSTD",
            predictor=3,
            interleave="pixel",
            num_threads="ALL_CPUS",
            bigtiff="YES",
        )
        prof.pop("nodata", None)

        with rasterio.open(tmp_path, "w", **prof) as dst:
            try:
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            except Exception:
                pass

            for _, win in src.block_windows(1):
                o = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)
                out = np.empty_like(o, dtype=np.float32)
                for c in range(3):
                    a, b = coeffs[c]
                    out[c] = a * o[c] + b
                out = _soft_clip01(out, margin=float(soft_margin))
                dst.write(out.astype(np.float32, copy=False), window=win)

    out_path.unlink()
    tmp_path.rename(out_path)


def _global_radiometric_finish_y_only(
    out_path: Path,
    guide_rgb_path: Path,
    *,
    p_lo: float,
    p_hi: float,
    soft_margin: float,
    verbose: bool,
) -> None:
    tmp_path = out_path.with_name(out_path.stem + "_radfixy_tmp.tif")

    with rasterio.open(out_path) as src, rasterio.open(guide_rgb_path) as gsrc:
        if src.width != gsrc.width or src.height != gsrc.height:
            raise ValueError("Radiometric finish Y-only requer saída e guide RGB alinhados.")

        sample_out = []
        sample_ref = []
        total = 0
        sample_max = 8_000_000
        rng = np.random.default_rng(0)

        for _, win in src.block_windows(1):
            o = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)
            r = gsrc.read([1, 2, 3], window=win).astype(np.float32, copy=False)

            yo, _, _ = rgb_to_ycbcr(np.clip(o, 0.0, 1.0))
            yr, _, _ = rgb_to_ycbcr(np.clip(r, 0.0, 1.0))

            m = np.isfinite(yo) & np.isfinite(yr)
            m &= (yr > 0.0)

            idx = np.flatnonzero(m.ravel())
            if idx.size == 0:
                continue

            remaining = sample_max - total
            if remaining <= 0:
                break

            take = min(idx.size, max(4096, remaining // 64))
            sel = rng.choice(idx, size=take, replace=False) if take < idx.size else idx

            sample_out.append(yo.ravel()[sel].astype(np.float32, copy=False))
            sample_ref.append(yr.ravel()[sel].astype(np.float32, copy=False))
            total += int(take)

        if sample_out and sample_ref:
            so = np.concatenate(sample_out).astype(np.float32, copy=False)
            sr = np.concatenate(sample_ref).astype(np.float32, copy=False)
            a, b = _robust_affine_match(so, sr, p_lo=float(p_lo), p_hi=float(p_hi))
        else:
            a, b = 1.0, 0.0

        if verbose:
            print("[RAD:Y] global robust affine coefficients:")
            print("        Y: a={:.6f} b={:.6f}".format(a, b))

        prof = src.profile.copy()
        prof.update(
            driver="GTiff",
            dtype="float32",
            count=3,
            nodata=None,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="ZSTD",
            predictor=3,
            interleave="pixel",
            num_threads="ALL_CPUS",
            bigtiff="YES",
        )
        prof.pop("nodata", None)

        with rasterio.open(tmp_path, "w", **prof) as dst:
            try:
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            except Exception:
                pass

            for _, win in src.block_windows(1):
                o = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)
                y, cb, cr = rgb_to_ycbcr(np.clip(o, 0.0, 1.0))
                y2 = a * y + b
                rgb = ycbcr_to_rgb_noclip(y2, cb, cr)
                rgb = _soft_clip01(rgb, margin=float(soft_margin))
                dst.write(rgb.astype(np.float32, copy=False), window=win)

    out_path.unlink()
    tmp_path.rename(out_path)


# ----------------------------
# Tile processing
# ----------------------------
def _process_window(
    *,
    src: rasterio.DatasetReader,
    gsrc: rasterio.DatasetReader,
    nirsrc: Optional[rasterio.DatasetReader],
    pansynsrc: Optional[rasterio.DatasetReader],
    win: Window,
    in_is_pan1: bool,
    norm_used: str,
    guide_div255: bool,
    nir_div255: bool,
    pansyn_div255: bool,
    out_range: str,
    tile: int,
    overlap: int,
    fusion: FusionParams,
    sanitize: bool,
) -> np.ndarray:
    H = int(win.height)
    W = int(win.width)

    if in_is_pan1:
        pan = src.read(1, window=win).astype(np.float32, copy=False)
        if sanitize:
            pan, _ = sanitize_array(pan, name="PAN(win)", verbose=False)
        pan = np.where(pan < SENTINEL_TH, 0.0, pan).astype(np.float32, copy=False)
        tile_in = pan[None, :, :].astype(np.float32, copy=False)
    else:
        tile_in = src.read(indexes=[1, 2, 3], window=win).astype(np.float32, copy=False)
        if sanitize:
            tile_in, _ = sanitize_array(tile_in, name="IN(win)", verbose=False)
        tile_in = np.where(tile_in < SENTINEL_TH, 0.0, tile_in).astype(np.float32, copy=False)

    tile_in = apply_norm_tile(tile_in, norm_used)

    guide = gsrc.read(indexes=[1, 2, 3], window=win).astype(np.float32, copy=False)
    if sanitize:
        guide, _ = sanitize_array(guide, name="GUIDE(win)", verbose=False)
    guide01_full = (guide / 255.0) if guide_div255 else guide
    guide01_full = np.clip(guide01_full, 0.0, 1.0).astype(np.float32, copy=False)

    if nirsrc is not None:
        nir = nirsrc.read(1, window=win).astype(np.float32, copy=False)
        if sanitize:
            nir, _ = sanitize_array(nir, name="NIR(win)", verbose=False)
        nir01_full = (nir / 255.0) if nir_div255 else nir
        nir01_full = np.clip(nir01_full, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        nir01_full = None

    if pansynsrc is not None:
        pans = pansynsrc.read(1, window=win).astype(np.float32, copy=False)
        if sanitize:
            pans, _ = sanitize_array(pans, name="PAN_SYN(win)", verbose=False)
        pansyn01_full = (pans / 255.0) if pansyn_div255 else pans
        pansyn01_full = np.clip(pansyn01_full, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        pansyn01_full = None

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap grande demais: tile - overlap precisa ser > 0")

    if H <= tile and W <= tile:
        if norm_used == "minus1_1":
            pan01 = np.clip((tile_in[0] + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            pan01 = np.clip(tile_in[0], 0.0, 1.0).astype(np.float32, copy=False)

        out = color_transfer_pan_rgbnir(
            pan01=pan01,
            guide01=guide01_full,
            nir01=nir01_full,
            pan_syn01=pansyn01_full,
            fp=fusion,
        )
        out = apply_out_range(out, out_range=out_range)
        if sanitize:
            out, _ = sanitize_array(out, name="OUT(win)", verbose=False)
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    xs = make_positions(W, tile, stride)
    ys = make_positions(H, tile, stride)

    sum_rgb = np.zeros((3, H, W), dtype=np.float32)
    sum_w = np.zeros((H, W), dtype=np.float32)
    wfull = make_weight(tile, tile, overlap)

    for y0 in ys:
        for x0 in xs:
            th = min(tile, H - y0)
            tw = min(tile, W - x0)

            in_crop = tile_in[:, y0 : y0 + th, x0 : x0 + tw]
            in_pad = pad_reflect(in_crop, tile, tile)

            g_crop = guide01_full[:, y0 : y0 + th, x0 : x0 + tw]
            g_pad = pad_reflect(g_crop, tile, tile)

            if nir01_full is not None:
                n_crop = nir01_full[y0 : y0 + th, x0 : x0 + tw]
                n_pad = pad_reflect_2d(n_crop, tile, tile)
            else:
                n_pad = None

            if pansyn01_full is not None:
                p_crop = pansyn01_full[y0 : y0 + th, x0 : x0 + tw]
                p_pad = pad_reflect_2d(p_crop, tile, tile)
            else:
                p_pad = None

            if norm_used == "minus1_1":
                pan01 = np.clip((in_pad[0] + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                pan01 = np.clip(in_pad[0], 0.0, 1.0).astype(np.float32, copy=False)

            pred = color_transfer_pan_rgbnir(
                pan01=pan01,
                guide01=g_pad,
                nir01=n_pad,
                pan_syn01=p_pad,
                fp=fusion,
            ).astype(np.float32, copy=False)

            pred = apply_out_range(pred, out_range=out_range)
            pred = pred[:, :th, :tw]
            w = wfull[:th, :tw]

            sum_rgb[:, y0 : y0 + th, x0 : x0 + tw] += pred * w[None, :, :]
            sum_w[y0 : y0 + th, x0 : x0 + tw] += w

    denom = np.maximum(sum_w, 1e-6)
    out = (sum_rgb / denom[None, :, :]).astype(np.float32, copy=False)
    if sanitize:
        out, _ = sanitize_array(out, name="OUT(win)", verbose=False)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


# ----------------------------
# QA2
# ----------------------------
def _grad_mag(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    gx = np.zeros_like(a, dtype=np.float32)
    gy = np.zeros_like(a, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
    gy[1:-1, :] = 0.5 * (a[2:, :] - a[:-2, :])
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _laplacian(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    lap = np.zeros_like(a, dtype=np.float32)
    lap[1:-1, 1:-1] = (
        -4.0 * a[1:-1, 1:-1]
        + a[1:-1, 2:]
        + a[1:-1, :-2]
        + a[2:, 1:-1]
        + a[:-2, 1:-1]
    )
    return lap


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2048:
        return float("nan")
    a = a[m].astype(np.float32, copy=False)
    b = b[m].astype(np.float32, copy=False)
    a -= float(a.mean())
    b -= float(b.mean())
    den = float(np.sqrt((a * a).mean()) * np.sqrt((b * b).mean()) + 1e-12)
    return float(((a * b).mean()) / den)


def _sample_windows_grid(W: int, H: int, win: int, n: int) -> list[Window]:
    win = min(int(win), int(W), int(H))
    if win <= 0:
        return []

    if n <= 1:
        x0 = max(0, (W - win) // 2)
        y0 = max(0, (H - win) // 2)
        return [Window(x0, y0, win, win)]

    g = int(round(np.sqrt(n)))
    g = max(g, 2)

    xs = np.linspace(0, max(0, W - win), g)
    ys = np.linspace(0, max(0, H - win), g)

    out = []
    seen = set()
    for y in ys:
        for x in xs:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            key = (xi, yi, int(win), int(win))
            if key in seen:
                continue
            seen.add(key)
            out.append(Window(xi, yi, win, win))
    return out


def _qa_spatial_non_tautological(
    src_path: str,
    out_path: Path,
    *,
    guide_path: Optional[str],
    verbose: bool,
    win: int,
    n: int,
    sample_step: int,
    min_corr: float,
    max_chroma_pan_corr: float,
    min_valid_frac: float = 0.70,
    eps_valid: float = 1e-6,
) -> None:
    with rasterio.open(src_path) as psrc, rasterio.open(out_path) as osrc:
        if psrc.width != osrc.width or psrc.height != osrc.height:
            print(f"[QA2] WARN: size mismatch PAN={psrc.width}x{psrc.height} OUT={osrc.width}x{osrc.height} (QA2 skipped)")
            return

        W, H = psrc.width, psrc.height
        wins = _sample_windows_grid(W, H, win=win, n=n)

        gI_corrs: list[float] = []
        lI_corrs: list[float] = []
        chroma_pan_corrs: list[float] = []
        guide_chroma_corrs: list[float] = []
        skipped_low_valid = 0
        skipped_small_mask = 0

        gsrc: Optional[rasterio.DatasetReader] = None
        if guide_path:
            gsrc = rasterio.open(guide_path)

        try:
            for w in wins:
                pan01 = np.clip(psrc.read(1, window=w).astype(np.float32), 0.0, 1.0)
                rgb = np.clip(osrc.read([1, 2, 3], window=w).astype(np.float32), 0.0, 1.0)

                I = ((rgb[0] + rgb[1] + rgb[2]) / 3.0).astype(np.float32)
                _y_out, cb, cr = rgb_to_ycbcr(rgb)
                cmag = _chroma_mag(cb, cr)

                valid_rgb = (
                    np.isfinite(rgb).all(axis=0)
                    & (rgb[0] > float(eps_valid))
                    & (rgb[1] > float(eps_valid))
                    & (rgb[2] > float(eps_valid))
                )

                valid_pan = np.isfinite(pan01) & (pan01 > float(eps_valid))
                valid = valid_rgb & valid_pan

                valid_frac = float(np.mean(valid))
                if valid_frac < float(min_valid_frac):
                    skipped_low_valid += 1
                    continue

                if sample_step > 1:
                    pan_s = pan01[::sample_step, ::sample_step]
                    I_s = I[::sample_step, ::sample_step]
                    cm_s = cmag[::sample_step, ::sample_step]
                    valid_s = valid[::sample_step, ::sample_step]
                else:
                    pan_s = pan01
                    I_s = I
                    cm_s = cmag
                    valid_s = valid

                m = np.isfinite(pan_s) & np.isfinite(I_s) & valid_s
                if m.sum() < 2048:
                    skipped_small_mask += 1
                    continue

                pan_m = np.where(m, pan_s, np.nan)
                I_m = np.where(m, I_s, np.nan)

                g_pan = _grad_mag(pan_m)
                g_I = _grad_mag(I_m)
                gI_corrs.append(_corr(g_pan, g_I))

                l_pan = _laplacian(pan_m)
                l_I = _laplacian(I_m)
                lI_corrs.append(_corr(l_pan, l_I))

                cm_m = np.where(m & np.isfinite(cm_s), cm_s, np.nan)
                g_c = _grad_mag(cm_m)
                chroma_pan_corrs.append(_corr(g_pan, g_c))

                if gsrc is not None:
                    g = np.clip(gsrc.read([1, 2, 3], window=w).astype(np.float32), 0.0, 1.0)
                    y_g, _, _ = rgb_to_ycbcr(g)
                    y_s = y_g[::sample_step, ::sample_step] if sample_step > 1 else y_g

                    mm = np.isfinite(y_s) & np.isfinite(cm_s) & valid_s
                    if mm.sum() >= 2048:
                        guide_chroma_corrs.append(
                            _corr(
                                _grad_mag(np.where(mm, y_s, np.nan)),
                                _grad_mag(np.where(mm, cm_s, np.nan)),
                            )
                        )
        finally:
            if gsrc is not None:
                gsrc.close()

        if not gI_corrs:
            print(
                f"[QA2] WARN: insufficient valid samples after masking "
                f"(skipped_low_valid={skipped_low_valid}, skipped_small_mask={skipped_small_mask}, "
                f"requested_windows={len(wins)})"
            )
            return

        def _stats(x: list[float]) -> tuple[float, float, float]:
            v = np.array(x, dtype=np.float32)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return float("nan"), float("nan"), float("nan")
            return float(v.mean()), float(v.min()), float(v.max())

        g_mean, g_min, g_max = _stats(gI_corrs)
        l_mean, l_min, l_max = _stats(lI_corrs)
        c_mean, c_min, c_max = _stats(chroma_pan_corrs)

        lvl = "OK"
        if np.isfinite(g_mean) and g_mean < float(min_corr):
            lvl = "WARN"
        if np.isfinite(l_mean) and l_mean < float(min_corr - 0.05):
            lvl = "WARN"
        if np.isfinite(c_mean) and c_mean > float(max_chroma_pan_corr):
            lvl = "WARN"

        print(
            f"[QA2:{lvl}] non-tautological | "
            f"corr(grad(PAN),grad(I)) mean={g_mean:.3f} min={g_min:.3f} max={g_max:.3f} | "
            f"corr(lap(PAN),lap(I)) mean={l_mean:.3f} min={l_min:.3f} max={l_max:.3f} | "
            f"leak corr(grad(PAN),grad(|CbCr|)) mean={c_mean:.3f} min={c_min:.3f} max={c_max:.3f} | "
            f"candidate_windows={len(wins)} used_windows={len(gI_corrs)} "
            f"skipped_low_valid={skipped_low_valid} skipped_small_mask={skipped_small_mask}"
        )

        if verbose and guide_chroma_corrs:
            gg_mean, gg_min, gg_max = _stats(guide_chroma_corrs)
            print(
                f"[QA2] (info) corr(grad(Y_guide),grad(|CbCr_out|)) mean={gg_mean:.3f} "
                f"min={gg_min:.3f} max={gg_max:.3f} | n={len(guide_chroma_corrs)}"
            )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_pan_tif", default=None)
    ap.add_argument("--in_tif", default=None)

    ap.add_argument("--out_tif", required=True)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)

    ap.add_argument("--norm", default="auto", choices=["auto", "none", "0_1", "0_255", "minus1_1"])
    ap.add_argument("--out_range", default="0_1", choices=["0_1", "minus1_1", "none"])

    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--guide_rgb", required=True)
    ap.add_argument("--guide_nir", default=None)
    ap.add_argument("--pan_syn_tif", default=None)

    ap.add_argument(
        "--fusion_mode",
        default="multiscale_ycbcr",
        choices=[
            "multiscale_ycbcr",
            "ratio",
            "guide_chroma_only",
            "pan_y_guide_cbcr",
            "blend_pan_guide_y",
        ],
    )
    ap.add_argument("--simple_y_mix", type=float, default=0.80)

    ap.add_argument("--chroma_strength", type=float, default=0.72)
    ap.add_argument("--sat", type=float, default=0.92)
    ap.add_argument("--max_gain", type=float, default=2.5)
    ap.add_argument("--cr_bias", type=float, default=-0.006)
    ap.add_argument("--cb_bias", type=float, default=-0.003)

    ap.add_argument("--chroma_blur_sigma", type=float, default=0.6)
    ap.add_argument("--chroma_blur_ksize", type=int, default=5)

    ap.add_argument("--veg_exg_th", type=float, default=0.10)
    ap.add_argument("--veg_sat", type=float, default=0.82)
    ap.add_argument("--veg_chroma", type=float, default=0.92)
    ap.add_argument("--veg_cr_bias", type=float, default=-0.010)
    ap.add_argument("--veg_cb_bias", type=float, default=0.003)
    ap.add_argument("--veg_luma_lift", type=float, default=0.0)
    ap.add_argument("--veg_luma_from_guide", type=float, default=0.0)

    ap.add_argument("--neutral_mag_thr", type=float, default=0.050)
    ap.add_argument("--neutral_strength", type=float, default=0.70)
    ap.add_argument("--neutral_cb_bias", type=float, default=0.0)
    ap.add_argument("--neutral_cr_bias", type=float, default=0.0)

    ap.add_argument("--shadow_y_lo", type=float, default=0.08)
    ap.add_argument("--shadow_y_hi", type=float, default=0.28)
    ap.add_argument("--shadow_strength", type=float, default=0.20)
    ap.add_argument("--shadow_cb_bias", type=float, default=-0.004)
    ap.add_argument("--shadow_cr_bias", type=float, default=0.003)
    ap.add_argument("--shadow_chroma", type=float, default=0.95)

    ap.add_argument("--hi_y", type=float, default=0.86)
    ap.add_argument("--hi_desat", type=float, default=0.45)
    ap.add_argument("--gamut_gain", type=float, default=4.0)

    ap.add_argument("--urban_hot_y", type=float, default=0.85)
    ap.add_argument("--urban_hot_strength", type=float, default=0.0)
    ap.add_argument("--final_softclip_margin", type=float, default=0.0)

    ap.add_argument("--luma_rolloff_knee", type=float, default=0.88)
    ap.add_argument("--luma_rolloff_strength", type=float, default=0.32)
    ap.add_argument("--luma_gamma", type=float, default=0.98)

    ap.add_argument("--detail_strength", type=float, default=0.72)
    ap.add_argument("--detail_strength_urban", type=float, default=1.15)
    ap.add_argument("--detail_strength_veg", type=float, default=0.55)
    ap.add_argument("--detail_strength_shadow", type=float, default=0.85)
    ap.add_argument("--detail_sigma1", type=float, default=1.2)
    ap.add_argument("--detail_sigma2", type=float, default=2.8)
    ap.add_argument("--detail_sigma3", type=float, default=5.6)
    ap.add_argument("--detail_alpha1", type=float, default=0.62)
    ap.add_argument("--detail_alpha2", type=float, default=0.28)
    ap.add_argument("--detail_alpha3", type=float, default=0.14)

    ap.add_argument("--pan_hp_sigma", type=float, default=2.2)
    ap.add_argument("--pan_hp_gain", type=float, default=0.16)
    ap.add_argument("--pan_hp_gain_urban", type=float, default=1.35)
    ap.add_argument("--pan_hp_gain_veg", type=float, default=0.50)
    ap.add_argument("--pan_hp_gain_shadow", type=float, default=0.75)
    ap.add_argument("--highlight_pan_damp_knee", type=float, default=0.82)
    ap.add_argument("--highlight_pan_damp_strength", type=float, default=0.75)

    ap.add_argument("--guided_radius", type=int, default=4)
    ap.add_argument("--guided_eps", type=float, default=1e-4)
    ap.add_argument("--guided_chroma_mix", type=float, default=0.65)

    ap.add_argument("--ndvi_veg_lo", type=float, default=0.18)
    ap.add_argument("--ndvi_veg_hi", type=float, default=0.38)
    ap.add_argument("--nir_detail_boost", type=float, default=0.20)
    ap.add_argument("--urban_detail_from_nir", type=float, default=0.12)

    ap.add_argument("--use_residual", action="store_true")
    ap.add_argument("--no_use_residual", action="store_true")
    ap.add_argument("--pan_syn_base_mix", type=float, default=0.15)
    ap.add_argument("--residual_guided_radius", type=int, default=8)
    ap.add_argument("--residual_guided_eps", type=float, default=1e-4)
    ap.add_argument("--residual_guided_mix", type=float, default=0.55)

    ap.add_argument("--radiometric_finish_mode", default="y_only", choices=["none", "y_only", "rgb_global"])
    ap.add_argument("--rad_p_lo", type=float, default=0.003)
    ap.add_argument("--rad_p_hi", type=float, default=0.997)
    ap.add_argument("--rad_soft_margin", type=float, default=0.020)

    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--io_block", type=int, default=None)
    ap.add_argument("--out_block", type=int, default=None)
    ap.add_argument("--compress", default="ZSTD", choices=["LZW", "DEFLATE", "ZSTD", "NONE"])

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--qa2", action="store_true")
    ap.add_argument("--no_qa2", action="store_true")
    ap.add_argument("--qa2_win", type=int, default=1024)
    ap.add_argument("--qa2_num_windows", type=int, default=25)
    ap.add_argument("--qa2_sample_step", type=int, default=2)
    ap.add_argument("--qa2_min_corr", type=float, default=0.85)
    ap.add_argument("--qa2_max_chroma_pan_corr", type=float, default=0.45)

    args = ap.parse_args()

    if (args.in_pan_tif is None) == (args.in_tif is None):
        raise ValueError("Use exatamente UM: --in_pan_tif (1 banda) OU --in_tif (3 bandas).")

    TILE = int(args.tile)
    OL = int(args.overlap)
    if TILE - OL <= 0:
        raise ValueError("overlap grande demais: tile - overlap precisa ser > 0")

    device = str(args.device)

    io_block = int(args.io_block) if args.io_block is not None else int(args.block)
    out_block = int(args.out_block) if args.out_block is not None else int(args.block)
    if io_block <= 0 or out_block <= 0:
        raise ValueError("--io_block/--out_block precisam ser > 0")

    if device == "cuda":
        amp_enabled = not bool(args.no_amp)
        if args.amp:
            amp_enabled = True
    else:
        amp_enabled = False

    use_residual = True
    if bool(args.no_use_residual):
        use_residual = False
    elif bool(args.use_residual):
        use_residual = True

    _set_cuda_perf_flags(device, verbose=bool(args.verbose))
    if args.verbose and device == "cuda":
        print(
            f"[RUN] FUSION starting with tile={TILE} overlap={OL} "
            f"amp={amp_enabled} io_block={io_block} out_block={out_block} "
            f"scipy_fast={'yes' if _HAS_SCIPY else 'no'} residual={use_residual}"
        )

    if args.in_tif:
        src_path = args.in_tif
        in_is_pan1 = False
    else:
        src_path = args.in_pan_tif
        in_is_pan1 = True

    with rasterio.open(src_path) as src:
        profile = src.profile
        H, W = src.height, src.width

        if in_is_pan1:
            if src.count != 1:
                raise ValueError(f"--in_pan_tif precisa ter 1 banda; veio {src.count}.")
            band_indexes = [1]
        else:
            if src.count != 3:
                raise ValueError(f"--in_tif precisa ter 3 bandas; veio {src.count}.")
            band_indexes = [1, 2, 3]

        est_max = _estimate_max(src, band_indexes=band_indexes, verbose=bool(args.verbose))
        norm_used = resolve_norm_mode(args.norm, est_max)
        if args.verbose:
            print(f"[INFO] norm solicitado={args.norm} | norm usado={norm_used}")

    out_path = Path(args.out_tif)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.guide_rgb) as gsrc0:
        if gsrc0.count != 3 or gsrc0.height != H or gsrc0.width != W:
            raise ValueError("guide_rgb precisa estar alinhado e com mesmo tamanho da PAN (3,H,W).")
        g_est = _estimate_max(gsrc0, band_indexes=[1, 2, 3], verbose=False)
        guide_div255 = bool(g_est > 1.5)
        if args.verbose:
            print(f"[INFO] guide RGB max estimado={g_est:.6f} -> {'/255' if guide_div255 else 'assume 0..1'}")

    nir_div255 = False
    if args.guide_nir:
        with rasterio.open(args.guide_nir) as nsrc0:
            if nsrc0.count != 1 or nsrc0.height != H or nsrc0.width != W:
                raise ValueError("guide_nir precisa estar alinhado e com mesmo tamanho da PAN (1,H,W).")
            n_est = _estimate_max(nsrc0, band_indexes=[1], verbose=False)
            nir_div255 = bool(n_est > 1.5)
            if args.verbose:
                print(f"[INFO] guide NIR max estimado={n_est:.6f} -> {'/255' if nir_div255 else 'assume 0..1'}")

    pansyn_div255 = False
    if args.pan_syn_tif:
        with rasterio.open(args.pan_syn_tif) as psrc0:
            if psrc0.count != 1 or psrc0.height != H or psrc0.width != W:
                raise ValueError("pan_syn_tif precisa estar alinhado e com mesmo tamanho da PAN (1,H,W).")
            p_est = _estimate_max(psrc0, band_indexes=[1], verbose=False)
            pansyn_div255 = bool(p_est > 1.5)
            if args.verbose:
                print(f"[INFO] PAN_syn max estimado={p_est:.6f} -> {'/255' if pansyn_div255 else 'assume 0..1'}")

    prof_out = _tiff_profile_like(
        profile,
        count=3,
        dtype="float32",
        height=H,
        width=W,
        block=out_block,
        compress=args.compress,
    )
    prof_out.update(photometric="RGB")

    fusion = FusionParams(
        mode=str(args.fusion_mode),
        simple_y_mix=float(args.simple_y_mix),

        chroma_strength=float(args.chroma_strength),
        sat=float(args.sat),
        max_gain=float(args.max_gain),
        chroma_blur_sigma=float(args.chroma_blur_sigma),
        chroma_blur_ksize=int(args.chroma_blur_ksize),

        veg_exg_th=float(args.veg_exg_th),
        veg_sat=float(args.veg_sat),
        veg_chroma=float(args.veg_chroma),
        cb_bias=float(args.cb_bias),
        cr_bias=float(args.cr_bias),
        veg_cb_bias=float(args.veg_cb_bias),
        veg_cr_bias=float(args.veg_cr_bias),

        neutral_mag_thr=float(args.neutral_mag_thr),
        neutral_strength=float(args.neutral_strength),
        neutral_cb_bias=float(args.neutral_cb_bias),
        neutral_cr_bias=float(args.neutral_cr_bias),

        shadow_y_lo=float(args.shadow_y_lo),
        shadow_y_hi=float(args.shadow_y_hi),
        shadow_strength=float(args.shadow_strength),
        shadow_cb_bias=float(args.shadow_cb_bias),
        shadow_cr_bias=float(args.shadow_cr_bias),
        shadow_chroma=float(args.shadow_chroma),

        hi_y=float(args.hi_y),
        hi_desat=float(args.hi_desat),
        gamut_gain=float(args.gamut_gain),

        luma_rolloff_knee=float(args.luma_rolloff_knee),
        luma_rolloff_strength=float(args.luma_rolloff_strength),
        luma_gamma=float(args.luma_gamma),

        detail_strength=float(args.detail_strength),
        detail_strength_urban=float(args.detail_strength_urban),
        detail_strength_veg=float(args.detail_strength_veg),
        detail_strength_shadow=float(args.detail_strength_shadow),
        detail_sigma1=float(args.detail_sigma1),
        detail_sigma2=float(args.detail_sigma2),
        detail_sigma3=float(args.detail_sigma3),
        detail_alpha1=float(args.detail_alpha1),
        detail_alpha2=float(args.detail_alpha2),
        detail_alpha3=float(args.detail_alpha3),

        pan_hp_sigma=float(args.pan_hp_sigma),
        pan_hp_gain=float(args.pan_hp_gain),
        pan_hp_gain_urban=float(args.pan_hp_gain_urban),
        pan_hp_gain_veg=float(args.pan_hp_gain_veg),
        pan_hp_gain_shadow=float(args.pan_hp_gain_shadow),
        highlight_pan_damp_knee=float(args.highlight_pan_damp_knee),
        highlight_pan_damp_strength=float(args.highlight_pan_damp_strength),

        guided_radius=int(args.guided_radius),
        guided_eps=float(args.guided_eps),
        guided_chroma_mix=float(args.guided_chroma_mix),

        ndvi_veg_lo=float(args.ndvi_veg_lo),
        ndvi_veg_hi=float(args.ndvi_veg_hi),
        nir_detail_boost=float(args.nir_detail_boost),
        urban_detail_from_nir=float(args.urban_detail_from_nir),

        use_residual=bool(use_residual),
        pan_syn_base_mix=float(args.pan_syn_base_mix),
        residual_guided_radius=int(args.residual_guided_radius),
        residual_guided_eps=float(args.residual_guided_eps),
        residual_guided_mix=float(args.residual_guided_mix),

        veg_luma_lift=float(args.veg_luma_lift),
        veg_luma_from_guide=float(args.veg_luma_from_guide),
        urban_hot_y=float(args.urban_hot_y),
        urban_hot_strength=float(args.urban_hot_strength),
        final_softclip_margin=float(args.final_softclip_margin),
    )

    with rasterio.open(src_path) as src, rasterio.open(args.guide_rgb) as gsrc, (
        rasterio.open(args.guide_nir) if args.guide_nir else None
    ) as nirsrc, (
        rasterio.open(args.pan_syn_tif) if args.pan_syn_tif else None
    ) as pansynsrc, rasterio.open(out_path, "w", **prof_out) as dst:
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)

        for row in range(0, H, io_block):
            hh = min(io_block, H - row)
            for col in range(0, W, io_block):
                ww = min(io_block, W - col)
                win = Window(col, row, ww, hh)

                out_win = _process_window(
                    src=src,
                    gsrc=gsrc,
                    nirsrc=nirsrc,
                    pansynsrc=pansynsrc,
                    win=win,
                    in_is_pan1=in_is_pan1,
                    norm_used=norm_used,
                    guide_div255=guide_div255,
                    nir_div255=nir_div255,
                    pansyn_div255=pansyn_div255,
                    out_range=str(args.out_range),
                    tile=int(args.tile),
                    overlap=int(args.overlap),
                    fusion=fusion,
                    sanitize=bool(args.sanitize),
                )

                dst.write(out_win.astype(np.float32, copy=False), window=win)

    if str(args.radiometric_finish_mode).lower() == "rgb_global":
        _global_radiometric_finish(
            out_path=out_path,
            guide_rgb_path=Path(args.guide_rgb),
            p_lo=float(args.rad_p_lo),
            p_hi=float(args.rad_p_hi),
            soft_margin=float(args.rad_soft_margin),
            verbose=bool(args.verbose),
        )
    elif str(args.radiometric_finish_mode).lower() == "y_only":
        _global_radiometric_finish_y_only(
            out_path=out_path,
            guide_rgb_path=Path(args.guide_rgb),
            p_lo=float(args.rad_p_lo),
            p_hi=float(args.rad_p_hi),
            soft_margin=float(args.rad_soft_margin),
            verbose=bool(args.verbose),
        )

    _write_embedded_stats(out_path, verbose=bool(args.verbose))

    print("OK:", str(out_path))

    qa2_enabled = (not bool(args.no_qa2)) or bool(args.qa2)
    if qa2_enabled:
        _qa_spatial_non_tautological(
            src_path=src_path,
            out_path=out_path,
            guide_path=args.guide_rgb,
            verbose=bool(args.verbose),
            win=int(args.qa2_win),
            n=int(args.qa2_num_windows),
            sample_step=int(args.qa2_sample_step),
            min_corr=float(args.qa2_min_corr),
            max_chroma_pan_corr=float(args.qa2_max_chroma_pan_corr),
        )


if __name__ == "__main__":
    main()