from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine
import torch
import torch.nn.functional as F

from cbers_colorize.rsinet.net import Kong

SENTINEL_TH = -1e20
EPS = 1e-6


# ----------------------------
# Model
# ----------------------------
def load_model(pkl_path: str, device: str, scale: int, model_half: bool = False):
    model = Kong(scale=scale, multi_out=True).to(device)
    ckpt = torch.load(pkl_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    if model_half and str(device).startswith("cuda"):
        model.half()

    return model


def select_pred(pred, head_mode: str):
    if not isinstance(pred, (list, tuple)):
        return pred

    if head_mode == "first":
        return pred[0]
    if head_mode == "last":
        return pred[-1]
    if head_mode == "mean":
        return torch.stack(list(pred), dim=0).mean(dim=0)

    raise ValueError(f"head_mode inválido: {head_mode}")


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


def make_positions(total: int, tile: int, stride: int, offset: int = 0) -> list[int]:
    if total <= tile:
        return [0]

    offset = int(max(0, min(offset, max(0, total - tile))))
    pos = []
    p = offset

    if p > 0:
        pos.append(0)

    while True:
        if p + tile >= total:
            pos.append(total - tile)
            break
        pos.append(p)
        p += stride

    return sorted(set(int(v) for v in pos))


def _gauss_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)

    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k) + EPS
    return k.astype(np.float32)


def _conv1d_reflect_axis_last(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    r = len(k) // 2
    ap = np.pad(a, ((0, 0), (0, 0), (r, r)), mode="reflect")
    out = np.empty_like(a, dtype=np.float32)
    for i in range(a.shape[2]):
        out[:, :, i] = np.sum(ap[:, :, i : i + 2 * r + 1] * k[None, None, :], axis=2)
    return out


def _conv1d_reflect_axis_mid(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    r = len(k) // 2
    ap = np.pad(a, ((0, 0), (r, r), (0, 0)), mode="reflect")
    out = np.empty_like(a, dtype=np.float32)
    for i in range(a.shape[1]):
        out[:, i, :] = np.sum(ap[:, i : i + 2 * r + 1, :] * k[None, :, None], axis=1)
    return out


def gaussian_blur_chw(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a.astype(np.float32, copy=False)

    k = _gauss_kernel1d(float(sigma))
    tmp = _conv1d_reflect_axis_last(a.astype(np.float32, copy=False), k)
    out = _conv1d_reflect_axis_mid(tmp, k)
    return out.astype(np.float32, copy=False)


def _tukey_1d(n: int, edge: int, floor: float) -> np.ndarray:
    x = np.ones((n,), dtype=np.float32)
    if edge <= 0:
        return x

    edge = int(min(edge, n // 2))
    if edge <= 0:
        return x

    t = np.linspace(0.0, 1.0, edge, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.pi * t)
    ramp = np.clip(ramp, floor, 1.0)

    x[:edge] = ramp
    x[-edge:] = ramp[::-1]
    return x


def make_weight(
    h: int,
    w: int,
    overlap: int,
    mode: str = "center_strong",
    center_power: float = 2.2,
    radial_power: float = 2.6,
    edge_floor: float = 1e-4,
    radial_mix: float = 0.50,
) -> np.ndarray:
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)

    overlap = int(min(overlap, h // 2, w // 2))
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)

    wy = _tukey_1d(h, overlap, floor=edge_floor)
    wx = _tukey_1d(w, overlap, floor=edge_floor)

    if mode == "sine":
        y = np.linspace(0.0, np.pi, h, dtype=np.float32)
        x = np.linspace(0.0, np.pi, w, dtype=np.float32)
        wy = np.maximum(np.sin(y), edge_floor)
        wx = np.maximum(np.sin(x), edge_floor)
        w2 = wy[:, None] * wx[None, :]
        w2 /= max(float(w2.max()), EPS)
        return w2.astype(np.float32)

    if mode not in ("tukey", "center_strong"):
        raise ValueError(f"weight_mode inválido: {mode}")

    w_sep = (wy[:, None] * wx[None, :]).astype(np.float32)

    if mode == "tukey":
        w_sep /= max(float(w_sep.max()), EPS)
        return w_sep.astype(np.float32)

    # center_strong
    w_sep = np.power(np.clip(w_sep, edge_floor, 1.0), float(max(center_power, 1.0)))

    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    rr = np.sqrt(X * X + Y * Y) / np.sqrt(2.0)
    rr = np.clip(rr, 0.0, 1.0)

    w_rad = np.power(np.clip(1.0 - rr, 0.0, 1.0), float(max(radial_power, 1.0)))
    w_rad = np.maximum(w_rad, edge_floor).astype(np.float32)

    mix = float(np.clip(radial_mix, 0.0, 1.0))
    w2 = ((1.0 - mix) * w_sep + mix * (w_sep * w_rad)).astype(np.float32)
    w2 = np.maximum(w2, edge_floor)
    w2 /= max(float(w2.max()), EPS)
    return w2.astype(np.float32)


def default_raw_path(out_tif: str) -> str:
    base, ext = os.path.splitext(out_tif)
    return f"{base}_raw{ext}"


def default_vis_path(out_tif: str, post_out: str) -> str:
    base, ext = os.path.splitext(out_tif)
    return f"{base}_vis_{post_out}{ext}"


def write_rgb_geotiff(path: str, arr: np.ndarray, profile_template: dict, transform_hr: Affine):
    profile_out = profile_template.copy()
    profile_out.pop("nodata", None)
    profile_out.update(
        count=3,
        dtype="float32",
        nodata=None,
        height=arr.shape[1],
        width=arr.shape[2],
        transform=transform_hr,
        photometric="RGB",
        tiled=True,
        BIGTIFF="IF_SAFER",
        compress="deflate",
    )

    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32))
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)


# ----------------------------
# Normalization
# ----------------------------
def compute_p2p98(img: np.ndarray, ignore_zeros: bool) -> tuple[np.ndarray, np.ndarray]:
    C = img.shape[0]
    p2 = np.zeros((C,), dtype=np.float32)
    p98 = np.zeros((C,), dtype=np.float32)

    for c in range(C):
        x = img[c].astype(np.float32)
        m = np.isfinite(x)
        if ignore_zeros:
            m &= (x != 0)
        vals = x[m]
        if vals.size == 0:
            p2[c], p98[c] = 0.0, 1.0
        else:
            p2[c] = float(np.quantile(vals, 0.02))
            p98[c] = float(np.quantile(vals, 0.98))
            if p98[c] <= p2[c]:
                p2[c], p98[c] = float(vals.min()), float(vals.max())
                if p98[c] <= p2[c]:
                    p2[c], p98[c] = 0.0, 1.0

    return p2, p98


def apply_norm_global(img: np.ndarray, mode: str, ignore_zeros: bool) -> tuple[np.ndarray, str, dict]:
    params = {}
    m = float(np.nanmax(img))
    chosen = mode

    if mode == "auto":
        chosen = "0_255" if m > 1.5 else "0_1"

    x = img.astype(np.float32)

    if chosen == "none":
        return x, chosen, params

    if chosen == "0_255":
        x = x / 255.0
        x = np.clip(x, 0.0, 1.0)
        return x, chosen, params

    if chosen == "0_1":
        x = np.clip(x, 0.0, 1.0)
        return x, chosen, params

    if chosen == "minus1_1":
        x = np.clip(x, 0.0, 1.0)
        x = x * 2.0 - 1.0
        return x, chosen, params

    if chosen in ("p2p98_01", "p2p98_m11"):
        p2, p98 = compute_p2p98(x, ignore_zeros=ignore_zeros)
        params["p2"] = p2
        params["p98"] = p98

        p2b = p2[:, None, None]
        p98b = p98[:, None, None]
        denom = np.maximum(p98b - p2b, 1e-6)

        y = (x - p2b) / denom
        y = np.clip(y, 0.0, 1.0)

        if chosen == "p2p98_m11":
            y = y * 2.0 - 1.0

        return y.astype(np.float32), chosen, params

    raise ValueError(f"Modo de normalização desconhecido: {mode}")


def apply_out_range(out: np.ndarray, out_range: str, norm_used: str) -> tuple[np.ndarray, str]:
    chosen = out_range
    y = out.astype(np.float32)

    if out_range == "auto":
        if norm_used.endswith("m11") or norm_used == "minus1_1":
            chosen = "auto_m11_to_01"
        else:
            chosen = "0_1"

    if chosen == "none":
        return y, chosen

    if chosen == "minus1_1":
        y = np.clip(y, -1.0, 1.0)
        return y, chosen

    if chosen == "0_1":
        y = np.clip(y, 0.0, 1.0)
        return y, chosen

    if chosen == "auto_m11_to_01":
        y = np.clip(y, -1.0, 1.0)
        y = (y + 1.0) / 2.0
        y = np.clip(y, 0.0, 1.0)
        return y, chosen

    raise ValueError(f"out_range desconhecido: {out_range}")


def postprocess_out(out: np.ndarray, mode: str, ignore_zeros: bool) -> tuple[np.ndarray, dict]:
    info = {}
    x = out.astype(np.float32)

    if mode == "none":
        return x, info

    if mode == "clip_01":
        return np.clip(x, 0.0, 1.0), info

    if mode == "m11_to_01":
        y = np.clip(x, -1.0, 1.0)
        y = (y + 1.0) / 2.0
        y = np.clip(y, 0.0, 1.0)
        return y.astype(np.float32), info

    if mode in ("p2p98_01", "p2p98_m11_to_01"):
        p2, p98 = compute_p2p98(x, ignore_zeros=ignore_zeros)
        info["p2"] = p2
        info["p98"] = p98

        p2b = p2[:, None, None]
        p98b = p98[:, None, None]
        denom = np.maximum(p98b - p2b, 1e-6)

        y = (x - p2b) / denom
        y = np.clip(y, 0.0, 1.0)

        if mode == "p2p98_m11_to_01":
            y = y * 2.0 - 1.0
            y = np.clip(y, -1.0, 1.0)
            y = (y + 1.0) / 2.0
            y = np.clip(y, 0.0, 1.0)

        return y.astype(np.float32), info

    raise ValueError(f"post_out desconhecido: {mode}")


# ----------------------------
# Grid / shave / harmonization
# ----------------------------
def get_grid_offsets(grid_mode: str, stride_lr: int) -> list[tuple[int, int]]:
    if grid_mode == "single":
        return [(0, 0)]

    if grid_mode == "dual":
        s2 = max(1, stride_lr // 2)
        return [(0, 0), (s2, s2)]

    if grid_mode == "triple":
        s2 = max(1, stride_lr // 2)
        s3a = max(1, stride_lr // 3)
        s3b = max(1, (2 * stride_lr) // 3)
        return [(0, 0), (s2, s2), (s3a, s3b)]

    raise ValueError(f"grid_mode inválido: {grid_mode}")


def resolve_shaves(
    *,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    H_lr: int,
    W_lr: int,
    ht_hr: int,
    wt_hr: int,
    shave_hr: int,
    overlap_hr: int,
    adaptive_shave: bool,
    adaptive_extra: int,
) -> tuple[int, int, int, int]:
    is_top = (y0 == 0)
    is_left = (x0 == 0)
    is_bottom = (y1 == H_lr)
    is_right = (x1 == W_lr)

    top = 0 if is_top else int(shave_hr)
    left = 0 if is_left else int(shave_hr)
    bottom = 0 if is_bottom else int(shave_hr)
    right = 0 if is_right else int(shave_hr)

    if adaptive_shave:
        extra = int(max(0, adaptive_extra))
        extra = min(extra, max(0, overlap_hr // 2))
        extra = min(extra, max(0, min(ht_hr, wt_hr) // 8))

        if not is_top:
            top += extra
        if not is_left:
            left += extra
        if not is_bottom:
            bottom += extra
        if not is_right:
            right += extra

    if ht_hr <= (top + bottom):
        top, bottom = 0, 0
    if wt_hr <= (left + right):
        left, right = 0, 0

    return top, left, bottom, right


def harmonize_lowfreq_proxy(
    out_raw: np.ndarray,
    ref_hr: np.ndarray,
    sigma: float,
    gain_clip: float,
    bias_clip: float,
    verbose: bool = False,
) -> np.ndarray:
    if sigma <= 0:
        return out_raw.astype(np.float32, copy=False)

    ds = max(1, int(round(max(1.0, sigma / 4.0))))
    sigma_proxy = max(0.5, float(sigma) / float(ds))

    if verbose:
        print(
            f"[INFO] lowfreq harmonize proxy: sigma={sigma:.2f} "
            f"ds={ds} sigma_proxy={sigma_proxy:.2f}"
        )

    out_p = out_raw[:, ::ds, ::ds].astype(np.float32, copy=False)
    ref_p = ref_hr[:, ::ds, ::ds].astype(np.float32, copy=False)

    low_out = gaussian_blur_chw(out_p, sigma=sigma_proxy)
    low_ref = gaussian_blur_chw(ref_p, sigma=sigma_proxy)

    gain_p = np.clip(low_ref / np.maximum(low_out, EPS), 1.0 - gain_clip, 1.0 + gain_clip).astype(np.float32)
    bias_p = np.clip(low_ref - low_out, -bias_clip, bias_clip).astype(np.float32)

    out = out_raw.copy().astype(np.float32, copy=False)
    H = out.shape[1]
    W = out.shape[2]

    for c in range(out.shape[0]):
        g = np.repeat(np.repeat(gain_p[c], ds, axis=0), ds, axis=1)[:H, :W]
        b = np.repeat(np.repeat(bias_p[c], ds, axis=0), ds, axis=1)[:H, :W]
        out[c] = out[c] * g + b

    return out.astype(np.float32, copy=False)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tif", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out_tif", required=True)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--scale", type=int, default=4)

    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--shave_hr", type=int, default=16)

    ap.add_argument("--grid_mode", default="single", choices=["single", "dual", "triple"])

    ap.add_argument("--weight_mode", default="center_strong", choices=["sine", "tukey", "center_strong"])
    ap.add_argument("--weight_center_power", type=float, default=2.4)
    ap.add_argument("--weight_radial_power", type=float, default=2.8)
    ap.add_argument("--weight_radial_mix", type=float, default=0.60)
    ap.add_argument("--weight_edge_floor", type=float, default=1e-4)

    ap.add_argument("--adaptive_shave", action="store_true")
    ap.add_argument("--adaptive_shave_extra", type=int, default=8)

    ap.add_argument(
        "--norm",
        default="auto",
        choices=["auto", "none", "0_1", "0_255", "minus1_1", "p2p98_01", "p2p98_m11"],
    )
    ap.add_argument("--ignore_zeros", action="store_true")

    ap.add_argument(
        "--out_range",
        default="none",
        choices=["auto", "0_1", "minus1_1", "none"],
    )

    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug_one_tile", action="store_true")

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--model_half", action="store_true")
    ap.add_argument("--gc_every", type=int, default=16)

    ap.add_argument("--head_mode", default="first", choices=["first", "last", "mean"])
    ap.add_argument("--residual_mode", default="add", choices=["add", "none"])

    ap.add_argument(
        "--post_out",
        default="p2p98_01",
        choices=["none", "clip_01", "m11_to_01", "p2p98_01", "p2p98_m11_to_01"],
    )
    ap.add_argument("--post_ignore_zeros", action="store_true")

    ap.add_argument("--harmonize_lowfreq", action="store_true")
    ap.add_argument("--harm_sigma", type=float, default=12.0)
    ap.add_argument("--harm_gain_clip", type=float, default=0.03)
    ap.add_argument("--harm_bias_clip", type=float, default=0.03)

    args = ap.parse_args()

    scale = int(args.scale)
    tile_lr = int(args.tile)
    overlap_lr = int(args.overlap)
    stride_lr = tile_lr - overlap_lr

    if stride_lr <= 0:
        raise ValueError("overlap grande demais: tile - overlap precisa ser > 0")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("AVISO: CUDA solicitada, mas não disponível. Fazendo fallback para CPU.")
        device = "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = load_model(args.pkl, device, scale=scale, model_half=args.model_half)

    with rasterio.open(args.in_tif) as src:
        img = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform

    if img.shape[0] != 3:
        raise ValueError(f"Entrada precisa ter 3 bandas (RGB). Veio {img.shape[0]}.")

    if args.sanitize:
        img, _ = sanitize_array(img, name="INPUT(global)", verbose=args.verbose)
    img = np.where(img < SENTINEL_TH, 0.0, img).astype(np.float32)

    img_n, norm_used, norm_params = apply_norm_global(img, args.norm, ignore_zeros=args.ignore_zeros)
    if args.verbose:
        mn, mx = float(np.nanmin(img_n)), float(np.nanmax(img_n))
        print(f"[INFO] norm solicitado={args.norm} | norm usado={norm_used} | min={mn:.6f} max={mx:.6f}")
        if "p2" in norm_params:
            print(f"[INFO] p2(entrada):  {norm_params['p2']}")
            print(f"[INFO] p98(entrada): {norm_params['p98']}")

    C, H_lr, W_lr = img_n.shape
    H_hr, W_hr = H_lr * scale, W_lr * scale

    out_sum = np.zeros((C, H_hr, W_hr), dtype=np.float32)
    w_sum = np.zeros((H_hr, W_hr), dtype=np.float32)

    overlap_hr = overlap_lr * scale
    tile_counter = 0

    grid_offsets = get_grid_offsets(args.grid_mode, stride_lr)
    total_tiles = 0
    for oy, ox in grid_offsets:
        ys = make_positions(H_lr, tile_lr, stride_lr, offset=oy)
        xs = make_positions(W_lr, tile_lr, stride_lr, offset=ox)
        total_tiles += len(ys) * len(xs)

    if args.verbose:
        print(f"[INFO] grid_mode={args.grid_mode} offsets={grid_offsets}")
        print(
            f"[INFO] weight_mode={args.weight_mode} center_power={args.weight_center_power} "
            f"radial_power={args.weight_radial_power} radial_mix={args.weight_radial_mix}"
        )
        print(
            f"[INFO] adaptive_shave={args.adaptive_shave} "
            f"shave_hr={args.shave_hr} adaptive_extra={args.adaptive_shave_extra}"
        )

    for grid_idx, (off_y, off_x) in enumerate(grid_offsets, start=1):
        ys = make_positions(H_lr, tile_lr, stride_lr, offset=off_y)
        xs = make_positions(W_lr, tile_lr, stride_lr, offset=off_x)

        if args.verbose:
            print(f"[INFO] grid {grid_idx}/{len(grid_offsets)} offset=({off_y},{off_x}) tiles={len(ys) * len(xs)}")

        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + tile_lr, H_lr)
                x1 = min(x0 + tile_lr, W_lr)

                tile = img_n[:, y0:y1, x0:x1].astype(np.float32)
                ht_lr, wt_lr = tile.shape[1], tile.shape[2]

                if args.sanitize:
                    tile, _ = sanitize_array(tile, name="INPUT(tile)", verbose=False)

                tile_p = pad_reflect(tile, tile_lr, tile_lr)

                x_dtype = torch.float16 if (device.startswith("cuda") and args.fp16) else torch.float32
                tile_t = torch.from_numpy(tile_p).unsqueeze(0).to(device=device, dtype=x_dtype)

                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=(device.startswith("cuda") and args.fp16),
                    ):
                        pred = model(tile_t)
                        pred_t = select_pred(pred, args.head_mode).squeeze(0)

                        if args.residual_mode == "add":
                            base_t = F.interpolate(tile_t, scale_factor=scale, mode="bicubic", align_corners=False)
                            pred_t = base_t.squeeze(0) + pred_t

                pred_np = pred_t.detach().float().cpu().numpy().astype(np.float32)

                ht_hr = ht_lr * scale
                wt_hr = wt_lr * scale
                pred_np = pred_np[:, :ht_hr, :wt_hr]

                if args.sanitize:
                    pred_np, _ = sanitize_array(pred_np, name="PRED(tile)", verbose=False)

                pred_np, _ = apply_out_range(pred_np, args.out_range, norm_used)

                Y0 = y0 * scale
                X0 = x0 * scale
                Y1 = Y0 + ht_hr
                X1 = X0 + wt_hr

                top, left, bottom, right = resolve_shaves(
                    y0=y0,
                    x0=x0,
                    y1=y1,
                    x1=x1,
                    H_lr=H_lr,
                    W_lr=W_lr,
                    ht_hr=ht_hr,
                    wt_hr=wt_hr,
                    shave_hr=int(args.shave_hr),
                    overlap_hr=overlap_hr,
                    adaptive_shave=bool(args.adaptive_shave),
                    adaptive_extra=int(args.adaptive_shave_extra),
                )

                pred_core = pred_np[:, top:ht_hr - bottom, left:wt_hr - right]

                Y0c = Y0 + top
                X0c = X0 + left
                Y1c = Y1 - bottom
                X1c = X1 - right

                htc = pred_core.shape[1]
                wtc = pred_core.shape[2]
                if htc <= 0 or wtc <= 0:
                    del tile_t, pred_t, pred_np, pred_core
                    continue

                ol_eff = min(overlap_hr, htc // 2, wtc // 2)
                weight = make_weight(
                    htc,
                    wtc,
                    ol_eff,
                    mode=args.weight_mode,
                    center_power=float(args.weight_center_power),
                    radial_power=float(args.weight_radial_power),
                    edge_floor=float(args.weight_edge_floor),
                    radial_mix=float(args.weight_radial_mix),
                )

                out_sum[:, Y0c:Y1c, X0c:X1c] += pred_core * weight[None, :, :]
                w_sum[Y0c:Y1c, X0c:X1c] += weight

                del tile_t, pred_t, pred_np, pred_core
                tile_counter += 1

                if device.startswith("cuda") and (tile_counter % int(args.gc_every) == 0):
                    torch.cuda.empty_cache()
                    gc.collect()

                print(f"[{tile_counter}/{total_tiles}] {(100.0 * tile_counter / total_tiles):6.2f}%")

                if args.debug_one_tile:
                    break
            if args.debug_one_tile:
                break
        if args.debug_one_tile:
            break

    missing = int((w_sum == 0).sum())
    print(f"Pixels sem cobertura (w_sum==0): {missing} de {H_hr * W_hr}")

    w_sum_safe = np.where(w_sum == 0, 1.0, w_sum).astype(np.float32)
    out_raw = out_sum / w_sum_safe[None, :, :]
    out_raw[:, w_sum == 0] = 0.0

    if args.harmonize_lowfreq:
        ref_hr = F.interpolate(
            torch.from_numpy(img_n).unsqueeze(0),
            scale_factor=scale,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).numpy().astype(np.float32)

        out_raw = harmonize_lowfreq_proxy(
            out_raw=out_raw,
            ref_hr=ref_hr,
            sigma=float(args.harm_sigma),
            gain_clip=float(args.harm_gain_clip),
            bias_clip=float(args.harm_bias_clip),
            verbose=bool(args.verbose),
        )
        del ref_hr

    if args.sanitize:
        out_raw, _ = sanitize_array(out_raw, name="OUTPUT(raw_final)", verbose=args.verbose)

    transform_hr = Affine(
        transform.a / scale, transform.b, transform.c,
        transform.d, transform.e / scale, transform.f
    )

    out_raw_path = default_raw_path(args.out_tif)
    write_rgb_geotiff(out_raw_path, out_raw, profile, transform_hr)
    print("OK (raw):", out_raw_path)

    out_vis, info = postprocess_out(out_raw, args.post_out, ignore_zeros=args.post_ignore_zeros)
    if args.verbose and "p2" in info:
        print(f"[INFO] p2(saída):  {info['p2']}")
        print(f"[INFO] p98(saída): {info['p98']}")

    out_vis_path = default_vis_path(args.out_tif, args.post_out)
    write_rgb_geotiff(out_vis_path, out_vis, profile, transform_hr)
    print("OK (vis):", out_vis_path)
    print("OK (base name):", args.out_tif)


if __name__ == "__main__":
    main()