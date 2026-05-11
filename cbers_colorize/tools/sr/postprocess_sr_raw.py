from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine


def compute_p2p98_from_array(img: np.ndarray, ignore_zeros: bool) -> tuple[np.ndarray, np.ndarray]:
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


def estimate_p2p98_streaming(src, ignore_zeros: bool, sample_max_dim: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    C = src.count
    H = src.height
    W = src.width

    scale = max(H / sample_max_dim, W / sample_max_dim, 1.0)
    out_h = max(1, int(round(H / scale)))
    out_w = max(1, int(round(W / scale)))

    arr = src.read(
        out_shape=(C, out_h, out_w),
        resampling=Resampling.nearest,
    ).astype(np.float32)

    return compute_p2p98_from_array(arr, ignore_zeros=ignore_zeros)


def apply_post(arr: np.ndarray, mode: str, p2: np.ndarray | None, p98: np.ndarray | None) -> np.ndarray:
    x = arr.astype(np.float32)

    if mode == "none":
        return x

    if mode == "clip_01":
        return np.clip(x, 0.0, 1.0)

    if mode == "m11_to_01":
        y = np.clip(x, -1.0, 1.0)
        y = (y + 1.0) / 2.0
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    if mode in ("p2p98_01", "p2p98_m11_to_01"):
        assert p2 is not None and p98 is not None
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

        return y.astype(np.float32)

    raise ValueError(f"post_out desconhecido: {mode}")


def default_vis_path(raw_tif: str, post_out: str) -> str:
    base, ext = os.path.splitext(raw_tif)
    return f"{base}_vis_{post_out}{ext}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_raw_tif", required=True)
    ap.add_argument("--out_vis_tif", default="")
    ap.add_argument(
        "--post_out",
        default="p2p98_01",
        choices=["none", "clip_01", "m11_to_01", "p2p98_01", "p2p98_m11_to_01"],
    )
    ap.add_argument("--ignore_zeros", action="store_true")
    ap.add_argument("--sample_max_dim", type=int, default=4096)
    args = ap.parse_args()

    with rasterio.open(args.in_raw_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        H = src.height
        W = src.width

        p2 = None
        p98 = None
        if args.post_out in ("p2p98_01", "p2p98_m11_to_01"):
            p2, p98 = estimate_p2p98_streaming(
                src,
                ignore_zeros=args.ignore_zeros,
                sample_max_dim=args.sample_max_dim,
            )
            print(f"[INFO] p2 global:  {p2}")
            print(f"[INFO] p98 global: {p98}")

        out_vis_tif = args.out_vis_tif.strip() or default_vis_path(args.in_raw_tif, args.post_out)
        out_path = Path(out_vis_tif)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        profile_out = profile.copy()
        profile_out.pop("nodata", None)
        profile_out.update(
            count=3,
            dtype="float32",
            nodata=None,
            photometric="RGB",
            tiled=True,
            BIGTIFF="IF_SAFER",
            compress="deflate",
        )

        block_h = profile.get("blockysize", 512) or 512
        block_w = profile.get("blockxsize", 512) or 512

        with rasterio.open(out_path, "w", **profile_out) as dst:
            for row in range(0, H, block_h):
                h = min(block_h, H - row)
                for col in range(0, W, block_w):
                    w = min(block_w, W - col)
                    win = rasterio.windows.Window(col, row, w, h)
                    arr = src.read(window=win).astype(np.float32)
                    vis = apply_post(arr, args.post_out, p2, p98)
                    dst.write(vis.astype(np.float32), window=win)

            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)

    print("OK (vis):", str(out_path))


if __name__ == "__main__":
    main()