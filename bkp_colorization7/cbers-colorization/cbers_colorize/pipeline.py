from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Iterable
from contextlib import contextmanager
import time
import os
import subprocess
import sys
import shutil

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window


@dataclass
class PipelineConfig:
    pan: Path
    blue: Path
    green: Path
    red: Path
    nir: Path

    outdir: Path
    workdir: Path

    device: str = "cpu"
    tile: int = 512
    overlap: int = 32
    sanitize: bool = False
    verbose: bool = False
    keep_tmp: bool = False

    amp: bool = True

    io_block: int = 1024
    out_block: int = 1024
    compress: str = "ZSTD"

    scale: int = 4
    color_tool: Path | str = "/app/cbers_colorize/tools/infer_geotiff_color.py"

    pan_p_lo: float = 0.02
    pan_p_hi: float = 0.998

    guide_p_lo: float = 0.02
    guide_p_hi: float = 0.98
    guide_norm: str = "per_band"

    fusion_mode: str = "multiscale_ycbcr"
    chroma_strength: float = 0.72
    sat: float = 0.92
    max_gain: float = 2.5
    chroma_blur_sigma: float = 0.6
    chroma_blur_ksize: int = 5

    cr_bias: float = -0.006
    cb_bias: float = -0.003

    veg_cr_bias: float = -0.010
    veg_cb_bias: float = 0.003
    veg_exg_th: float = 0.10
    veg_sat: float = 0.82
    veg_chroma: float = 0.92

    neutral_mag_thr: float = 0.050
    neutral_strength: float = 0.70
    neutral_cb_bias: float = 0.0
    neutral_cr_bias: float = 0.0

    shadow_y_lo: float = 0.08
    shadow_y_hi: float = 0.28
    shadow_strength: float = 0.20
    shadow_cb_bias: float = -0.004
    shadow_cr_bias: float = 0.003
    shadow_chroma: float = 0.95

    hi_y: float = 0.86
    hi_desat: float = 0.45
    gamut_gain: float = 4.0

    luma_rolloff_knee: float = 0.88
    luma_rolloff_strength: float = 0.32
    luma_gamma: float = 0.98

    detail_strength: float = 0.72
    detail_strength_urban: float = 1.15
    detail_strength_veg: float = 0.55
    detail_strength_shadow: float = 0.85
    detail_sigma1: float = 1.2
    detail_sigma2: float = 2.8
    detail_sigma3: float = 5.6
    detail_alpha1: float = 0.62
    detail_alpha2: float = 0.28
    detail_alpha3: float = 0.14

    pan_hp_sigma: float = 2.2
    pan_hp_gain: float = 0.16
    pan_hp_gain_urban: float = 1.35
    pan_hp_gain_veg: float = 0.50
    pan_hp_gain_shadow: float = 0.75
    highlight_pan_damp_knee: float = 0.82
    highlight_pan_damp_strength: float = 0.75

    guided_radius: int = 4
    guided_eps: float = 1e-4
    guided_chroma_mix: float = 0.65

    ndvi_veg_lo: float = 0.18
    ndvi_veg_hi: float = 0.38
    nir_detail_boost: float = 0.20
    urban_detail_from_nir: float = 0.12

    rad_p_lo: float = 0.003
    rad_p_hi: float = 0.997
    rad_soft_margin: float = 0.020

    qa_num_windows: int = 25
    qa_win: int = 1024
    qa_sample_step: int = 2
    qa_min_corr: float = 0.85

    export_vis: bool = False
    vis_format: str = "GTIFF"
    vis_compress: str = "JPEG"
    vis_p_lo: float = 0.01
    vis_p_hi: float = 0.995
    vis_gamma: float = 1.00
    vis_overviews: bool = False

    export_cog: bool = False
    cog_quality: int = 90
    cog_blocksize: int = 512
    cog_overviews: bool = False

    export_ecw: bool = False
    ecw_target_mb: int = 0


@contextmanager
def _timed(label: str, verbose: bool, store: dict[str, float] | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[label] = dt
    print(f"[TIME] {label}: {dt:.2f}s")


def _run(cmd: Sequence[str], verbose: bool) -> None:
    if verbose:
        print("[PIPE] RUN:", " ".join(map(str, cmd)))
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    needed = "/app"
    if needed not in py_path.split(":"):
        env["PYTHONPATH"] = needed + (":" + py_path if py_path else "")
    subprocess.run(list(map(str, cmd)), check=True, env=env)


def _run_capture(cmd: Sequence[str], verbose: bool) -> subprocess.CompletedProcess:
    if verbose:
        print("[PIPE] RUN/CAPTURE:", " ".join(map(str, cmd)))
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    needed = "/app"
    if needed not in py_path.split(":"):
        env["PYTHONPATH"] = needed + (":" + py_path if py_path else "")
    return subprocess.run(list(map(str, cmd)), check=False, env=env, text=True, capture_output=True)


def _safe_unlink(paths: Iterable[Path], verbose: bool) -> None:
    for p in paths:
        try:
            if p and p.exists():
                p.unlink()
                if verbose:
                    print("[PIPE] CLEAN:", p)
        except Exception as e:
            if verbose:
                print("[PIPE] CLEAN WARN:", p, "->", repr(e))


def _safe_rmtree(path: Path, verbose: bool) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
            if verbose:
                print("[PIPE] CLEAN DIR:", path)
    except Exception as e:
        if verbose:
            print("[PIPE] CLEAN DIR WARN:", path, "->", repr(e))


def _assert_exists(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"[PIPE] Missing {label}: {p}")


def _log_inputs(cfg: PipelineConfig) -> None:
    if not cfg.verbose:
        return

    def _info(p: Path) -> str:
        with rasterio.open(p) as src:
            crs = src.crs.to_string() if src.crs else "None"
            return f"{p.name} | bands={src.count} | {src.width}x{src.height} | dtype={src.dtypes[0]} | crs={crs}"

    print("[PIPE] INPUTS:")
    print("  PAN  :", _info(cfg.pan))
    print("  BLUE :", _info(cfg.blue))
    print("  GREEN:", _info(cfg.green))
    print("  RED  :", _info(cfg.red))
    print("  NIR  :", _info(cfg.nir))
    print("[PIPE] NOTE: multispectral guide is RGB + NIR.")
    print("[PIPE] NOTE: internal RGB order is ALWAYS [RED, GREEN, BLUE].")


def _log_rgb_band_order(vrt_path: Path, verbose: bool) -> None:
    if not verbose:
        return
    try:
        with rasterio.open(vrt_path) as src:
            count = src.count
            w, h = src.width, src.height
            print(f"[PIPE] VRT sanity: {vrt_path.name} bands={count} size={w}x{h}")
            if count != 3:
                print("[PIPE] VRT WARN: expected 3 bands (RGB order).")

            win_w = min(512, w)
            win_h = min(512, h)
            col0 = max(0, (w - win_w) // 2)
            row0 = max(0, (h - win_h) // 2)
            window = Window(col0, row0, win_w, win_h)

            arr = src.read(indexes=[1, 2, 3], window=window).astype(np.float32)
            means = [float(np.nanmean(arr[i])) for i in range(3)]
            mins = [float(np.nanmin(arr[i])) for i in range(3)]
            maxs = [float(np.nanmax(arr[i])) for i in range(3)]

            print("[PIPE] VRT band stats (sample window):")
            print("       band1 (R): mean={:.3f} min={:.3f} max={:.3f}".format(means[0], mins[0], maxs[0]))
            print("       band2 (G): mean={:.3f} min={:.3f} max={:.3f}".format(means[1], mins[1], maxs[1]))
            print("       band3 (B): mean={:.3f} min={:.3f} max={:.3f}".format(means[2], mins[2], maxs[2]))

            eq01 = float(np.nanmean(np.abs(arr[0] - arr[1]) < 1e-6))
            eq12 = float(np.nanmean(np.abs(arr[1] - arr[2]) < 1e-6))
            eq02 = float(np.nanmean(np.abs(arr[0] - arr[2]) < 1e-6))
            if max(eq01, eq12, eq02) > 0.95:
                print("[PIPE] VRT WARN: >=95% pixels identical between some bands (possible band wiring issue).")
    except Exception as e:
        print("[PIPE] VRT sanity WARN:", repr(e))


def _percentiles_streaming(
    src: rasterio.DatasetReader,
    *,
    band: int,
    width_c: int,
    height_c: int,
    p_lo: float,
    p_hi: float,
    sample_step: int = 16,
    sample_max: int = 12_000_000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    total = 0

    for _, win in src.block_windows(band):
        x0, y0 = int(win.col_off), int(win.row_off)
        if x0 >= width_c or y0 >= height_c:
            continue
        w = min(int(win.width), width_c - x0)
        h = min(int(win.height), height_c - y0)
        if w <= 0 or h <= 0:
            continue
        winc = Window(x0, y0, w, h)

        arr = src.read(band, window=winc).astype(np.float32, copy=False)
        arr = arr[::sample_step, ::sample_step].ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue

        remaining = sample_max - total
        if remaining <= 0:
            break

        take = min(arr.size, max(4096, remaining // 64))
        if take < arr.size:
            idx = rng.choice(arr.size, size=take, replace=False)
            arr = arr[idx]

        samples.append(arr)
        total += int(arr.size)

    if total <= 0:
        return 0.0, 1.0

    s = np.concatenate(samples).astype(np.float32, copy=False)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0, 1.0

    lo = float(np.quantile(s, float(p_lo)))
    hi = float(np.quantile(s, float(p_hi)))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo + 1.0
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _write_pan_3band_float01_streaming(
    pan_path: Path,
    out_path: Path,
    p_lo: float,
    p_hi: float,
    scale: int,
    verbose: bool,
) -> tuple[float, float]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(pan_path) as src:
        if src.count != 1:
            raise ValueError(f"[PIPE] PAN esperado 1 banda, veio {src.count}")

        W, H = int(src.width), int(src.height)
        Wc = (W // scale) * scale if scale > 1 else W
        Hc = (H // scale) * scale if scale > 1 else H

        if scale > 1 and (Wc != W or Hc != H) and verbose:
            print(f"[PIPE] WARN: PAN {W}x{H} not multiple of {scale}; cropping to {Wc}x{Hc} (bottom/right).")

        lo, hi = _percentiles_streaming(
            src,
            band=1,
            width_c=Wc,
            height_c=Hc,
            p_lo=p_lo,
            p_hi=p_hi,
            sample_step=16,
            sample_max=12_000_000,
            seed=0,
        )

        if verbose:
            print(f"[PIPE] PAN robust normalize p_lo={p_lo:.4f}->{lo:.3f} p_hi={p_hi:.4f}->{hi:.3f}")

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=Wc,
            height=Hc,
            count=3,
            dtype="float32",
            nodata=None,
            tiled=True,
            bigtiff="YES",
            interleave="pixel",
            compress="ZSTD",
            predictor=3,
            num_threads="ALL_CPUS",
        )
        profile.pop("nodata", None)

        with rasterio.open(out_path, "w", **profile) as dst:
            try:
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            except Exception:
                pass

            for _, win in src.block_windows(1):
                x0, y0 = int(win.col_off), int(win.row_off)
                if x0 >= Wc or y0 >= Hc:
                    continue
                w = min(int(win.width), Wc - x0)
                h = min(int(win.height), Hc - y0)
                if w <= 0 or h <= 0:
                    continue
                winc = Window(x0, y0, w, h)

                pan = src.read(1, window=winc).astype(np.float32, copy=False)
                pan01 = (pan - lo) / (hi - lo + 1e-6)
                pan01 = np.clip(pan01, 0.0, 1.0).astype(np.float32, copy=False)

                rgb = np.stack([pan01, pan01, pan01], axis=0).astype(np.float32, copy=False)
                dst.write(rgb, window=winc)

    return lo, hi


def _normalize_single_band_p2p98_01(
    band_path: Path,
    out_path: Path,
    p_lo: float,
    p_hi: float,
    verbose: bool,
    sample_max: int = 8_000_000,
    seed: int = 0,
    eps: float = 1e-6,
) -> None:
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    total = 0

    with rasterio.open(band_path) as src:
        if src.count != 1:
            raise RuntimeError("Banda única esperada")

        for _, win in src.block_windows(1):
            arr = src.read(1, window=win).astype(np.float32, copy=False)
            arr = arr[np.isfinite(arr)]
            arr = arr[arr > 0.0]
            if arr.size == 0:
                continue

            remaining = sample_max - total
            if remaining <= 0:
                break

            take = min(arr.size, max(4096, remaining // 64))
            if take < arr.size:
                idx = rng.choice(arr.size, size=take, replace=False)
                arr = arr[idx]

            samples.append(arr.astype(np.float32, copy=False))
            total += int(arr.size)

    lo, hi = 0.0, 1.0
    if total > 0:
        s = np.concatenate(samples).astype(np.float32, copy=False)
        s = s[np.isfinite(s)]
        if s.size > 0:
            lo = float(np.quantile(s, p_lo))
            hi = float(np.quantile(s, p_hi))
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0

    if verbose:
        print(f"[PIPE] NIR robust p_lo={p_lo:.4f}->{lo:.3f} p_hi={p_hi:.4f}->{hi:.3f}")

    with rasterio.open(band_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="float32",
            nodata=None,
            compress="ZSTD",
            predictor=3,
            num_threads="ALL_CPUS",
            bigtiff="YES",
        )
        profile.pop("nodata", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, win in src.block_windows(1):
                band = src.read(1, window=win).astype(np.float32, copy=False)
                valid = np.isfinite(band) & (band > 0.0)
                norm = (band - lo) / (hi - lo + eps)
                norm = np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)
                out = np.where(valid, norm, 0.0).astype(np.float32, copy=False)
                dst.write(out, 1, window=win)


def _normalize_rgb_joint_p2p98_01(
    rgb_path: Path,
    out_path: Path,
    p_lo: float,
    p_hi: float,
    verbose: bool,
    sample_max: int = 8_000_000,
    seed: int = 0,
    eps: float = 1e-6,
    border_frac: float = 0.03,
    min_valid_frac_window: float = 0.05,
) -> None:
    rng = np.random.default_rng(seed)
    samples = []
    total = 0

    with rasterio.open(rgb_path) as src:
        if src.count < 3:
            raise RuntimeError("RGB esperado com 3 bandas")

        H, W = src.height, src.width
        bx = max(0, int(round(W * float(border_frac))))
        by = max(0, int(round(H * float(border_frac))))

        for _, win in src.block_windows(1):
            rgb = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)
            h, w = rgb.shape[1], rgb.shape[2]
            if h <= 0 or w <= 0:
                continue

            rows = np.arange(int(win.row_off), int(win.row_off) + h, dtype=np.int32)[:, None]
            cols = np.arange(int(win.col_off), int(win.col_off) + w, dtype=np.int32)[None, :]
            inner = (rows >= by) & (rows < (H - by)) & (cols >= bx) & (cols < (W - bx))

            valid = np.isfinite(rgb).all(axis=0) & (rgb[0] > 0.0) & (rgb[1] > 0.0) & (rgb[2] > 0.0) & inner
            valid_frac = float(np.mean(valid))
            if valid_frac < float(min_valid_frac_window):
                continue

            y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            y = y[valid]
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue

            remaining = sample_max - total
            if remaining <= 0:
                break

            take = min(y.size, max(4096, remaining // 64))
            if take < y.size:
                idx = rng.choice(y.size, size=take, replace=False)
                y = y[idx]

            samples.append(y.astype(np.float32, copy=False))
            total += int(y.size)

    if total == 0:
        lo, hi = 0.0, 1.0
    else:
        s = np.concatenate(samples).astype(np.float32, copy=False)
        s = s[np.isfinite(s) & (s > 0.0)]
        if s.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.quantile(s, p_lo))
            hi = float(np.quantile(s, p_hi))
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0

    if verbose:
        print(f"[PIPE] GUIDE joint robust on Y: p_lo={p_lo:.4f}->{lo:.3f} p_hi={p_hi:.4f}->{hi:.3f}")

    with rasterio.open(rgb_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=3,
            dtype="float32",
            nodata=None,
            compress="ZSTD",
            predictor=3,
            num_threads="ALL_CPUS",
            bigtiff="YES",
        )
        profile.pop("nodata", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            try:
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            except Exception:
                pass

            for _, win in src.block_windows(1):
                rgb = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)
                valid = np.isfinite(rgb).all(axis=0) & (rgb[0] > 0.0) & (rgb[1] > 0.0) & (rgb[2] > 0.0)
                out = np.zeros_like(rgb, dtype=np.float32)

                y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                y01 = (y - lo) / (hi - lo + eps)
                y01 = np.clip(y01, 0.0, 1.0).astype(np.float32)

                gain = np.zeros_like(y, dtype=np.float32)
                gain[valid] = y01[valid] / (y[valid] + eps)

                for c in range(3):
                    band = rgb[c] * gain
                    out[c] = np.where(valid, np.clip(band, 0.0, 1.0), 0.0).astype(np.float32)

                dst.write(out, window=win)


def _normalize_rgb_per_band_p2p98_01(
    rgb_path: Path,
    out_path: Path,
    p_lo: float,
    p_hi: float,
    verbose: bool,
    sample_max: int = 8_000_000,
    seed: int = 0,
    eps: float = 1e-6,
    border_frac: float = 0.03,
    min_valid_frac_window: float = 0.05,
) -> None:
    rng = np.random.default_rng(seed)
    samples: list[list[np.ndarray]] = [[], [], []]
    total = 0

    with rasterio.open(rgb_path) as src:
        if src.count < 3:
            raise RuntimeError("RGB esperado com 3 bandas")

        H, W = src.height, src.width
        bx = max(0, int(round(W * float(border_frac))))
        by = max(0, int(round(H * float(border_frac))))

        for _, win in src.block_windows(1):
            rgb = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)

            h, w = rgb.shape[1], rgb.shape[2]
            if h <= 0 or w <= 0:
                continue

            rows = np.arange(int(win.row_off), int(win.row_off) + h, dtype=np.int32)[:, None]
            cols = np.arange(int(win.col_off), int(win.col_off) + w, dtype=np.int32)[None, :]
            inner = (rows >= by) & (rows < (H - by)) & (cols >= bx) & (cols < (W - bx))

            m = np.isfinite(rgb).all(axis=0) & (rgb[0] > 0.0) & (rgb[1] > 0.0) & (rgb[2] > 0.0) & inner
            valid_frac = float(np.mean(m))
            if valid_frac < float(min_valid_frac_window):
                continue

            idx = np.flatnonzero(m.ravel())
            if idx.size == 0:
                continue

            remaining = sample_max - total
            if remaining <= 0:
                break

            take = min(idx.size, max(4096, remaining // 64))
            if take <= 0:
                break

            sel = rng.choice(idx, size=take, replace=False) if take < idx.size else idx

            for c in range(3):
                v = rgb[c].ravel()[sel]
                v = v[np.isfinite(v)]
                if v.size > 0:
                    samples[c].append(v.astype(np.float32, copy=False))

            total += int(take)

    lo = [0.0, 0.0, 0.0]
    hi = [1.0, 1.0, 1.0]
    for c in range(3):
        if not samples[c]:
            continue
        s = np.concatenate(samples[c]).astype(np.float32, copy=False)
        s = s[np.isfinite(s) & (s > 0.0)]
        if s.size:
            lo[c] = float(np.quantile(s, p_lo))
            hi[c] = float(np.quantile(s, p_hi))
            if not np.isfinite(lo[c]):
                lo[c] = 0.0
            if not np.isfinite(hi[c]) or hi[c] <= lo[c]:
                hi[c] = lo[c] + 1.0

    if verbose:
        print(
            "[PIPE] GUIDE per-band robust:"
            f" R p_lo={p_lo:.4f}->{lo[0]:.3f} p_hi={p_hi:.4f}->{hi[0]:.3f} |"
            f" G p_lo={p_lo:.4f}->{lo[1]:.3f} p_hi={p_hi:.4f}->{hi[1]:.3f} |"
            f" B p_lo={p_lo:.4f}->{lo[2]:.3f} p_hi={p_hi:.4f}->{hi[2]:.3f}"
        )

    with rasterio.open(rgb_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=3,
            dtype="float32",
            nodata=None,
            compress="ZSTD",
            predictor=3,
            num_threads="ALL_CPUS",
            bigtiff="YES",
        )
        profile.pop("nodata", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            try:
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            except Exception:
                pass

            for _, win in src.block_windows(1):
                rgb = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)

                valid = np.isfinite(rgb).all(axis=0) & (rgb[0] > 0.0) & (rgb[1] > 0.0) & (rgb[2] > 0.0)
                out = np.zeros_like(rgb, dtype=np.float32)

                for c in range(3):
                    band = rgb[c]
                    norm = (band - lo[c]) / (hi[c] - lo[c] + eps)
                    norm = np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)
                    out[c] = np.where(valid, norm, 0.0)

                dst.write(out, window=win)


def _rgb_percentiles_streaming(
    src_path: Path,
    p_lo: float,
    p_hi: float,
    sample_step: int = 8,
    sample_max: int = 12_000_000,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    samples: list[list[np.ndarray]] = [[], [], []]
    total = 0

    with rasterio.open(src_path) as src:
        for _, win in src.block_windows(1):
            arr = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)
            arr = np.clip(arr, 0.0, 1.0)
            arr = arr[:, ::sample_step, ::sample_step]
            m = np.isfinite(arr).all(axis=0)
            if not np.any(m):
                continue

            idx = np.flatnonzero(m.ravel())
            remaining = sample_max - total
            if remaining <= 0:
                break

            take = min(idx.size, max(4096, remaining // 64))
            sel = rng.choice(idx, size=take, replace=False)

            for c in range(3):
                samples[c].append(arr[c].ravel()[sel].astype(np.float32, copy=False))
            total += take

    lo = [0.0, 0.0, 0.0]
    hi = [1.0, 1.0, 1.0]
    for c in range(3):
        if not samples[c]:
            continue
        s = np.concatenate(samples[c]).astype(np.float32, copy=False)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        lo[c] = float(np.quantile(s, p_lo))
        hi[c] = float(np.quantile(s, p_hi))
        if hi[c] <= lo[c]:
            hi[c] = lo[c] + 1e-3
    return lo, hi


def _write_visual_rgb_byte(
    src_path: Path,
    out_path: Path,
    *,
    p_lo: float,
    p_hi: float,
    gamma: float,
    compress: str,
    verbose: bool,
) -> tuple[list[float], list[float]]:
    lo, hi = _rgb_percentiles_streaming(src_path, p_lo=p_lo, p_hi=p_hi)

    if verbose:
        print(
            "[PIPE] VIS stretch:"
            f" R p{p_lo:.3f}={lo[0]:.4f} p{p_hi:.3f}={hi[0]:.4f} |"
            f" G p{p_lo:.3f}={lo[1]:.4f} p{p_hi:.3f}={hi[1]:.4f} |"
            f" B p{p_lo:.3f}={lo[2]:.4f} p{p_hi:.3f}={hi[2]:.4f} | gamma={gamma:.3f}"
        )

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=3,
            nodata=None,
            tiled=True,
            bigtiff="YES",
            interleave="pixel",
            compress=str(compress).upper(),
            photometric="YCBCR" if str(compress).upper() == "JPEG" else "RGB",
            blockxsize=512,
            blockysize=512,
            num_threads="ALL_CPUS",
        )
        if str(compress).upper() in ("DEFLATE", "LZW"):
            profile["predictor"] = 2
        profile.pop("nodata", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            try:
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            except Exception:
                pass

            for _, win in src.block_windows(1):
                rgb = src.read([1, 2, 3], window=win).astype(np.float32, copy=False)
                out = np.empty_like(rgb, dtype=np.uint8)
                for c in range(3):
                    x = (rgb[c] - lo[c]) / (hi[c] - lo[c] + 1e-6)
                    x = np.clip(x, 0.0, 1.0)
                    if gamma != 1.0:
                        x = np.power(x, gamma, dtype=np.float32)
                    out[c] = np.clip(np.round(x * 255.0), 0.0, 255.0).astype(np.uint8)
                dst.write(out, window=win)

    return lo, hi


def _gdal_driver_available(driver_name: str, verbose: bool) -> bool:
    exe = shutil.which("gdalinfo")
    if not exe:
        if verbose:
            print(f"[PIPE] WARN: gdalinfo não encontrado; não dá para testar driver {driver_name}.")
        return False

    cp = _run_capture([exe, "--formats"], verbose=False)
    if cp.returncode != 0:
        if verbose:
            print(f"[PIPE] WARN: falha ao consultar drivers GDAL para {driver_name}.")
        return False

    txt = (cp.stdout or "") + "\n" + (cp.stderr or "")
    return driver_name.upper() in txt.upper()


def _build_overviews(path: Path, levels: list[int], verbose: bool) -> None:
    exe = shutil.which("gdaladdo")
    if not exe:
        if verbose:
            print("[PIPE] WARN: gdaladdo não encontrado; overviews não criadas.")
        return
    cmd = [exe, "-r", "average", str(path)] + [str(v) for v in levels]
    _run(cmd, verbose=verbose)


def _translate_to_cog_jpeg(
    src: Path,
    dst: Path,
    *,
    quality: int,
    blocksize: int,
    verbose: bool,
) -> None:
    exe = shutil.which("gdal_translate")
    if not exe:
        raise RuntimeError("gdal_translate não encontrado para exportar COG.")

    quality = int(np.clip(quality, 1, 100))
    blocksize = int(max(blocksize, 128))

    cmd = [
        exe,
        "-of", "COG",
        "-co", "COMPRESS=JPEG",
        "-co", f"QUALITY={quality}",
        "-co", f"BLOCKSIZE={blocksize}",
        "-co", "BIGTIFF=IF_SAFER",
        "-co", "OVERVIEWS=IGNORE_EXISTING",
        src,
        dst,
    ]
    _run(cmd, verbose=verbose)


def _translate_to_ecw(src: Path, dst: Path, target_mb: int, verbose: bool) -> None:
    exe = shutil.which("gdal_translate")
    if not exe:
        raise RuntimeError("gdal_translate não encontrado para exportar ECW.")

    if not _gdal_driver_available("ECW", verbose=verbose):
        raise RuntimeError("Driver ECW não disponível neste GDAL.")

    cmd = [exe, "-of", "ECW"]
    if target_mb > 0 and src.exists():
        sz_mb = max(src.stat().st_size / (1024.0 * 1024.0), 1.0)
        ratio = max(sz_mb / float(target_mb), 2.0)
        ratio = min(ratio, 200.0)
        cmd += ["-co", f"TARGET={ratio:.2f}"]
    else:
        cmd += ["-co", "TARGET=20"]

    cmd += [src, dst]
    _run(cmd, verbose=verbose)


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


def _sample_windows_grid(W: int, H: int, win: int, n: int):
    import rasterio.windows

    win = min(win, W, H)
    if win <= 0:
        return []

    if n <= 1:
        x0 = max(0, (W - win) // 2)
        y0 = max(0, (H - win) // 2)
        return [rasterio.windows.Window(x0, y0, win, win)]

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
            out.append(rasterio.windows.Window(xi, yi, win, win))

    return out


def _rgb_to_ycbcr(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb01[0], rgb01[1], rgb01[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def _chroma_mag(cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb0 = cb - 0.5
    cr0 = cr - 0.5
    return np.sqrt(cb0 * cb0 + cr0 * cr0).astype(np.float32)


def _qa_vs_pan_reference(
    pan_ref_path: Path,
    out_rgb: Path,
    *,
    guide_path: Path | None,
    verbose: bool,
    win: int,
    n: int,
    sample_step: int,
    min_corr: float,
    max_chroma_pan_corr: float,
    min_valid_frac: float = 0.70,
    eps_valid: float = 1e-6,
) -> None:
    with rasterio.open(pan_ref_path) as psrc, rasterio.open(out_rgb) as osrc:
        if psrc.width != osrc.width or psrc.height != osrc.height:
            print(f"[QA3] WARN: size mismatch PANref={psrc.width}x{psrc.height} OUT={osrc.width}x{osrc.height} (QA3 skipped)")
            return

        W, H = psrc.width, psrc.height
        wins = _sample_windows_grid(W, H, win=win, n=n)

        gI_corrs: list[float] = []
        lI_corrs: list[float] = []
        chroma_pan_corrs: list[float] = []
        guide_chroma_corrs: list[float] = []

        skipped_low_valid = 0
        skipped_small_mask = 0
        gsrc = rasterio.open(guide_path) if guide_path else None

        try:
            for w in wins:
                pan01 = psrc.read(1, window=w).astype(np.float32)
                pan01 = np.clip(pan01, 0.0, 1.0)

                rgb = osrc.read([1, 2, 3], window=w).astype(np.float32)
                rgb = np.clip(rgb, 0.0, 1.0)

                I = ((rgb[0] + rgb[1] + rgb[2]) / 3.0).astype(np.float32)
                _y_out, cb, cr = _rgb_to_ycbcr(rgb)
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
                    g = gsrc.read([1, 2, 3], window=w).astype(np.float32)
                    g = np.clip(g, 0.0, 1.0)
                    y_g, _, _ = _rgb_to_ycbcr(g)

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
                f"[QA3] WARN: insufficient valid samples after masking "
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
            f"[QA3:{lvl}] vs PAN reference | "
            f"corr(grad(PAN_ref01),grad(I_out)) mean={g_mean:.3f} min={g_min:.3f} max={g_max:.3f} | "
            f"corr(lap(PAN_ref01),lap(I_out)) mean={l_mean:.3f} min={l_min:.3f} max={l_max:.3f} | "
            f"leak corr(grad(PAN_ref01),grad(|CbCr|)) mean={c_mean:.3f} min={c_min:.3f} max={c_max:.3f} | "
            f"candidate_windows={len(wins)} used_windows={len(gI_corrs)} "
            f"skipped_low_valid={skipped_low_valid} skipped_small_mask={skipped_small_mask}"
        )

        if verbose and guide_chroma_corrs:
            gg_mean, gg_min, gg_max = _stats(guide_chroma_corrs)
            print(
                f"[QA3] (info) corr(grad(Y_guide),grad(|CbCr_out|)) mean={gg_mean:.3f} "
                f"min={gg_min:.3f} max={gg_max:.3f} | n={len(guide_chroma_corrs)}"
            )


def _export_visual_products(cfg: PipelineConfig, master_path: Path) -> None:
    if not cfg.export_vis:
        return

    vis_tif = cfg.outdir / "pan_2m_color_guided_vis.tif"
    with _timed("VIS export byte GeoTIFF", cfg.verbose):
        _write_visual_rgb_byte(
            src_path=master_path,
            out_path=vis_tif,
            p_lo=float(cfg.vis_p_lo),
            p_hi=float(cfg.vis_p_hi),
            gamma=float(cfg.vis_gamma),
            compress=str(cfg.vis_compress).upper(),
            verbose=cfg.verbose,
        )

    if cfg.vis_overviews:
        with _timed("VIS overviews", cfg.verbose):
            _build_overviews(vis_tif, levels=[2, 4, 8, 16, 32], verbose=cfg.verbose)

    if str(cfg.vis_format).upper() == "COG":
        vis_cog = cfg.outdir / "pan_2m_color_guided_vis.cog.tif"
        with _timed("VIS export COG", cfg.verbose):
            _translate_to_cog_jpeg(
                vis_tif,
                vis_cog,
                quality=90,
                blocksize=512,
                verbose=cfg.verbose,
            )


def _export_cog_product(cfg: PipelineConfig, master_path: Path) -> None:
    if not cfg.export_cog:
        return

    temp_light = cfg.outdir / "pan_2m_color_guided_light.tif"
    cog_path = cfg.outdir / "pan_2m_color_guided.cog.tif"

    try:
        with _timed("COG light export byte GeoTIFF", cfg.verbose):
            _write_visual_rgb_byte(
                src_path=master_path,
                out_path=temp_light,
                p_lo=float(cfg.vis_p_lo),
                p_hi=float(cfg.vis_p_hi),
                gamma=float(cfg.vis_gamma),
                compress="JPEG",
                verbose=cfg.verbose,
            )

        if cfg.cog_overviews:
            with _timed("COG light overviews", cfg.verbose):
                _build_overviews(temp_light, levels=[2, 4, 8, 16, 32], verbose=cfg.verbose)

        with _timed("COG export", cfg.verbose):
            _translate_to_cog_jpeg(
                temp_light,
                cog_path,
                quality=int(cfg.cog_quality),
                blocksize=int(cfg.cog_blocksize),
                verbose=cfg.verbose,
            )
    finally:
        _safe_unlink([temp_light], verbose=cfg.verbose)


def _export_ecw_product(cfg: PipelineConfig, master_path: Path) -> None:
    if not cfg.export_ecw:
        return

    out_ecw = cfg.outdir / "pan_2m_color_guided.ecw"
    try:
        with _timed("ECW export", cfg.verbose):
            _translate_to_ecw(master_path, out_ecw, target_mb=int(cfg.ecw_target_mb), verbose=cfg.verbose)
    except Exception as e:
        print(f"[PIPE] WARN: ECW indisponível/falhou ({repr(e)}). O master TIFF foi mantido.")


def _run_pipeline_impl(cfg: PipelineConfig) -> Path:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    _assert_exists(cfg.pan, "PAN")
    _assert_exists(cfg.blue, "BLUE band")
    _assert_exists(cfg.green, "GREEN band")
    _assert_exists(cfg.red, "RED band")
    _assert_exists(cfg.nir, "NIR band")

    if cfg.verbose:
        print("[PIPE] outdir:", cfg.outdir)
        print("[PIPE] workdir:", cfg.workdir)
        print("[PIPE] keep_tmp:", cfg.keep_tmp)
        print("[PIPE] device:", cfg.device, "| tile:", cfg.tile, "| overlap:", cfg.overlap, "| scale:", cfg.scale)
        print("[PIPE] io_block:", cfg.io_block, "| out_block:", cfg.out_block, "| compress:", cfg.compress)
        print("[PIPE] amp:", cfg.amp)
        print("[PIPE] guide_norm:", cfg.guide_norm)
        print("[PIPE] fusion_mode:", cfg.fusion_mode)
        print("[PIPE] PAN p_lo/hi:", cfg.pan_p_lo, cfg.pan_p_hi)
        print("[PIPE] guide p_lo/hi:", cfg.guide_p_lo, cfg.guide_p_hi)
        print("[PIPE] detail:", cfg.detail_strength, cfg.detail_alpha1, cfg.detail_alpha2, cfg.detail_alpha3)
        print("[PIPE] pan_hp:", cfg.pan_hp_sigma, cfg.pan_hp_gain, cfg.pan_hp_gain_urban, cfg.pan_hp_gain_veg, cfg.pan_hp_gain_shadow)
        print("[PIPE] pan_hi_damp:", cfg.highlight_pan_damp_knee, cfg.highlight_pan_damp_strength)
        print("[PIPE] guided:", cfg.guided_radius, cfg.guided_eps, cfg.guided_chroma_mix)
        print("[PIPE] radiometry:", cfg.rad_p_lo, cfg.rad_p_hi, cfg.rad_soft_margin)
        print("[PIPE] export_vis:", cfg.export_vis, "| vis_format:", cfg.vis_format, "| vis_compress:", cfg.vis_compress)
        print("[PIPE] export_cog:", cfg.export_cog, "| cog_quality:", cfg.cog_quality, "| cog_blocksize:", cfg.cog_blocksize)
        print("[PIPE] export_ecw:", cfg.export_ecw, "| ecw_target_mb:", cfg.ecw_target_mb)

    _log_inputs(cfg)

    step_times: dict[str, float] = {}
    t_total0 = time.perf_counter()
    tmp_files: list[Path] = []

    from cbers_colorize.ops_gdal import (
        build_rgb_lr_vrt_aligned_to_pan,
        cleanup_rgb_lr_tmp_files,
        make_env as make_gdal_env,
        GdalEnv,
        gdalwarp_to_ref_gtiff,
    )

    gdal_env = make_gdal_env(cfg=GdalEnv(gdal_cachemax_mb=None))

    pan_3band_f01 = cfg.workdir / "pan_3band_f01.tif"
    if cfg.verbose:
        print("[PIPE] STEP0 PAN->3band float01:", pan_3band_f01)

    with _timed("STEP0 PAN->3band float01", cfg.verbose, step_times):
        _write_pan_3band_float01_streaming(
            pan_path=cfg.pan,
            out_path=pan_3band_f01,
            p_lo=cfg.pan_p_lo,
            p_hi=cfg.pan_p_hi,
            scale=cfg.scale,
            verbose=cfg.verbose,
        )

    rgb_lr_vrt = cfg.workdir / "rgb_lr_8m_aligned.vrt"
    if cfg.verbose:
        print("[PIPE] STEP1 Build LR RGB VRT ->", rgb_lr_vrt)

    with _timed("STEP1 build LR RGB VRT", cfg.verbose, step_times):
        build_rgb_lr_vrt_aligned_to_pan(
            blue=cfg.blue,
            green=cfg.green,
            red=cfg.red,
            pan_ref_tif=pan_3band_f01,
            out_rgb_vrt=rgb_lr_vrt,
            scale=cfg.scale,
            verbose=cfg.verbose,
            env=gdal_env,
        )

    _log_rgb_band_order(rgb_lr_vrt, verbose=cfg.verbose)

    sensor_lr_01 = cfg.workdir / "rgb_sensor_lr_8m_01.tif"
    if cfg.verbose:
        print("[PIPE] STEP1b sensor LR 0..1 ->", sensor_lr_01)

    with _timed("STEP1b sensor LR normalize", cfg.verbose, step_times):
        tmp_sensor_lr = cfg.workdir / "rgb_sensor_lr_8m_tmp.tif"
        gdalwarp_to_ref_gtiff(
            in_path=rgb_lr_vrt,
            out_tif=tmp_sensor_lr,
            ref_path=rgb_lr_vrt,
            resample="nearest",
            dtype="Float32",
            overwrite=True,
            verbose=cfg.verbose,
            env=gdal_env,
        )
        if str(cfg.guide_norm).lower() in ("per_band", "perband", "per-band"):
            _normalize_rgb_per_band_p2p98_01(
                rgb_path=tmp_sensor_lr,
                out_path=sensor_lr_01,
                p_lo=cfg.guide_p_lo,
                p_hi=cfg.guide_p_hi,
                verbose=cfg.verbose,
            )
        else:
            _normalize_rgb_joint_p2p98_01(
                rgb_path=tmp_sensor_lr,
                out_path=sensor_lr_01,
                p_lo=cfg.guide_p_lo,
                p_hi=cfg.guide_p_hi,
                verbose=cfg.verbose,
            )
        tmp_files.append(tmp_sensor_lr)

    guide_sensor_hr = cfg.workdir / "rgb_sensor_hr_2m_01.tif"
    if cfg.verbose:
        print("[PIPE] STEP2 GUIDE sensor RGB ->", guide_sensor_hr)

    with _timed("STEP2 resample SENSOR RGB guide", cfg.verbose, step_times):
        gdalwarp_to_ref_gtiff(
            in_path=sensor_lr_01,
            out_tif=guide_sensor_hr,
            ref_path=pan_3band_f01,
            resample="cubic",
            dtype="Float32",
            overwrite=True,
            verbose=cfg.verbose,
            env=gdal_env,
        )

    nir_lr_01 = cfg.workdir / "nir_sensor_lr_8m_01.tif"
    if cfg.verbose:
        print("[PIPE] STEP2b sensor LR NIR 0..1 ->", nir_lr_01)

    with _timed("STEP2b sensor LR NIR normalize", cfg.verbose, step_times):
        _normalize_single_band_p2p98_01(
            band_path=cfg.nir,
            out_path=nir_lr_01,
            p_lo=cfg.guide_p_lo,
            p_hi=cfg.guide_p_hi,
            verbose=cfg.verbose,
        )

    nir_hr_01 = cfg.workdir / "nir_sensor_hr_2m_01.tif"
    if cfg.verbose:
        print("[PIPE] STEP3 GUIDE sensor NIR ->", nir_hr_01)

    with _timed("STEP3 resample SENSOR NIR guide", cfg.verbose, step_times):
        gdalwarp_to_ref_gtiff(
            in_path=nir_lr_01,
            out_tif=nir_hr_01,
            ref_path=pan_3band_f01,
            resample="cubic",
            dtype="Float32",
            overwrite=True,
            verbose=cfg.verbose,
            env=gdal_env,
        )

    out_pan_color = cfg.outdir / "pan_2m_color_guided.tif"
    if cfg.verbose:
        print("[PIPE] STEP4 FUSE upgraded PAN+RGB+NIR ->", out_pan_color)

    fuse_cmd = [
        sys.executable,
        str(cfg.color_tool),
        "--in_tif", str(pan_3band_f01),
        "--out_tif", str(out_pan_color),
        "--device", cfg.device,
        "--tile", str(cfg.tile),
        "--overlap", str(cfg.overlap),
        *(["--sanitize"] if cfg.sanitize else []),
        *(["--verbose"] if cfg.verbose else []),
        "--norm", "0_1",
        "--out_act", "sigmoid",
        "--out_temp", "1.0",
        "--out_range", "0_1",
        "--guide_rgb", str(guide_sensor_hr),
        "--guide_nir", str(nir_hr_01),
        "--guide_mode", "pan_luminance_injection",
        "--guide_only",
        "--fusion_mode", str(cfg.fusion_mode),
        "--chroma_strength", str(cfg.chroma_strength),
        "--sat", str(cfg.sat),
        "--max_gain", str(cfg.max_gain),
        "--chroma_blur_sigma", str(cfg.chroma_blur_sigma),
        "--chroma_blur_ksize", str(cfg.chroma_blur_ksize),
        "--veg_exg_th", str(cfg.veg_exg_th),
        "--veg_sat", str(cfg.veg_sat),
        "--veg_chroma", str(cfg.veg_chroma),
        "--veg_cr_bias", str(float(cfg.veg_cr_bias)),
        "--veg_cb_bias", str(float(cfg.veg_cb_bias)),
        "--cb_bias", str(float(cfg.cb_bias)),
        "--cr_bias", str(float(cfg.cr_bias)),
        "--neutral_mag_thr", str(float(cfg.neutral_mag_thr)),
        "--neutral_strength", str(float(cfg.neutral_strength)),
        "--neutral_cb_bias", str(float(cfg.neutral_cb_bias)),
        "--neutral_cr_bias", str(float(cfg.neutral_cr_bias)),
        "--shadow_y_lo", str(float(cfg.shadow_y_lo)),
        "--shadow_y_hi", str(float(cfg.shadow_y_hi)),
        "--shadow_strength", str(float(cfg.shadow_strength)),
        "--shadow_cb_bias", str(float(cfg.shadow_cb_bias)),
        "--shadow_cr_bias", str(float(cfg.shadow_cr_bias)),
        "--shadow_chroma", str(float(cfg.shadow_chroma)),
        "--hi_y", str(float(cfg.hi_y)),
        "--hi_desat", str(float(cfg.hi_desat)),
        "--gamut_gain", str(float(cfg.gamut_gain)),
        "--luma_rolloff_knee", str(float(cfg.luma_rolloff_knee)),
        "--luma_rolloff_strength", str(float(cfg.luma_rolloff_strength)),
        "--luma_gamma", str(float(cfg.luma_gamma)),
        "--detail_strength", str(float(cfg.detail_strength)),
        "--detail_strength_urban", str(float(cfg.detail_strength_urban)),
        "--detail_strength_veg", str(float(cfg.detail_strength_veg)),
        "--detail_strength_shadow", str(float(cfg.detail_strength_shadow)),
        "--detail_sigma1", str(float(cfg.detail_sigma1)),
        "--detail_sigma2", str(float(cfg.detail_sigma2)),
        "--detail_sigma3", str(float(cfg.detail_sigma3)),
        "--detail_alpha1", str(float(cfg.detail_alpha1)),
        "--detail_alpha2", str(float(cfg.detail_alpha2)),
        "--detail_alpha3", str(float(cfg.detail_alpha3)),
        "--pan_hp_sigma", str(float(cfg.pan_hp_sigma)),
        "--pan_hp_gain", str(float(cfg.pan_hp_gain)),
        "--pan_hp_gain_urban", str(float(cfg.pan_hp_gain_urban)),
        "--pan_hp_gain_veg", str(float(cfg.pan_hp_gain_veg)),
        "--pan_hp_gain_shadow", str(float(cfg.pan_hp_gain_shadow)),
        "--highlight_pan_damp_knee", str(float(cfg.highlight_pan_damp_knee)),
        "--highlight_pan_damp_strength", str(float(cfg.highlight_pan_damp_strength)),
        "--guided_radius", str(int(cfg.guided_radius)),
        "--guided_eps", str(float(cfg.guided_eps)),
        "--guided_chroma_mix", str(float(cfg.guided_chroma_mix)),
        "--ndvi_veg_lo", str(float(cfg.ndvi_veg_lo)),
        "--ndvi_veg_hi", str(float(cfg.ndvi_veg_hi)),
        "--nir_detail_boost", str(float(cfg.nir_detail_boost)),
        "--urban_detail_from_nir", str(float(cfg.urban_detail_from_nir)),
        "--rad_p_lo", str(float(cfg.rad_p_lo)),
        "--rad_p_hi", str(float(cfg.rad_p_hi)),
        "--rad_soft_margin", str(float(cfg.rad_soft_margin)),
        "--io_block", str(int(cfg.io_block)),
        "--out_block", str(int(cfg.out_block)),
        "--compress", str(cfg.compress).upper(),
    ]

    if cfg.device == "cuda":
        fuse_cmd.append("--amp" if cfg.amp else "--no_amp")

    with _timed("STEP4 FUSE upgraded", cfg.verbose, step_times):
        _run(fuse_cmd, verbose=cfg.verbose)

    with _timed("QA3 vs PAN reference", cfg.verbose, step_times):
        _qa_vs_pan_reference(
            pan_ref_path=pan_3band_f01,
            out_rgb=out_pan_color,
            guide_path=guide_sensor_hr,
            verbose=cfg.verbose,
            win=int(cfg.qa_win),
            n=int(cfg.qa_num_windows),
            sample_step=int(cfg.qa_sample_step),
            min_corr=float(cfg.qa_min_corr),
            max_chroma_pan_corr=0.45,
            min_valid_frac=0.70,
        )

    _export_visual_products(cfg, out_pan_color)
    _export_cog_product(cfg, out_pan_color)
    _export_ecw_product(cfg, out_pan_color)

    if not cfg.keep_tmp:
        cleanup_rgb_lr_tmp_files(cfg.workdir, verbose=cfg.verbose)
        tmp_files.extend([
            rgb_lr_vrt,
            guide_sensor_hr,
            sensor_lr_01,
            nir_lr_01,
            nir_hr_01,
            pan_3band_f01,
        ])
        _safe_unlink(tmp_files, verbose=cfg.verbose)
        _safe_rmtree(cfg.workdir, verbose=cfg.verbose)
    else:
        if cfg.verbose:
            print("[PIPE] KEEP tmp files:")
            for p in [
                rgb_lr_vrt,
                guide_sensor_hr,
                sensor_lr_01,
                nir_lr_01,
                nir_hr_01,
                pan_3band_f01,
                cfg.workdir / "_tmp_red_lr.vrt",
                cfg.workdir / "_tmp_green_lr.vrt",
                cfg.workdir / "_tmp_blue_lr.vrt",
                cfg.workdir,
            ]:
                print("       ", p)

    t_total = time.perf_counter() - t_total0
    if cfg.verbose:
        print(f"[TIME] TOTAL: {t_total:.2f}s")
        if step_times:
            s = sum(step_times.values())
            if s > 0:
                print("[TIME] Breakdown:")
                for k, v in step_times.items():
                    print(f"       {k:30s} {v:8.2f}s  ({100.0*v/s:5.1f}%)")

    return out_pan_color


def run_pipeline(args, workdir: Path, outdir: Path) -> Path:
    cfg = PipelineConfig(
        pan=Path(args.pan),
        blue=Path(args.blue),
        green=Path(args.green),
        red=Path(args.red),
        nir=Path(args.nir),
        outdir=outdir,
        workdir=workdir,
        device=str(args.device),
        tile=int(args.tile),
        overlap=int(args.overlap),
        sanitize=bool(getattr(args, "sanitize", False)),
        verbose=bool(getattr(args, "verbose", False)),
        keep_tmp=bool(getattr(args, "keep_tmp", False)),
        amp=bool(getattr(args, "amp", True)),
        scale=int(getattr(args, "scale", 4)),
        fusion_mode=str(getattr(args, "fusion_mode", "multiscale_ycbcr")),
        chroma_strength=float(getattr(args, "chroma_strength", 0.72)),
        sat=float(getattr(args, "sat", 0.92)),
        max_gain=float(getattr(args, "max_gain", 2.5)),
        chroma_blur_sigma=float(getattr(args, "chroma_blur_sigma", 0.6)),
        chroma_blur_ksize=int(getattr(args, "chroma_blur_ksize", 5)),
        veg_exg_th=float(getattr(args, "veg_exg_th", 0.10)),
        veg_sat=float(getattr(args, "veg_sat", 0.82)),
        veg_chroma=float(getattr(args, "veg_chroma", 0.92)),
        io_block=int(getattr(args, "io_block", getattr(args, "block", 1024))),
        out_block=int(getattr(args, "out_block", getattr(args, "block", 1024))),
        compress=str(getattr(args, "compress", "ZSTD")).upper(),
        cr_bias=float(getattr(args, "cr_bias", -0.006)),
        cb_bias=float(getattr(args, "cb_bias", -0.003)),
        veg_cr_bias=float(getattr(args, "veg_cr_bias", -0.010)),
        veg_cb_bias=float(getattr(args, "veg_cb_bias", 0.003)),
        guide_norm=str(getattr(args, "guide_norm", "per_band")),
        pan_p_lo=float(getattr(args, "pan_p_lo", 0.02)),
        pan_p_hi=float(getattr(args, "pan_p_hi", 0.998)),
        guide_p_lo=float(getattr(args, "guide_p_lo", 0.02)),
        guide_p_hi=float(getattr(args, "guide_p_hi", 0.98)),
        neutral_mag_thr=float(getattr(args, "neutral_mag_thr", 0.050)),
        neutral_strength=float(getattr(args, "neutral_strength", 0.70)),
        neutral_cb_bias=float(getattr(args, "neutral_cb_bias", 0.0)),
        neutral_cr_bias=float(getattr(args, "neutral_cr_bias", 0.0)),
        shadow_y_lo=float(getattr(args, "shadow_y_lo", 0.08)),
        shadow_y_hi=float(getattr(args, "shadow_y_hi", 0.28)),
        shadow_strength=float(getattr(args, "shadow_strength", 0.20)),
        shadow_cb_bias=float(getattr(args, "shadow_cb_bias", -0.004)),
        shadow_cr_bias=float(getattr(args, "shadow_cr_bias", 0.003)),
        shadow_chroma=float(getattr(args, "shadow_chroma", 0.95)),
        hi_y=float(getattr(args, "hi_y", 0.86)),
        hi_desat=float(getattr(args, "hi_desat", 0.45)),
        gamut_gain=float(getattr(args, "gamut_gain", 4.0)),
        luma_rolloff_knee=float(getattr(args, "luma_rolloff_knee", 0.88)),
        luma_rolloff_strength=float(getattr(args, "luma_rolloff_strength", 0.32)),
        luma_gamma=float(getattr(args, "luma_gamma", 0.98)),
        detail_strength=float(getattr(args, "detail_strength", 0.72)),
        detail_strength_urban=float(getattr(args, "detail_strength_urban", 1.15)),
        detail_strength_veg=float(getattr(args, "detail_strength_veg", 0.55)),
        detail_strength_shadow=float(getattr(args, "detail_strength_shadow", 0.85)),
        detail_sigma1=float(getattr(args, "detail_sigma1", 1.2)),
        detail_sigma2=float(getattr(args, "detail_sigma2", 2.8)),
        detail_sigma3=float(getattr(args, "detail_sigma3", 5.6)),
        detail_alpha1=float(getattr(args, "detail_alpha1", 0.62)),
        detail_alpha2=float(getattr(args, "detail_alpha2", 0.28)),
        detail_alpha3=float(getattr(args, "detail_alpha3", 0.14)),
        pan_hp_sigma=float(getattr(args, "pan_hp_sigma", 2.2)),
        pan_hp_gain=float(getattr(args, "pan_hp_gain", 0.16)),
        pan_hp_gain_urban=float(getattr(args, "pan_hp_gain_urban", 1.35)),
        pan_hp_gain_veg=float(getattr(args, "pan_hp_gain_veg", 0.50)),
        pan_hp_gain_shadow=float(getattr(args, "pan_hp_gain_shadow", 0.75)),
        highlight_pan_damp_knee=float(getattr(args, "highlight_pan_damp_knee", 0.82)),
        highlight_pan_damp_strength=float(getattr(args, "highlight_pan_damp_strength", 0.75)),
        guided_radius=int(getattr(args, "guided_radius", 4)),
        guided_eps=float(getattr(args, "guided_eps", 1e-4)),
        guided_chroma_mix=float(getattr(args, "guided_chroma_mix", 0.65)),
        ndvi_veg_lo=float(getattr(args, "ndvi_veg_lo", 0.18)),
        ndvi_veg_hi=float(getattr(args, "ndvi_veg_hi", 0.38)),
        nir_detail_boost=float(getattr(args, "nir_detail_boost", 0.20)),
        urban_detail_from_nir=float(getattr(args, "urban_detail_from_nir", 0.12)),
        rad_p_lo=float(getattr(args, "rad_p_lo", 0.003)),
        rad_p_hi=float(getattr(args, "rad_p_hi", 0.997)),
        rad_soft_margin=float(getattr(args, "rad_soft_margin", 0.020)),
        export_vis=bool(getattr(args, "export_vis", False)),
        vis_format=str(getattr(args, "vis_format", "GTIFF")).upper(),
        vis_compress=str(getattr(args, "vis_compress", "JPEG")).upper(),
        vis_p_lo=float(getattr(args, "vis_p_lo", 0.01)),
        vis_p_hi=float(getattr(args, "vis_p_hi", 0.995)),
        vis_gamma=float(getattr(args, "vis_gamma", 1.00)),
        vis_overviews=bool(getattr(args, "vis_overviews", False)),
        export_cog=bool(getattr(args, "export_cog", False)),
        cog_quality=int(getattr(args, "cog_quality", 90)),
        cog_blocksize=int(getattr(args, "cog_blocksize", 512)),
        cog_overviews=bool(getattr(args, "cog_overviews", False)),
        export_ecw=bool(getattr(args, "export_ecw", False)),
        ecw_target_mb=int(getattr(args, "ecw_target_mb", 0)),
    )
    return _run_pipeline_impl(cfg)