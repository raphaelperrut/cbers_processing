# cbers_colorize/ops_color.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional
import os
import subprocess

import numpy as np
import rasterio
from rasterio.enums import ColorInterp


@dataclass
class GuideStretch:
    """
    Config do 'guide byte cut' (visual guidance):
    cada banda é esticada entre p_lo..p_hi para 0..255.
    """
    p_lo: float = 0.02
    p_hi: float = 0.98
    sample_stride: int = 16  # amostra rápida para percentis


def _run(cmd: Sequence[str], verbose: bool = False, env: Optional[dict] = None) -> None:
    if verbose:
        print("[COLOR] RUN:", " ".join(cmd))
    subprocess.run(list(cmd), check=True, env=env)


def make_env_with_pythonpath(base_env: Optional[dict] = None) -> dict:
    """
    Garante que scripts em /app/cbers_colorize/tools consigam importar:
      - cbers_colorize.*
      - cbers_colorize.rsinet.*
    """
    env = dict(os.environ if base_env is None else base_env)
    py_path = env.get("PYTHONPATH", "")
    needed = "/app"
    if needed not in py_path.split(":"):
        env["PYTHONPATH"] = needed + (":" + py_path if py_path else "")
    return env


def _percentiles_from_sample(sample: np.ndarray, p_lo: float, p_hi: float) -> tuple[float, float]:
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(sample, p_lo))
    hi = float(np.quantile(sample, p_hi))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def write_guide_byte_cut(
    rgb_sr_2m_path: Path,
    out_path: Path,
    stretch: GuideStretch,
    verbose: bool = False,
) -> None:
    """
    Converte rgb_sr_2m (float/qualquer escala) para 3-band uint8
    com stretch por banda (p_lo..p_hi) usando amostragem.

    Isso é exatamente o que você vinha usando como:
      rgb_sr_2m_frombyte_01_vis_p2p98_01_byte_cut.tif
    """
    with rasterio.open(rgb_sr_2m_path) as src:
        rgb = src.read().astype(np.float32)  # (3,H,W)
        profile = src.profile

    if rgb.shape[0] != 3:
        raise ValueError(f"Esperado 3 bandas em {rgb_sr_2m_path}, veio {rgb.shape[0]}")

    out = np.zeros_like(rgb, dtype=np.uint8)

    for c in range(3):
        x = rgb[c]
        samp = x[::stretch.sample_stride, ::stretch.sample_stride].ravel()
        lo, hi = _percentiles_from_sample(samp, stretch.p_lo, stretch.p_hi)
        if verbose:
            print(f"[COLOR] GUIDE band{c} stretch p{int(stretch.p_lo*100)}={lo:.6f} "
                  f"p{int(stretch.p_hi*100)}={hi:.6f}")
        y = (x - lo) / (hi - lo + 1e-6)
        y = np.clip(y, 0.0, 1.0)
        out[c] = (y * 255.0 + 0.5).astype(np.uint8)

    prof = profile.copy()
    prof.update(dtype="uint8", count=3, nodata=None)
    prof.pop("nodata", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)


@dataclass
class ColorizeParams:
    """
    Parâmetros equivalentes ao seu comando 'v3.1 que funcionou'.
    """
    norm: str = "0_255"
    out_act: str = "sigmoid"
    out_temp: float = 1.0
    out_range: str = "0_1"

    guide_mode: str = "pan_luminance_injection"
    fusion_mode: str = "ycbcr"

    chroma_strength: float = 0.85
    sat: float = 0.95
    max_gain: float = 2.5  # só no ratio

    chroma_blur_sigma: float = 0.9
    chroma_blur_ksize: int = 7

    veg_exg_th: float = 0.12
    veg_sat: float = 0.55
    veg_chroma: float = 0.65


def run_colorize_tool(
    *,
    color_tool: Path,
    pan_3band_byte: Path,
    guide_rgb_byte: Path,
    color_pkl: Path,
    out_tif: Path,
    device: str,
    tile: int,
    overlap: int,
    sanitize: bool,
    verbose: bool,
    params: ColorizeParams,
) -> None:
    """
    Chama tools/infer_geotiff_color.py com os parâmetros afinados.
    """
    env = make_env_with_pythonpath()

    cmd = [
        "python",
        str(color_tool),
        "--in_tif",
        str(pan_3band_byte),
        "--pkl",
        str(color_pkl),
        "--out_tif",
        str(out_tif),
        "--device",
        str(device),
        "--tile",
        str(int(tile)),
        "--overlap",
        str(int(overlap)),
        "--norm",
        params.norm,
        "--out_act",
        params.out_act,
        "--out_temp",
        str(float(params.out_temp)),
        "--out_range",
        params.out_range,
        "--guide_rgb",
        str(guide_rgb_byte),
        "--guide_mode",
        params.guide_mode,
        "--fusion_mode",
        params.fusion_mode,
        "--chroma_strength",
        str(float(params.chroma_strength)),
        "--sat",
        str(float(params.sat)),
        "--chroma_blur_sigma",
        str(float(params.chroma_blur_sigma)),
        "--chroma_blur_ksize",
        str(int(params.chroma_blur_ksize)),
        "--veg_exg_th",
        str(float(params.veg_exg_th)),
        "--veg_sat",
        str(float(params.veg_sat)),
        "--veg_chroma",
        str(float(params.veg_chroma)),
    ]

    # max_gain só interessa no modo ratio, mas não faz mal passar também (se seu script aceita).
    cmd += ["--max_gain", str(float(params.max_gain))]

    if sanitize:
        cmd.append("--sanitize")
    if verbose:
        cmd.append("--verbose")

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    _run(cmd, verbose=verbose, env=env)