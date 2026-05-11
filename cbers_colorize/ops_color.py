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
    Stretch visual para produtos RGB em 0..1 ou float arbitrário.

    Uso recomendado:
      - apenas para derivados visuais
      - não para o master de fusão
    """
    p_lo: float = 0.02
    p_hi: float = 0.98
    sample_stride: int = 16
    gamma: float = 1.0


def _run(cmd: Sequence[str], verbose: bool = False, env: Optional[dict] = None) -> None:
    if verbose:
        print("[COLOR] RUN:", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, env=env)


def make_env_with_pythonpath(base_env: Optional[dict] = None) -> dict:
    """
    Garante import de /app/cbers_colorize/tools e /app/cbers_colorize/*
    dentro do container/ambiente de execução.
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
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def write_guide_byte_cut(
    rgb_path: Path,
    out_path: Path,
    stretch: GuideStretch,
    verbose: bool = False,
) -> None:
    """
    Converte um RGB qualquer para uint8 com stretch por banda.

    Atenção:
      isso é um utilitário de VISUALIZAÇÃO.
      Não use esse produto como guide master da fusão nova.
    """
    with rasterio.open(rgb_path) as src:
        rgb = src.read().astype(np.float32, copy=False)
        profile = src.profile.copy()

    if rgb.shape[0] != 3:
        raise ValueError(f"Esperado 3 bandas em {rgb_path}, veio {rgb.shape[0]}")

    out = np.zeros_like(rgb, dtype=np.uint8)

    for c in range(3):
        x = rgb[c]
        samp = x[::stretch.sample_stride, ::stretch.sample_stride].ravel()
        lo, hi = _percentiles_from_sample(samp, stretch.p_lo, stretch.p_hi)

        if verbose:
            print(
                f"[COLOR] GUIDE band{c+1} stretch "
                f"p_lo={stretch.p_lo:.4f}->{lo:.6f} "
                f"p_hi={stretch.p_hi:.4f}->{hi:.6f} "
                f"gamma={stretch.gamma:.3f}"
            )

        y = (x - lo) / (hi - lo + 1e-6)
        y = np.clip(y, 0.0, 1.0)
        if float(stretch.gamma) != 1.0:
            y = np.power(y, float(stretch.gamma), dtype=np.float32)
        out[c] = np.clip(np.round(y * 255.0), 0.0, 255.0).astype(np.uint8)

    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=3,
        nodata=None,
        interleave="pixel",
    )
    profile.pop("nodata", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out)
        try:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        except Exception:
            pass


@dataclass
class ColorizeParams:
    """
    Wrapper opcional para chamar o infer_geotiff_color.py novo fora do pipeline principal.

    Este dataclass agora reflete a arquitetura nova:
      - PAN 1 banda
      - guide RGB 0..1 ou compatível
      - guide NIR opcional
      - PAN sintética opcional
      - acabamento radiométrico configurável
      - residual-based fusion
    """
    norm: str = "0_1"
    out_range: str = "0_1"

    fusion_mode: str = "multiscale_ycbcr"
    chroma_strength: float = 0.72
    sat: float = 0.92
    max_gain: float = 2.5

    chroma_blur_sigma: float = 0.6
    chroma_blur_ksize: int = 5

    veg_exg_th: float = 0.10
    veg_sat: float = 0.82
    veg_chroma: float = 0.92

    cr_bias: float = -0.006
    cb_bias: float = -0.003
    veg_cr_bias: float = -0.010
    veg_cb_bias: float = 0.003

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

    use_residual: bool = True
    pan_syn_base_mix: float = 0.15
    residual_guided_radius: int = 8
    residual_guided_eps: float = 1e-4
    residual_guided_mix: float = 0.55

    radiometric_finish_mode: str = "y_only"
    rad_p_lo: float = 0.003
    rad_p_hi: float = 0.997
    rad_soft_margin: float = 0.020

    compress: str = "ZSTD"
    io_block: int = 1024
    out_block: int = 1024


def run_colorize_tool(
    *,
    color_tool: Path,
    pan_1band_tif: Path,
    guide_rgb_tif: Path,
    out_tif: Path,
    device: str,
    tile: int,
    overlap: int,
    sanitize: bool,
    verbose: bool,
    params: ColorizeParams,
    guide_nir_tif: Path | None = None,
    pan_syn_tif: Path | None = None,
    amp: bool | None = None,
) -> None:
    """
    Chama o infer_geotiff_color.py NOVO, já alinhado com a arquitetura atual.

    Esse wrapper é opcional. O pipeline principal não depende dele.
    """
    env = make_env_with_pythonpath()

    cmd = [
        "python",
        str(color_tool),
        "--in_pan_tif",
        str(pan_1band_tif),
        "--out_tif",
        str(out_tif),
        "--device",
        str(device),
        "--tile",
        str(int(tile)),
        "--overlap",
        str(int(overlap)),
        "--norm",
        str(params.norm),
        "--out_range",
        str(params.out_range),
        "--guide_rgb",
        str(guide_rgb_tif),
        "--fusion_mode",
        str(params.fusion_mode),
        "--chroma_strength",
        str(float(params.chroma_strength)),
        "--sat",
        str(float(params.sat)),
        "--max_gain",
        str(float(params.max_gain)),
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
        "--cr_bias",
        str(float(params.cr_bias)),
        "--cb_bias",
        str(float(params.cb_bias)),
        "--veg_cr_bias",
        str(float(params.veg_cr_bias)),
        "--veg_cb_bias",
        str(float(params.veg_cb_bias)),
        "--neutral_mag_thr",
        str(float(params.neutral_mag_thr)),
        "--neutral_strength",
        str(float(params.neutral_strength)),
        "--neutral_cb_bias",
        str(float(params.neutral_cb_bias)),
        "--neutral_cr_bias",
        str(float(params.neutral_cr_bias)),
        "--shadow_y_lo",
        str(float(params.shadow_y_lo)),
        "--shadow_y_hi",
        str(float(params.shadow_y_hi)),
        "--shadow_strength",
        str(float(params.shadow_strength)),
        "--shadow_cb_bias",
        str(float(params.shadow_cb_bias)),
        "--shadow_cr_bias",
        str(float(params.shadow_cr_bias)),
        "--shadow_chroma",
        str(float(params.shadow_chroma)),
        "--hi_y",
        str(float(params.hi_y)),
        "--hi_desat",
        str(float(params.hi_desat)),
        "--gamut_gain",
        str(float(params.gamut_gain)),
        "--luma_rolloff_knee",
        str(float(params.luma_rolloff_knee)),
        "--luma_rolloff_strength",
        str(float(params.luma_rolloff_strength)),
        "--luma_gamma",
        str(float(params.luma_gamma)),
        "--detail_strength",
        str(float(params.detail_strength)),
        "--detail_strength_urban",
        str(float(params.detail_strength_urban)),
        "--detail_strength_veg",
        str(float(params.detail_strength_veg)),
        "--detail_strength_shadow",
        str(float(params.detail_strength_shadow)),
        "--detail_sigma1",
        str(float(params.detail_sigma1)),
        "--detail_sigma2",
        str(float(params.detail_sigma2)),
        "--detail_sigma3",
        str(float(params.detail_sigma3)),
        "--detail_alpha1",
        str(float(params.detail_alpha1)),
        "--detail_alpha2",
        str(float(params.detail_alpha2)),
        "--detail_alpha3",
        str(float(params.detail_alpha3)),
        "--pan_hp_sigma",
        str(float(params.pan_hp_sigma)),
        "--pan_hp_gain",
        str(float(params.pan_hp_gain)),
        "--pan_hp_gain_urban",
        str(float(params.pan_hp_gain_urban)),
        "--pan_hp_gain_veg",
        str(float(params.pan_hp_gain_veg)),
        "--pan_hp_gain_shadow",
        str(float(params.pan_hp_gain_shadow)),
        "--highlight_pan_damp_knee",
        str(float(params.highlight_pan_damp_knee)),
        "--highlight_pan_damp_strength",
        str(float(params.highlight_pan_damp_strength)),
        "--guided_radius",
        str(int(params.guided_radius)),
        "--guided_eps",
        str(float(params.guided_eps)),
        "--guided_chroma_mix",
        str(float(params.guided_chroma_mix)),
        "--ndvi_veg_lo",
        str(float(params.ndvi_veg_lo)),
        "--ndvi_veg_hi",
        str(float(params.ndvi_veg_hi)),
        "--nir_detail_boost",
        str(float(params.nir_detail_boost)),
        "--urban_detail_from_nir",
        str(float(params.urban_detail_from_nir)),
        *(["--use_residual"] if params.use_residual else ["--no_use_residual"]),
        "--pan_syn_base_mix",
        str(float(params.pan_syn_base_mix)),
        "--residual_guided_radius",
        str(int(params.residual_guided_radius)),
        "--residual_guided_eps",
        str(float(params.residual_guided_eps)),
        "--residual_guided_mix",
        str(float(params.residual_guided_mix)),
        "--radiometric_finish_mode",
        str(params.radiometric_finish_mode),
        "--rad_p_lo",
        str(float(params.rad_p_lo)),
        "--rad_p_hi",
        str(float(params.rad_p_hi)),
        "--rad_soft_margin",
        str(float(params.rad_soft_margin)),
        "--io_block",
        str(int(params.io_block)),
        "--out_block",
        str(int(params.out_block)),
        "--compress",
        str(params.compress).upper(),
    ]

    if guide_nir_tif is not None:
        cmd += ["--guide_nir", str(guide_nir_tif)]

    if pan_syn_tif is not None:
        cmd += ["--pan_syn_tif", str(pan_syn_tif)]

    if sanitize:
        cmd.append("--sanitize")
    if verbose:
        cmd.append("--verbose")

    if amp is not None and str(device).lower() == "cuda":
        cmd.append("--amp" if bool(amp) else "--no_amp")

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    _run(cmd, verbose=verbose, env=env)