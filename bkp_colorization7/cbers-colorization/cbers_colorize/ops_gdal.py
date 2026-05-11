# cbers_colorize/ops_gdal.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional
import os
import subprocess


@dataclass(frozen=True)
class GdalEnv:
    """
    Ajustes de ambiente para subprocessos GDAL dentro do container.

    - Em Docker slim, geralmente já funciona sem nada.
    - Se você quiser setar GDAL_CACHEMAX, CPL_DEBUG, etc, faça aqui.
    """
    gdal_cachemax_mb: Optional[int] = None


def _run(cmd: Sequence[str], verbose: bool = False, env: Optional[dict] = None) -> None:
    if verbose:
        print("[GDAL] RUN:", " ".join(cmd))
    subprocess.run(list(cmd), check=True, env=env)


def make_env(base_env: Optional[dict] = None, cfg: Optional[GdalEnv] = None) -> dict:
    env = dict(os.environ if base_env is None else base_env)
    if cfg and cfg.gdal_cachemax_mb is not None:
        # GDAL_CACHEMAX is in MB
        env["GDAL_CACHEMAX"] = str(int(cfg.gdal_cachemax_mb))
    return env


def gdalwarp_to_ref_vrt(
    in_path: Path,
    out_vrt: Path,
    ref_bounds: tuple[float, float, float, float],
    ref_crs_wkt: str,
    out_w: int,
    out_h: int,
    resample: str = "bilinear",
    overwrite: bool = True,
    verbose: bool = False,
    env: Optional[dict] = None,
) -> None:
    """
    Warp de um raster para coincidir com um "grid de referência" (extent/CRS e tamanho exato),
    escrevendo saída em VRT (leve e rápido).

    ref_bounds: (xmin, ymin, xmax, ymax) no CRS do ref_crs_wkt
    out_w/out_h: dimensões finais do VRT
    """
    if not ref_crs_wkt:
        raise RuntimeError("ref_crs_wkt vazio: impossível alinhar via gdalwarp.")

    xmin, ymin, xmax, ymax = ref_bounds
    out_vrt.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gdalwarp",
        "-of", "VRT",
        "-t_srs", ref_crs_wkt,
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-ts", str(int(out_w)), str(int(out_h)),
        "-r", resample,
    ]
    if overwrite:
        cmd.append("-overwrite")

    cmd += [str(in_path), str(out_vrt)]
    _run(cmd, verbose=verbose, env=env)


def build_rgb_lr_vrt_aligned_to_pan(
    *,
    blue: Path,
    green: Path,
    red: Path,
    pan_ref_tif: Path,
    out_rgb_vrt: Path,
    scale: int,
    verbose: bool = False,
    env: Optional[dict] = None,
) -> None:
    """
    Constrói um VRT 3 bandas (RGB) em baixa resolução (LR) alinhado ao PAN de referência.

    Importante (CBERS):
      - você comentou que band1, band2, band3 correspondem a B,G,R.
      - aqui o VRT final sempre sai na ordem RGB = (R,G,B),
        então chamamos gdalbuildvrt com (red, green, blue).

    pan_ref_tif:
      - idealmente já recortado para dimensões múltiplas de scale (ex.: 4),
        para que LR = PAN/scale seja inteiro.
    """
    import rasterio  # import local pra evitar custo quando não usado

    if scale <= 0:
        raise ValueError("scale precisa ser > 0")

    with rasterio.open(pan_ref_tif) as src:
        bounds = src.bounds
        crs_wkt = src.crs.to_wkt() if src.crs is not None else ""
        pan_w, pan_h = src.width, src.height

    if not crs_wkt:
        raise RuntimeError("PAN reference não tem CRS. Não dá para alinhar B/G/R com segurança.")

    lr_w = pan_w // scale
    lr_h = pan_h // scale
    if lr_w <= 0 or lr_h <= 0:
        raise RuntimeError(f"LR inválido: PAN {pan_w}x{pan_h}, scale={scale} -> {lr_w}x{lr_h}")

    if verbose:
        print(f"[GDAL] PAN grid: {pan_w}x{pan_h} | LR grid: {lr_w}x{lr_h} | scale={scale}")
        print("[GDAL] Band mapping for VRT: output RGB = (R,G,B) using inputs (red, green, blue)")

    ref_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    tmp_dir = out_rgb_vrt.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    b_vrt = tmp_dir / "_tmp_blue_lr.vrt"
    g_vrt = tmp_dir / "_tmp_green_lr.vrt"
    r_vrt = tmp_dir / "_tmp_red_lr.vrt"

    gdalwarp_to_ref_vrt(
        blue, b_vrt, ref_bounds, crs_wkt, lr_w, lr_h,
        resample="bilinear", overwrite=True, verbose=verbose, env=env
    )
    gdalwarp_to_ref_vrt(
        green, g_vrt, ref_bounds, crs_wkt, lr_w, lr_h,
        resample="bilinear", overwrite=True, verbose=verbose, env=env
    )
    gdalwarp_to_ref_vrt(
        red, r_vrt, ref_bounds, crs_wkt, lr_w, lr_h,
        resample="bilinear", overwrite=True, verbose=verbose, env=env
    )

    # Ordem RGB = R,G,B
    cmd = ["gdalbuildvrt", "-separate", str(out_rgb_vrt), str(r_vrt), str(g_vrt), str(b_vrt)]
    _run(cmd, verbose=verbose, env=env)


def safe_unlink(path: Path, verbose: bool = False) -> None:
    try:
        path.unlink()
        if verbose:
            print("[GDAL] RM:", path)
    except FileNotFoundError:
        return


def cleanup_rgb_lr_tmp_files(workdir: Path, verbose: bool = False) -> None:
    """
    Remove os VRTs temporários criados por build_rgb_lr_vrt_aligned_to_pan.
    (mantém o VRT final, se existir)
    """
    for name in ("_tmp_blue_lr.vrt", "_tmp_green_lr.vrt", "_tmp_red_lr.vrt"):
        safe_unlink(workdir / name, verbose=verbose)

def _read_ref_grid(ref_path: Path) -> tuple[tuple[float, float, float, float], str, int, int]:
    """Lê bounds/CRS/dims de um raster (inclui VRT)."""
    import rasterio  # import local para evitar custo quando não usado
    with rasterio.open(ref_path) as src:
        bounds = src.bounds
        crs_wkt = src.crs.to_wkt() if src.crs is not None else ""
        w, h = src.width, src.height
    if not crs_wkt:
        raise RuntimeError(f"ref_path sem CRS: {ref_path}")
    return (bounds.left, bounds.bottom, bounds.right, bounds.top), crs_wkt, int(w), int(h)


def gdalwarp_to_ref_gtiff(
    in_path: Path,
    out_tif: Path,
    ref_path: Path,
    *,
    resample: str = "cubic",
    dtype: Optional[str] = None,
    overwrite: bool = True,
    verbose: bool = False,
    env: Optional[dict] = None,
    num_threads: str | None = "ALL_CPUS",
) -> None:
    """
    Warp de um raster para coincidir exatamente com um raster de referência (extent/CRS e tamanho),
    escrevendo saída em GeoTIFF.

    - ref_path pode ser TIF ou VRT.
    - dtype: opcional (ex.: "Float32", "UInt16"). Se None, GDAL decide.
    - num_threads: usa -multi + -wo NUM_THREADS=... para acelerar.
    """
    ref_bounds, ref_wkt, out_w, out_h = _read_ref_grid(ref_path)

    xmin, ymin, xmax, ymax = ref_bounds
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gdalwarp",
        "-of", "GTiff",
        "-t_srs", ref_wkt,
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-ts", str(int(out_w)), str(int(out_h)),
        "-r", resample,
    ]

    if num_threads:
        cmd += ["-multi", "-wo", f"NUM_THREADS={num_threads}"]

    if dtype:
        cmd += ["-ot", str(dtype)]

    if overwrite:
        cmd.append("-overwrite")

    cmd += [str(in_path), str(out_tif)]
    _run(cmd, verbose=verbose, env=env)
