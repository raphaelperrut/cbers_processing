from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional
import os
import subprocess


@dataclass(frozen=True)
class GdalEnv:
    """
    Ajustes de ambiente para subprocessos GDAL.

    gdal_cachemax_mb:
      Se definido, exporta GDAL_CACHEMAX em MB.

    cpl_debug:
      Se True/False, força CPL_DEBUG=ON/OFF. Se None, não altera.

    num_threads:
      Valor default para warps/translates que suportam NUM_THREADS.
    """
    gdal_cachemax_mb: Optional[int] = None
    cpl_debug: Optional[bool] = None
    num_threads: str = "ALL_CPUS"


def _run(cmd: Sequence[str], verbose: bool = False, env: Optional[dict] = None) -> None:
    if verbose:
        print("[GDAL] RUN:", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, env=env)


def make_env(base_env: Optional[dict] = None, cfg: Optional[GdalEnv] = None) -> dict:
    env = dict(os.environ if base_env is None else base_env)

    if cfg is not None:
        if cfg.gdal_cachemax_mb is not None:
            env["GDAL_CACHEMAX"] = str(int(cfg.gdal_cachemax_mb))
        if cfg.cpl_debug is not None:
            env["CPL_DEBUG"] = "ON" if bool(cfg.cpl_debug) else "OFF"

    return env


def _append_nodata_args(
    cmd: list[str],
    *,
    src_nodata: float | int | str | None,
    dst_nodata: float | int | str | None,
) -> None:
    if src_nodata is not None:
        cmd += ["-srcnodata", str(src_nodata)]
    if dst_nodata is not None:
        cmd += ["-dstnodata", str(dst_nodata)]


def _read_ref_grid(ref_path: Path) -> tuple[tuple[float, float, float, float], str, int, int]:
    """
    Lê bounds/CRS/dimensões de um raster de referência (TIF ou VRT).
    """
    import rasterio

    with rasterio.open(ref_path) as src:
        bounds = src.bounds
        crs_wkt = src.crs.to_wkt() if src.crs is not None else ""
        w, h = int(src.width), int(src.height)

    if not crs_wkt:
        raise RuntimeError(f"ref_path sem CRS: {ref_path}")

    return (bounds.left, bounds.bottom, bounds.right, bounds.top), crs_wkt, w, h


def gdalwarp_to_ref_vrt(
    in_path: Path,
    out_vrt: Path,
    ref_bounds: tuple[float, float, float, float],
    ref_crs_wkt: str,
    out_w: int,
    out_h: int,
    *,
    resample: str = "bilinear",
    overwrite: bool = True,
    verbose: bool = False,
    env: Optional[dict] = None,
    src_nodata: float | int | str | None = None,
    dst_nodata: float | int | str | None = None,
    num_threads: str | None = "ALL_CPUS",
) -> None:
    """
    Warp para um grid de referência, escrevendo VRT.

    ref_bounds: (xmin, ymin, xmax, ymax)
    ref_crs_wkt: WKT do CRS de destino
    out_w/out_h: dimensões exatas do grid destino
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
        "-r", str(resample),
    ]

    if num_threads:
        cmd += ["-multi", "-wo", f"NUM_THREADS={num_threads}"]

    _append_nodata_args(cmd, src_nodata=src_nodata, dst_nodata=dst_nodata)

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
    downsample_resample: str = "average",
    src_nodata: float | int | str | None = None,
    dst_nodata: float | int | str | None = None,
    num_threads: str | None = "ALL_CPUS",
) -> None:
    """
    Constrói um VRT 3 bandas (RGB) em baixa resolução alinhado ao PAN de referência.

    Saída final:
      band1 = R
      band2 = G
      band3 = B

    Estratégia:
      - usa o grid do PAN de referência
      - reduz para LR = PAN/scale
      - cria VRT separado em ordem RGB

    Nota:
      Para downsampling radiometricamente mais coerente, o default aqui é "average".
    """
    import rasterio

    if scale <= 0:
        raise ValueError("scale precisa ser > 0")

    with rasterio.open(pan_ref_tif) as src:
        bounds = src.bounds
        crs_wkt = src.crs.to_wkt() if src.crs is not None else ""
        pan_w, pan_h = int(src.width), int(src.height)

    if not crs_wkt:
        raise RuntimeError("PAN reference não tem CRS. Não dá para alinhar B/G/R com segurança.")

    lr_w = pan_w // scale
    lr_h = pan_h // scale
    if lr_w <= 0 or lr_h <= 0:
        raise RuntimeError(f"LR inválido: PAN {pan_w}x{pan_h}, scale={scale} -> {lr_w}x{lr_h}")

    if verbose:
        print(f"[GDAL] PAN grid: {pan_w}x{pan_h} | LR grid: {lr_w}x{lr_h} | scale={scale}")
        print("[GDAL] Band mapping for VRT: output RGB = (R,G,B) using inputs (red, green, blue)")
        print(f"[GDAL] Downsample resampling for LR guide: {downsample_resample}")

    ref_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    tmp_dir = out_rgb_vrt.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    b_vrt = tmp_dir / "_tmp_blue_lr.vrt"
    g_vrt = tmp_dir / "_tmp_green_lr.vrt"
    r_vrt = tmp_dir / "_tmp_red_lr.vrt"

    gdalwarp_to_ref_vrt(
        blue,
        b_vrt,
        ref_bounds,
        crs_wkt,
        lr_w,
        lr_h,
        resample=downsample_resample,
        overwrite=True,
        verbose=verbose,
        env=env,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        num_threads=num_threads,
    )
    gdalwarp_to_ref_vrt(
        green,
        g_vrt,
        ref_bounds,
        crs_wkt,
        lr_w,
        lr_h,
        resample=downsample_resample,
        overwrite=True,
        verbose=verbose,
        env=env,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        num_threads=num_threads,
    )
    gdalwarp_to_ref_vrt(
        red,
        r_vrt,
        ref_bounds,
        crs_wkt,
        lr_w,
        lr_h,
        resample=downsample_resample,
        overwrite=True,
        verbose=verbose,
        env=env,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        num_threads=num_threads,
    )

    cmd = ["gdalbuildvrt", "-separate", str(out_rgb_vrt), str(r_vrt), str(g_vrt), str(b_vrt)]
    _run(cmd, verbose=verbose, env=env)


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
    src_nodata: float | int | str | None = None,
    dst_nodata: float | int | str | None = None,
    creation_options: Optional[Sequence[str]] = None,
) -> None:
    """
    Warp para coincidir exatamente com um raster de referência (extent/CRS/dimensões),
    escrevendo saída em GeoTIFF.

    ref_path pode ser TIF ou VRT.
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
        "-r", str(resample),
    ]

    if num_threads:
        cmd += ["-multi", "-wo", f"NUM_THREADS={num_threads}"]

    if dtype:
        cmd += ["-ot", str(dtype)]

    _append_nodata_args(cmd, src_nodata=src_nodata, dst_nodata=dst_nodata)

    if creation_options:
        for co in creation_options:
            cmd += ["-co", str(co)]

    if overwrite:
        cmd.append("-overwrite")

    cmd += [str(in_path), str(out_tif)]
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
    Mantém o VRT final, se existir.
    """
    for name in ("_tmp_blue_lr.vrt", "_tmp_green_lr.vrt", "_tmp_red_lr.vrt"):
        safe_unlink(workdir / name, verbose=verbose)