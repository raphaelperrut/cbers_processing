import argparse
from pathlib import Path

from cbers_colorize.pipeline import run_pipeline


def _resolve_device(device: str) -> str:
    device = (device or "").lower()
    if device == "cpu":
        return "cpu"

    try:
        import torch  # noqa
    except Exception as e:
        if device in ("cuda", "auto"):
            raise RuntimeError(
                "Torch não está disponível/importável neste ambiente, "
                "então não é possível usar device=cuda/auto. "
                f"Erro: {repr(e)}"
            )
        return "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda solicitado, mas CUDA não está disponível neste ambiente.")
        return "cuda"

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    raise ValueError(f"device inválido: {device}")


def _resolve_amp(args) -> None:
    if args.device != "cuda":
        args.amp = False
        return

    if getattr(args, "no_amp", False):
        args.amp = False
    elif getattr(args, "amp", False):
        args.amp = True
    else:
        args.amp = True


def _resolve_blocks(args) -> None:
    legacy_block = getattr(args, "block", None)

    if getattr(args, "io_block", None) is None:
        args.io_block = legacy_block if legacy_block is not None else 1024
    if getattr(args, "out_block", None) is None:
        args.out_block = legacy_block if legacy_block is not None else 1024

    args.io_block = int(args.io_block)
    args.out_block = int(args.out_block)

    if args.io_block <= 0 or args.out_block <= 0:
        raise ValueError("--io_block e --out_block precisam ser > 0")


def _validate_shadow(args) -> None:
    if float(getattr(args, "shadow_strength", 0.0)) <= 0.0:
        return

    y_lo = float(getattr(args, "shadow_y_lo", 0.0))
    y_hi = float(getattr(args, "shadow_y_hi", 0.0))
    if not (0.0 <= y_lo <= 1.0 and 0.0 <= y_hi <= 1.0):
        raise ValueError("--shadow_y_lo e --shadow_y_hi devem estar em 0..1")

    if y_hi <= y_lo:
        raise ValueError("--shadow_y_hi precisa ser > --shadow_y_lo quando shadow_strength > 0")

    sc = float(getattr(args, "shadow_chroma", 1.0))
    if sc < 0.0:
        raise ValueError("--shadow_chroma precisa ser >= 0")

    st = float(getattr(args, "shadow_strength", 0.0))
    if not (0.0 <= st <= 1.0):
        raise ValueError("--shadow_strength deve estar em 0..1")


def _validate_highlights(args) -> None:
    hi_y = float(getattr(args, "hi_y", 0.0))
    hi_desat = float(getattr(args, "hi_desat", 0.0))
    gamut_gain = float(getattr(args, "gamut_gain", 0.0))
    damp_knee = float(getattr(args, "highlight_pan_damp_knee", 0.0))
    damp_strength = float(getattr(args, "highlight_pan_damp_strength", 0.0))

    if not (0.0 <= hi_y <= 1.0):
        raise ValueError("--hi_y deve estar em 0..1 (0 desliga)")
    if not (0.0 <= hi_desat <= 1.0):
        raise ValueError("--hi_desat deve estar em 0..1 (0 desliga)")
    if gamut_gain < 0.0:
        raise ValueError("--gamut_gain deve ser >= 0 (0 desliga)")
    if not (0.0 <= damp_knee <= 1.0):
        raise ValueError("--highlight_pan_damp_knee deve estar em 0..1")
    if not (0.0 <= damp_strength <= 1.0):
        raise ValueError("--highlight_pan_damp_strength deve estar em 0..1")


def _validate_luma(args) -> None:
    if not (0.0 <= float(args.luma_rolloff_knee) <= 1.0):
        raise ValueError("--luma_rolloff_knee deve estar em 0..1")
    if not (0.0 <= float(args.luma_rolloff_strength) <= 1.0):
        raise ValueError("--luma_rolloff_strength deve estar em 0..1")
    if float(args.luma_gamma) <= 0.0:
        raise ValueError("--luma_gamma deve ser > 0")


def _validate_detail(args) -> None:
    for k in (
        "detail_strength",
        "detail_strength_urban",
        "detail_strength_veg",
        "detail_strength_shadow",
        "detail_sigma1",
        "detail_sigma2",
        "detail_sigma3",
        "detail_alpha1",
        "detail_alpha2",
        "detail_alpha3",
        "nir_detail_boost",
        "urban_detail_from_nir",
        "pan_hp_sigma",
        "pan_hp_gain",
        "pan_hp_gain_urban",
        "pan_hp_gain_veg",
        "pan_hp_gain_shadow",
    ):
        v = float(getattr(args, k))
        if v < 0.0:
            raise ValueError(f"--{k} deve ser >= 0")

    if float(args.detail_sigma1) <= 0.0:
        raise ValueError("--detail_sigma1 deve ser > 0")
    if float(args.detail_sigma2) <= float(args.detail_sigma1):
        raise ValueError("--detail_sigma2 deve ser > --detail_sigma1")
    if float(args.detail_sigma3) <= float(args.detail_sigma2):
        raise ValueError("--detail_sigma3 deve ser > --detail_sigma2")
    if float(args.pan_hp_sigma) <= 0.0:
        raise ValueError("--pan_hp_sigma deve ser > 0")


def _validate_guided(args) -> None:
    if int(args.guided_radius) < 0:
        raise ValueError("--guided_radius deve ser >= 0")
    if float(args.guided_eps) <= 0.0:
        raise ValueError("--guided_eps deve ser > 0")
    if not (0.0 <= float(args.guided_chroma_mix) <= 1.0):
        raise ValueError("--guided_chroma_mix deve estar em 0..1")


def _validate_ndvi(args) -> None:
    if float(args.ndvi_veg_lo) < -1.0 or float(args.ndvi_veg_lo) > 1.0:
        raise ValueError("--ndvi_veg_lo deve estar em -1..1")
    if float(args.ndvi_veg_hi) < -1.0 or float(args.ndvi_veg_hi) > 1.0:
        raise ValueError("--ndvi_veg_hi deve estar em -1..1")
    if float(args.ndvi_veg_hi) <= float(args.ndvi_veg_lo):
        raise ValueError("--ndvi_veg_hi deve ser > --ndvi_veg_lo")


def _validate_radiometry(args) -> None:
    if not (0.0 <= float(args.rad_p_lo) <= 1.0):
        raise ValueError("--rad_p_lo deve estar em 0..1")
    if not (0.0 <= float(args.rad_p_hi) <= 1.0):
        raise ValueError("--rad_p_hi deve estar em 0..1")
    if not (0.0 <= float(args.rad_soft_margin) <= 1.0):
        raise ValueError("--rad_soft_margin deve estar em 0..1")
    if float(args.rad_p_hi) <= float(args.rad_p_lo):
        raise ValueError("--rad_p_hi deve ser > --rad_p_lo")


def _validate_general(args) -> None:
    if not (0.0 <= float(args.pan_p_lo) <= 1.0):
        raise ValueError("--pan_p_lo deve estar em 0..1")
    if not (0.0 <= float(args.pan_p_hi) <= 1.0):
        raise ValueError("--pan_p_hi deve estar em 0..1")
    if float(args.pan_p_hi) <= float(args.pan_p_lo):
        raise ValueError("--pan_p_hi deve ser > --pan_p_lo")

    if not (0.0 <= float(args.guide_p_lo) <= 1.0):
        raise ValueError("--guide_p_lo deve estar em 0..1")
    if not (0.0 <= float(args.guide_p_hi) <= 1.0):
        raise ValueError("--guide_p_hi deve estar em 0..1")
    if float(args.guide_p_hi) <= float(args.guide_p_lo):
        raise ValueError("--guide_p_hi deve ser > --guide_p_lo")

    if float(args.max_gain) <= 0.0:
        raise ValueError("--max_gain deve ser > 0")
    if float(args.chroma_blur_sigma) < 0.0:
        raise ValueError("--chroma_blur_sigma deve ser >= 0")
    if int(args.chroma_blur_ksize) <= 0:
        raise ValueError("--chroma_blur_ksize deve ser > 0")
    if float(args.sat) < 0.0:
        raise ValueError("--sat deve ser >= 0")
    if float(args.veg_sat) < 0.0:
        raise ValueError("--veg_sat deve ser >= 0")
    if float(args.veg_chroma) < 0.0:
        raise ValueError("--veg_chroma deve ser >= 0")
    if not (0.0 <= float(args.veg_exg_th) <= 1.0):
        raise ValueError("--veg_exg_th deve estar em 0..1")


def main():
    ap = argparse.ArgumentParser("cbers-colorize")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run upgraded CBERS PAN colorization / fusion pipeline")

    run.add_argument("--pan", required=True)
    run.add_argument("--blue", required=True)
    run.add_argument("--green", required=True)
    run.add_argument("--red", required=True)
    run.add_argument("--nir", required=True)

    run.add_argument("--outdir", required=True)
    run.add_argument("--workdir", default=None)

    run.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    run.add_argument("--tile", type=int, default=512)
    run.add_argument("--overlap", type=int, default=32)
    run.add_argument("--sanitize", action="store_true")
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--keep_tmp", action="store_true")

    run.add_argument("--block", type=int, default=1024)
    run.add_argument("--io_block", type=int, default=None)
    run.add_argument("--out_block", type=int, default=None)

    run.add_argument("--compress", default="ZSTD", choices=["LZW", "DEFLATE", "ZSTD", "NONE"])

    run.add_argument("--guide_norm", default="per_band", choices=["per_band", "joint_y"])

    run.add_argument("--pan_p_lo", type=float, default=0.02, help="Percentil baixo do PAN para normalização.")
    run.add_argument("--pan_p_hi", type=float, default=0.998, help="Percentil alto do PAN para normalização.")

    run.add_argument("--guide_p_lo", type=float, default=0.02, help="Percentil baixo do guide para normalização.")
    run.add_argument("--guide_p_hi", type=float, default=0.98, help="Percentil alto do guide para normalização.")

    run.add_argument("--amp", action="store_true")
    run.add_argument("--no_amp", action="store_true")

    run.add_argument("--fusion_mode", default="multiscale_ycbcr", choices=["multiscale_ycbcr", "ratio"])
    run.add_argument("--chroma_strength", type=float, default=0.72)
    run.add_argument("--sat", type=float, default=0.92)
    run.add_argument("--max_gain", type=float, default=2.5)
    run.add_argument("--chroma_blur_sigma", type=float, default=0.6)
    run.add_argument("--chroma_blur_ksize", type=int, default=5)

    run.add_argument("--veg_exg_th", type=float, default=0.10)
    run.add_argument("--veg_sat", type=float, default=0.82)
    run.add_argument("--veg_chroma", type=float, default=0.92)
    run.add_argument("--veg_cr_bias", type=float, default=-0.010)
    run.add_argument("--veg_cb_bias", type=float, default=0.003)

    run.add_argument("--cr_bias", type=float, default=-0.006)
    run.add_argument("--cb_bias", type=float, default=-0.003)

    run.add_argument("--neutral_mag_thr", type=float, default=0.050)
    run.add_argument("--neutral_strength", type=float, default=0.70)
    run.add_argument("--neutral_cb_bias", type=float, default=0.0)
    run.add_argument("--neutral_cr_bias", type=float, default=0.0)

    run.add_argument("--shadow_y_lo", type=float, default=0.08)
    run.add_argument("--shadow_y_hi", type=float, default=0.28)
    run.add_argument("--shadow_strength", type=float, default=0.20)
    run.add_argument("--shadow_cb_bias", type=float, default=-0.004)
    run.add_argument("--shadow_cr_bias", type=float, default=0.003)
    run.add_argument("--shadow_chroma", type=float, default=0.95)

    run.add_argument("--hi_y", type=float, default=0.86)
    run.add_argument("--hi_desat", type=float, default=0.45)
    run.add_argument("--gamut_gain", type=float, default=4.0)

    run.add_argument("--luma_rolloff_knee", type=float, default=0.88)
    run.add_argument("--luma_rolloff_strength", type=float, default=0.32)
    run.add_argument("--luma_gamma", type=float, default=0.98)

    run.add_argument("--detail_strength", type=float, default=0.72)
    run.add_argument("--detail_strength_urban", type=float, default=1.15)
    run.add_argument("--detail_strength_veg", type=float, default=0.55)
    run.add_argument("--detail_strength_shadow", type=float, default=0.85)
    run.add_argument("--detail_sigma1", type=float, default=1.2)
    run.add_argument("--detail_sigma2", type=float, default=2.8)
    run.add_argument("--detail_sigma3", type=float, default=5.6)
    run.add_argument("--detail_alpha1", type=float, default=0.62)
    run.add_argument("--detail_alpha2", type=float, default=0.28)
    run.add_argument("--detail_alpha3", type=float, default=0.14)

    run.add_argument("--pan_hp_sigma", type=float, default=2.2)
    run.add_argument("--pan_hp_gain", type=float, default=0.16)
    run.add_argument("--pan_hp_gain_urban", type=float, default=1.35)
    run.add_argument("--pan_hp_gain_veg", type=float, default=0.50)
    run.add_argument("--pan_hp_gain_shadow", type=float, default=0.75)
    run.add_argument("--highlight_pan_damp_knee", type=float, default=0.82)
    run.add_argument("--highlight_pan_damp_strength", type=float, default=0.75)

    run.add_argument("--guided_radius", type=int, default=4)
    run.add_argument("--guided_eps", type=float, default=1e-4)
    run.add_argument("--guided_chroma_mix", type=float, default=0.65)

    run.add_argument("--ndvi_veg_lo", type=float, default=0.18)
    run.add_argument("--ndvi_veg_hi", type=float, default=0.38)
    run.add_argument("--nir_detail_boost", type=float, default=0.20)
    run.add_argument("--urban_detail_from_nir", type=float, default=0.12)

    run.add_argument("--rad_p_lo", type=float, default=0.003)
    run.add_argument("--rad_p_hi", type=float, default=0.997)
    run.add_argument("--rad_soft_margin", type=float, default=0.020)

    run.add_argument("--export_vis", action="store_true", help="Gera derivado leve legado GeoTIFF/COG.")
    run.add_argument("--vis_format", default="GTIFF", choices=["GTIFF", "COG"])
    run.add_argument("--vis_compress", default="JPEG", choices=["JPEG", "DEFLATE", "LZW"])
    run.add_argument("--vis_p_lo", type=float, default=0.01)
    run.add_argument("--vis_p_hi", type=float, default=0.995)
    run.add_argument("--vis_gamma", type=float, default=1.00)
    run.add_argument("--vis_overviews", action="store_true")

    run.add_argument("--export_cog", action="store_true", help="Gera um COG leve em JPEG a partir do master.")
    run.add_argument("--cog_quality", type=int, default=90)
    run.add_argument("--cog_blocksize", type=int, default=512)
    run.add_argument("--cog_overviews", action="store_true", help="Cria overviews no derivado leve antes do COG.")

    run.add_argument("--export_ecw", action="store_true", help="Gera pan_2m_color_guided.ecw como derivado opcional.")
    run.add_argument("--ecw_target_mb", type=int, default=0, help="Meta opcional de tamanho final em MB.")

    args = ap.parse_args()

    if args.cmd == "run":
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        workdir = Path(args.workdir) if args.workdir else (outdir / "_work")
        workdir.mkdir(parents=True, exist_ok=True)

        args.device = _resolve_device(args.device)
        _resolve_amp(args)
        _resolve_blocks(args)
        _validate_general(args)
        _validate_shadow(args)
        _validate_highlights(args)
        _validate_luma(args)
        _validate_detail(args)
        _validate_guided(args)
        _validate_ndvi(args)
        _validate_radiometry(args)

        args.compress = (args.compress or "ZSTD").upper()
        args.vis_format = (args.vis_format or "GTIFF").upper()
        args.vis_compress = (args.vis_compress or "JPEG").upper()

        if args.verbose:
            print("[CLI] cmd=run")
            print("[CLI] device:", args.device)
            print("[CLI] outdir :", outdir)
            print("[CLI] workdir:", workdir)
            print("[CLI] tile/overlap:", args.tile, args.overlap)
            print("[CLI] io_block:", args.io_block, "| out_block:", args.out_block)
            print("[CLI] compress:", args.compress)
            print("[CLI] amp:", args.amp)
            print("[CLI] fusion_mode:", args.fusion_mode)
            print("[CLI] pan_p_lo/hi:", args.pan_p_lo, args.pan_p_hi)
            print("[CLI] guide_p_lo/hi:", args.guide_p_lo, args.guide_p_hi)
            print("[CLI] detail:", args.detail_strength, args.detail_alpha1, args.detail_alpha2, args.detail_alpha3)
            print("[CLI] pan_hp:", args.pan_hp_sigma, args.pan_hp_gain, args.pan_hp_gain_urban, args.pan_hp_gain_veg, args.pan_hp_gain_shadow)
            print("[CLI] pan_hi_damp:", args.highlight_pan_damp_knee, args.highlight_pan_damp_strength)
            print("[CLI] guided:", args.guided_radius, args.guided_eps, args.guided_chroma_mix)
            print("[CLI] ndvi:", args.ndvi_veg_lo, args.ndvi_veg_hi, args.nir_detail_boost, args.urban_detail_from_nir)
            print("[CLI] radiometry:", args.rad_p_lo, args.rad_p_hi, args.rad_soft_margin)
            print("[CLI] export_vis:", args.export_vis, "| vis_format:", args.vis_format, "| vis_compress:", args.vis_compress)
            print("[CLI] export_cog:", args.export_cog, "| cog_quality:", args.cog_quality, "| cog_blocksize:", args.cog_blocksize)
            print("[CLI] export_ecw:", args.export_ecw, "| ecw_target_mb:", args.ecw_target_mb)

        run_pipeline(args, workdir=workdir, outdir=outdir)
        return

    raise ValueError(f"Subcomando não suportado: {args.cmd}")


if __name__ == "__main__":
    main()