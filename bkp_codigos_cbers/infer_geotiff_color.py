import argparse
import numpy as np
import torch
import rasterio
from rasterio.enums import ColorInterp
from model import net

SENTINEL_TH = -1e20  # tudo menor que isso vira 0
EPS = 1e-6


# ----------------------------
# Utils
# ----------------------------
def load_model(pkl_path: str, device: str):
    model = net.Kong(scale=1).to(device)
    ckpt = torch.load(pkl_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


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

    def ramp(n):
        t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
        return np.sin(t)  # 0..1

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


# ----------------------------
# Norm / Output
# ----------------------------
def apply_norm(img: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    m = float(np.nanmax(img))
    chosen = mode

    out = img.astype(np.float32)
    if mode == "auto":
        chosen = "0_255" if m > 1.5 else "0_1"

    if chosen == "none":
        return out, chosen

    if chosen == "0_255":
        out = out / 255.0
        out = np.clip(out, 0.0, 1.0)
        return out, chosen

    if chosen == "0_1":
        out = np.clip(out, 0.0, 1.0)
        return out, chosen

    if chosen == "minus1_1":
        out = np.clip(out, 0.0, 1.0)
        out = out * 2.0 - 1.0
        return out, chosen

    raise ValueError(f"Modo de normalização desconhecido: {mode}")


def apply_out_act(pred_t: torch.Tensor, out_act: str, out_temp: float) -> torch.Tensor:
    if out_act == "none":
        return pred_t
    if out_act == "sigmoid":
        return torch.sigmoid(pred_t / out_temp)
    if out_act == "tanh01":
        y = torch.tanh(pred_t / out_temp)
        return (y + 1.0) * 0.5
    raise ValueError(f"out_act desconhecido: {out_act}")


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
# Color space helpers (YCbCr)
# ----------------------------
def rgb_to_ycbcr(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb01[0], rgb01[1], rgb01[2]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb2 = cb - 0.5
    cr2 = cr - 0.5
    r = y + 1.402 * cr2
    g = y - 0.344136 * cb2 - 0.714136 * cr2
    b = y + 1.772 * cb2
    out = np.stack([r, g, b], axis=0).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def apply_sat_ycbcr(cb: np.ndarray, cr: np.ndarray, sat: float) -> tuple[np.ndarray, np.ndarray]:
    cb2 = (cb - 0.5) * sat + 0.5
    cr2 = (cr - 0.5) * sat + 0.5
    return np.clip(cb2, 0.0, 1.0), np.clip(cr2, 0.0, 1.0)


# ----------------------------
# Simple separable Gaussian blur (no OpenCV dependency)
# ----------------------------
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
    """
    img: (H,W) float32
    blur separável com padding reflect
    """
    if sigma <= 0:
        return img.astype(np.float32)

    k = _gauss_kernel1d(sigma, ksize)
    r = len(k) // 2

    a = img.astype(np.float32)

    # horizontal
    a_pad = np.pad(a, ((0, 0), (r, r)), mode="reflect")
    tmp = np.empty_like(a, dtype=np.float32)
    for j in range(a.shape[1]):
        tmp[:, j] = (a_pad[:, j:j + 2 * r + 1] * k[None, :]).sum(axis=1)

    # vertical
    t_pad = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.empty_like(a, dtype=np.float32)
    for i in range(a.shape[0]):
        out[i, :] = (t_pad[i:i + 2 * r + 1, :] * k[:, None]).sum(axis=0)

    return out.astype(np.float32)


# ----------------------------
# Updated color transfer using PAN luminance (SAFE)
# ----------------------------
def color_transfer_pan_rgb(
    pan01: np.ndarray,
    guide01: np.ndarray,
    mode: str,
    chroma_strength: float,
    sat: float,
    max_gain: float,
    chroma_blur_sigma: float,
    chroma_blur_ksize: int,
    veg_exg_th: float,
    veg_sat: float,
    veg_chroma: float,
) -> np.ndarray:
    """
    pan01: (H,W) 0..1
    guide01: (3,H,W) 0..1

    mode:
      - ycbcr (recomendado): Y = PAN, Cb/Cr = guia (controlado) + blur opcional no chroma
      - ratio (legado): out = guide * clip(pan/I(guide), 0..max_gain)  (pode imprimir textura do guia!)

    controles:
      chroma_strength: 0..1
      sat: saturação global do chroma (<=1 costuma ajudar)
      chroma_blur_*: blur só em Cb/Cr (mata pontilhado do SR sem borrar PAN)
      veg_*: reduz sat/força em vegetação (ExG), útil para “verde neon”
    """
    if mode not in ("ycbcr", "ratio"):
        raise ValueError(f"fusion_mode inválido: {mode}")

    g = np.clip(guide01, 0.0, 1.0).astype(np.float32)
    pan = np.clip(pan01, 0.0, 1.0).astype(np.float32)

    chroma_strength = float(np.clip(chroma_strength, 0.0, 1.0))
    sat = float(max(sat, 0.0))
    max_gain = float(max(max_gain, 1e-3))

    if mode == "ratio":
        I = (g[0] + g[1] + g[2]) / 3.0
        ratio = pan / (I + EPS)
        ratio = np.clip(ratio, 0.0, max_gain)
        out = g * ratio[None, :, :]
        return np.clip(out, 0.0, 1.0)

    # --- ycbcr (SAFE)
    _, cb, cr = rgb_to_ycbcr(g)

    # saturação global
    cb, cr = apply_sat_ycbcr(cb, cr, sat)

    # blur apenas no chroma (remove checker/pontilhado do SR)
    if chroma_blur_sigma > 0:
        cb = gaussian_blur2d(cb, chroma_blur_sigma, chroma_blur_ksize)
        cr = gaussian_blur2d(cr, chroma_blur_sigma, chroma_blur_ksize)
        cb = np.clip(cb, 0.0, 1.0)
        cr = np.clip(cr, 0.0, 1.0)

    # vegetação: reduz sat e força do chroma onde ExG alto
    if veg_exg_th > 0:
        exg = 2.0 * g[1] - g[0] - g[2]  # (H,W)
        t = (exg - veg_exg_th) / (0.5 + EPS)  # escala heurística
        t = np.clip(t, 0.0, 1.0).astype(np.float32)

        # sat local
        sat_local = (1.0 - t) * 1.0 + t * float(np.clip(veg_sat, 0.0, 2.0))
        cb = (cb - 0.5) * sat_local + 0.5
        cr = (cr - 0.5) * sat_local + 0.5

        # força local do chroma
        veg_chroma = float(np.clip(veg_chroma, 0.0, 1.0))
        strength_map = (1.0 - t) * chroma_strength + t * (chroma_strength * veg_chroma)
    else:
        strength_map = chroma_strength

    # mistura com neutro (cinza) via strength_map
    cb_f = 0.5 + (cb - 0.5) * strength_map
    cr_f = 0.5 + (cr - 0.5) * strength_map

    out = ycbcr_to_rgb(pan, cb_f, cr_f)
    return np.clip(out, 0.0, 1.0)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_pan_tif", default=None, help="GeoTIFF PAN 1 banda (2m real).")
    ap.add_argument("--in_tif", default=None, help="GeoTIFF 3 bandas (se já tiver PAN replicada).")

    ap.add_argument("--pkl", required=True, help="Checkpoint do modelo de colorização (color.pkl / colorx2.pkl).")
    ap.add_argument("--out_tif", required=True, help="Saída RGB (float32).")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)

    ap.add_argument("--norm", default="auto",
                    choices=["auto", "none", "0_1", "0_255", "minus1_1"],
                    help="Normalização aplicada na entrada (PAN replicada).")

    ap.add_argument("--out_act", default="sigmoid", choices=["none", "sigmoid", "tanh01"],
                    help="Ativação na saída do modelo.")
    ap.add_argument("--out_temp", type=float, default=1.0)
    ap.add_argument("--out_range", default="0_1", choices=["0_1", "minus1_1", "none"])

    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # Guia
    ap.add_argument("--guide_rgb", default=None, help="GeoTIFF RGB guia alinhado na PAN.")
    ap.add_argument("--guide_mode", default="none", choices=["none", "pan_luminance_injection"],
                    help="Como usar o guide_rgb. (pan_luminance_injection aplica a fusão)")

    # Novos controles da fusão (aplicados dentro de color_transfer_pan_rgb)
    ap.add_argument("--fusion_mode", default="ycbcr", choices=["ycbcr", "ratio"],
                    help="Modo de fusão PAN+guia: ycbcr (recomendado) ou ratio (legado).")

    ap.add_argument("--chroma_strength", type=float, default=0.85, help="0..1 força da crominância do guia.")
    ap.add_argument("--sat", type=float, default=0.95, help="Saturação global do chroma do guia.")
    ap.add_argument("--max_gain", type=float, default=3.0,
                    help="Só no modo ratio: ganho máximo de pan/I.")

    ap.add_argument("--chroma_blur_sigma", type=float, default=0.0,
                    help="Sigma do blur aplicado SOMENTE no chroma (Cb/Cr) do guia. 0 desliga.")
    ap.add_argument("--chroma_blur_ksize", type=int, default=7,
                    help="Kernel size (ímpar) do blur do chroma.")

    # Hack vegetação
    ap.add_argument("--veg_exg_th", type=float, default=0.12,
                    help="Threshold do ExG (2G-R-B). <=0 desliga.")
    ap.add_argument("--veg_sat", type=float, default=0.65,
                    help="Saturação em áreas de vegetação (0..1 costuma ajudar).")
    ap.add_argument("--veg_chroma", type=float, default=0.75,
                    help="Multiplica chroma_strength em vegetação (0..1).")

    args = ap.parse_args()

    if (args.in_pan_tif is None) == (args.in_tif is None):
        raise ValueError("Use exatamente UM: --in_pan_tif (1 banda) OU --in_tif (3 bandas).")

    TILE = int(args.tile)
    OL = int(args.overlap)
    STRIDE = TILE - OL
    if STRIDE <= 0:
        raise ValueError("overlap grande demais: tile - overlap precisa ser > 0")

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False

    device = args.device
    model = load_model(args.pkl, device)

    # ---- read input
    if args.in_pan_tif:
        with rasterio.open(args.in_pan_tif) as src:
            pan = src.read(1).astype(np.float32)  # (H,W)
            profile = src.profile
        if args.sanitize:
            pan, _ = sanitize_array(pan, name="PAN(global)", verbose=args.verbose)
        pan = np.where(pan < SENTINEL_TH, 0.0, pan).astype(np.float32)
        img = np.stack([pan, pan, pan], axis=0)  # (3,H,W)
    else:
        with rasterio.open(args.in_tif) as src:
            img = src.read().astype(np.float32)  # (3,H,W)
            profile = src.profile
        if img.shape[0] != 3:
            raise ValueError(f"--in_tif precisa ter 3 bandas; veio {img.shape[0]}.")
        if args.sanitize:
            img, _ = sanitize_array(img, name="INPUT(global)", verbose=args.verbose)
        img = np.where(img < SENTINEL_TH, 0.0, img).astype(np.float32)
        pan = None

    # Normalização para o modelo
    img_n, norm_used = apply_norm(img, args.norm)
    if args.verbose:
        print(f"[INFO] norm solicitado={args.norm} | norm usado={norm_used} | "
              f"min={float(np.nanmin(img_n)):.6f} max={float(np.nanmax(img_n)):.6f}")

    _, H, W = img_n.shape
    out_sum = np.zeros((3, H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)

    ys = make_positions(H, TILE, STRIDE)
    xs = make_positions(W, TILE, STRIDE)

    tiles_bad_in = 0
    tiles_bad_out = 0
    weight_cache: dict[tuple[int, int, int], np.ndarray] = {}

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + TILE, H)
            x1 = min(x0 + TILE, W)

            tile = img_n[:, y0:y1, x0:x1].astype(np.float32)
            ht, wt = tile.shape[1], tile.shape[2]

            if args.sanitize:
                tile, had_bad = sanitize_array(tile, name="INPUT(tile)", verbose=False)
                if had_bad:
                    tiles_bad_in += 1

            tile_p = pad_reflect(tile, TILE, TILE)
            tile_t = torch.from_numpy(tile_p).unsqueeze(0).to(device)

            with torch.no_grad():
                y = model(tile_t)

            pred_t = y[0] if isinstance(y, (list, tuple)) else y
            if pred_t.ndim != 4 or pred_t.shape[1] != 3:
                raise RuntimeError(f"Saída inesperada do modelo: shape={tuple(pred_t.shape)}")

            pred_t = apply_out_act(pred_t, args.out_act, args.out_temp)

            pred = pred_t[0].detach().float().cpu().numpy().astype(np.float32)
            pred = pred[:, :ht, :wt]

            if args.sanitize:
                pred, had_bad = sanitize_array(pred, name="PRED(tile)", verbose=False)
                if had_bad:
                    tiles_bad_out += 1

            pred = apply_out_range(pred, args.out_range)

            ol_eff = min(OL, ht // 2, wt // 2)
            key = (ht, wt, ol_eff)
            if key not in weight_cache:
                weight_cache[key] = make_weight(ht, wt, ol_eff)
            weight = weight_cache[key]

            out_sum[:, y0:y1, x0:x1] += pred * weight[None, :, :]
            w_sum[y0:y1, x0:x1] += weight

    missing = int((w_sum == 0).sum())
    print(f"Pixels sem cobertura (w_sum==0): {missing} de {H*W}")
    if args.sanitize:
        print(f"Tiles com NaN/Inf na entrada (antes do saneamento): {tiles_bad_in}")
        print(f"Tiles com NaN/Inf na saída do modelo (antes do saneamento): {tiles_bad_out}")

    w_sum_safe = np.where(w_sum == 0, 1.0, w_sum).astype(np.float32)
    out = out_sum / w_sum_safe[None, :, :]
    out[:, w_sum == 0] = 0.0
    if args.sanitize:
        out, _ = sanitize_array(out, name="OUTPUT(final)", verbose=args.verbose)

    # ----------------------------
    # Guide fusion (PAN luminance injection)
    # ----------------------------
    if args.guide_rgb:
        with rasterio.open(args.guide_rgb) as gsrc:
            guide = gsrc.read().astype(np.float32)  # (3,H,W)
        if guide.shape[0] != 3 or guide.shape[1] != H or guide.shape[2] != W:
            raise ValueError("guide_rgb precisa estar alinhado e com mesmo tamanho da PAN (3,H,W).")
        if args.sanitize:
            guide, _ = sanitize_array(guide, name="GUIDE(global)", verbose=args.verbose)

        # guia em 0..1
        gmax = float(np.nanmax(guide))

        if args.verbose:
            print(f"[INFO] guide_rgb: dtype={guide.dtype} min={float(np.nanmin(guide)):.6f} max={gmax:.6f}")

        # Avisos de faixa típica
        if gmax <= 1.5:
            # já parece estar em 0..1
            pass
        elif gmax <= 300.0:
            # muito provavelmente byte 0..255
            if args.verbose:
                print("[WARN] guide_rgb parece estar em 0..255 (byte). Convertendo para 0..1 via /255.")
        else:
            # muito provavelmente reflectance/escala 0..10000 (Sentinel/Landsat-like) ou algo fora do padrão
            print("[WARN] guide_rgb tem valores muito altos (>300). "
                "Isso não parece byte. Confirme a escala (ex: 0..10000). "
                "O script vai dividir por 255 mesmo assim — você pode querer reescalar antes ou adaptar a conversão.")

        guide01 = guide / 255.0 if gmax > 1.5 else guide
        guide01 = np.clip(guide01, 0.0, 1.0).astype(np.float32)

        # pan01 em 0..1 (derivado do que efetivamente foi para o modelo)
        if pan is None:
            pan01 = ((img_n[0] + img_n[1] + img_n[2]) / 3.0).astype(np.float32)
        else:
            pan01 = img_n[0].astype(np.float32)

        # se o norm usado foi -1..1, volta para 0..1 antes do clip
        if norm_used == "minus1_1":
            pan01 = (pan01 + 1.0) * 0.5

        pan01 = np.clip(pan01, 0.0, 1.0).astype(np.float32)

        if args.guide_mode == "pan_luminance_injection":
            out = color_transfer_pan_rgb(
                pan01=pan01,
                guide01=guide01,
                mode=args.fusion_mode,
                chroma_strength=float(args.chroma_strength),
                sat=float(args.sat),
                max_gain=float(args.max_gain),
                chroma_blur_sigma=float(args.chroma_blur_sigma),
                chroma_blur_ksize=int(args.chroma_blur_ksize),
                veg_exg_th=float(args.veg_exg_th),
                veg_sat=float(args.veg_sat),
                veg_chroma=float(args.veg_chroma),
            )

    # ---- write GeoTIFF
    profile_out = profile.copy()
    profile_out.pop("nodata", None)
    profile_out.update(count=3, dtype="float32", nodata=None)
    profile_out.update(photometric="RGB")

    with rasterio.open(args.out_tif, "w", **profile_out) as dst:
        dst.write(out.astype(np.float32))
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)

    print("OK:", args.out_tif)


if __name__ == "__main__":
    main()