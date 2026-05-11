import argparse
import os
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.enums import ColorInterp
from model import net

SENTINEL_TH = -1e20  # tudo menor que isso vira 0


# ----------------------------
# Utils
# ----------------------------
def load_model(pkl_path: str, device: str, scale: int):
    """
    Carrega modelo SR (Kong) com scale configurável.
    """
    model = net.Kong(scale=scale).to(device)
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
    """Pad (C,H,W) para (C,target_h,target_w) com reflect."""
    _, h, w = tile.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return tile
    return np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")


def make_positions(total: int, tile: int, stride: int) -> list[int]:
    """Posições no espaço LR garantindo encostar no final."""
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
    """
    Máscara de feathering no espaço onde está sendo aplicada.
    overlap = largura da rampa nas bordas.
    """
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    if overlap > 0:
        eps = 1e-3
        ramp = np.linspace(eps, 1.0, overlap, dtype=np.float32)
        wy[:overlap] = ramp
        wy[-overlap:] = ramp[::-1]
        wx[:overlap] = ramp
        wx[-overlap:] = ramp[::-1]

    return (wy[:, None] * wx[None, :]).astype(np.float32)


# ----------------------------
# Normalization (GLOBAL)
# ----------------------------
def compute_p2p98(img: np.ndarray, ignore_zeros: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    img: (C,H,W) float32
    retorna p2 e p98 por banda (shape C,)
    """
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
    """
    Normaliza a imagem inteira (GLOBAL) e retorna params p/ logs.
    modes:
      - none
      - 0_1       : clamp 0..1
      - 0_255     : /255 + clamp 0..1
      - minus1_1  : clamp 0..1 -> *2-1
      - p2p98_01  : percentil 2-98 -> 0..1
      - p2p98_m11 : percentil 2-98 -> -1..1
      - auto      : se max>1.5 assume 0_255 senão 0_1
    """
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
    """
    Converte a SAÍDA do modelo para o TIFF (por tile, antes do blending).
    out: float32 (C,H,W)
    out_range:
      - none     : não mexe
      - 0_1      : clamp 0..1
      - minus1_1 : clamp -1..1
      - auto     : se norm_used terminou em m11 -> map -1..1 => 0..1 ; senão clamp 0..1
    """
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
    """
    Pós-processamento no output FINAL (após blending).
    mode:
      - none
      - p2p98_01       : p2/p98 por banda -> 0..1
      - p2p98_m11_to_01: p2/p98 por banda -> -1..1 -> 0..1 (equivalente “m11 mas pronto p/ QGIS”)
    """
    info = {}
    x = out.astype(np.float32)

    if mode == "none":
        return x, info

    if mode not in ("p2p98_01", "p2p98_m11_to_01"):
        raise ValueError(f"post_out desconhecido: {mode}")

    p2, p98 = compute_p2p98(x, ignore_zeros=ignore_zeros)
    info["p2"] = p2
    info["p98"] = p98

    p2b = p2[:, None, None]
    p98b = p98[:, None, None]
    denom = np.maximum(p98b - p2b, 1e-6)

    y = (x - p2b) / denom
    y = np.clip(y, 0.0, 1.0)

    if mode == "p2p98_m11_to_01":
        # vira -1..1 e volta pra 0..1 (mantém “estética m11”, mas gravado pronto p/ QGIS)
        y = y * 2.0 - 1.0
        y = np.clip(y, -1.0, 1.0)
        y = (y + 1.0) / 2.0
        y = np.clip(y, 0.0, 1.0)

    return y.astype(np.float32), info


def default_raw_path(out_tif: str) -> str:
    base, ext = os.path.splitext(out_tif)
    return f"{base}_raw{ext}"


def default_vis_path(out_tif: str, post_out: str) -> str:
    base, ext = os.path.splitext(out_tif)
    tag = post_out
    return f"{base}_vis_{tag}{ext}"


def write_rgb_geotiff(path: str, arr: np.ndarray, profile_template: dict, transform_hr: Affine):
    """
    Escreve GeoTIFF RGB float32 com colorinterp correto.
    """
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
    )

    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32))
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tif", required=True, help="RGB LR GeoTIFF (3 bandas)")
    ap.add_argument("--pkl", required=True, help="Checkpoint .pkl do modelo SR (ex: x4.pkl)")
    ap.add_argument("--out_tif", required=True, help="GeoTIFF HR (float32) BASE (usado p/ naming)")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--scale", type=int, default=4, help="Fator SR (8m->2m = 4)")

    ap.add_argument("--tile", type=int, default=256, help="Tile size no espaço LR")
    ap.add_argument("--overlap", type=int, default=16, help="Overlap no espaço LR (feathering)")

    ap.add_argument(
        "--norm",
        default="auto",
        choices=["auto", "none", "0_1", "0_255", "minus1_1", "p2p98_01", "p2p98_m11"],
        help="Normalização aplicada na ENTRADA do SR (GLOBAL).",
    )
    ap.add_argument(
        "--ignore_zeros",
        action="store_true",
        help="Ao calcular p2/p98 (entrada), ignora pixels == 0.",
    )

    ap.add_argument(
        "--out_range",
        default="none",
        choices=["auto", "0_1", "minus1_1", "none"],
        help="Como tratar a SAÍDA do modelo por tile ANTES do blending (diagnóstico/compat).",
    )

    ap.add_argument("--sanitize", action="store_true", help="Sanitiza NaN/Inf na entrada/saída.")
    ap.add_argument("--verbose", action="store_true", help="Logs detalhados.")
    ap.add_argument("--debug_one_tile", action="store_true", help="Roda apenas o primeiro tile e para (diagnóstico).")

    # shave no HR
    ap.add_argument(
        "--shave_hr",
        type=int,
        default=32,
        help="Corta bordas do tile no espaço HR antes do blending (reduz 'tabuleiro'). Ex: 16/32/64.",
    )

    # >>> NOVO: pós-processamento da saída FINAL (após blending)
    ap.add_argument(
        "--post_out",
        default="p2p98_01",
        choices=["none", "p2p98_01", "p2p98_m11_to_01"],
        help="Pós-processamento no output FINAL: p2/p98 por banda -> pronto pro QGIS.",
    )
    ap.add_argument(
        "--post_ignore_zeros",
        action="store_true",
        help="Ao calcular p2/p98 no output FINAL, ignora pixels == 0.",
    )

    # >>> NOVO: paths opcionais (senão auto)
    ap.add_argument(
        "--out_raw_tif",
        default="",
        help="Se vazio, grava <out_tif>_raw.tif com a saída crua.",
    )
    ap.add_argument(
        "--out_vis_tif",
        default="",
        help="Se vazio, grava <out_tif>_vis_<post_out>.tif com a saída pós-processada (se post_out!=none).",
    )

    args = ap.parse_args()

    SCALE = int(args.scale)
    TILE_LR = int(args.tile)
    OL_LR = int(args.overlap)
    STRIDE_LR = TILE_LR - OL_LR
    if STRIDE_LR <= 0:
        raise ValueError("overlap grande demais: tile - overlap precisa ser > 0")

    device = args.device
    if device == "cuda":
        torch.backends.cudnn.benchmark = False

    model = load_model(args.pkl, device, scale=SCALE)

    # --- read LR
    with rasterio.open(args.in_tif) as src:
        img = src.read().astype(np.float32)  # (C,H,W)
        profile = src.profile
        transform = src.transform

    if img.shape[0] != 3:
        raise ValueError(f"Entrada precisa ter 3 bandas (RGB). Veio {img.shape[0]}.")

    # saneamento + sentinel
    if args.sanitize:
        img, _ = sanitize_array(img, name="INPUT(global)", verbose=args.verbose)
    img = np.where(img < SENTINEL_TH, 0.0, img).astype(np.float32)

    # --- GLOBAL normalization
    img_n, norm_used, norm_params = apply_norm_global(img, args.norm, ignore_zeros=args.ignore_zeros)
    if args.verbose:
        mn, mx = float(np.nanmin(img_n)), float(np.nanmax(img_n))
        print(f"[INFO] norm solicitado={args.norm} | norm usado={norm_used} | min={mn:.6f} max={mx:.6f}")
        if "p2" in norm_params:
            print(f"[INFO] p2(entrada) por banda:  {norm_params['p2']}")
            print(f"[INFO] p98(entrada) por banda: {norm_params['p98']}")

    C, H_lr, W_lr = img_n.shape
    H_hr, W_hr = H_lr * SCALE, W_lr * SCALE

    out_sum = np.zeros((C, H_hr, W_hr), dtype=np.float32)
    w_sum = np.zeros((H_hr, W_hr), dtype=np.float32)

    ys = make_positions(H_lr, TILE_LR, STRIDE_LR)
    xs = make_positions(W_lr, TILE_LR, STRIDE_LR)

    tiles_bad_in = 0
    tiles_bad_out = 0

    # overlap no HR
    OL_HR = OL_LR * SCALE
    shave_hr = int(args.shave_hr)

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + TILE_LR, H_lr)
            x1 = min(x0 + TILE_LR, W_lr)

            tile = img_n[:, y0:y1, x0:x1].astype(np.float32)  # (3, ht, wt)
            ht_lr, wt_lr = tile.shape[1], tile.shape[2]

            if args.sanitize:
                tile, had_bad = sanitize_array(tile, name="INPUT(tile)", verbose=False)
                if had_bad:
                    tiles_bad_in += 1

            # pad LR para TILE_LR
            tile_p = pad_reflect(tile, TILE_LR, TILE_LR)
            tile_t = torch.from_numpy(tile_p).unsqueeze(0).to(device)  # (1,3,T,T)

            with torch.no_grad():
                pred = model(tile_t)
                pred_t = pred[0] if isinstance(pred, (list, tuple)) else pred
                pred_t = pred_t.squeeze(0)  # (3, T*scale, T*scale)

            pred_np = pred_t.detach().float().cpu().numpy().astype(np.float32)

            # recorta HR pro tamanho real correspondente ao tile LR
            ht_hr, wt_hr = ht_lr * SCALE, wt_lr * SCALE
            pred_np = pred_np[:, :ht_hr, :wt_hr]

            import torch.nn.functional as F

            # base upsample do tile de entrada (NO MESMO DOMÍNIO do img_n)
            # tile_p é (3, TILE_LR, TILE_LR) já normalizado, e tile_t é (1,3,T,T)
            base_t = F.interpolate(tile_t, scale_factor=SCALE, mode="bicubic", align_corners=False)
            base_np = base_t.squeeze(0).detach().float().cpu().numpy().astype(np.float32)

            # recorta base para o tamanho real do tile
            base_np = base_np[:, :ht_hr, :wt_hr]

            # >>> hipótese residual: soma
            pred_np = base_np + pred_np

            if args.sanitize:
                pred_np, had_bad = sanitize_array(pred_np, name="PRED(tile)", verbose=False)
                if had_bad:
                    tiles_bad_out += 1

            # (opcional) out_range por tile ANTES do blending (em geral deixe "none" aqui)
            pred_np, _out_used = apply_out_range(pred_np, args.out_range, norm_used)

            # coords HR (tile inteiro)
            Y0 = y0 * SCALE
            X0 = x0 * SCALE
            Y1 = Y0 + ht_hr
            X1 = X0 + wt_hr

            # ============================================================
            # SHAVE "AWARE DE BORDA" (sem w_sum==0)
            # ============================================================
            is_top = (y0 == 0)
            is_left = (x0 == 0)
            is_bottom = (y1 == H_lr)
            is_right = (x1 == W_lr)

            top = 0 if is_top else shave_hr
            left = 0 if is_left else shave_hr
            bottom = 0 if is_bottom else shave_hr
            right = 0 if is_right else shave_hr

            # segurança
            if ht_hr <= (top + bottom):
                top, bottom = 0, 0
            if wt_hr <= (left + right):
                left, right = 0, 0

            pred_core = pred_np[:, top:ht_hr - bottom, left:wt_hr - right]

            Y0c = Y0 + top
            X0c = X0 + left
            Y1c = Y1 - bottom
            X1c = X1 - right

            htc = pred_core.shape[1]
            wtc = pred_core.shape[2]
            if htc <= 0 or wtc <= 0:
                continue

            ol_eff = min(OL_HR, htc // 2, wtc // 2)
            weight = make_weight(htc, wtc, ol_eff)

            out_sum[:, Y0c:Y1c, X0c:X1c] += pred_core * weight[None, :, :]
            w_sum[Y0c:Y1c, X0c:X1c] += weight
            # ============================================================

            if args.debug_one_tile:
                if args.verbose:
                    print("[DBG] Stopping after first tile (--debug_one_tile).")
                break
        if args.debug_one_tile:
            break

    missing = int((w_sum == 0).sum())
    print(f"Pixels sem cobertura (w_sum==0): {missing} de {H_hr * W_hr}")

    if args.sanitize:
        print(f"Tiles com NaN/Inf na entrada (antes do saneamento): {tiles_bad_in}")
        print(f"Tiles com NaN/Inf na saída do modelo (antes do saneamento): {tiles_bad_out}")

    w_sum_safe = np.where(w_sum == 0, 1.0, w_sum).astype(np.float32)
    out_raw = out_sum / w_sum_safe[None, :, :]
    out_raw[:, w_sum == 0] = 0.0

    if args.sanitize:
        out_raw, _ = sanitize_array(out_raw, name="OUTPUT(raw_final)", verbose=args.verbose)

    # --- transform HR: pixel size / SCALE (mantém origem)
    a = transform.a / SCALE
    e = transform.e / SCALE
    transform_hr = Affine(a, transform.b, transform.c, transform.d, e, transform.f)

    # --- paths
    out_raw_path = args.out_raw_tif.strip() or default_raw_path(args.out_tif)
    write_rgb_geotiff(out_raw_path, out_raw, profile, transform_hr)
    print("OK (raw):", out_raw_path)

    # --- pós-processamento final para visual
    if args.post_out != "none":
        out_vis, post_info = postprocess_out(out_raw, args.post_out, ignore_zeros=args.post_ignore_zeros)
        if args.verbose:
            mn, mx = float(np.nanmin(out_vis)), float(np.nanmax(out_vis))
            print(f"[INFO] post_out={args.post_out} | min={mn:.6f} max={mx:.6f}")
            if "p2" in post_info:
                print(f"[INFO] p2(saída) por banda:  {post_info['p2']}")
                print(f"[INFO] p98(saída) por banda: {post_info['p98']}")

        out_vis_path = args.out_vis_tif.strip() or default_vis_path(args.out_tif, args.post_out)
        write_rgb_geotiff(out_vis_path, out_vis, profile, transform_hr)
        print("OK (vis):", out_vis_path)
    else:
        print("[INFO] post_out=none -> não gerou arquivo vis.")

    # compat: mantém “OK:” original apontando pra base (pra seu hábito)
    print("OK (base name):", args.out_tif)


if __name__ == "__main__":
    main()
