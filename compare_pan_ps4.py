import argparse
import csv
import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


@dataclass
class RunningStats:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    sum_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_sq_err: float = 0.0

    sum_gx2: float = 0.0
    sum_gy2: float = 0.0
    sum_gxy: float = 0.0

    ssim_sum: float = 0.0
    ssim_n: int = 0

    def update(self, x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray, ssim_val: float | None):
        self.n += x.size
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float((x * x).sum())
        self.sum_y2 += float((y * y).sum())
        self.sum_xy += float((x * y).sum())

        err = y - x
        self.sum_err += float(err.sum())
        self.sum_abs_err += float(np.abs(err).sum())
        self.sum_sq_err += float((err * err).sum())

        self.sum_gx2 += float((gx * gx).sum())
        self.sum_gy2 += float((gy * gy).sum())
        self.sum_gxy += float((gx * gy).sum())

        if ssim_val is not None and np.isfinite(ssim_val):
            self.ssim_sum += float(ssim_val)
            self.ssim_n += 1

    def finalize(self):
        if self.n == 0:
            return {}

        mean_x = self.sum_x / self.n
        mean_y = self.sum_y / self.n
        var_x = self.sum_x2 / self.n - mean_x * mean_x
        var_y = self.sum_y2 / self.n - mean_y * mean_y
        cov_xy = self.sum_xy / self.n - mean_x * mean_y

        std_x = math.sqrt(max(var_x, 0.0))
        std_y = math.sqrt(max(var_y, 0.0))
        corr = cov_xy / (std_x * std_y) if std_x > 0 and std_y > 0 else float("nan")

        bias = self.sum_err / self.n
        mae = self.sum_abs_err / self.n
        rmse = math.sqrt(self.sum_sq_err / self.n)

        # Gradiente: correlação via energia e cross-term (streaming)
        grad_corr = (
            self.sum_gxy / math.sqrt(self.sum_gx2 * self.sum_gy2)
            if self.sum_gx2 > 0 and self.sum_gy2 > 0
            else float("nan")
        )
        grad_energy_ratio = (
            (self.sum_gy2 / self.n) / (self.sum_gx2 / self.n)
            if self.sum_gx2 > 0
            else float("nan")
        )

        ssim_mean = self.ssim_sum / self.ssim_n if self.ssim_n > 0 else float("nan")

        return {
            "n_pixels": self.n,
            "pan_mean": mean_x,
            "ps_pan_mean": mean_y,
            "pan_std": std_x,
            "ps_pan_std": std_y,
            "pearson_corr": corr,
            "bias(ps-pan_minus_pan)": bias,
            "mae": mae,
            "rmse": rmse,
            "ssim_mean(blockwise)": ssim_mean,
            "grad_corr(sobel_mag)": grad_corr,
            "grad_energy_ratio(ps/pan)": grad_energy_ratio,
        }


@dataclass
class RunningLinFit:
    n: int = 0
    sum_x: float = 0.0  # PAN
    sum_y: float = 0.0  # SYN
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray):
        self.n += x.size
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_y2 += float((y * y).sum())
        self.sum_xy += float((x * y).sum())

    def solve(self):
        """
        Ajusta y_adj = a*y + b para minimizar ||x - (a*y + b)||^2
        Retorna (a, b).
        """
        if self.n == 0:
            return (1.0, 0.0)

        mx = self.sum_x / self.n
        my = self.sum_y / self.n
        var_y = self.sum_y2 / self.n - my * my
        cov_xy = self.sum_xy / self.n - mx * my

        if var_y <= 0:
            return (1.0, mx - my)

        a = cov_xy / var_y
        b = mx - a * my
        return (float(a), float(b))


def to_float32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32, copy=False) if a.dtype != np.float32 else a


def parse_list_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_pan_synthetic(ps_block: np.ndarray, mode: str, weights: list[float] | None, rgb_bands_are: str):
    """
    ps_block: (bands, h, w) float32
    mode:
      - rgb: luminância Rec.709 nas 3 bandas (ignorando a 4ª)
      - mean3: média das 3 bandas (ignora 4ª)
      - mean4: média 4 bandas
      - weights: combinação linear com pesos fornecidos
    rgb_bands_are:
      - "rgb": assume bandas [1,2,3] = R,G,B
      - "bgr": assume bandas [1,2,3] = B,G,R
    """
    b, _, _ = ps_block.shape
    if b < 3:
        raise ValueError("PS precisa ter pelo menos 3 bandas para modos rgb/mean3")

    if mode == "rgb":
        # Rec.709
        if rgb_bands_are.lower() == "rgb":
            R, G, B = ps_block[0], ps_block[1], ps_block[2]
        else:  # bgr
            B, G, R = ps_block[0], ps_block[1], ps_block[2]
        return 0.2126 * R + 0.7152 * G + 0.0722 * B

    if mode == "mean3":
        return ps_block[0:3].mean(axis=0)

    if mode == "mean4":
        if b < 4:
            raise ValueError("PS precisa ter 4 bandas para mean4")
        return ps_block[0:4].mean(axis=0)

    if mode == "weights":
        if weights is None:
            raise ValueError("mode=weights requer --weights")
        if len(weights) != b:
            raise ValueError(f"weights tem {len(weights)} valores, mas PS tem {b} bandas. Ajuste --weights.")
        w = np.array(weights, dtype=np.float32).reshape((b, 1, 1))
        return (ps_block * w).sum(axis=0)

    raise ValueError(f"Modo inválido: {mode}")


def sobel_mag(img2d: np.ndarray) -> np.ndarray:
    sx = sobel(img2d, axis=1, mode="nearest")
    sy = sobel(img2d, axis=0, mode="nearest")
    return np.hypot(sx, sy).astype(np.float32, copy=False)


def compute_block_ssim(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.shape[0] < 11 or x.shape[1] < 11:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.mean() < 0.5:
        return None

    x2 = x.copy()
    y2 = y.copy()
    mx = np.nanmean(x2)
    my = np.nanmean(y2)
    x2[~mask] = mx
    y2[~mask] = my

    lo = np.nanpercentile(x2, 1)
    hi = np.nanpercentile(x2, 99)
    dr = float(max(hi - lo, 1e-6))
    return float(ssim(x2, y2, data_range=dr))


def read_ps_block_reprojected(ps_ds, pan_ds, win: Window, out_bands: list[int], resampling: Resampling):
    pan_window_transform = rasterio.windows.transform(win, pan_ds.transform)
    h = int(win.height)
    w = int(win.width)
    out = np.zeros((len(out_bands), h, w), dtype=np.float32)

    for i, b in enumerate(out_bands):
        dst = np.zeros((h, w), dtype=np.float32)
        reproject(
            source=rasterio.band(ps_ds, b),
            destination=dst,
            src_transform=ps_ds.transform,
            src_crs=ps_ds.crs,
            dst_transform=pan_window_transform,
            dst_crs=pan_ds.crs,
            dst_nodata=np.nan,
            resampling=resampling,
            num_threads=2,
        )
        out[i] = dst
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pan", required=True)
    ap.add_argument("--ps", required=True)
    ap.add_argument("--pan_band", type=int, default=1)
    ap.add_argument("--ps_bands", default="1,2,3,4", help="Bandas do PS (1-based). Ex: '1,2,3,4'")
    ap.add_argument(
        "--pan_syn",
        default="rgb",
        choices=["rgb", "mean3", "mean4", "weights"],
        help="Como gerar PAN sintética a partir do PS.",
    )
    ap.add_argument(
        "--rgb_bands_are",
        default="rgb",
        choices=["rgb", "bgr"],
        help="Interpretação das 3 primeiras bandas ao usar pan_syn=rgb.",
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="Pesos se pan_syn=weights. Ex: '0.2126,0.7152,0.0722,0.0' (4 valores).",
    )
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--resampling", default="bilinear", choices=["nearest", "bilinear", "cubic"])
    ap.add_argument("--out_csv", default=None)
    ap.add_argument(
        "--radiometric_match",
        default="none",
        choices=["none", "global"],
        help="Opcional: ajusta PAN_sintética com y_adj=a*y+b antes das métricas (2-pass).",
    )
    args = ap.parse_args()

    resampling_map = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "cubic": Resampling.cubic}
    rs = resampling_map[args.resampling]

    ps_bands = [int(x.strip()) for x in args.ps_bands.split(",") if x.strip()]
    weights = parse_list_floats(args.weights) if args.weights else None

    fit = RunningLinFit()

    with rasterio.open(args.pan) as pan_ds, rasterio.open(args.ps) as ps_ds:
        width, height = pan_ds.width, pan_ds.height
        bs = args.block
        total_windows = ((width + bs - 1) // bs) * ((height + bs - 1) // bs)

        def iter_windows():
            for row_off in range(0, height, bs):
                for col_off in range(0, width, bs):
                    h = min(bs, height - row_off)
                    w = min(bs, width - col_off)
                    yield Window(col_off, row_off, w, h)

        # ---------------- PASS 1: estima a,b (se global) ----------------
        a, b = 1.0, 0.0
        if args.radiometric_match == "global":
            pbar = tqdm(total=total_windows, desc="Passo 1/2 (fit a,b)")
            for win in iter_windows():
                pan = to_float32(pan_ds.read(args.pan_band, window=win, boundless=False))
                ps_block = read_ps_block_reprojected(ps_ds, pan_ds, win, ps_bands, rs)
                ps_block = to_float32(ps_block)
                pan_syn = build_pan_synthetic(ps_block, args.pan_syn, weights, args.rgb_bands_are)

                mask = np.isfinite(pan) & np.isfinite(pan_syn)
                if mask.sum() < 1000:
                    pbar.update(1)
                    continue

                fit.update(pan[mask].ravel(), pan_syn[mask].ravel())
                pbar.update(1)
            pbar.close()

            a, b = fit.solve()
            print(f"\nRadiometric match global: y_adj = {a:.8f}*y + {b:.8f}")

        # ---------------- PASS 2: métricas finais ----------------
        stats = RunningStats()
        pbar = tqdm(total=total_windows, desc="Passo 2/2 (métricas)" if args.radiometric_match == "global" else "Comparando por blocos")

        for win in iter_windows():
            pan = to_float32(pan_ds.read(args.pan_band, window=win, boundless=False))
            ps_block = read_ps_block_reprojected(ps_ds, pan_ds, win, ps_bands, rs)
            ps_block = to_float32(ps_block)

            pan_syn = build_pan_synthetic(ps_block, args.pan_syn, weights, args.rgb_bands_are)

            # aplica ajuste radiométrico global se pedido
            if args.radiometric_match == "global":
                pan_syn = a * pan_syn + b

            mask = np.isfinite(pan) & np.isfinite(pan_syn)
            if mask.sum() < 1000:
                pbar.update(1)
                continue

            pan_v = pan[mask].ravel()
            syn_v = pan_syn[mask].ravel()

            # --- NaN-safe Sobel: evita vazamento de NaN na convolução ---
            pan_f = pan.copy()
            syn_f = pan_syn.copy()

            m = np.isfinite(pan_f) & np.isfinite(syn_f)
            if m.sum() < 1000:
                pbar.update(1)
                continue

            pan_fill = float(np.nanmean(pan_f[m]))
            syn_fill = float(np.nanmean(syn_f[m]))
            pan_f[~np.isfinite(pan_f)] = pan_fill
            syn_f[~np.isfinite(syn_f)] = syn_fill

            g_pan = sobel_mag(pan_f)
            g_syn = sobel_mag(syn_f)

            g_pan_v = g_pan[mask].ravel()
            g_syn_v = g_syn[mask].ravel()

            ssim_val = compute_block_ssim(pan, pan_syn)
            stats.update(pan_v, syn_v, g_pan_v, g_syn_v, ssim_val)

            pbar.update(1)

        pbar.close()

    results = stats.finalize()

    print("\n=== Métricas PAN vs PAN-sintética(PS) ===")
    for k, v in results.items():
        print(f"{k:28s}: {v:.6f}" if isinstance(v, float) else f"{k:28s}: {v}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["metric", "value"])
            for k, v in results.items():
                wri.writerow([k, v])
        print(f"\nSalvo em: {args.out_csv}")


if __name__ == "__main__":
    main()