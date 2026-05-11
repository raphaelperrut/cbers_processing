import numpy as np
import rasterio

p = r"C:\Users\DGEO2CGEO\Documents\Colorizacao\cbers\output\dbg_residual_sum_raw.tif"
qs = [0.001, 0.01, 0.5, 0.99, 0.999]
ignore_zeros = True

with rasterio.open(p) as s:
    a = s.read().astype(np.float32)
    nodata = s.nodata

for i, name in enumerate(["R", "G", "B"]):
    x = a[i]
    m = np.isfinite(x)
    if nodata is not None:
        m &= (x != nodata)
    if ignore_zeros:
        m &= (x != 0)

    v = x[m]
    qv = np.quantile(v, qs) if v.size else [np.nan]*len(qs)
    print(
        name,
        "q0.1%", qv[0],
        "q1%",   qv[1],
        "q50%",  qv[2],
        "q99%",  qv[3],
        "q99.9%",qv[4],
        "mean",  (v.mean() if v.size else np.nan),
        "std",   (v.std() if v.size else np.nan),
        "n",     int(v.size)
    )
