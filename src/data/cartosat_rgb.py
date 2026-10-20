"""Stack Cartosat-3 BAND1..4 (B,G,R,NIR; 11-bit in uint16) into an 8-bit RGB COG for labelling/YOLO,
plus a small PNG preview.

    python -m src.data.cartosat_rgb data/raw/cartosat3/nrsc_samples/extracted/266335021 --name lucknow_20260115
"""
import argparse
import pathlib

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

ROOT = pathlib.Path(__file__).resolve().parents[2]
RGB_BANDS = ("BAND3.tif", "BAND2.tif", "BAND1.tif")


def percentiles(srcs, step=32):
    """2–98 % stretch per band from a decimated read (ignores nodata=0)."""
    lo, hi = [], []
    for s in srcs:
        a = s.read(1, out_shape=(s.height // step, s.width // step), resampling=Resampling.nearest)
        v = a[a > 0]
        lo.append(np.percentile(v, 2))
        hi.append(np.percentile(v, 98))
    return np.array(lo), np.array(hi)


def to8(a, lo, hi):
    out = np.clip((a.astype(np.float32) - lo) / (hi - lo) * 254 + 1, 1, 255).astype(np.uint8)
    out[a == 0] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("product_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--block", type=int, default=4096)
    a = ap.parse_args()

    pdir = pathlib.Path(a.product_dir)
    out_dir = ROOT / "data/raw/cartosat3/rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = [rasterio.open(pdir / b) for b in RGB_BANDS]
    lo, hi = percentiles(srcs)

    # preview (~2k px on the long side)
    f = max(srcs[0].width, srcs[0].height) // 2048 + 1
    shape = (srcs[0].height // f, srcs[0].width // f)
    prev = np.stack([to8(s.read(1, out_shape=shape, resampling=Resampling.average), lo[i], hi[i])
                     for i, s in enumerate(srcs)])
    prof = dict(driver="PNG", width=shape[1], height=shape[0], count=3, dtype="uint8")
    with rasterio.open(out_dir / f"{a.name}_preview.png", "w", **prof) as d:
        d.write(prev)
    print("preview ->", out_dir / f"{a.name}_preview.png")
    if a.preview_only:
        return

    # full-res RGB GeoTIFF written in blocks, then converted to COG
    tmp = out_dir / f"{a.name}_tmp.tif"
    prof = srcs[0].profile | dict(count=3, dtype="uint8", nodata=0, tiled=True, blockxsize=512,
                                  blockysize=512, compress="DEFLATE", BIGTIFF="YES", photometric="RGB")
    with rasterio.open(tmp, "w", **prof) as d:
        for r in range(0, srcs[0].height, a.block):
            for c in range(0, srcs[0].width, a.block):
                w = Window(c, r, min(a.block, srcs[0].width - c), min(a.block, srcs[0].height - r))
                d.write(np.stack([to8(s.read(1, window=w), lo[i], hi[i]) for i, s in enumerate(srcs)]), window=w)
    from rasterio.shutil import copy as rio_copy
    out = out_dir / f"{a.name}_rgb.tif"
    rio_copy(tmp, out, driver="COG", compress="JPEG", QUALITY=92, BIGTIFF="YES")
    tmp.unlink()
    print("rgb cog ->", out)


if __name__ == "__main__":
    main()
