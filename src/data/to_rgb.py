"""Convert multi-band 16-bit imagery (e.g. Planet SkySat pansharpened B,G,R,NIR) into an 8-bit RGB COG + preview.

    python -m src.data.to_rgb data/raw/planet/delhi_ncr/SkySatCollect_pilot --pattern '*_pansharpened_clip.tif' \
        --bands 3 2 1 --out data/raw/planet/delhi_ncr/rgb
"""
import argparse
import pathlib

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window

from src.data.cartosat_rgb import to8

ROOT = pathlib.Path(__file__).resolve().parents[2]


def convert(src_path, out_dir, bands, block=4096, step=32):
    name = src_path.name.split("_pansharpened")[0]
    out = out_dir / f"{name}_rgb.tif"
    if out.exists():
        return out
    with rasterio.open(src_path) as s:
        dec = s.read(bands, out_shape=(len(bands), s.height // step, s.width // step), resampling=Resampling.nearest)
        valid = dec.min(0) > 0
        lo = np.array([np.percentile(b[valid], 2) for b in dec])
        hi = np.array([np.percentile(b[valid], 98) for b in dec])
        stretch = lambda a: np.stack([to8(a[i], lo[i], hi[i]) for i in range(len(bands))])

        f = max(s.width, s.height) // 2048 + 1
        prev = stretch(s.read(bands, out_shape=(len(bands), s.height // f, s.width // f), resampling=Resampling.average))
        with rasterio.open(out_dir / f"{name}_preview.png", "w", driver="PNG", width=prev.shape[2],
                           height=prev.shape[1], count=3, dtype="uint8") as d:
            d.write(prev)

        tmp = out_dir / f"{name}_tmp.tif"
        prof = s.profile | dict(count=3, dtype="uint8", nodata=0, tiled=True, blockxsize=512, blockysize=512,
                                compress="DEFLATE", BIGTIFF="YES", photometric="RGB")
        with rasterio.open(tmp, "w", **prof) as d:
            for r in range(0, s.height, block):
                for c in range(0, s.width, block):
                    w = Window(c, r, min(block, s.width - c), min(block, s.height - r))
                    d.write(stretch(s.read(bands, window=w)), window=w)
    rio_copy(tmp, out, driver="COG", compress="JPEG", QUALITY=92, BIGTIFF="YES")
    tmp.unlink()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("--pattern", default="*_pansharpened_clip.tif")
    ap.add_argument("--bands", type=int, nargs=3, default=[3, 2, 1])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(pathlib.Path(a.src_dir).rglob(a.pattern)):
        print("->", convert(p, out_dir, a.bands), flush=True)


if __name__ == "__main__":
    main()
