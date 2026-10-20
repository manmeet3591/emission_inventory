"""Cut an RGB GeoTIFF into overlapping chips for labelling / YOLO, keeping a georeferenced index.

    python -m src.data.tile data/raw/cartosat3/rgb/lucknow_20260115_rgb.tif --sensor cartosat3 --size 1024 --overlap 128

Writes data/tiles/<stem>/images/*.jpg and data/tiles/<stem>/index.json
(chip name -> pixel window + affine transform + CRS), used to map YOLO boxes back to map coordinates.
"""
import argparse
import json
import pathlib

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raster")
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--min-valid", type=float, default=0.5, help="skip chips with less valid (non-zero) area")
    a = ap.parse_args()

    src = pathlib.Path(a.raster)
    out = ROOT / "data/tiles" / src.stem
    (out / "images").mkdir(parents=True, exist_ok=True)
    index = {"source": str(src), "sensor": a.sensor, "chips": {}}
    step = a.size - a.overlap
    with rasterio.open(src) as ds:
        index["crs"] = str(ds.crs)
        index["gsd_m"] = abs(ds.transform.a)
        for r in range(0, ds.height - a.overlap, step):
            for c in range(0, ds.width - a.overlap, step):
                w = Window(c, r, a.size, a.size)
                arr = ds.read([1, 2, 3], window=w, boundless=True, fill_value=0)
                if (arr.max(0) > 0).mean() < a.min_valid:
                    continue
                name = f"{src.stem}_r{r:06d}_c{c:06d}.jpg"
                Image.fromarray(np.moveaxis(arr, 0, -1)).save(out / "images" / name, quality=95)
                index["chips"][name] = {"row": r, "col": c, "transform": list(ds.window_transform(w))[:6]}
    json.dump(index, open(out / "index.json", "w"))
    print(f"{len(index['chips'])} chips -> {out}")


if __name__ == "__main__":
    main()
