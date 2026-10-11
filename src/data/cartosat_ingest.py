"""Ingest Cartosat-3 products downloaded from NRSC Bhoonidhi (https://bhoonidhi.nrsc.gov.in).

Bhoonidhi requires an interactive login, so products are ordered/downloaded through the portal
and dropped (zip or extracted) into data/raw/cartosat3/. This script:
  1. unzips archives,
  2. finds the band rasters (GeoTIFF / JP2) and converts each to a Cloud-Optimised GeoTIFF,
  3. writes data/raw/cartosat3/manifest.json with footprint, GSD, CRS and acquisition metadata.

    python -m src.data.cartosat_ingest
"""
import json
import pathlib
import zipfile

import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.warp import transform_bounds

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/cartosat3"
COG = RAW / "cog"


def main():
    for z in RAW.glob("*.zip"):
        dst = RAW / z.stem
        if not dst.exists():
            zipfile.ZipFile(z).extractall(dst)

    COG.mkdir(exist_ok=True)
    manifest = []
    for p in sorted(list(RAW.rglob("*.tif")) + list(RAW.rglob("*.TIF")) + list(RAW.rglob("*.jp2"))):
        if COG in p.parents:
            continue
        out = COG / (p.stem + ".tif")
        if not out.exists():
            rio_copy(p, out, driver="COG", compress="DEFLATE", BIGTIFF="IF_SAFER")
        with rasterio.open(out) as ds:
            manifest.append({
                "source": str(p.relative_to(ROOT)),
                "cog": str(out.relative_to(ROOT)),
                "crs": str(ds.crs),
                "gsd_m": abs(ds.transform.a),
                "bands": ds.count,
                "bbox_wgs84": transform_bounds(ds.crs, "EPSG:4326", *ds.bounds),
                # Bhoonidhi ships a META/*.txt or BAND_META.txt per product; parse date/sensor from it here
                "meta_files": [str(m.relative_to(ROOT)) for m in p.parent.glob("*META*")],
            })
    json.dump(manifest, open(RAW / "manifest.json", "w"), indent=2)
    print(f"{len(manifest)} rasters -> {RAW/'manifest.json'}")


if __name__ == "__main__":
    main()
