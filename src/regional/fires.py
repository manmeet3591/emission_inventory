"""Layer D (regional): VIIRS active fires and coal power plants around Delhi.

Inputs (public): NASA FIRMS VIIRS S-NPP yearly country files (375 m), WRI Global Power Plant Database.
Outputs: data/derived/regional/fires_region.parquet, plants_300km.csv, and summary figures.

    python -m src.regional.fires
"""
import glob
import math
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/regional"
OUT = ROOT / "data/derived/regional"
FIG = ROOT / "outputs/figs"
BBOX = (73.8, 27.0, 80.0, 32.6)  # Punjab, Haryana, Delhi, west UP
DELHI = (77.21, 28.61)
NCR = (76.80, 28.20, 77.70, 28.95)
SITES = {"Ghazipur landfill": (77.327, 28.625), "Bhalswa landfill": (77.161, 28.739), "Okhla landfill": (77.288, 28.512)}


def km(lon1, lat1, lon2, lat2):
    p = math.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def bearing(lon1, lat1, lon2, lat2):
    p = math.pi / 180
    y = np.sin((lon2 - lon1) * p) * np.cos(lat2 * p)
    x = np.cos(lat1 * p) * np.sin(lat2 * p) - np.sin(lat1 * p) * np.cos(lat2 * p) * np.cos((lon2 - lon1) * p)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def load_fires():
    parts = []
    for f in sorted(glob.glob(str(RAW / "firms/viirs-snpp_*_India.csv"))):
        d = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight", "type"])
        d = d[(d.longitude.between(BBOX[0], BBOX[2])) & (d.latitude.between(BBOX[1], BBOX[3]))]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["t"] = pd.to_datetime(d.acq_date)
    d["dist_km"] = km(DELHI[0], DELHI[1], d.longitude.values, d.latitude.values)
    d["bearing"] = bearing(DELHI[0], DELHI[1], d.longitude.values, d.latitude.values)
    d["in_ncr"] = d.longitude.between(NCR[0], NCR[2]) & d.latitude.between(NCR[1], NCR[3])
    d["nw_sector"] = (d.bearing >= 270) & (d.dist_km <= 450)  # winter upwind: Punjab/Haryana direction
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_fires()
    d.to_parquet(OUT / "fires_region.parquet")
    veg = d[(d.type == 0) & (d.confidence != "l")]
    static = d[d.type == 2]
    print(f"{len(d):,} detections in region; vegetation/other (type 0, nominal+high conf) {len(veg):,}; static land sources (type 2) {len(static):,}")

    # 1. seasonality and trend of the post-monsoon crop-burning season (upwind NW sector)
    nw = veg[veg.nw_sector]
    season = nw[nw.t.dt.month.isin([10, 11])].groupby(nw.t.dt.year).agg(fires=("frp", "size"), frp_sum_MW=("frp", "sum"))
    print("\nOct–Nov fires in NW upwind sector (≤450 km):\n", season.round(0).to_string())
    monthly = nw.groupby(nw.t.dt.month).size() / nw.t.dt.year.nunique()
    print("\nmean fires per month (NW sector):", monthly.round(0).astype(int).to_dict())

    # 2. fires inside NCR, incl. landfills
    ncr = veg[veg.in_ncr]
    print(f"\nfires inside Delhi-NCR box: {len(ncr):,} over {ncr.t.dt.year.nunique()} years")
    for n, (x, y) in SITES.items():
        near = d[km(x, y, d.longitude.values, d.latitude.values) <= 0.6]
        print(f"  {n}: {len(near)} detections within 600 m; years {near.t.dt.year.value_counts().sort_index().to_dict()}")

    # 3. persistent industrial heat sources (FIRMS type 2), clustered on a ~1 km grid
    s = static.assign(gx=(static.longitude / 0.01).round().astype(int), gy=(static.latitude / 0.01).round().astype(int))
    clusters = s.groupby(["gx", "gy"]).agg(n=("frp", "size"), frp_med=("frp", "median"), lon=("longitude", "mean"),
                                           lat=("latitude", "mean"), first=("t", "min"), last=("t", "max")).reset_index()
    clusters["dist_km"] = km(DELHI[0], DELHI[1], clusters.lon.values, clusters.lat.values)
    clusters = clusters[clusters.n >= 10].sort_values("n", ascending=False)
    clusters.to_csv(OUT / "static_heat_sources.csv", index=False)
    print(f"\npersistent static heat sources (≥10 detections): {len(clusters)}; within 300 km of Delhi: {(clusters.dist_km <= 300).sum()}")

    # 4. coal power plants within 300 km
    g = pd.read_csv(RAW / "gppd.csv", low_memory=False)
    g = g[g.country == "IND"].copy()
    g["dist_km"] = km(DELHI[0], DELHI[1], g.longitude.values, g.latitude.values)
    p = g[(g.dist_km <= 300) & (g.primary_fuel.isin(["Coal", "Gas", "Oil"]))].sort_values("dist_km")
    p[["name", "primary_fuel", "capacity_mw", "commissioning_year", "latitude", "longitude", "dist_km"]].to_csv(OUT / "plants_300km.csv", index=False)
    print(f"\nthermal plants within 300 km (GPPD v1.3, 2021): {len(p)}; coal {int((p.primary_fuel == 'Coal').sum())} "
          f"totalling {p[p.primary_fuel == 'Coal'].capacity_mw.sum():,.0f} MW")
    print(p[p.primary_fuel == "Coal"][["name", "capacity_mw", "dist_km"]].round(0).to_string(index=False))
    # match plants to static heat clusters (≤3 km)
    if len(clusters):
        dmin = [km(r.longitude, r.latitude, clusters.lon.values, clusters.lat.values).min() for r in p.itertuples()]
        p["static_fire_within_3km"] = np.array(dmin) <= 3
        print(f"coal plants with a persistent VIIRS heat source within 3 km: {int(p[p.primary_fuel == 'Coal'].static_fire_within_3km.sum())}"
              f"/{int((p.primary_fuel == 'Coal').sum())}")

    # figures
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    sub = veg[veg.t.dt.month.isin([10, 11])].sample(min(150000, len(veg)), random_state=0)
    ax[0].scatter(sub.longitude, sub.latitude, s=0.2, c="#D85A30", alpha=0.25, lw=0, label="Oct–Nov vegetation fires 2017–25")
    ax[0].scatter(clusters.lon, clusters.lat, s=12, marker="s", c="#3C3489", label="persistent industrial heat (VIIRS type 2)")
    coal = p[p.primary_fuel == "Coal"]
    ax[0].scatter(coal.longitude, coal.latitude, s=coal.capacity_mw / 15, facecolors="none", edgecolors="#185FA5", lw=1.2, label="coal plants (size ∝ MW)")
    ax[0].plot(*DELHI, "k*", ms=14, label="Delhi")
    for r in (150, 300):
        th = np.linspace(0, 2 * np.pi, 200)
        ax[0].plot(DELHI[0] + r / (111 * np.cos(np.radians(DELHI[1]))) * np.cos(th), DELHI[1] + r / 111 * np.sin(th), "k:", lw=0.6)
    ax[0].set_xlim(BBOX[0], BBOX[2]); ax[0].set_ylim(BBOX[1], BBOX[3]); ax[0].set_aspect(1 / np.cos(np.radians(29.5)))
    ax[0].legend(loc="lower left", fontsize=8, markerscale=3); ax[0].set_title("Regional sources upwind of Delhi")
    ym = nw.groupby(nw.t.dt.to_period("M")).size()
    ax[1].bar(ym.index.to_timestamp(), ym.values, width=25, color="#D85A30")
    ax[1].set_title("Monthly VIIRS fires in the NW upwind sector (≤450 km)"); ax[1].set_ylabel("detections / month")
    plt.tight_layout(); plt.savefig(FIG / "regional_fires_plants.png", dpi=120)
    print("\nfigure ->", FIG / "regional_fires_plants.png")


if __name__ == "__main__":
    main()
