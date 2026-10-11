# Emission Inventory from Very-High-Res Imagery (CVPR 2027 target)

**Goal:** a bottom-up, object-level **300 m emission inventory** for **Delhi-NCR (India)** and
selected **US** regions. Emitting objects are detected in sub-metre/metre imagery
(Planet SkySat/PlanetScope, Cartosat-3, Maxar WorldView) with YOLO-style detectors,
and the detections are converted to emissions with activity proxies and emission factors.

## Why this is a CVPR paper (not just an application)

1. **New benchmark** – multi-sensor (Planet 3 m / 0.5 m SkySat, Cartosat-3 0.28–0.6 m, Maxar 0.3 m),
   two continents, an emission-relevant taxonomy (~25 classes, long-tailed). Nothing like this exists:
   current datasets are single-class (brick kilns) or generic (DOTA, xView).
2. **Cross-sensor / cross-continent generalisation** – train on India, test on US (and vice-versa);
   train on Maxar, test on Cartosat. Resolution-agnostic detection is an open problem.
3. **Detection → physical quantity** – the evaluation metric is not just mAP but *emission error at 300 m*
   versus reference inventories (EDGAR, CAMS-GLOB-ANT, EPA NEI / GHGRP, SAFAR-Delhi). Proposes a
   "task-aware" loss weighting objects by emission mass rather than count.
4. **Activity state from imagery** – is the kiln firing (plume, fresh soot, green-brick stacks)?
   Is the crematorium active? State classification multiplies the emission factor.

## Taxonomy (see `configs/taxonomy.yaml`)
Point/area sources that are visible from orbit, grouped by sector:
- **Industry (India-heavy):** brick kilns (FCBTK / zig-zag / clamp – kiln *type* matters for EF),
  stone crushers, steel re-rolling mills, cement plants, thermal power plants (stacks + cooling towers + coal yards).
- **Combustion / waste:** crematoria (wood pyres vs electric/CNG), landfills (Ghazipur, Bhalswa, Okhla) and
  landfill fires, open-burning patches, crop-residue burn scars.
- **Transport:** truck parking / logistics hubs, bus depots, rail yards, airports (aircraft count),
  traffic-congested junctions (vehicle counting at 0.3 m).
- **Built environment / dust:** active construction sites, unpaved roads, sand/aggregate stockpiles.
- **Distributed combustion (out-of-the-box):** diesel-genset clusters (rooftop at commercial blocks),
  dhabas / tandoor clusters, tyre-pyrolysis units, informal e-waste burning.
- **Agriculture / CH4:** dairy colonies & cattle sheds, CAFOs/feedlots (US), rice paddies, biogas plants,
  wastewater treatment plants.
- **US-specific:** oil & gas well pads, flare stacks, compressor stations, tank batteries, refineries,
  data centres (backup gensets), ports & anchored ships.

## Pipeline
```
src/data/       planet_download.py   (Planet Data/Orders API — CSDA key)
                cartosat_ingest.py   (Bhoonidhi/NRSC products → COG + manifest)
                maxar_download.py    (Maxar via CSDA / Maxar Open Data STAC)
                tile.py              (reproject, tile 1024² chips, per-sensor GSD metadata)
src/detect/     weak_labels.py       (OSM / EPA GHGRP / EIA-860 / published kiln lists → seed boxes)
                train_yolo.py        (ultralytics YOLO-OBB, multi-sensor)
                infer.py             (sliding window + NMS → GeoPackage of detections)
src/inventory/  grid300.py           (detections × activity × EF → 300 m raster per species)
```
Species: CO2, CO, PM2.5, PM10, BC, OC, SO2, NOx, CH4, NMVOC.

Grids: Delhi-NCR on **EPSG:32643 (UTM 43N)**, US on **EPSG:5070 (CONUS Albers)**, 300 m cells.
Optional: also deliver on 0.0025° lat/lon for model coupling (WRF-Chem / CMAQ).

## Data access status
| Source | Access | Status |
|---|---|---|
| Planet (PlanetScope, SkySat) | NASA CSDA | need `PL_API_KEY` (check Outlook for CSDA/Planet onboarding email) |
| Cartosat-3 | NRSC Bhoonidhi | need login token; products downloaded via portal → `data/raw/cartosat3/` |
| Maxar WorldView | NASA CSDA | need CSDA/Maxar credentials; Maxar Open Data STAC usable now for US events |

## Seed labels (to bootstrap before manual annotation)
- Brick kilns: published Indo-Gangetic-Plain kiln datasets (e.g. IIT Gandhinagar Sustainability Lab
  kiln work; Boyd et al. 2018 "Slavery from Space") – verify licences before use.
- Power plants: WRI Global Power Plant DB, EIA-860 (US). Facilities: EPA GHGRP, NEI point sources.
- OSM tags: `landuse=landfill`, `amenity=crematorium`, `man_made=chimney|kiln|works`, `aeroway`, `railway=yard`.
- Oil & gas (US): state well databases, HIFLD.

## Milestones (CVPR 2027 deadline ≈ mid-Nov 2026 — confirm on cvpr.thecvf.com)
1. Week 1: credentials + AOI imagery pulled for Delhi-NCR and 1–2 US AOIs (Permian / LA basin / Central Valley).
2. Week 1–2: weak labels + 1–2k manual OBB annotations; baseline YOLO-OBB.
3. Week 2–3: cross-sensor experiments, activity-state head, 300 m inventory, comparison to EDGAR/NEI.
4. Week 3–4: paper writing (`paper/`).
