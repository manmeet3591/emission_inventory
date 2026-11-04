# Delhi emission source atlas — plan (decided 2026-11-03)

**Goal:** an independent, object-level and multi-scale account of *all* sources behind Delhi's air pollution,
checked against measured source contributions. Target venue: CVPR EarthVision 2027 (≈March) or NeurIPS 2027
Datasets & Benchmarks (≈May), not CVPR 2027 main track. No manual labelling: labels come from existing datasets,
maps, pre-trained models and automatic signals; a small human-checked set (58 Delhi chips, 44 Lucknow kilns) is
kept for evaluation only.

## Why multi-scale
Published apportionment for Delhi PM2.5 (method- and season-dependent, studies disagree):
- Real-time study, winter 2022–23 (IIT Kanpur, IIT Delhi, TERI; DPCC): secondary inorganic aerosol ≈32%,
  biomass burning ≈24% (in and outside Delhi), vehicles ≈17%, coal/fly ash ≈7%; most secondary aerosol from outside Delhi.
- IIT Kanpur 2016 inventory: road dust ≈38%, vehicles ≈20%, domestic fuel ≈12%, industry ≈11%.
So local point sources are only part of the story; regional burning and secondary-aerosol precursors must be in the atlas.

## Layers
| Layer | Sources | Observation | Status |
|---|---|---|---|
| A. Point/area sources in NCR | kilns, industry, power plants, landfills (+fires), construction, stone crushers, crematoria, truck hubs | SkySat/PlanetScope detection, OSM, facility lists | pilot done (kilns, vehicles, tanks) |
| B. Roads and traffic | exhaust, paved/unpaved road dust | OSM roads × imagery vehicle counts × time profiles | not started |
| C. Households / diffuse | cooking & heating fuel, garbage burning, DG sets, restaurants | Census household fuel × settlement type from imagery | not started |
| D. Regional, transported | crop-residue fires, coal plants ≤300 km, regional kiln belt, dust, Diwali | VIIRS/MODIS active fires, PlanetScope burn scars, plant registries, aerosol products | **step 1 — in progress** |
| E. Secondary precursors | NOx, SO₂, NH₃ | TROPOMI NO₂/SO₂, IASI/CrIS NH₃ | not started |

Validation: CPCB CAAQMS stations, DPCC real-time apportionment, TROPOMI, EDGAR/CAMS.

## Lessons carried over (see memory / commits)
- Kiln *presence* is trackable monthly at 3 m (PlanetScope); kiln *firing* is not visible in PlanetScope colour,
  HLS SWIR (30 m) or Landsat thermal (100 m). Next activity test: smoke-plume detection; SkySat repeats when access returns.
- v0 kiln detector fails cross-sensor/region (Lucknow recall 0.09) → diversify training, colour augmentation.
- SkySat download permission lost ~2026-10-20; PlanetScope works (quota A ≈241k km² left).

## Order of work
1. Layer D: VIIRS fires 2017–2026 (Punjab, Haryana, Delhi, west UP) + coal plants within 300 km + regional kiln belt.
2. Layer A at NCR scale with automatic labels.
3. Layers B and C.
4. Layer E + validation against apportionment shares.
5. Paper.
