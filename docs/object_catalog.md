# Objects of interest for the emission inventory

How each object becomes emissions: **count (or area/size) × activity (active days, state) × emission factor**,
summed on a 300 m grid per month. SkySat (0.5 m) finds and sizes objects; PlanetScope (3 m, ~168 clear days/yr)
tracks their activity over time.

Priority: ★★★ core for the CVPR paper (big emitter, clearly visible) · ★★ include if time allows · ★ mention / future work.
Label Studio: **box** = rotated rectangle (`obj`), **poly** = polygon (`area`), **—** = not labelled by hand
(derived automatically or out of scope).

| # | Object | Label Studio class | Geometry | Count / measure | Main pollutants | Imagery | Priority |
|---|---|---|---|---|---|---|---|
| **Industry** | | | | | | | |
| 1 | Brick kiln | `brick_kiln` (+ `kiln_type`, `state`) | box | number, type, size, active/idle | PM2.5, BC, CO, SO₂, CO₂ | SkySat; PlanetScope activity | ★★★ |
| 2 | Industrial chimney / stack | `chimney_stack` | box | number, height from shadow | PM, SO₂, NOx | SkySat | ★★ |
| 3 | Stone crusher | `stone_crusher` | box | number, dust-heap size | PM10 | SkySat | ★★ |
| 4 | Steel re-rolling / foundry / other factory | `industrial_unit` | poly | area | PM, CO, metals | SkySat | ★ |
| 5 | Power plant (coal yard, cooling towers) | `power_plant`, `cooling_tower` | poly, box | coal-yard area, towers, plume | SO₂, NOx, PM, CO₂ | SkySat + PlanetScope | ★ |
| 6 | Cement / hot-mix asphalt plant | `cement_or_hotmix_plant` | poly | number, stockpiles | PM, VOCs | SkySat | ★ |
| **Transport** | | | | | | | |
| 7 | Truck (heavy goods vehicle) | `truck` | box | count; parked vs moving | NOx, PM2.5, BC | SkySat only | ★★★ |
| 8 | Bus | `bus` | box | count at depots / roads | NOx, PM | SkySat only | ★★ |
| 9 | Car / two-wheeler / auto-rickshaw | `car` | box | density | NOx, CO, VOCs | needs ≤0.3 m; marginal at 0.5 m | ★ |
| 10 | Truck parking hub / logistics yard | `truck_parking_hub` | poly | area + occupancy | idling NOx, PM | SkySat; occupancy PlanetScope | ★★ |
| 11 | Aircraft | `aircraft` | box | count at gates | NOx, CO, PM | SkySat | ★ |
| 12 | Rail yard / diesel locomotives | `rail_yard`, `locomotive` | poly, box | count of locos | NOx, PM | SkySat | ★ |
| 13 | Bus depot | `bus_depot` | poly | area | NOx, PM | SkySat | ★ |
| **Waste and burning** | | | | | | | |
| 14 | Landfill | `landfill` | poly | area, growth over time | CH₄; PM, CO when burning | SkySat + PlanetScope | ★★★ |
| 15 | Landfill fire / smoke | `landfill_fire`, `smoke_plume` | poly, box | burning area, frequency | PM2.5, CO, dioxins | PlanetScope daily | ★★★ |
| 16 | Open garbage burning | `open_burning` | poly | number, burned area | PM2.5, BC | SkySat | ★★ |
| 17 | Crop-residue burn scars | `crop_burn_scar` (mostly automatic) | poly | burned field area (Oct–Nov) | PM2.5, CO, BC | PlanetScope | ★★★ regional |
| 18 | Crematorium (wood pyres) | `crematorium_site`, `cremation_pyre` | poly, box | active pyres | PM, CO | SkySat, needs known locations | ★ |
| 19 | Wastewater treatment plant | `wastewater_plant` | poly | tank area | CH₄ | SkySat | ★ |
| **Construction and dust** | | | | | | | |
| 20 | Construction site | `construction_site` | poly | disturbed-ground area | PM10 | SkySat + PlanetScope | ★★★ |
| 21 | Sand / aggregate / coal stockpile | `aggregate_stockpile` | poly | heap area (volume from shadow) | PM10 | SkySat | ★★ |
| 22 | Unpaved road | — | — | length | PM10 | SkySat | ★ (future) |
| **Agriculture** | | | | | | | |
| 23 | Dairy colony / cattle sheds | `dairy_colony`, `cattle_shed` | poly, box | shed area, animals | CH₄, NH₃ | SkySat | ★★ |
| 24 | Biogas digester / open slurry tank | `open_tank_or_digester` | box | number | CH₄ | SkySat | ★ |
| **Fuel and energy** | | | | | | | |
| 25 | Storage tank (closed) | `storage_tank` | box | number, diameter, height | VOCs (only if fuel) | SkySat | ★ |
| 26 | Petrol pump / fuel depot | `fuel_station_or_depot` | poly | number | VOCs | SkySat | ★ |
| 27 | Diesel generator clusters | `diesel_genset` | box | count | NOx, PM | needs ≤0.3 m | ★ |
| 28 | Restaurant / tandoor clusters | — | — | count | PM, CO | not visible from space | ★ (future) |

## Status (25 Oct 2026)
- Labelled so far: brick_kiln (105), truck (333), storage_tank (239), aircraft (21), car (2) — on 249 SkySat chips.
- Next to label: landfill (3 polygons: Ghazipur, Bhalswa, Okhla), construction_site, bus (split from truck),
  more kilns incl. oval and circular types.
- Automatic (no hand labels): crop burn scars, landfill fire frequency, kiln activity — from PlanetScope time series.

## Kiln types (`kiln_type`)
`zigzag` (sharp rectangular corners) · `fcbtk_oval` (rounded ends, like a running track) ·
`circular_cfcbk` (round ring) · `clamp` (irregular heap, no chimney) · `unsure`.
