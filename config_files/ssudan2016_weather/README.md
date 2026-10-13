# South Sudan 2016–2017 — clean single-scale input set (ICCS 2026 model)

A cleaned copy of the correct single-scale 2026 model, ready to use as the base for
the journal paper's runs. Cruft removed: LibreOffice lock files, `plots*/`,
`sf_outputs/`, `out/`, `SWEEP/`, backups, `refugees_old.csv`, and the analysis
scripts (kept separately in your working copy).

## Contents
- `input_csv/`
  - `locations.csv` (143 locations), `routes.csv` (189 routes), `closures.csv`,
    `conflicts.csv`, `registration_corrections.csv`, `sim_period.csv`
    (StartDate 2016-06-01, Length 426)
  - `precipitation.csv` — per-link daily total precipitation, 426 days
  - `river_discharge.csv` — daily GloFAS discharge grid (lat, lon, dis24)
- `source_data/` — `data_layout.csv`, `refugees.csv`, and the per-camp UNHCR files
  (validation data)
- `simsetting.yml` — movement/spawn settings (no weather/flood/seed keys)

## Important: how weather currently enters the model
Your runner `run_par.py` executes a **standard conflict-driven Flee simulation**.
It does **not** read `precipitation.csv` or `river_discharge.csv` — in the 2026
pipeline those files are used only by the **post-processing** scripts
(`new_evacuees*.py`, `precipitation_vs_evacuations.py`, `river_discharge.py`,
`heatmap.py`) to correlate weather with the simulation output.

Consequence for this paper:
- The descriptive climate–mobility analysis (§6.1: correlations, lag, regimes) works
  from this pipeline as-is.
- The **M0–M3 counterfactuals** (Δ_P, Δ_D, Δ_PD and their interaction) — the
  paper's central new contribution — **require weather to affect agent movement
  inside the simulation**, which this pipeline does not yet do. This needs a small,
  self-contained addition to the single-scale run (a weather-aware `Link` that
  reuses the threshold logic), independent of the multiscale coupling.

## Coverage and conventions (from the data)
- Precipitation covers the **83 interior South Sudan links** that appear in
  `precipitation.csv`; **106 of the 189 routes have no precipitation data**
  (the long international legs to camps). Weather should therefore modulate only the
  covered interior links; the rest stay normal — appropriate, since flooding is an
  interior phenomenon.
- The river crossing is **Tonga–Malakal**, identified by coordinates in your code
  (Tonga ≈ 9.4744, 31.0451; Malakal ≈ 9.5410, 31.6520), not by a routes column.
  Discharge closure therefore acts on this single crossing.
- No random seeding exists anywhere in the code, so the uncertainty ensemble (RQ2)
  needs seed control added before matched-seed M0–M3 replicates are possible.

## Producing M0–M3 (after weather-in-sim is added)
The input files here are shared across all four configurations; only the weather
inputs are switched:

| Config | Precipitation | Discharge |
|--------|---------------|-----------|
| M0 baseline | off / neutral | off / neutral |
| M1 precip only | on | off / neutral |
| M2 discharge only | off / neutral | on |
| M3 combined | on | on |

Two equivalent ways to switch, once the simulation reads the weather files:
1. **Neutral input files** (no further code): a zero `precipitation.csv` makes the
   precip mechanism inert; a zeroed `river_discharge.csv` keeps the crossing open.
   Swap these per configuration.
2. **Flags** (`--use_precipitation` / `--use_discharge`) if you add them to the
   runner.

## Next step
Add the single-scale "weather → movement" step, then generate the M0–M3 variants.
That one change unblocks the whole counterfactual campaign while keeping the model
single-scale.
