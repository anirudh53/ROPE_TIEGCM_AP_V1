# ROPE-TIEGCM V1

**Reduced-Order Probabilistic Emulator** of TIE-GCM thermospheric density, driven by
F10.7, its 41-day average and ap. It forecasts global 3-D density (72 LST × 36 lat × 45 alt,
100–980 km, hourly) with an uncertainty estimate.

This is the simplest version: the initial condition comes from the IC table and the drivers from a
CSV.

* **Latent dynamics:** 15-model ensemble (5 LSTM, 5 GRU, 5 Transformer) fused by a meta-model, in the
  10-dimensional latent space of a convolutional orthogonal autoencoder (COAE)
* **Uncertainty:** model (ensemble) uncertainty from an unscented transform, scaled by α
  (see [Uncertainty scaling](#uncertainty-scaling))

See **[demo.ipynb](demo.ipynb)**.

## Quick start

```python
from rope import ROPE

rope = ROPE()                                            # CelesTrak drivers
out  = rope.run("2024-05-10 18:00", horizon=72)          # uncertainty=True by default

out["mean_density"]   # (72, 72, 36, 45)  kg/m³   (hour, LST, lat, alt)
out["density_std"]    # (72, 72, 36, 45)  1σ, scaled by α
out["window_df"]      # datetime, f10, f10_41day_avg, ap used

out = rope.run("2024-05-10 18:00", horizon=72, warmup_hours=48)    # optional 48 h warm-up
out = rope.run("2024-05-10 18:00", horizon=72, uncertainty=False)  # fast deterministic run
```

Density at a point (time, LST, lat, alt), with σ:

```python
from interpolator import DensityInterpolator
DensityInterpolator(out).query("2024-05-11 06:30", lst=14.0, lat=30.0, alt_km=450.0, time_mode="interp_time")
```

### Options

| Argument | Default | Meaning |
|---|---|---|
| `drivers` | `"celestrak"` | `"celestrak"`: `data/sw_celestrack_1957_ap.csv.gz` (1957-10 → 2026-10); `"tiegcm"`: TIE-GCM training drivers (1996–2009) |
| `driver_csv` | – | your own hourly CSV with `datetime, f10, ap` |
| `device` | `"cuda"` | falls back to CPU if no GPU |
| `warmup_hours` | `0` | spin-up hours before the start (also per `run()`) |

`run(start, horizon=120, uncertainty=True, warmup_hours=None)` returns `window_df`, `mean_density`,
`density_std` (if `uncertainty`), `ic_latents` and `warmup_hours`.

**Warm-up.** The IC table gives only an approximation of the state at the start. `warmup_hours=48`
starts the run 48 h earlier on the drivers and returns only the requested window, so that error has
faded by the start.

## Installation

```bash
conda create -n rope python=3.11 -y
conda activate rope
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
```

Run from the repository root. A GPU is optional; a 120-hour run with uncertainty takes about 10 s on a GPU.

## How it works

1. **Drivers.** Hourly F10.7, its 41-day trailing mean and ap, plus diurnal and annual harmonics.
2. **Initial condition.** Three hourly latent states (T−2 h … T), interpolated from the IC table on (F10.7, ap).
3. **Forecast.** The 15 ensemble members roll out autoregressively in latent space; the meta-model
   weights them per step and per latent coefficient.
4. **Decode.** The COAE decoder maps latents to log10 density.
5. **Uncertainty.** An unscented transform over the ensemble's 10 × 10 latent covariance at each hour
   (21 sigma points, decoded), then scaled by α.

## Uncertainty scaling

`density_std` is always scaled by the sigma-scaling parameter α:

    density_std = σ_UT / α,     α = 0.747

α is `SIGMA_ALPHA` in `rope.py`.

## Notes on the drivers

* **CelesTrak file.** Built by `tools/build_celestrak_drivers.py` from CelesTrak `SW-All.csv`:
  * F10.7 is the observed value, anchored at 00 UT, linear between days.
  * ap is on 3-hour nodes, linear between them.
* **ap.** The training ap was built by converting 3-hourly Kp to ap and interpolating ap linearly
  between the 3-hour nodes. CelesTrak's 3-hourly ap, interpolated the same way, matches it to about 1 nT.
* **F10.7.** ROPE was trained on the F10.7 that drove the TIE-GCM runs, which is smoother than the observed
  daily F10.7. ROPE is run with observed F10.7, as was the original ROPE.

## Repository layout

```
rope.py               ROPE class (forecast, uncertainty)
interpolator.py       point queries on ROPE output
demo.ipynb            demo
ae_utils/ ts_utils/   COAE and custom-layer definitions
weights/              COAE weights
Models/StormTuned/    15 ensemble members        Meta Models/   meta-model
data/                 normalization stats, IC table, driver CSVs
tools/                build_celestrak_drivers.py (update the CelesTrak driver file)
```

## Updating the drivers

```bash
python tools/build_celestrak_drivers.py      # downloads SW-All.csv, rewrites data/sw_celestrack_1957_ap.csv.gz
```

## Contact

Anirudh Tapedia, West Virginia University, anirudh.tapedia@mail.wvu.edu
