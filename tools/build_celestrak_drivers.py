"""
build_celestrak_drivers.py -- hourly CelesTrak driver CSV for ROPE-TIEGCM (f10, f10_41day_avg, ap).

Downloads (or reads) CelesTrak SW-All.csv and writes the same hourly layout as the old
original ROPE CelesTrak driver file (f10, kp), with ap added:

    datetime, year, doy, hour, f10, kp, ap, f10_41day_avg

Conventions (identical to the old sw_celestrack_1957.csv, checked by --verify-old):
  f10  : F10.7_OBS, daily value anchored at 00 UT, linear in time between days
  kp   : KPn / 10 (CelesTrak stores Kp x 10), anchored at the start of each 3-h interval
         (00, 03, ..., 21 UT), linear between nodes
  ap   : APn, same 3-hourly anchoring and linear interpolation as kp
  f10_41day_avg : trailing mean of the previous 984 hourly f10 values (41 days), the same
         definition rope.py's DriverWindowBuilder uses. Rows without a full window are NaN.

Usage:
    python tools/build_celestrak_drivers.py                          # download, observed only
    python tools/build_celestrak_drivers.py --sw-all SW-All.csv      # use a local copy
    python tools/build_celestrak_drivers.py --include-predicted      # + 45-day PRD forecast
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd

URL = "https://celestrak.org/SpaceData/SW-All.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIN_41D = 41 * 24

# Daily F10.7_OBS overrides. Current CelesTrak carries the flare-contaminated 560.9 sfu for
# 2003-11-04 (X28 flare); the old sw_celestrack_1957.csv had 160.9, which is kept here so the
# two files agree. Other flare-day spikes (e.g. 2005-09-09, 2006-12-06, 2011-03-07) are left
# exactly as CelesTrak and the old file have them.
F10_OVERRIDES = {"2003-11-04": 160.9}


def load_sw_all(path: str | None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(ROOT, "data", "SW-All.csv")
        print(f"downloading {URL} -> {path}")
        urllib.request.urlretrieve(URL, path)
    sw = pd.read_csv(path, parse_dates=["DATE"])
    return sw


def to_hourly(sw: pd.DataFrame, include_predicted: bool = False, start: str | None = None) -> pd.DataFrame:
    types = ["OBS", "INT"] + (["PRD"] if include_predicted else [])
    sw = sw[sw["F10.7_DATA_TYPE"].isin(types)].copy()
    kp_cols = [f"KP{i}" for i in range(1, 9)]
    ap_cols = [f"AP{i}" for i in range(1, 9)]
    sw = sw.dropna(subset=kp_cols + ap_cols + ["F10.7_OBS"]).sort_values("DATE").reset_index(drop=True)
    for day, val in F10_OVERRIDES.items():
        sw.loc[sw["DATE"] == pd.Timestamp(day), "F10.7_OBS"] = val

    # 3-hourly nodes at 00, 03, ..., 21 UT
    t3 = (sw["DATE"].values[:, None] + (np.arange(8) * 3 * 3600 * 10**9).astype("timedelta64[ns]")).ravel()
    kp3 = (sw[kp_cols].to_numpy(float) / 10.0).ravel()
    ap3 = sw[ap_cols].to_numpy(float).ravel()

    hours = pd.date_range(sw["DATE"].iloc[0], sw["DATE"].iloc[-1] + pd.Timedelta(hours=23), freq="h")
    x = hours.values.astype("datetime64[s]").astype(np.int64)

    def lin(tn, yn):
        return np.interp(x, np.asarray(tn).astype("datetime64[s]").astype(np.int64), yn)

    out = pd.DataFrame({"datetime": hours})
    out["year"] = out["datetime"].dt.year
    out["doy"] = out["datetime"].dt.dayofyear
    out["hour"] = out["datetime"].dt.hour
    out["f10"] = lin(sw["DATE"].values, sw["F10.7_OBS"].to_numpy(float))
    out["kp"] = lin(t3, kp3)
    out["ap"] = lin(t3, ap3)
    out["f10_41day_avg"] = out["f10"].rolling(WIN_41D, min_periods=WIN_41D).mean()
    if start is not None:
        out = out[out["datetime"] >= pd.Timestamp(start)].reset_index(drop=True)
    return out


def verify_against_old(new: pd.DataFrame, old_csv: str) -> None:
    old = pd.read_csv(old_csv, parse_dates=["datetime"])
    m = old.merge(new, on="datetime", suffixes=("_old", "_new"))
    for c in ("f10", "kp"):
        d = np.abs(m[f"{c}_old"] - m[f"{c}_new"])
        print(f"  vs {os.path.basename(old_csv)}: {c} rows={len(m)}  max|diff|={d.max():.4g}  "
              f"rows with |diff|>1e-6: {(d > 1e-6).sum()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sw-all", default=None, help="local SW-All.csv (default: download)")
    p.add_argument("--out", default=os.path.join(ROOT, "data", "sw_celestrack_1957_ap.csv.gz"))
    p.add_argument("--start", default=None, help="first hour to keep, e.g. 1996-01-01")
    p.add_argument("--include-predicted", action="store_true")
    p.add_argument("--verify-old", default=None, help="old sw_celestrack_1957.csv to compare f10/kp with")
    a = p.parse_args()

    hourly = to_hourly(load_sw_all(a.sw_all), a.include_predicted, a.start)
    hourly.to_csv(a.out, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"wrote {a.out}: {len(hourly)} rows, {hourly.datetime.iloc[0]} .. {hourly.datetime.iloc[-1]}")
    if a.verify_old:
        verify_against_old(hourly, a.verify_old)


if __name__ == "__main__":
    main()
