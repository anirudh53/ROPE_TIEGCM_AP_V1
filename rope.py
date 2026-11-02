"""
rope.py  --  ROPE-TIEGCM V1 (F10.7, 41-day average F10.7, ap)
--------------------------------------------------------------
The simplest ROPE: initial condition from the IC table, drivers from a CSV, no data assimilation and
no F10.7 forecaster.

    from rope import ROPE
    rope = ROPE()                                       # CelesTrak drivers (1957-10 .. 2026-10)
    out  = rope.run("2024-05-10 18:00", horizon=72)
    out["mean_density"], out["density_std"]             # (72, 72, 36, 45) kg/m^3, (hour, LST, lat, alt)

Model uncertainty: an unscented transform over the 15-member ensemble's latent covariance at every hour,
then CALIBRATED: density_std = sigma_UT / SIGMA_ALPHA, SIGMA_ALPHA = 0.747 (always applied). The factor
minimises the calibration error score (CES) of the reliability diagram on the test years 1997, 1998,
1999, 2001 (292 back-to-back 120 h runs from the true TIE-GCM state, TIE-GCM drivers, every grid cell):
CES 9.3 % unscaled -> 1.9 % scaled. See README.md.

drivers="celestrak" (default, data/sw_celestrack_1957_ap.csv.gz) or "tiegcm" (SW_TIEGCM.csv, 1996..2009),
or driver_csv=<your hourly CSV with datetime, f10, ap>.
warmup_hours (default 0): start the run that many hours earlier from the IC table on the drivers and
return only the requested window, so the IC-table error has faded by the start.

Output keys: window_df, mean_density, density_std (if uncertainty), ic_latents, warmup_hours
Driver set:  f10, f10_41day_avg, ap, t1, t2, t3, t4
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata

import tensorflow as tf
from tensorflow import keras

from ts_utils.custom_layers import PositionalEncoding
from ae_utils import utils as utils_cae
from ae_utils.attn_models import COAE

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")

# Let TensorFlow take GPU memory as needed instead of all of it, so torch (COAE decoder) and
# other processes can share the GPU.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:          # GPUs already initialized by an earlier import
        pass


# ── Driver columns (fixed for TIE-GCM) ────────────────────────────────────────
TIEGCM_DRIVER_COLS: List[str] = ["f10", "f10_41day_avg", "ap", "t1", "t2", "t3", "t4"]
F10_AVG_ROWS = 984   # 41 days of hourly rows

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVER_CSVS = {
    "celestrak": os.path.join(THIS_DIR, "data", "sw_celestrack_1957_ap.csv.gz"),
    "tiegcm":    os.path.join(THIS_DIR, "data", "SW_TIEGCM.csv"),
}


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ROPEConfig:
    latent_dim:        int = 10
    seq_len:           int = 3
    decode_batch_size: int = 4

    @property
    def total_dim(self) -> int:
        return self.latent_dim + len(TIEGCM_DRIVER_COLS)   # 17


# ============================================================
# Small utils
# ============================================================
def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)


def _safe_device(device: str) -> str:
    return device if not (device.startswith("cuda") and not torch.cuda.is_available()) else "cpu"


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Feature normalizer
# ============================================================
class FeatureNormalizer:
    def __init__(self, stats_ts: Dict[str, Any], latent_dim: int = 10):
        self.K     = latent_dim
        self.mu    = _to_numpy(stats_ts["mu"]).astype(np.float32)
        self.sigma = _to_numpy(stats_ts["sigma"]).astype(np.float32)

        if len(self.mu) != len(self.sigma):
            raise ValueError(f"stats_ts mu/sigma length mismatch: {len(self.mu)} vs {len(self.sigma)}")

        self.total_dim  = len(self.mu)
        self.driver_dim = self.total_dim - self.K

        if self.driver_dim != len(TIEGCM_DRIVER_COLS):
            raise ValueError(
                f"stats_ts driver_dim={self.driver_dim} but TIE-GCM expects {len(TIEGCM_DRIVER_COLS)}. "
                f"Check you loaded the correct stats_ts file."
            )

    def norm_full(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sigma).astype(np.float32)

    def norm_driver(self, drv: np.ndarray) -> np.ndarray:
        return ((drv - self.mu[self.K:]) / self.sigma[self.K:]).astype(np.float32)

    def denorm_latents(self, lat_norm: np.ndarray) -> np.ndarray:
        return (lat_norm * self.sigma[: self.K] + self.mu[: self.K]).astype(np.float32)


# ============================================================
# IC-table interpolator
# ============================================================
class ICTableInterpolator:
    def __init__(self, ic_table: pd.DataFrame):
        self.pts  = ic_table[["F10", "Ap"]].values
        self.vals = ic_table.drop(columns=["F10", "Ap"]).values

    def get_latent_coeffs(self, f10: float, ap: float) -> np.ndarray:
        query = np.array([[f10, ap]])
        pred  = griddata(self.pts, self.vals, query, method="linear")
        if np.isnan(pred).any():
            pred = griddata(self.pts, self.vals, query, method="nearest")
        return pred.flatten().astype(np.float32)


# ============================================================
# Driver window builder
# ============================================================
class DriverWindowBuilder:
    def build(
        self,
        driver_df:      pd.DataFrame,
        start_datetime,
        horizon:        int,
        seq_len:        int,
    ) -> pd.DataFrame:
        start_dt   = pd.Timestamp(start_datetime)
        hist_start = pd.Timestamp(start_dt - timedelta(hours=(seq_len - 1)))
        end_dt     = pd.Timestamp(start_dt + timedelta(hours=(horizon - 1)))
        timeline   = pd.date_range(hist_start, end_dt, freq="h")

        df = (
            driver_df.copy()
            .assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
            .set_index("datetime")
            .reindex(timeline)
            .reset_index()
            .rename(columns={"index": "datetime"})
        )

        hour       = df["datetime"].dt.hour
        doy        = df["datetime"].dt.dayofyear.astype(float)
        df["t1"]   = np.sin(2 * np.pi * hour / 24)
        df["t2"]   = np.cos(2 * np.pi * hour / 24)
        df["t3"]   = np.sin(2 * np.pi * doy / 365.25)
        df["t4"]   = np.cos(2 * np.pi * doy / 365.25)

        # f10_41day_avg at dt = mean of the last 984 driver rows at or before dt
        # (same as drv.loc[:dt, "f10"].tail(984).mean(), computed on a slice for speed)
        drv = (
            driver_df.assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
            .sort_values("datetime")
        )
        t   = drv["datetime"].to_numpy()
        lo  = max(int(np.searchsorted(t, hist_start.to_datetime64(), side="left")) - F10_AVG_ROWS, 0)
        hi  = int(np.searchsorted(t, end_dt.to_datetime64(), side="right"))
        f10 = drv["f10"].iloc[lo:hi].reset_index(drop=True)
        roll = pd.Series(
            f10.rolling(F10_AVG_ROWS, min_periods=1).mean().to_numpy(),
            index=pd.DatetimeIndex(t[lo:hi]),
        )
        df["f10_41day_avg"] = roll.reindex(timeline, method="ffill").to_numpy()

        missing = df[["f10", "ap"]].isna().any(axis=1)
        if missing.any():
            bad = df.loc[missing, ["datetime", "f10", "ap"]].head(10)
            raise ValueError(f"Driver CSV has gaps near {start_datetime}.\n{bad}")

        return df


# ============================================================
# Sequence builder
# ============================================================
class SequenceBuilder:
    def __init__(
        self,
        normalizer: FeatureNormalizer,
        ic_interp:  ICTableInterpolator,
        latent_dim: int = 10,
        seq_len:    int = 3,
    ):
        self.norm = normalizer
        self.ic   = ic_interp
        self.K    = latent_dim
        self.S    = seq_len
        self.D    = normalizer.total_dim

    def build_X_init_norm(self, window_df: pd.DataFrame, latents=None) -> np.ndarray:
        """latents: optional (S, K) IC in ROPE latent units; None -> IC-table lookup."""
        coeff_rows, drv_rows = [], []
        for i in range(self.S):
            row = window_df.iloc[i]
            coeff_rows.append(np.asarray(latents[i], dtype=np.float32) if latents is not None
                              else self.ic.get_latent_coeffs(row["f10"], row["ap"]))
            drv_rows.append(row[TIEGCM_DRIVER_COLS].to_numpy(dtype=np.float32))
        X = np.hstack([np.vstack(coeff_rows), np.vstack(drv_rows)])
        return self.norm.norm_full(X)

    def build_x_chunk(
        self, X_init_norm: np.ndarray, forecast_df: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        x_chunk    = np.zeros((horizon, self.S, self.D), dtype=np.float32)
        x_chunk[0] = X_init_norm
        raw_drv    = forecast_df[TIEGCM_DRIVER_COLS].to_numpy(dtype=np.float32)
        for t in range(1, horizon):
            drv_norm     = self.norm.norm_driver(raw_drv[t])
            row          = np.zeros(self.D, dtype=np.float32)
            row[self.K:] = drv_norm
            x_chunk[t]   = np.vstack([x_chunk[t - 1][1:], row])
        return x_chunk


# ============================================================
# Keras helpers
# ============================================================
def _load_keras_ensemble(
    model_dir:      str,
    n_models:       int  = 5,
    custom_objects: Optional[dict] = None,
) -> List[keras.Model]:
    return [
        keras.models.load_model(
            os.path.join(model_dir, f"best_model_{i}.keras"),
            compile=False,
            custom_objects=custom_objects,
        )
        for i in range(1, n_models + 1)
    ]


def _make_infer_fn(model: keras.Model):
    @tf.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)
    return infer


# ============================================================
# Dynamic rollout
# ============================================================
class DynamicRollout:
    def __init__(self, latent_dim: int = 10):
        self.K = latent_dim

    def run(self, infer_fn, x_chunk_np: np.ndarray, horizon: int) -> np.ndarray:
        return self.run_batch(infer_fn, x_chunk_np[None], horizon)[0]

    def run_batch(self, infer_fn, x_chunks: np.ndarray, horizon: int) -> np.ndarray:
        """x_chunks: (B, H, S, D) -> (B, H-1, K); B independent rollouts in one pass."""
        inp   = x_chunks[:, 0].copy()
        preds = np.zeros((x_chunks.shape[0], horizon - 1, self.K), dtype=np.float32)
        for t in range(1, horizon):
            p               = infer_fn(tf.constant(inp)).numpy()
            preds[:, t - 1] = p.astype(np.float32)
            if (t + 1) < horizon:
                # the new last row pairs the latent just predicted (hour t0+t) with the drivers of
                # the same hour, as in training (and in rope-framework). x_chunks[:, t] ends with the
                # drivers of hour t0+t; the original rope.py used t+1 here (drivers one hour ahead).
                inp[:, :-1]          = inp[:, 1:]
                inp[:, -1, : self.K] = p
                inp[:, -1, self.K:]  = x_chunks[:, t, -1, self.K:]
        return preds


# ============================================================
# Meta fusion
# ============================================================
class MetaFusion:
    def __init__(self, coeff_level: bool = True):
        self.coeff_level = coeff_level

    def fuse(
        self,
        meta_infer_fn,
        x_chunk_np: np.ndarray,
        all_preds:  np.ndarray,
    ) -> np.ndarray:
        W         = meta_infer_fn(tf.constant(x_chunk_np)).numpy()
        preds_TMK = all_preds.transpose(1, 0, 2)
        if self.coeff_level:
            return np.sum(W * preds_TMK, axis=1).astype(np.float32)
        return np.sum(W[:, :, None] * preds_TMK, axis=1).astype(np.float32)


# ============================================================
# Latent decoder
# ============================================================
class LatentDecoder:
    def __init__(self, cae: COAE, stats_cae: Any, device: str, batch_size: int = 4):
        self.cae        = cae
        self.stats_cae  = stats_cae
        self.device     = device
        self.batch_size = batch_size

    @torch.no_grad()
    def decode(self, latent_series: np.ndarray) -> np.ndarray:
        loader = DataLoader(
            torch.tensor(latent_series, dtype=torch.float32),
            batch_size=self.batch_size,
            shuffle=False,
        )
        dens = [
            np.power(
                10,
                utils_cae.denormalize(
                    self.cae.decoder(batch.to(self.device)).detach().cpu().numpy(),
                    self.stats_cae,
                ),
            )
            for batch in loader
        ]
        return np.concatenate(dens, axis=0).squeeze(1)


# ============================================================
# Unscented Transform
# ============================================================
def _unscented_transform_density(
    base_latents_norm: np.ndarray,
    mu_latents_full:   np.ndarray,
    init_lat_phys:     np.ndarray,
    normalizer:        FeatureNormalizer,
    decoder:           LatentDecoder,
) -> tuple[np.ndarray, np.ndarray]:
    M, _, K = base_latents_norm.shape
    H       = mu_latents_full.shape[0]

    base_phys    = normalizer.denorm_latents(
        base_latents_norm.reshape(-1, K)
    ).reshape(M, H - 1, K).astype(np.float32)
    init_rep     = np.repeat(init_lat_phys[None, None, :], M, axis=0)
    lat_full_all = np.concatenate([init_rep, base_phys], axis=1)

    alpha, beta, kappa = 1.0, 2.0, 0.0
    lam = alpha ** 2 * (K + kappa) - K
    c   = K + lam
    Wm    = np.full(2 * K + 1, 1.0 / (2.0 * c), dtype=np.float32)
    Wc    = np.full(2 * K + 1, 1.0 / (2.0 * c), dtype=np.float32)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha ** 2 + beta)

    sigma_lat          = np.zeros((H, 2 * K + 1, K), dtype=np.float32)
    sigma_lat[:, 0, :] = mu_latents_full

    for t in range(H):
        Xt = lat_full_all[:, t, :].astype(np.float32)
        mt = mu_latents_full[t].astype(np.float32)
        d  = Xt - mt[None, :]
        Pt = (d.T @ d) / float(max(M - 1, 1))
        Pt = 0.5 * (Pt + Pt.T) + 1e-6 * np.eye(K, dtype=np.float32)
        try:
            S = np.linalg.cholesky(c * Pt).astype(np.float32)
        except np.linalg.LinAlgError:
            Pt += 1e-3 * np.eye(K, dtype=np.float32)
            S   = np.linalg.cholesky(c * Pt).astype(np.float32)
        for i in range(K):
            sigma_lat[t, 1 + i,     :] = mt + S[:, i]
            sigma_lat[t, 1 + K + i, :] = mt - S[:, i]

    lat_flat  = sigma_lat.reshape(H * (2 * K + 1), K)
    dens_flat = decoder.decode(lat_flat)
    dens_sig  = dens_flat.reshape(H, 2 * K + 1, *dens_flat.shape[1:]).astype(np.float32)

    ut_mean = np.tensordot(Wm, dens_sig, axes=(0, 1)).astype(np.float32)
    diff    = dens_sig - ut_mean[:, None, ...]
    ut_var  = np.tensordot(Wc, diff * diff, axes=(0, 1)).astype(np.float32)

    return ut_mean, np.sqrt(np.maximum(ut_var, 0.0)).astype(np.float32)


# ============================================================
# ROPE  (IC table, meta fusion, calibrated model uncertainty)
# ============================================================
# Global scale on the model sigma: density_std = sigma_UT / SIGMA_ALPHA (sigma x 1.339). CES-minimising
# factor on the test years 1997, 1998, 1999, 2001; CES 9.3 % -> 1.9 %. Valid for this UT (alpha=1,
# beta=2, kappa=0) and these model weights only.
SIGMA_ALPHA = 0.747


class ROPE:
    """
    ROPE for TIE-GCM: IC table, meta-model fusion, calibrated model uncertainty.

        rope = ROPE()
        out  = rope.run("2024-05-10 18:00", horizon=72)
    """

    def __init__(
        self,
        device:            str   = "cuda",
        decode_batch_size: int   = 4,
        use_xla:           bool  = False,

        # ── Stats / COAE ──────────────────────────────────────
        stats_ts_path:    str = os.path.join(THIS_DIR, "data", "stats_ts.pt"),
        stats_cae_path:   str = os.path.join(THIS_DIR, "data", "stats_cae.pt"),
        coae_config_yaml: str = os.path.join(THIS_DIR, "weights", "finetuned_coae", "config.yaml"),
        coae_weights_pth: str = os.path.join(THIS_DIR, "weights", "finetuned_coae", "best_weights_1gpu.pth"),

        # ── Ensemble model dirs ───────────────────────────────
        lstm_dir:        str = os.path.join(THIS_DIR, "Models", "StormTuned", "LSTM MODELS"),
        gru_dir:         str = os.path.join(THIS_DIR, "Models", "StormTuned", "GRU MODELS"),
        transformer_dir: str = os.path.join(THIS_DIR, "Models", "StormTuned", "TRANSFORMER MODELS"),

        # ── Meta model ────────────────────────────────────────
        meta_model_path: str = os.path.join(THIS_DIR, "Meta Models", "meta_model_41df10_ap_stormtuned.keras"),

        # ── Drivers + IC table ────────────────────────────────
        drivers:      str = "celestrak",          # "celestrak" | "tiegcm" (ignored if driver_csv given)
        driver_csv:   Optional[str] = None,
        ic_table_csv: str = os.path.join(THIS_DIR, "data", "IC_Table.csv"),
        warmup_hours: int = 0,                    # spin-up hours before the start
    ):
        self.device = _safe_device(device)
        self.cfg    = ROPEConfig(decode_batch_size=decode_batch_size)

        tf.config.optimizer.set_jit(bool(use_xla))

        # ── Stats ─────────────────────────────────────────────
        self.stats_ts = torch.load(stats_ts_path, map_location="cpu")
        try:
            self.stats_cae = torch.load(stats_cae_path, weights_only=True, map_location="cpu")
        except TypeError:
            self.stats_cae = torch.load(stats_cae_path, map_location="cpu")

        self.normalizer = FeatureNormalizer(self.stats_ts, latent_dim=self.cfg.latent_dim)

        # ── IC + driver pipeline ──────────────────────────────
        driver_csv = driver_csv or DRIVER_CSVS[drivers]
        df = pd.read_csv(driver_csv)
        df["datetime"]  = pd.to_datetime(df["datetime"])
        self._driver_df = df.sort_values("datetime").reset_index(drop=True)
        self.driver_df  = self._driver_df
        self.driver_csv = driver_csv

        self._ic_interp      = ICTableInterpolator(pd.read_csv(ic_table_csv))
        self._window_builder = DriverWindowBuilder()
        self._seq_builder    = SequenceBuilder(
            normalizer=self.normalizer,
            ic_interp=self._ic_interp,
            latent_dim=self.cfg.latent_dim,
            seq_len=self.cfg.seq_len,
        )

        # ── COAE ──────────────────────────────────────────────
        coae_cfg = _load_yaml(coae_config_yaml)
        self.cae = COAE(config=coae_cfg.get("model"))
        try:
            sd = torch.load(coae_weights_pth, weights_only=True, map_location="cpu")
        except TypeError:
            sd = torch.load(coae_weights_pth, map_location="cpu")
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
        self.cae.load_state_dict(sd)
        self.cae.to(self.device).eval()

        # ── Ensemble models ─────────────────────────────────────
        self._all_models = (
            _load_keras_ensemble(lstm_dir, n_models=5)
            + _load_keras_ensemble(gru_dir, n_models=5)
            + _load_keras_ensemble(
                transformer_dir, n_models=5,
                custom_objects={"PositionalEncoding": PositionalEncoding},
            )
        )
        self._infer_fns = [_make_infer_fn(m) for m in self._all_models]

        # ── Meta model ──────────────────────────────────────────
        self._meta_infer_fn = _make_infer_fn(
            keras.models.load_model(meta_model_path, compile=False)
        )

        # ── Shared pipeline components ────────────────────────
        self._rollout = DynamicRollout(latent_dim=self.cfg.latent_dim)
        self._fuser   = MetaFusion(coeff_level=True)
        self._decoder = LatentDecoder(
            self.cae, self.stats_cae,
            device=self.device,
            batch_size=self.cfg.decode_batch_size,
        )

        if int(warmup_hours) < 0:
            raise ValueError("warmup_hours must be >= 0")
        self.warmup_hours = int(warmup_hours)

        print(f"\nROPE ready  [drivers={os.path.basename(driver_csv)}  device={self.device}]\n")

    # ----------------------------------------------------------
    def _prepare(self, start_datetime, horizon: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        internal_df = self._window_builder.build(
            self._driver_df, start_datetime, horizon=horizon, seq_len=self.cfg.seq_len,
        )
        start_idx   = self.cfg.seq_len - 1
        forecast_df = internal_df.iloc[start_idx: start_idx + horizon].reset_index(drop=True)
        X_init_norm = self._seq_builder.build_X_init_norm(internal_df.iloc[: self.cfg.seq_len])
        x_chunk_np  = self._seq_builder.build_x_chunk(X_init_norm, forecast_df, horizon)
        return x_chunk_np, X_init_norm, forecast_df

    # ----------------------------------------------------------
    def run(
        self,
        start_datetime,
        horizon:      int  = 120,
        uncertainty:  bool = True,
        warmup_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a ROPE forecast (IC table, meta fusion).

        horizon      : hours, the first being start_datetime
        uncertainty  : calibrated model uncertainty (density_std); False = fast deterministic run
        warmup_hours : spin-up hours before the start for this call (None = constructor value)

        Returns dict: window_df (datetime, f10, f10_41day_avg, ap), mean_density, density_std (if
        uncertainty), ic_latents, warmup_hours. Time series cover start .. start+horizon-1 only.
        """
        K = self.cfg.latent_dim
        W = self.warmup_hours if warmup_hours is None else int(warmup_hours)
        if W < 0:
            raise ValueError("warmup_hours must be >= 0")
        H = int(horizon) + W
        run_start = pd.Timestamp(start_datetime) - pd.Timedelta(hours=W)

        x_chunk_np, X_init_norm, forecast_df = self._prepare(run_start, H)

        base = np.stack([self._rollout.run(fn, x_chunk_np, H) for fn in self._infer_fns],
                        axis=0).astype(np.float32)                              # (M, H-1, K)
        fused_norm = self._fuser.fuse(self._meta_infer_fn, x_chunk_np[: H - 1], base)
        init_lat_phys = self.normalizer.denorm_latents(X_init_norm[-1, :K])
        fused_full = np.vstack([init_lat_phys[None, :], self.normalizer.denorm_latents(fused_norm)])

        cols = [c for c in ["datetime", "f10", "f10_41day_avg", "ap"] if c in forecast_df.columns]
        out: Dict[str, Any] = {
            "window_df":    forecast_df[cols].iloc[W:].reset_index(drop=True),
            "ic_latents":   self.normalizer.denorm_latents(X_init_norm[:, :K]),
            "warmup_hours": W,
        }
        if uncertainty:
            ut_mean, ut_std = _unscented_transform_density(
                base_latents_norm=base,
                mu_latents_full=fused_full,
                init_lat_phys=init_lat_phys,
                normalizer=self.normalizer,
                decoder=self._decoder,
            )
            out["mean_density"] = ut_mean[W:]
            out["density_std"]  = (ut_std[W:] / SIGMA_ALPHA).astype(np.float32)   # calibrated
        else:
            out["mean_density"] = self._decoder.decode(fused_full)[W:]
        return out
