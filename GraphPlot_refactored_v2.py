#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphPlot (refactored)

Goal: preserve all existing functionality of the original GraphPlot.py while
making architecture cleaner: explicit inputs, separated concerns, minimal globals,
and isolated side effects inside main() / run().

Dependencies preserved:
- pandas, numpy, matplotlib, yfinance, tkinter
- sklearn (LinearRegression, PolynomialFeatures)
- sqlalchemy + psycopg2 driver for Postgres connections

Original behaviors preserved (as closely as practical):
- Prompt user for ALGO ticker and NUMERAIRE ticker (or USD MMAcct if blank)
- Prompt for day lag
- Tkinter dropdown for intraday interval (minutes)
- Prompt for polynomial degree (default 10)
- Fetch intraday history via yfinance
- If numeraire provided: normalize price by numeraire and by initial value
- Plot:
  - price, polynomial fit, MAV04/08/16 (+ MAV50 if day_lag>60)
  - volatility bands (upper/lower)
  - fibonacci levels
  - support/resistance trendlines inferred from derivative turning points
  - earnings date vertical lines (with optional amendment + DB persist)
- Additionally create the "derivatives" plot (forecast + 1st/2nd derivative)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Sequence, Tuple

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


# -----------------------------
# Utilities (kept small & pure)
# -----------------------------

def set_df_index_from_first_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Original Set_DF + DD_Index intent:
      - set index from first column
      - drop that column
      - remove duplicated index rows
      - sort by index
    """
    if df.empty:
        return df
    first_col = df.columns[0]
    out = df.copy()
    out.index = out[first_col]
    out.index.name = first_col
    out = out.drop(columns=[first_col])
    out = out[~out.index.duplicated(keep="first")]
    out = out.sort_index()
    return out


def to_hour_index(df: pd.DataFrame) -> pd.DataFrame:
    """Match original behavior: truncate timestamps to hour resolution."""
    out = df.copy()
    out.index = pd.to_datetime(out.index).map(lambda x: datetime(x.year, x.month, x.day, x.hour))
    return out


def safe_int(s: str, default: int) -> int:
    s = (s or "").strip()
    if not s:
        return default
    return int(s)


# -----------------------------
# Configuration / Inputs
# -----------------------------

@dataclass(frozen=True)
class DBConfig:
    # NOTE: to preserve functionality, we keep the original default values
    # but allow overriding with env vars (recommended).
    visiblealpha_db: str = os.environ.get("VISIBLEALPHA_DB", "visiblealpha_laptop")
    factormodel_db: str = os.environ.get("FACTORMODEL_DB", "factormodel_db")
    host: str = os.environ.get("PGHOST", "localhost")
    port: int = int(os.environ.get("PGPORT", "5432"))
    username: str = os.environ.get("PGUSER", "postgres")
    password: str = os.environ.get("PGPASSWORD", "akf7a7j5")

    @property
    def visiblealpha_url(self) -> str:
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.visiblealpha_db}"

    @property
    def factormodel_url(self) -> str:
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.factormodel_db}"


@dataclass(frozen=True)
class PlotConfig:
    window: int = 42
    band_m: float = 2.0
    max_limit: float = 1.05
    min_limit: float = 0.95
    fibonacci_seq: Tuple[float, ...] = (0.00, 23.60, 38.20, 50.00, 61.80, 78.60, 100.00)
    lag_choices: Tuple[str, ...] = ("5", "15", "30", "45", "60")


@dataclass(frozen=True)
class UserInputs:
    algo_ticker_raw: str
    numeraire_ticker_raw: str  # may be empty => USD MMAcct
    day_lag: int
    lag_minutes: str
    poly_degree: int


# -----------------------------
# Data access layer
# -----------------------------

class PostgresRepo:
    def __init__(self, engine: Engine):
        self.engine = engine

    def read_table(self, table: str) -> pd.DataFrame:
        df = pd.read_sql_query(f"SELECT * FROM {table}", self.engine)
        return set_df_index_from_first_column(df)

    def write_table_replace(self, table: str, df: pd.DataFrame) -> None:
        df.to_sql(table, self.engine, index=True, if_exists="replace")


def build_engines(db_cfg: DBConfig) -> Tuple[Engine, Engine]:
    engine = create_engine(db_cfg.visiblealpha_url)
    fengine = create_engine(db_cfg.factormodel_url)
    return engine, fengine


# -----------------------------
# YFinance layer
# -----------------------------

def fetch_intraday_history(ticker: str, start: date, end: date, lag_minutes: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval=f"{lag_minutes}m")
    if df.empty:
        raise ValueError(f"No YF data returned for {ticker} at interval {lag_minutes}m.")
    return df


def resolve_yf_ticker(vayf_df: pd.DataFrame, user_ticker_raw: str) -> str:
    """
    Preserve original: map VisibleAlpha ticker to YF ticker if present,
    else take the first token from user's input.
    """
    user_ticker_raw = (user_ticker_raw or "").upper().strip()
    if not user_ticker_raw:
        return ""
    key = user_ticker_raw.split(" ")[0]
    try:
        return str(vayf_df.loc[key, "YF Ticker"])
    except Exception:
        return key


# -----------------------------
# Analytics layer
# -----------------------------

@dataclass(frozen=True)
class SeriesBundle:
    # indexing: "plot_index" is datetime index used for x-axis labels
    plot_index: pd.DatetimeIndex
    price_df: pd.DataFrame          # index: integer 0..n-1, col: display_name
    plot_price_df: pd.DataFrame     # index: datetime hourly, col: display_name
    display_name: str


def build_price_series(
    algo_yf: str,
    algo_display_raw: str,
    num_yf: Optional[str],
    num_display_raw: Optional[str],
    start: date,
    end: date,
    lag_minutes: str,
) -> SeriesBundle:
    algo_hist = fetch_intraday_history(algo_yf, start=start, end=end, lag_minutes=lag_minutes)

    if num_yf:
        num_hist = fetch_intraday_history(num_yf, start=start, end=end, lag_minutes=lag_minutes)

        # match original: truncate to hour and intersect
        algo_hist_h = to_hour_index(algo_hist)
        num_hist_h = to_hour_index(num_hist)
        common = algo_hist_h.index.intersection(num_hist_h.index)
        algo_hist_h = algo_hist_h.loc[common]
        num_hist_h = num_hist_h.loc[common]

        num_close = pd.DataFrame(num_hist_h["Close"].values, index=np.arange(len(num_hist_h.index)), columns=[num_yf])
        display_name = f"{algo_yf} in {num_display_raw}"
        algo_close = pd.DataFrame(algo_hist_h["Close"].values, index=np.arange(len(algo_hist_h.index)), columns=[display_name])

        # normalize
        algo_close = algo_close / num_close.values
        algo_close = algo_close / algo_close.iloc[0]

        plot_df = pd.DataFrame(algo_close.values, index=algo_hist_h.index, columns=[display_name])
        return SeriesBundle(plot_index=algo_hist_h.index, price_df=algo_close, plot_price_df=plot_df, display_name=display_name)

    # no numeraire
    display_name = algo_yf if algo_display_raw is None else algo_yf
    algo_close = pd.DataFrame(algo_hist["Close"].values, index=np.arange(len(algo_hist.index)), columns=[algo_yf])
    plot_df = pd.DataFrame(algo_close.values, index=algo_hist.index, columns=[algo_yf])
    return SeriesBundle(plot_index=algo_hist.index, price_df=algo_close, plot_price_df=plot_df, display_name=algo_yf)


@dataclass(frozen=True)
class FitBundle:
    predictions_df: pd.DataFrame
    upper_b: pd.DataFrame
    lower_b: pd.DataFrame
    mavs: dict
    fib_levels: list
    deriv1: np.ndarray
    deriv2: np.ndarray


def polynomial_fit_and_derivatives(price_df: pd.DataFrame, poly_degree: int, cfg: PlotConfig) -> FitBundle:
    """
    Mirrors original polynomial fit, volatility bands, MAVs, fibonacci,
    and derivative coefficient construction.
    """
    lin_reg = LinearRegression()
    x_values = price_df.index.values.reshape(-1, 1)
    y_values = np.array(price_df)

    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
    poly_x = poly_features.fit_transform(x_values)
    lr_fit = lin_reg.fit(poly_x, y_values)
    predictions_df = pd.DataFrame(lr_fit.predict(poly_x), index=price_df.index, columns=["Forecast"])

    # bands
    window = cfg.window
    band_m = cfg.band_m
    upper_b = ((1 + band_m * price_df.pct_change().rolling(window).std() * np.sqrt(window)).shift(window) * price_df.shift(window)).dropna()
    lower_b = ((1 - band_m * price_df.pct_change().rolling(window).std() * np.sqrt(window)).shift(window) * price_df.shift(window)).dropna()

    # MAVs (original: 4/8/16; 50 is added later based on day_lag)
    mavs = {
        "MAV04": price_df.rolling(4, min_periods=1).mean(),
        "MAV08": price_df.rolling(8, min_periods=1).mean(),
        "MAV16": price_df.rolling(16, min_periods=1).mean(),
    }

    # fibonacci
    col = price_df.columns[0]
    local_max = float(price_df[col].max())
    local_min = float(price_df[col].min())
    hl_gap = local_max - local_min
    fib_levels = []
    for f_level in cfg.fibonacci_seq:
        fib_levels.append(round(local_min + (f_level / 100.0) * hl_gap, 3))

    # derivative coefficient arrays (kept very close to original)
    first_deriv = 1
    second_deriv = 2

    # Derivative coefficient computation is delicate; preserve original method.
    poly_features_1d = PolynomialFeatures(degree=(poly_degree - first_deriv), include_bias=False)
    poly_dx = poly_features_1d.fit_transform(x_values)
    n, _m = poly_dx.shape
    x0 = np.ones((n, 1))
    poly_dx_base = np.c_[x0, poly_dx]

    # Build polynomial coefficients from LR_Fit coefficients:
    # original: np.poly1d(np.array(list(LR_Fit.coef_[0][::-1]) + [1])).deriv()
    # Note: This assumes a specific ordering; we keep it to preserve output style.
    poly_deriv = np.array(list(np.poly1d(np.array(list(lr_fit.coef_[0][::-1]) + [1])).deriv())[::-1])

    poly_features_2d = PolynomialFeatures(degree=(poly_degree - second_deriv), include_bias=False)
    poly_2dx = poly_features_2d.fit_transform(x_values)
    n2, _m2 = poly_2dx.shape
    x0_2 = np.ones((n2, 1))
    poly_2dx_base = np.c_[x0_2, poly_2dx]
    poly_2deriv = np.array(list(np.poly1d(np.array(list(poly_deriv)[::-1])).deriv())[::-1])

    # Evaluate derivatives to series-like arrays (for turning point detection)
    deriv1 = poly_dx_base.dot(poly_deriv)
    deriv2 = poly_2dx_base.dot(poly_2deriv)

    return FitBundle(
        predictions_df=predictions_df,
        upper_b=upper_b,
        lower_b=lower_b,
        mavs=mavs,
        fib_levels=fib_levels,
        deriv1=deriv1,
        deriv2=deriv2,
    )


def turning_points_from_derivative(deriv1: np.ndarray) -> pd.DataFrame:
    """
    Zero-crossing turning point detection (vectorized).

    Preserves original semantics:
      - When derivative crosses from negative to positive => 'Min'
      - When derivative crosses from positive to negative => 'Max'

    To mimic the original prev-index comparison in the presence of exact zeros,
    we forward-fill 0 signs with the previous non-zero sign.
    Output DataFrame indexed by integer positions with column 'Local'.
    """
    d = np.asarray(deriv1, dtype=float).ravel()
    if d.size == 0:
        return pd.DataFrame(columns=["Local"])

    s = np.sign(d)
    # Forward-fill zeros to preserve "compare to previous" behavior
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]

    flips = (s[1:] * s[:-1]) < 0
    idx = np.where(flips)[0] + 1
    if idx.size == 0:
        return pd.DataFrame(columns=["Local"])

    labels = np.where((s[idx] > 0) & (s[idx - 1] < 0), "Min",
             np.where((s[idx] < 0) & (s[idx - 1] > 0), "Max", None))

    out = pd.DataFrame({"Local": labels}, index=idx)
    out = out.dropna()
    return out



def compute_support_resistance_lines(
    price_df: pd.DataFrame,
    turning_pts: pd.DataFrame,
    min_limit: float,
    max_limit: float,
) -> Tuple[Sequence[pd.DataFrame], Sequence[pd.DataFrame]]:
    """
    Faster + more robust support/resistance construction.

    Keeps original intent and outputs:
      - Determine segment extrema (highs/lows) between turning points
      - Build pairwise lines through extrema and extend to chart edges
      - Return lists of 4-point DataFrames with columns 'Low' / 'High'
      - Clip to [min_limit * price_min, max_limit * price_max]

    Improvements vs legacy:
      - Vectorized extrema position selection (idxmin/idxmax) instead of float equality scans
      - Avoid O(k^2) sklearn fits by using closed-form line through two points
      - Reduced intermediate object overhead
    """
    col = price_df.columns[0]
    price = price_df[col]

    if turning_pts is None or len(turning_pts.index) <= 1:
        return [], []

    idx0 = int(price_df.index[0])
    idxN = int(price_df.index[-1])

    tp_idx = turning_pts.index.astype(int).tolist()
    # boundaries include start and end
    boundaries = [idx0] + tp_idx + [idxN]

    lows: list[tuple[float, int]] = []
    highs: list[tuple[float, int]] = []

    # Segment processing: mimic original "sign trick" but explicitly.
    # Special handling for the first segment depends on the first turning point label.
    first_tp = tp_idx[0]
    first_label = str(turning_pts.loc[first_tp].iloc[0])

    seg0 = price.loc[idx0:first_tp]
    if not seg0.empty:
        if first_label == "Max":
            pos = int(seg0.idxmin())
            lows.append((float(seg0.loc[pos]), pos))
        else:
            pos = int(seg0.idxmax())
            highs.append((float(seg0.loc[pos]), pos))

    # Internal segments: decide based on label at the left turning point
    for k in range(1, len(tp_idx)):
        left = tp_idx[k - 1]
        right = tp_idx[k]
        seg = price.loc[left:right]
        if seg.empty:
            continue
        label = str(turning_pts.loc[left].iloc[0])
        if label == "Min":
            pos = int(seg.idxmin())
            lows.append((float(seg.loc[pos]), pos))
        elif label == "Max":
            pos = int(seg.idxmax())
            highs.append((float(seg.loc[pos]), pos))

    # Tail segment: original uses p_last = turning_pts.index[-2] and last label to decide
    if len(tp_idx) >= 2:
        tail_left = tp_idx[-2]
        tail = price.loc[tail_left:idxN]
        last_label = str(turning_pts.iloc[-1].iloc[0])
        if not tail.empty:
            if last_label == "Max":
                pos = int(tail.idxmax())
                highs.append((float(tail.loc[pos]), pos))
            else:
                pos = int(tail.idxmin())
                lows.append((float(tail.loc[pos]), pos))

    pmin = float(price.min()) * min_limit
    pmax = float(price.max()) * max_limit

    def mk_lines(extrema: list[tuple[float, int]], colname: str) -> list[pd.DataFrame]:
        out: list[pd.DataFrame] = []
        if len(extrema) <= 1:
            return out

        for j in range(len(extrema) - 1):
            y1, x1i = extrema[j]
            x1 = float(x1i)
            for c in range(j + 1, len(extrema)):
                y2, x2i = extrema[c]
                x2 = float(x2i)

                xs = np.array([idx0, x1, x2, idxN], dtype=float)
                if x2 == x1:
                    ys = np.array([y1, y1, y1, y1], dtype=float)
                else:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    ys = m * xs + b

                ys = np.clip(ys, pmin, pmax)
                out.append(pd.DataFrame(ys, index=[idx0, x1, x2, idxN], columns=[colname]))
        return out

    support_lines = mk_lines(lows, "Low")
    resistance_lines = mk_lines(highs, "High")
    return support_lines, resistance_lines



# -----------------------------
# Earnings dates (DB + prompt)
# -----------------------------

def load_earnings_relation(repo: PostgresRepo) -> pd.DataFrame:
    return repo.read_table("ed_relation")


def compute_earnings_dates_for_ticker(ed_df: pd.DataFrame, algo_ticker_raw: str) -> pd.Series:
    """
    Preserve original logic:
      - If algo ticker is a column in ed_relation, parse 'mmdd' strings
      - year is computed from index string like 'Q1-2025'?:
            int(x.split('-')[1]) + int(x[0]) // 4
        (kept as-is)
    Returns Series of datetime.date.
    """
    if algo_ticker_raw not in ed_df.columns:
        return pd.Series([], dtype="object")

    s = ed_df[algo_ticker_raw]
    yr_s = pd.Series(s.index.map(lambda x: int(str(x).split("-")[1]) + int(str(x)[0]) // 4), index=s.index)
    e_dt_df = pd.DataFrame(s).join(yr_s.rename("index"))
    e_dt_df = e_dt_df.replace({"": np.nan}).dropna()

    dates: list[date] = []
    for idx in e_dt_df.index:
        y = int(e_dt_df.loc[idx, "index"])
        mmdd = str(e_dt_df.loc[idx, algo_ticker_raw])
        dates.append(datetime(y, int(mmdd[:2]), int(mmdd[-2:])).date())

    return pd.Series(dates, index=e_dt_df.index)


def prompt_amend_earnings_dates(
    ed_df: pd.DataFrame,
    algo_ticker_raw: str,
    earnings_dates: pd.Series,
    repo: PostgresRepo,
) -> pd.Series:
    """
    Keeps the original interactive editing:
      - show last N earnings dates within range
      - allow user to amend by entering mmdd for recent quarters
      - persist ed_relation back to DB (replace)
    """
    if earnings_dates.empty:
        return earnings_dates

    print(f"Last {earnings_dates.shape[0]} Earnings Dates:")
    print(earnings_dates)
    print("")
    amend = input("Ammend Earnings Dates (empty or Y/y)? ").upper().strip()
    if amend != "Y":
        return earnings_dates

    # iterate backwards, same semantics as original
    for e, saved_dt in enumerate(earnings_dates[::-1].values, start=1):
        c_ed_str = str(input(f"Enter {e} ED back (saved {saved_dt.strftime('%d%b%y')}) mmdd or empty to keep: ")).strip()
        if not c_ed_str:
            continue
        try:
            _ = int(c_ed_str)  # validate
            # ed_df's index corresponds to quarters; original updated by position [-e]
            ed_df.loc[ed_df.index[-e], algo_ticker_raw] = c_ed_str
            # also update the derived earnings_dates series
            qidx = earnings_dates.index[-e]
            year = int(str(ed_df.index[-e]).split("-")[1]) + int(str(ed_df.index[-e])[0]) // 4
            earnings_dates.loc[qidx] = datetime(year, int(c_ed_str[:2]), int(c_ed_str[-2:])).date()
            repo.write_table_replace("ed_relation", ed_df)
        except Exception:
            print(f"{c_ed_str} not saved")

    return earnings_dates


# -----------------------------
# UI layer (Tkinter lag select)
# -----------------------------

def choose_lag_minutes(choices: Sequence[str]) -> str:
    """
    Preserve original selection UI.
    Returns selected lag as string minutes.
    """
    selected = {"value": None}

    def on_selection(value: str):
        selected["value"] = value
        root.destroy()

    root = tk.Tk()
    root.title("Choose lag (minutes)")
    tkvar = tk.StringVar(root)
    popup = tk.OptionMenu(root, tkvar, *choices, command=on_selection)
    tk.Label(root, text="Choose a lag (minutes)").grid(row=0, column=0)
    popup.grid(row=1, column=0)
    root.mainloop()

    if not selected["value"]:
        # fallback to first choice
        return str(choices[0])
    return str(selected["value"])


# -----------------------------
# Plotting layer
# -----------------------------

def plot_main_chart(
    series: SeriesBundle,
    fit: FitBundle,
    cfg: PlotConfig,
    day_lag: int,
    lag_minutes: str,
    earnings_dates: Optional[Sequence[date]] = None,
) -> None:
    col = series.price_df.columns[0]
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)

    # MAV50 special case (original behavior)
    mavs = dict(fit.mavs)
    if day_lag > 60:
        step = (series.price_df.shape[0] // day_lag) + 1
        mav50 = series.price_df[::step].rolling(50, min_periods=1).mean()
        mavs["MAV50"] = mav50

    # price and overlays
    ax.plot(series.price_df, c="k", label=f"{col} {round(float(series.price_df.iloc[-1, 0]), 2)}")
    ax.plot(fit.predictions_df, c="b", alpha=0.15, label=f"Polynomial Fit {round(float(fit.predictions_df.iloc[-1, 0]), 2)}")

    # MAVs
    style_map = {
        "MAV04": dict(c="g", alpha=0.6, linestyle="dashed"),
        "MAV08": dict(c="b", alpha=0.6, linestyle="dashed"),
        "MAV16": dict(c="r", alpha=0.6, linestyle="dashed"),
        "MAV50": dict(c="k", alpha=0.6, linestyle="dashed"),
    }
    for name, s in mavs.items():
        st = style_map.get(name, dict(alpha=0.6, linestyle="dashed"))
        ax.plot(s, label=f"{name} {round(float(s.iloc[-1, 0]), 2)}", **st)

    # Bands clipped
    pmin = float(series.price_df.min().iloc[0]) * cfg.min_limit
    pmax = float(series.price_df.max().iloc[0]) * cfg.max_limit
    ax.plot(fit.upper_b.clip(pmin, pmax, axis=1), c="g", linestyle="dotted", alpha=0.5, label="Upper")
    ax.plot(fit.lower_b.clip(pmin, pmax, axis=1), c="r", linestyle="dotted", alpha=0.5, label="Lower")

    # Fibonacci series lines + collect tick values
    fib_tick_vals = []
    local_max = float(series.price_df[col].max())
    local_min = float(series.price_df[col].min())
    hl_gap = local_max - local_min
    for f_level in cfg.fibonacci_seq:
        f_array = local_min + np.ones(len(series.price_df.index)) * (f_level / 100.0) * hl_gap
        f_series = pd.Series(f_array, index=series.price_df.index)
        fib_tick_vals.append(round(float(f_series.iloc[0]), 3))
        ax.plot(f_series, alpha=0.5)

    # Support/resistance
    turning = turning_points_from_derivative(fit.deriv1)
    support_lines, resistance_lines = compute_support_resistance_lines(
        series.price_df, turning_pts=turning, min_limit=cfg.min_limit, max_limit=cfg.max_limit
    )
    for df in support_lines:
        ax.plot(df, c="g", alpha=0.35, label=f"Support {round(float(df.iloc[-1, 0]), 3)}")
    for df in resistance_lines:
        ax.plot(df, c="r", alpha=0.35, label=f"Resistance {round(float(df.iloc[-1, 0]), 3)}")

    # Annotations (same text as original)
    max_pos = series.price_df[series.price_df[col] == series.price_df[col].max()].index[0]
    ax.text(max_pos, series.price_df[col].max(), str(round(100 * (series.price_df[col].max() / series.price_df.iloc[0, 0] - 1), 2)))
    ax.text(series.price_df.index[-1], series.price_df.iloc[-1, 0], str(round(100 * (series.price_df.iloc[-1, 0] / series.price_df.max().iloc[0] - 1), 2)))

    # X ticks mapped to datetime dates for labeling
    step = int(np.floor(series.price_df.shape[0] / 10)) if series.price_df.shape[0] >= 10 else 1
    ax.set_xticks(series.price_df.index[::step])
    ax.set_xticklabels(series.plot_price_df.index.map(lambda x: x.date())[::step], rotation=45)

    # Earnings date verticals (match original: try to find by date in plot index)
    if earnings_dates:
        plot_dates = series.plot_price_df.index.map(lambda x: x.date()).tolist()
        for e_dt in list(earnings_dates):
            try:
                idx = plot_dates.index(e_dt)
                ax.axvline(x=idx, c="0.25", label=e_dt.strftime("%d%b%y"), linestyle="dotted")
            except Exception:
                pass

    # Y ticks set to fibonacci levels
    ax.set_yticks(fib_tick_vals)
    ax.set_yticklabels(fib_tick_vals)

    ax.set_title(f"{col} - {day_lag} days, {lag_minutes} minutes")
    ax.legend(loc="best")
    plt.show()


def plot_derivatives_view(price_df: pd.DataFrame, fit: FitBundle, poly_degree: int) -> None:
    """
    Preserve the secondary plot block:
      pd.DataFrame(poly_X.dot(coef)+intercept).plot(...).twinx()
      plt.plot(deriv1)
      plt.plot(20*deriv2)
    We rebuild equivalent plots explicitly.
    """
    # Recompute fitted curve directly from predictions_df (simpler, equivalent for view)
    fig, ax_left = plt.subplots(figsize=(14, 10))
    ax_right = ax_left.twinx()

    ax_left.plot(fit.predictions_df.values, c="g")
    ax_right.plot(fit.deriv1, c="k")
    ax_right.plot(20 * fit.deriv2, c="r")

    ax_left.set_title(f"Polynomial Fit + Derivatives (deg={poly_degree})")
    plt.show()


# -----------------------------
# Orchestration
# -----------------------------

def gather_user_inputs(cfg: PlotConfig) -> UserInputs:
    algo_raw = input("Enter ALGO Ticker: ").upper().strip()
    num_raw = input("Enter NUMERAIRE Ticker, empty for USD MMAcct: ").upper().strip()
    day_lag = int(input("Enter Days: ").strip())
    lag_minutes = choose_lag_minutes(cfg.lag_choices)
    poly_degree = safe_int(input("Enter Pol Degree (empty for 10): "), 10)

    return UserInputs(
        algo_ticker_raw=algo_raw,
        numeraire_ticker_raw=num_raw,
        day_lag=day_lag,
        lag_minutes=lag_minutes,
        poly_degree=poly_degree,
    )


def run(inputs: UserInputs, db_cfg: DBConfig, cfg: PlotConfig) -> None:
    warnings.filterwarnings("ignore")

    today = date.today()
    start_date = today - timedelta(days=inputs.day_lag)

    # DB: load mappings and earnings relation
    engine, fengine = build_engines(db_cfg)
    repo = PostgresRepo(engine)
    frepo = PostgresRepo(fengine)

    ed_df = load_earnings_relation(repo)
    vayf_df = frepo.read_table("va_yf")

    algo_yf = resolve_yf_ticker(vayf_df, inputs.algo_ticker_raw)

    num_raw = (inputs.numeraire_ticker_raw or "").strip()
    if num_raw:
        num_yf = resolve_yf_ticker(vayf_df, num_raw)
        num_display = num_raw.split(" ")[0]
    else:
        print("USD MMAcct Numeraire")
        num_yf = None
        num_display = None

    series = build_price_series(
        algo_yf=algo_yf,
        algo_display_raw=inputs.algo_ticker_raw,
        num_yf=num_yf,
        num_display_raw=num_display,
        start=start_date - timedelta(days=1),
        end=today + timedelta(days=1),
        lag_minutes=inputs.lag_minutes,
    )

    fit = polynomial_fit_and_derivatives(series.price_df, inputs.poly_degree, cfg)

    # Earnings dates logic (only if ticker column exists)
    earnings_dates_full = compute_earnings_dates_for_ticker(ed_df, inputs.algo_ticker_raw)
    earnings_dates = pd.Series([], dtype="object")
    if not earnings_dates_full.empty:
        # Keep only earnings dates >= first plot date, like original
        earnings_dates = earnings_dates_full[earnings_dates_full >= series.plot_price_df.index[0].date()]
        # Optional amendment + persist
        earnings_dates = prompt_amend_earnings_dates(ed_df, inputs.algo_ticker_raw, earnings_dates, repo)


    plot_main_chart(
        series=series,
        fit=fit,
        cfg=cfg,
        day_lag=inputs.day_lag,
        lag_minutes=inputs.lag_minutes,
        earnings_dates=list(earnings_dates.values) if not earnings_dates.empty else None,
    )
    plot_derivatives_view(series.price_df, fit, inputs.poly_degree)


def main() -> None:
    cfg = PlotConfig()
    db_cfg = DBConfig()
    inputs = gather_user_inputs(cfg)
    run(inputs, db_cfg, cfg)


if __name__ == "__main__":
    main()
