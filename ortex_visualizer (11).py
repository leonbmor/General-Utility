#!/usr/bin/env python
# coding: utf-8

"""
Ortex Fundamentals Visualizer
==============================
Plots normalized fundamentals metrics across stocks. Two modes:

  Mode 1 - Estimate Revision (Calendar Quarter):
    Tracks how the value of a SINGLE quarter evolved across successive
    update dates (x-axis = download_date). Shows analyst estimate revisions
    leading up to and after the earnings release for that quarter.

  Mode 2 - Snapshot (Update Dates):
    Shows the full quarterly time series as it appeared on one or two
    specific download_dates (x-axis = quarter).

Both modes cap estimates at 16 quarters ahead (4 years).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine, text
from datetime import date
from dateutil.relativedelta import relativedelta

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONNECTION_STRING = "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db"
ENGINE = create_engine(CONNECTION_STRING)

ORTEX_CUTOFF_DATE = pd.Timestamp('2026-02-06')

# ==============================================================================
# METRICS CATALOGUE
# ==============================================================================

METRICS = {
    # --- INCOME ---
    1:  ('income', 'totalRevenues',       'Total Revenues'),
    2:  ('income', 'costOfRevenues',      'Cost of Revenues'),
    3:  ('income', 'grossProfit',         'Gross Profit'),
    4:  ('income', 'sg&a',                'SG&A'),
    5:  ('income', 'r&d',                 'R&D'),
    6:  ('income', 'd&a',                 'D&A'),
    7:  ('income', 'totalOperatingExp',   'Total Operating Expenses'),
    8:  ('income', 'operatingIncome',     'Operating Income'),
    9:  ('income', 'interestExp',         'Interest Expense'),
    10: ('income', 'interestIncome',      'Interest Income'),
    11: ('income', 'earningsBeforeTax',   'Earnings Before Tax'),
    12: ('income', 'netIncome',           'Net Income'),
    13: ('income', 'normalizedNetIncome', 'Normalized Net Income'),
    14: ('income', 'dilutedAverageShares','Diluted Average Shares'),
    15: ('income', 'eps',                 'EPS'),
    16: ('income', 'dilutedEps',          'Diluted EPS'),
    # --- CASH ---
    17: ('cash',    'cashFromOperations',  'Cash from Operations'),
    18: ('cash',    'capitalExpenditure',  'Capital Expenditure'),
    19: ('cash',    'cashFromInvesting',   'Cash from Investing'),
    20: ('cash',    'cashFromFinancing',   'Cash from Financing'),
    21: ('cash',    'netChangeInCash',     'Net Change in Cash'),
    # --- SUMMARY ---
    22: ('summary', 'cashAndCashEquivalents', 'Cash & Cash Equivalents'),
    23: ('summary', 'debt',                   'Debt'),
    24: ('summary', 'netDebt',                'Net Debt'),
    25: ('summary', 'ebitda',                 'EBITDA'),
}

# ==============================================================================
# DATABASE HELPERS
# ==============================================================================

def clean_ticker(ticker: str) -> str:
    return ticker.split(' ')[0].upper()


def ticker_exists(ticker: str) -> bool:
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT 1 FROM income_data WHERE ticker = :t LIMIT 1"
        ), {"t": t}).fetchone()
        if row:
            return True
        row = conn.execute(text(
            "SELECT 1 FROM cash_data WHERE ticker = :t LIMIT 1"
        ), {"t": t}).fetchone()
        if row:
            return True
        row = conn.execute(text(
            "SELECT 1 FROM summary_data WHERE ticker = :t LIMIT 1"
        ), {"t": t}).fetchone()
    return bool(row)


def get_first_estimated_period(ticker: str) -> str:
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        row = conn.execute(text("""
            SELECT first_estimated_period FROM estimation_status
            WHERE ticker = :t AND category = 'income'
        """), {"t": t}).fetchone()
    return row[0] if row else None


def get_available_download_dates(ticker: str, table: str) -> list:
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT download_date FROM {table}
            WHERE ticker = :t ORDER BY download_date
        """), {"t": t}).fetchall()
    return sorted([r[0] for r in rows])


def load_quarter_series(ticker: str, table: str, metric: str) -> pd.Series:
    """
    Load full time series for a metric: for each quarter, take the most recent
    download_date value, then ffill to handle Ortex NaN-for-unchanged quarters.
    Returns a Series indexed by period (sorted).
    """
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            WITH ranked AS (
                SELECT period, value,
                       ROW_NUMBER() OVER (PARTITION BY period ORDER BY download_date DESC) rn
                FROM {table}
                WHERE ticker = :t AND metric_name = :m
            )
            SELECT period, value FROM ranked WHERE rn = 1 ORDER BY period
        """), conn, params={"t": t, "m": metric})
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df['value'].values, index=df['period'])
    s = pd.to_numeric(s, errors='coerce').ffill().dropna()
    return s


def load_snapshot_series(ticker: str, table: str, metric: str,
                          download_date) -> pd.Series:
    """
    Load the quarterly series as it appeared on a specific download_date
    (most recent snapshot on or before that date per quarter), then ffill/dropna.
    """
    t  = clean_ticker(ticker)
    dt = pd.Timestamp(download_date)
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            WITH ranked AS (
                SELECT period, value,
                       ROW_NUMBER() OVER (PARTITION BY period ORDER BY download_date DESC) rn
                FROM {table}
                WHERE ticker = :t AND metric_name = :m AND download_date <= :dt
            )
            SELECT period, value FROM ranked WHERE rn = 1 ORDER BY period
        """), conn, params={"t": t, "m": metric, "dt": dt})
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df['value'].values, index=df['period'])
    s = pd.to_numeric(s, errors='coerce').ffill().dropna()
    return s


def load_revision_series(ticker: str, table: str, metric: str,
                          target_period: str) -> pd.Series:
    """
    For a single target_period (e.g. '2025Q4'), return how its metric value
    changed across successive download_dates.
    x-axis = download_date, y-axis = metric value on that date.
    Uses the most recent value available on or before each download_date (ffill).
    """
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT download_date, value
            FROM {table}
            WHERE ticker = :t AND metric_name = :m AND period = :p
            ORDER BY download_date ASC
        """), conn, params={"t": t, "m": metric, "p": target_period})
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(
        pd.to_numeric(df['value'], errors='coerce').values,
        index=pd.to_datetime(df['download_date'])
    ).ffill().dropna()
    return s


# ==============================================================================
# USER INPUT HELPERS
# ==============================================================================

def prompt_mode() -> int:
    print("\n" + "="*60)
    print("SELECT MODE")
    print("="*60)
    print("1. Estimate Revision  (how did a quarter\'s value evolve over time?)")
    print("2. Snapshot           (how does the full quarterly series look on a given date?)")
    while True:
        v = input("\nSelect mode (1/2): ").strip()
        if v in ('1', '2'):
            return int(v)


def prompt_target_quarter() -> list:
    """
    Returns a list of integer offsets from each stock's own last_actual.
    Each stock resolves its own absolute target quarter(s) at build time.
    """
    print("\n" + "="*60)
    print("SELECT TARGET QUARTER (Estimate Revision mode)")
    print("="*60)
    print("Each stock independently tracks its OWN version of the selected quarter(s).")
    print()
    opts = {
        'a': ([0],       "Last actual"),
        'b': ([-1],      "1 quarter ago"),
        'c': ([-2],      "2 quarters ago"),
        'd': ([1],       "1 quarter ahead  (first estimate)"),
        'e': ([2],       "2 quarters ahead"),
        'f': ([0, 2],    "Last actual + 2Q ahead"),
    }
    for k, (_, desc) in opts.items():
        print(f"{k}) {desc}")
    while True:
        v = input("\nSelect option (a-f): ").strip().lower()
        if v in opts:
            return opts[v][0]


def prompt_date_option() -> str:
    print("\n" + "="*60)
    print("SELECT UPDATE DATE(S)")
    print("="*60)
    print("a) Last date available")
    print("b) Date before last date available")
    print("c) Date before 1 month ago")
    print("d) Date before 6 months ago")
    print("e) (a) + (b)")
    print("f) (a) + (c)")
    print("g) (a) + (d)")
    while True:
        v = input("\nSelect option (a/b/c/d/e/f/g): ").strip().lower()
        if v in ('a', 'b', 'c', 'd', 'e', 'f', 'g'):
            return v


def prompt_start_date_revision() -> int:
    """Returns max number of update dates to show (None = all)."""
    print("\n" + "="*60)
    print("SET HISTORY LENGTH (Estimate Revision mode)")
    print("="*60)
    print("1. All update dates available")
    print("2. Last 5 years of updates")
    print("3. Last 2 years of updates")
    print("4. Last 1 year of updates")
    print("5. Last 6 months of updates")
    print("6. Last 3 months of updates")
    while True:
        v = input("\nSelect option (1-6): ").strip()
        if v not in ('1','2','3','4','5','6'):
            continue
        if v == '1':
            return None
        months = {'2': 60, '3': 24, '4': 12, '5': 6, '6': 3}[v]
        return (pd.Timestamp.today() - pd.DateOffset(months=months))


def prompt_start_quarter_dates() -> int:
    """Returns number of quarters to keep (None = all)."""
    print("\n" + "="*60)
    print("SET START QUARTER (Update Dates mode)")
    print("="*60)
    print("1. All quarters available")
    print("2. 40 quarters back")
    print("3. 20 quarters back")
    print("4. 8 quarters back")
    print("5. 4 quarters back")
    while True:
        v = input("\nSelect option (1-5): ").strip()
        if v not in ('1','2','3','4','5'):
            continue
        return {'1': None, '2': 40, '3': 20, '4': 8, '5': 4}[v]


def prompt_display_mode() -> int:
    """Returns 1 for normalized, 2 for YoY growth."""
    print("\n" + "="*60)
    print("SELECT DISPLAY MODE (Snapshot mode)")
    print("="*60)
    print("1. Normalized (base = 1)")
    print("2. YoY quarterly growth (%)")
    while True:
        v = input("\nSelect option (1/2): ").strip()
        if v in ('1', '2'):
            return int(v)


def prompt_tickers() -> list:
    print("\n" + "="*60)
    print("SELECT STOCKS (up to 5, empty input to stop)")
    print("="*60)
    tickers = []
    while len(tickers) < 5:
        raw = input(f"  Stock {len(tickers)+1}: ").strip()
        if raw == '':
            if not tickers:
                print("  Please enter at least one stock.")
                continue
            break
        t = clean_ticker(raw)
        if ticker_exists(t):
            tickers.append(t)
            print(f"  OK: {t} added")
        else:
            print(f"  '{t}' not found in database -- try again or press Enter to stop")
    return tickers


def prompt_metric() -> tuple:
    print("\n" + "="*60)
    print("SELECT METRIC")
    print("="*60)
    print(f"  {'#':<4} {'Group':<10} {'Metric'}")
    print("  " + "-"*40)
    for num, (group, metric, label) in METRICS.items():
        print(f"  {num:<4} {group:<10} {label}")
    while True:
        raw = input("\nEnter metric number: ").strip()
        try:
            n = int(raw)
            if n in METRICS:
                return METRICS[n]
            print(f"  Invalid number, choose between 1 and {max(METRICS)}")
        except ValueError:
            print("  Please enter a number")


# ==============================================================================
# PERIOD HELPERS
# ==============================================================================

def add_quarters(period: str, n: int) -> str:
    year, q = int(period[:4]), int(period[5])
    q += n
    while q > 4:
        q -= 4
        year += 1
    while q < 1:
        q += 4
        year -= 1
    return f"{year}Q{q}"


def period_to_date(period: str) -> pd.Timestamp:
    """Convert 'YYYYQn' to approximate end-of-quarter date for plotting."""
    year, q = int(period[:4]), int(period[5])
    month = q * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


# ==============================================================================
# CORE: BUILD SERIES FOR PLOTTING
# ==============================================================================

def get_date_options(ticker: str, table: str, option: str) -> list:
    """Return list of (label, download_date) tuples for update-date mode."""
    dates = get_available_download_dates(ticker, table)
    if not dates:
        return []

    today = date.today()
    last  = dates[-1]

    def latest_before(target_date):
        candidates = [d for d in dates if d <= target_date]
        return candidates[-1] if candidates else None

    one_month_ago = (pd.Timestamp(today) - pd.DateOffset(months=1)).date()
    six_months_ago = (pd.Timestamp(today) - pd.DateOffset(months=6)).date()

    mapping = {
        'a': [(f"Latest ({last})",                   last)],
        'b': [(f"2nd latest ({dates[-2] if len(dates)>=2 else last})",
                dates[-2] if len(dates) >= 2 else last)],
        'c': [(f"Before 1mo ago ({latest_before(one_month_ago)})",
                latest_before(one_month_ago))],
        'd': [(f"Before 6mo ago ({latest_before(six_months_ago)})",
                latest_before(six_months_ago))],
    }
    mapping['e'] = mapping['a'] + mapping['b']
    mapping['f'] = mapping['a'] + mapping['c']
    mapping['g'] = mapping['a'] + mapping['d']

    return [(lbl, dt) for lbl, dt in mapping[option] if dt is not None]


def build_series_revision_mode(tickers: list, table: str, metric: str,
                                q_offsets: list, start_dt=None) -> dict:
    """
    Mode 1 - Estimate Revision:
    For each ticker and each offset in q_offsets, independently resolves:
        target = add_quarters(last_actual, offset)
    Then plots how that quarter's metric value evolved across download_dates.
    x-axis = download_date (real calendar time), so cross-stock comparison is valid.
    start_dt: if set (pd.Timestamp), clip to dates >= start_dt.
    """
    series_dict = {}

    for ticker in tickers:
        fep = get_first_estimated_period(ticker)
        last_actual = add_quarters(fep, -1) if fep else None

        if last_actual is None:
            print(f"  Cannot determine last_actual for {ticker} -- skipping")
            continue

        for q_offset in q_offsets:
            target_period = add_quarters(last_actual, q_offset)

            s = load_revision_series(ticker, table, metric, target_period)
            if s.empty:
                print(f"  No revision data for {ticker} | {metric} | {target_period}")
                continue

            if start_dt is not None:
                s = s[s.index >= start_dt]

            if s.empty:
                print(f"  No data after start date for {ticker} | {target_period}")
                continue

            label = f"{ticker} ({target_period})"
            series_dict[label] = (s, fep, target_period)

    return series_dict


MAX_ESTIMATE_QUARTERS = 16  # cap forward estimates at 4 years

def build_series_date_mode(tickers: list, table: str, metric: str,
                            date_option: str, n_quarters: int = None,
                            display_mode: int = 1) -> dict:
    """
    Mode 2 - Snapshot:
    Full quarterly series as it appeared on specific download_date(s).
    Estimates capped at 16 quarters ahead. n_quarters clips from the left.
    display_mode: 1 = normalized, 2 = YoY growth (computed before clipping).
    """
    series_dict = {}

    for ticker in tickers:
        fep = get_first_estimated_period(ticker)
        last_actual = add_quarters(fep, -1) if fep else None
        est_cap = add_quarters(fep, MAX_ESTIMATE_QUARTERS) if fep else None

        date_pairs = get_date_options(ticker, table, date_option)
        if not date_pairs:
            print(f"  No update dates found for {ticker}")
            continue

        for lbl, dt in date_pairs:
            s = load_snapshot_series(ticker, table, metric, dt)
            if s.empty:
                print(f"  No snapshot data for {ticker} on {dt}")
                continue

            # Cap estimates at 16Q ahead (right side)
            if est_cap:
                s = s[[p for p in s.index if p < est_cap]]

            # Compute YoY growth on FULL series before any left-side clipping
            # so that shift(4) has all the history it needs
            if display_mode == 2:
                s = _yoy_growth(s, f"{ticker} | {lbl}")
                if s.empty:
                    continue

            # Left-side clip: anchor to last_actual and count n_quarters back
            # Applied AFTER growth calc in both modes
            if n_quarters and last_actual:
                start_period = add_quarters(last_actual, -n_quarters)
                s = s[[p for p in s.index if p >= start_period]]

            if s.empty:
                print(f"  No data after clip for {ticker} | {lbl}")
                continue

            series_dict[f"{ticker} | {lbl}"] = s

    return series_dict


# ==============================================================================
# PLOTTING
# ==============================================================================

def _normalize(s: pd.Series, label: str):
    """Normalize series to start at 1. Returns None if first value is 0/NaN."""
    s = s.dropna()
    if s.empty:
        return None
    first_val = s.iloc[0]
    if first_val == 0 or pd.isna(first_val):
        print(f"  Cannot normalize {label} (first value is 0 or NaN) -- skipping")
        return None
    return s / first_val


def _yoy_growth(s: pd.Series, label: str) -> pd.Series:
    """
    Midpoint YoY growth on the FULL series (so shift(4) doesn't waste early data),
    then clip to +-100% to prevent outliers from distorting the chart.
    Returns series as % values.
    """
    if len(s) < 5:
        print(f"  Not enough quarters for YoY growth on {label} -- skipping")
        return pd.Series(dtype=float)
    growth = (s - s.shift(4)) / ((s + s.shift(4)) / 2).abs()
    growth = (growth * 100).dropna().clip(-100, 100)
    if growth.empty:
        print(f"  YoY growth is all NaN for {label} -- skipping")
    return growth


def plot_revision(series_dict: dict, metric_label: str, q_offsets: list):
    """
    Mode 1 plot: x-axis = download_date (real calendar time), y-axis = normalized value.
    Each series tracks its own ticker-specific quarter (shown in legend).
    Vertical dashed line marks when a quarter transitioned from estimate to actual.
    series_dict = {label: (pd.Series indexed by download_date, fep, target_period)}
    """
    if not series_dict:
        print("No data to plot.")
        return

    offset_map = {0: "Last Actual", -1: "1Q Ago", -2: "2Q Ago",
                  1: "1Q Ahead", 2: "2Q Ahead"}
    offset_desc = " + ".join(offset_map.get(o, f"Q{o:+d}") for o in q_offsets)

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = plt.cm.tab10.colors

    for i, (label, (raw_s, fep, target_period)) in enumerate(series_dict.items()):
        s_norm = _normalize(raw_s, label)
        if s_norm is None:
            continue

        color = colors[i % len(colors)]
        ax.plot(s_norm.index.to_numpy(), s_norm.values, color=color, linewidth=2, label=label)

        # Mark earnings release with a vertical line (when target_period became actual)
        # Best proxy: last download_date where target_period < fep (still estimated)
        # i.e. the value is still being revised = dashed; after = solid dot
        if fep and target_period < fep:
            # target_period is now actual -- mark first date in series as reference
            ax.axvline(s_norm.index[0], color=color, linestyle=':', linewidth=1, alpha=0.35)

    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_title(f"{metric_label} | {offset_desc} - Estimate Revision (normalized)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Update Date", fontsize=12)
    ax.set_ylabel("Normalized Value (base = 1)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.15)
    fig.text(0.13, 0.01, 'Each line tracks its own company-specific quarter (shown in legend)',
             fontsize=9, color='gray', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


def plot_snapshot(series_dict: dict, metric_label: str, first_estimated_periods: dict,
                  display_mode: int = 1):
    """
    Mode 2 plot: x-axis = quarter.
    display_mode 1: y-axis = normalized (base=1), solid=actual, dashed=estimated.
    display_mode 2: y-axis = YoY growth % (clipped +-100), no normalization.
    """
    if not series_dict:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = plt.cm.tab10.colors

    for i, (label, raw_series) in enumerate(series_dict.items()):
        if display_mode == 1:
            s_plot = _normalize(raw_series, label)
            if s_plot is None:
                continue
        else:
            s_plot = raw_series.dropna()
            if s_plot.empty:
                continue

        dates = [period_to_date(p) for p in s_plot.index]
        color = colors[i % len(colors)]
        ticker = label.split(' ')[0]
        fep    = first_estimated_periods.get(ticker)

        if fep and display_mode == 1:
            actual_mask = [p < fep for p in s_plot.index]
            est_mask    = [p >= fep for p in s_plot.index]

            act_d = [d for d, m in zip(dates, actual_mask) if m]
            act_v = [v for v, m in zip(s_plot.values, actual_mask) if m]
            est_d = [d for d, m in zip(dates, est_mask) if m]
            est_v = [v for v, m in zip(s_plot.values, est_mask) if m]

            if act_d and est_d:
                ax.plot(act_d, act_v, color=color, linewidth=2, linestyle='-', label=label)
                ax.plot([act_d[-1]] + est_d, [act_v[-1]] + est_v,
                        color=color, linewidth=2, linestyle='--')
            elif act_d:
                ax.plot(act_d, act_v, color=color, linewidth=2, linestyle='-', label=label)
            else:
                ax.plot(est_d, est_v, color=color, linewidth=2, linestyle='--', label=label)
        else:
            ax.plot(dates, s_plot.values, color=color, linewidth=2, linestyle='-', label=label)

    if display_mode == 1:
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_ylabel("Normalized Value (base = 1)", fontsize=12)
        ax.set_title(f"{metric_label} - Snapshot (normalized)", fontsize=14, fontweight='bold')
        footer = 'Solid = actual  |  Dashed = estimated'
    else:
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_ylabel("YoY Growth (%)", fontsize=12)
        ax.set_title(f"{metric_label} - Snapshot (YoY growth, clipped +/-100%)",
                     fontsize=14, fontweight='bold')
        footer = 'YoY midpoint growth  |  Clipped at +-100%'

    ax.set_xlabel("Quarter", fontsize=12)
    all_periods_flat = sorted(set(p for s in series_dict.values() for p in s.index))
    tick_dates  = [period_to_date(p) for p in all_periods_flat]
    tick_labels = [f"{p[:4]} Q{p[5]}" for p in all_periods_flat]
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10, loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.15)
    fig.text(0.13, 0.01, footer, fontsize=9, color='gray', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("  ORTEX FUNDAMENTALS VISUALIZER")
    print("="*60)

    # Step 1: Mode
    mode = prompt_mode()

    # Step 2: Tickers (needed early so mode 1 can use first ticker as quarter reference)
    tickers = prompt_tickers()
    if not tickers:
        print("No valid tickers selected. Exiting.")
        return

    # Step 3: Mode-specific options
    if mode == 1:
        q_offsets = prompt_target_quarter()
        start_dt = prompt_start_date_revision()
    else:
        d_option     = prompt_date_option()
        n_quarters   = prompt_start_quarter_dates()
        display_mode = prompt_display_mode()

    # Step 4: Metric
    group, metric, metric_label = prompt_metric()
    table = f"{group}_data"

    print(f"\nBuilding chart: [{metric_label}] for {tickers}...")

    # Collect first_estimated_period per ticker for boundary marking
    feps = {t: get_first_estimated_period(t) for t in tickers}

    # Build series and plot
    if mode == 1:
        series_dict = build_series_revision_mode(tickers, table, metric,
                                                  q_offsets, start_dt)
        plot_revision(series_dict, metric_label, q_offsets)
    else:
        series_dict = build_series_date_mode(tickers, table, metric,
                                              d_option, n_quarters, display_mode)
        plot_snapshot(series_dict, metric_label, feps, display_mode)


if __name__ == "__main__":
    main()
