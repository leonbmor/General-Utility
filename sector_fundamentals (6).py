#!/usr/bin/env python
# coding: utf-8

"""
sector_fundamentals.py
======================
Calculates and visualizes cap-weighted sector valuation or growth metrics
over a user-selected historical period, using fundamentals from the Ortex DB.

Metric types:
  - Valuation : mkt_cap / LTM or NTM metric  (e.g. EV/Sales, P/E, EV/EBITDA)
  - Growth    : symmetric (mid-point) YoY growth of LTM or NTM metric

Metrics available: totalRevenues (income), normalizedNetIncome (income), ebitda (summary)

FEQ mapping: for each back-date, determines the correct First Estimated Quarter
             by anchoring from today's FEQ and verifying via revenue changes
             across adjacent download_date snapshots.

Results cached in DB tables: sector_valuation / sector_growth
Re-runs only compute missing dates by default; set override=True to recompute all.

Usage:
    from sector_fundamentals import run
    df, fig = run(Pxs_df, sectors_s)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine, text
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================

ENGINE     = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")
MIN_STOCKS = 3    # minimum stocks with valid data to include a sector on a date


# ==============================================================================
# HELPERS
# ==============================================================================

def clean_ticker(t: str) -> str:
    return str(t).split(' ')[0].strip().upper()


def add_quarters(quarter: str, n: int) -> str:
    year, q = int(quarter[:4]), int(quarter[5])
    q += n
    while q > 4:
        q -= 4
        year += 1
    while q < 1:
        q += 4
        year -= 1
    return f"{year}Q{q}"


def quarter_to_approx_date(quarter: str) -> pd.Timestamp:
    """Approximate mid-point date of a quarter for ordering."""
    year, q = int(quarter[:4]), int(quarter[5])
    month = q * 3 - 1
    return pd.Timestamp(year=year, month=month, day=15)


# ==============================================================================
# DB HELPERS
# ==============================================================================

def _get_all_download_dates() -> pd.DatetimeIndex:
    """Get all distinct download_dates across income_data and summary_data."""
    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT download_date FROM income_data
            UNION
            SELECT DISTINCT download_date FROM summary_data
            ORDER BY download_date
        """)).fetchall()
    return pd.DatetimeIndex(sorted([pd.Timestamp(r[0]) for r in rows]))


def _get_ticker_download_dates(ticker: str) -> pd.DatetimeIndex:
    """All download_dates for a specific ticker in income_data."""
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT download_date FROM income_data
            WHERE ticker = :t ORDER BY download_date
        """), {"t": t}).fetchall()
    return pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])


def _get_current_feq(ticker: str) -> tuple:
    """
    Returns (current_feq, last_checked) from estimation_status for income category.
    """
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        row = conn.execute(text("""
            SELECT first_estimated_period, last_checked
            FROM estimation_status
            WHERE ticker = :t AND category = 'income'
        """), {"t": t}).fetchone()
    if row is None:
        return None, None
    return row[0], pd.Timestamp(row[1])


def _get_metric_values_at_date(ticker: str, table: str, metric: str,
                                download_date: pd.Timestamp) -> pd.Series:
    """
    Returns a Series {period -> value} for a ticker/metric at a specific download_date,
    forward-filled from prior download_dates (most recent value per period as of that date).
    """
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            WITH ranked AS (
                SELECT period, value,
                       ROW_NUMBER() OVER (PARTITION BY period ORDER BY download_date DESC) rn
                FROM {table}
                WHERE ticker = :t
                  AND metric_name = :m
                  AND download_date <= :d
                  AND value IS NOT NULL
            )
            SELECT period, value FROM ranked WHERE rn = 1
            ORDER BY period
        """), conn, params={"t": t, "m": metric, "d": download_date})
    if df.empty:
        return pd.Series(dtype=float)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.set_index('period')['value'].dropna()


def _get_metric_at_two_dates(ticker: str, table: str, metric: str,
                              date1: pd.Timestamp,
                              date2: pd.Timestamp) -> tuple:
    """
    Returns two Series {period -> value} at two specific download_dates (strict, no ffill).
    Used for FEQ verification.
    """
    t = clean_ticker(ticker)
    results = []
    for dt in [date1, date2]:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT period, value FROM {table}
                WHERE ticker = :t
                  AND metric_name = :m
                  AND download_date = :d
                  AND value IS NOT NULL
            """), conn, params={"t": t, "m": metric, "d": dt})
        if df.empty:
            results.append(pd.Series(dtype=float))
        else:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            results.append(df.set_index('period')['value'].dropna())
    return results[0], results[1]


# ==============================================================================
# FEQ MAPPING
# ==============================================================================

def resolve_past_feq(ticker: str,
                     calc_date: pd.Timestamp,
                     ticker_dl_dates: pd.DatetimeIndex) -> object:
    """
    Determines the First Estimated Quarter (FEQ) that was current as of calc_date.

    Algorithm:
    1. Find update_before = latest download_date <= calc_date
    2. Find update_after  = earliest download_date > calc_date
    3. Walk back from current FEQ using (last_checked - update_before) / 90 days
    4. Verify by checking totalRevenues changes across update_before / update_after
       for estimated_past_FEQ and ±1 quarters (4 candidates total)
    5. The most recent candidate with a changed value is the confirmed past FEQ
    6. If none changed, fall back to estimated_past_FEQ

    Returns None if update_before doesn't exist (skip this stock/date).
    """
    if ticker_dl_dates.empty:
        return None

    # Step 1 & 2: find straddling download dates
    before_mask = ticker_dl_dates[ticker_dl_dates <= calc_date]
    after_mask  = ticker_dl_dates[ticker_dl_dates > calc_date]

    if len(before_mask) == 0:
        return None   # no data yet at calc_date — skip

    update_before = before_mask[-1]
    update_after  = after_mask[0] if len(after_mask) > 0 else None

    # Step 3: estimate past FEQ by walking back from current
    current_feq, last_checked = _get_current_feq(ticker)
    if current_feq is None:
        return None

    days_delta    = (last_checked - update_before).days
    quarters_back = round(days_delta / 90)
    est_past_feq  = add_quarters(current_feq, -quarters_back)

    # Step 4: verify using totalRevenues comparison (4 candidates)
    if update_after is None:
        # No later snapshot to compare against — trust the estimate
        return est_past_feq

    # Candidates: one quarter before, est_past_feq, one after, two after
    candidates = [
        add_quarters(est_past_feq, -1),
        est_past_feq,
        add_quarters(est_past_feq, 1),
        add_quarters(est_past_feq, 2),
    ]

    s_before, s_after = _get_metric_at_two_dates(
        ticker, 'income_data', 'totalRevenues', update_before, update_after
    )

    # Find the most recent candidate where the value changed
    confirmed_feq = None
    for candidate in reversed(candidates):   # check most recent first
        v_before = s_before.get(candidate)
        v_after  = s_after.get(candidate)
        if v_before is not None and v_after is not None:
            if abs(float(v_before) - float(v_after)) > 0.01:
                confirmed_feq = candidate
                break

    return confirmed_feq if confirmed_feq is not None else est_past_feq


# ==============================================================================
# METRIC FETCHING
# ==============================================================================

def _get_ltm_values(ticker: str, table: str, metric: str,
                    feq: str, calc_date: pd.Timestamp) -> object:
    """
    Returns the last 4 actual quarters (LTM) ending at feq-1.
    feq is the first estimated quarter, so actuals are feq-4 .. feq-1.
    """
    quarters = [add_quarters(feq, -i) for i in range(1, 5)]   # feq-1 .. feq-4
    vals = _get_metric_values_at_date(ticker, table, metric, calc_date)
    result = {}
    for q in quarters:
        if q in vals.index:
            result[q] = vals[q]
    if len(result) < 4:
        return None
    return pd.Series(result)


def _get_ntm_values(ticker: str, table: str, metric: str,
                    feq: str, calc_date: pd.Timestamp) -> object:
    """
    Returns the first 4 estimated quarters (NTM) starting at feq.
    """
    quarters = [add_quarters(feq, i) for i in range(0, 4)]    # feq .. feq+3
    vals = _get_metric_values_at_date(ticker, table, metric, calc_date)
    result = {}
    for q in quarters:
        if q in vals.index:
            result[q] = vals[q]
    if len(result) < 4:
        return None
    return pd.Series(result)


def _get_prior_ltm_values(ticker: str, table: str, metric: str,
                           feq: str, calc_date: pd.Timestamp) -> object:
    """
    Returns the 4 actual quarters prior to LTM: feq-8 .. feq-5.
    Used for LTM growth calculation.
    """
    quarters = [add_quarters(feq, -i) for i in range(5, 9)]   # feq-5 .. feq-8
    vals = _get_metric_values_at_date(ticker, table, metric, calc_date)
    result = {}
    for q in quarters:
        if q in vals.index:
            result[q] = vals[q]
    if len(result) < 4:
        return None
    return pd.Series(result)


def _get_shares(ticker: str, feq: str, calc_date: pd.Timestamp) -> object:
    """
    Most recent actual dilutedAverageShares as of calc_date (feq-1 quarter).
    """
    last_actual = add_quarters(feq, -1)
    vals = _get_metric_values_at_date(ticker, 'income_data', 'dilutedAverageShares', calc_date)
    # Try last actual first, then feq-2 as fallback
    for q in [last_actual, add_quarters(feq, -2)]:
        if q in vals.index and not np.isnan(vals[q]):
            return float(vals[q])
    return None


# ==============================================================================
# SINGLE STOCK CALCULATION
# ==============================================================================

def _calc_stock_valuation(ticker: str, table: str, metric: str,
                           basis: str, feq: str,
                           calc_date: pd.Timestamp,
                           price: float) -> object:
    """
    Returns valuation = mkt_cap / metric_sum for one stock on one date.
    basis: 'LTM' or 'NTM'
    """
    shares = _get_shares(ticker, feq, calc_date)
    if shares is None or shares <= 0:
        return None

    mkt_cap = price * shares
    if mkt_cap <= 0:
        return None

    if basis == 'LTM':
        vals = _get_ltm_values(ticker, table, metric, feq, calc_date)
    else:
        vals = _get_ntm_values(ticker, table, metric, feq, calc_date)

    if vals is None:
        return None

    metric_sum = vals.sum()
    if metric_sum <= 0:
        return None

    return mkt_cap / metric_sum


def _calc_stock_growth(ticker: str, table: str, metric: str,
                       basis: str, feq: str,
                       calc_date: pd.Timestamp) -> object:
    """
    Returns symmetric YoY growth for one stock on one date.
    LTM: np.median( (last4 - prior4) / ((last4 + prior4) / 2) )  per-quarter
    NTM: np.median( (ntm4 - last4) / ((ntm4 + last4) / 2) )      per-quarter
    """
    if basis == 'LTM':
        numerator_vals   = _get_ltm_values(ticker, table, metric, feq, calc_date)
        denominator_vals = _get_prior_ltm_values(ticker, table, metric, feq, calc_date)
    else:
        numerator_vals   = _get_ntm_values(ticker, table, metric, feq, calc_date)
        denominator_vals = _get_ltm_values(ticker, table, metric, feq, calc_date)

    if numerator_vals is None or denominator_vals is None:
        return None

    # Align by position (both are sorted 4-quarter series)
    num  = numerator_vals.values
    den  = denominator_vals.values

    mid  = (num + den) / 2.0
    # Avoid division by zero
    valid = mid != 0
    if valid.sum() == 0:
        return None

    growth = (num[valid] - den[valid]) / mid[valid]
    return float(np.median(growth))


# ==============================================================================
# SECTOR AGGREGATION FOR ONE DATE
# ==============================================================================

def _calc_sector_date(metric_type: str, table: str, metric: str,
                      basis: str, calc_date: pd.Timestamp,
                      universe: list, sectors_s: pd.Series,
                      Pxs_df: pd.DataFrame,
                      ticker_dl_dates: dict) -> dict:
    """
    For a single calc_date, computes the cap-weighted sector metric.
    Returns dict {sector: value}.
    """
    sector_results = {}
    all_sectors    = sectors_s.unique()

    # Get prices at calc_date (most recent available)
    price_dates = Pxs_df.index[Pxs_df.index <= calc_date]
    if len(price_dates) == 0:
        return {}
    price_row = Pxs_df.loc[price_dates[-1]]

    for sector in sorted(all_sectors):
        sector_tickers = [t for t in universe if sectors_s.get(t) == sector]
        if not sector_tickers:
            continue

        weighted_vals = []
        weights       = []

        for ticker in sector_tickers:
            t = clean_ticker(ticker)

            # Get price
            price = price_row.get(t) or price_row.get(ticker)
            if price is None or np.isnan(price) or price <= 0:
                continue

            # Resolve FEQ for this back-date
            dl_dates = ticker_dl_dates.get(t, pd.DatetimeIndex([]))
            feq = resolve_past_feq(t, calc_date, dl_dates)
            if feq is None:
                continue

            # Get shares for cap weight
            shares = _get_shares(t, feq, calc_date)
            if shares is None or shares <= 0:
                continue
            mkt_cap = price * shares
            if mkt_cap <= 0:
                continue

            # Calculate metric
            if metric_type == 'valuation':
                val = _calc_stock_valuation(t, table, metric, basis, feq,
                                             calc_date, price)
            else:
                val = _calc_stock_growth(t, table, metric, basis, feq, calc_date)

            if val is None or np.isnan(val) or np.isinf(val):
                continue

            weighted_vals.append(val * mkt_cap)
            weights.append(mkt_cap)

        if len(weights) >= MIN_STOCKS:
            raw = sum(weighted_vals) / sum(weights)
            sector_results[sector] = round(raw * 100, 2) if metric_type == 'growth' else raw

    return sector_results


# ==============================================================================
# DB CACHE
# ==============================================================================

def _ensure_cache_table(table_name: str):
    with ENGINE.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                calc_date   DATE    NOT NULL,
                sector      TEXT    NOT NULL,
                value       NUMERIC,
                PRIMARY KEY (calc_date, sector)
            )
        """))
        conn.commit()


def _load_cached_dates_sf(table_name: str) -> set:
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                f"SELECT DISTINCT calc_date FROM {table_name}"
            )).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _save_to_cache(table_name: str, calc_date: pd.Timestamp, sector_vals: dict):
    if not sector_vals:
        return
    records = [{'calc_date': calc_date, 'sector': s, 'value': v}
               for s, v in sector_vals.items()]
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            DELETE FROM {table_name} WHERE calc_date = :d
        """), {"d": calc_date})
    pd.DataFrame(records).to_sql(table_name, ENGINE, if_exists='append', index=False)


def _load_from_cache(table_name: str) -> pd.DataFrame:
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT calc_date, sector, value FROM {table_name}
            ORDER BY calc_date, sector
        """), conn)
    if df.empty:
        return pd.DataFrame()
    df['calc_date'] = pd.to_datetime(df['calc_date'])
    df['value']     = pd.to_numeric(df['value'], errors='coerce')
    return df.pivot(index='calc_date', columns='sector', values='value')


# ==============================================================================
# PLOTTING
# ==============================================================================

def _plot_results(df: pd.DataFrame, metric_type: str, metric: str,
                  basis: str) -> plt.Figure:
    """Plot cap-weighted sector metric over time."""
    sectors = df.columns.tolist()
    colors  = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(16, 8))

    INDEX_STYLES = {
        'SPX': {'color': 'black',  'linewidth': 2.2, 'linestyle': '--', 'zorder': 5},
        'QQQ': {'color': 'dimgrey','linewidth': 2.0, 'linestyle': ':', 'zorder': 5},
    }
    sector_cols = [c for c in sorted(sectors) if c not in INDEX_STYLES]
    index_cols  = [c for c in ['SPX', 'QQQ'] if c in sectors]

    for i, sector in enumerate(sector_cols):
        if sector not in df.columns:
            continue
        s = df[sector].dropna()
        if s.empty:
            continue
        ax.plot(s.index.to_numpy(), s.values,
                label=sector,
                color=colors[i % len(colors)],
                linewidth=1.5)

    for idx_name in index_cols:
        s = df[idx_name].dropna()
        if s.empty:
            continue
        style = INDEX_STYLES[idx_name]
        ax.plot(s.index.to_numpy(), s.values, label=idx_name, **style)

    metric_labels = {
        'totalRevenues':      'Sales',
        'normalizedNetIncome': 'Net Income',
        'ebitda':              'EBITDA',
    }
    type_label  = metric_type.capitalize()
    metric_label = metric_labels.get(metric, metric)
    title = f"Cap-Weighted Sector {type_label}: {basis} {metric_label}"

    ax.set_title(title, fontsize=13, fontweight='bold')
    pct_suffix = " (%)" if metric_type == "growth" else ""
    ax.set_ylabel(f"{type_label} ({basis}){pct_suffix}", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    return fig


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run(Pxs_df: pd.DataFrame,
        sectors_s: pd.Series,
        override: bool = False,
        spx_df: pd.DataFrame = None,
        qqq_df: pd.DataFrame = None) -> tuple:
    """
    Interactive entry point for sector fundamentals analysis.

    Parameters
    ----------
    Pxs_df    : DataFrame (dates x tickers), daily prices, bare tickers
    sectors_s : Series (ticker -> sector), bare tickers
    override  : if True, recompute all dates ignoring DB cache
    spx_df    : DataFrame (dates x positions), SPX constituents over time (tickers with ' US' suffix)
    qqq_df    : DataFrame (dates x positions), QQQ constituents over time (tickers with ' US' suffix)

    Returns
    -------
    (df, fig) where:
      df  : DataFrame (calc_dates x sectors) with cap-weighted metric
      fig : matplotlib Figure
    """

    # ------------------------------------------------------------------
    # USER PROMPTS
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  SECTOR FUNDAMENTALS ANALYSIS")
    print("="*60)

    # Prompt 1: metric type
    while True:
        ans = input("\n  Metric type — valuation (v) or growth (g)? ").strip().lower()
        if ans in ('v', 'valuation'):
            metric_type = 'valuation'
            break
        elif ans in ('g', 'growth'):
            metric_type = 'growth'
            break
        print("  Please enter 'v' or 'g'.")

    # Prompt 2: metric
    print("\n  Available metrics:")
    print("    s  = Sales  (totalRevenues)")
    print("    ni = Net Income (normalizedNetIncome)")
    print("    e  = EBITDA")
    while True:
        ans = input("\n  Select metric (s / ni / e): ").strip().lower()
        if ans == 's':
            metric = 'totalRevenues'
            table  = 'income_data'
            break
        elif ans == 'ni':
            metric = 'normalizedNetIncome'
            table  = 'income_data'
            break
        elif ans == 'e':
            metric = 'ebitda'
            table  = 'summary_data'
            break
        print("  Please enter 's', 'ni', or 'e'.")

    # Prompt 3: LTM or NTM
    while True:
        ans = input("\n  Basis — LTM or NTM? ").strip().upper()
        if ans in ('LTM', 'NTM'):
            basis = ans
            break
        print("  Please enter 'LTM' or 'NTM'.")

    # Prompt 4: lookback period
    while True:
        ans = input("\n  Lookback period in years (e.g. 5): ").strip()
        try:
            n_years = int(ans)
            if n_years > 0:
                break
        except ValueError:
            pass
        print("  Please enter a positive integer.")

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    # Cache table encodes metric_type + metric + basis to avoid cross-contamination
    metric_short = {'totalRevenues': 'sales', 'normalizedNetIncome': 'ni', 'ebitda': 'ebitda'}
    cache_table  = f"sector_{metric_type}_{metric_short[metric]}_{basis.lower()}"
    _ensure_cache_table(cache_table)

    last_date  = Pxs_df.index[-1]
    start_date = last_date - relativedelta(years=n_years)

    # Build calculation dates: 1st of each month + last available price date
    calc_dates = []
    d = start_date.replace(day=1)
    while d <= last_date:
        if d >= Pxs_df.index[0]:
            calc_dates.append(pd.Timestamp(d))
        d += relativedelta(months=1)
    calc_dates.append(pd.Timestamp(last_date))
    calc_dates = sorted(set(calc_dates))

    print(f"\n  Metric type : {metric_type}")
    print(f"  Metric      : {metric}")
    print(f"  Basis       : {basis}")
    print(f"  Period      : {start_date.date()} → {last_date.date()}")
    print(f"  Calc dates  : {len(calc_dates)}")
    print(f"  Cache table : {cache_table}")

    # ------------------------------------------------------------------
    # DETERMINE WHICH DATES TO COMPUTE
    # ------------------------------------------------------------------
    if override:
        print("\n  override=True: recomputing all dates")
        dates_to_compute = calc_dates
    else:
        cached = _load_cached_dates_sf(cache_table)
        dates_to_compute = [d for d in calc_dates if d not in cached]
        print(f"\n  {len(cached)} dates already cached, "
              f"{len(dates_to_compute)} to compute")

    # ------------------------------------------------------------------
    # PRE-LOAD DOWNLOAD DATES PER TICKER (avoid N×M DB hits)
    # ------------------------------------------------------------------
    universe = [clean_ticker(t) for t in sectors_s.index
                if clean_ticker(t) in [clean_ticker(c) for c in Pxs_df.columns]]
    # Remap Pxs_df columns to bare tickers
    Pxs_bare = Pxs_df.copy()
    Pxs_bare.columns = [clean_ticker(c) for c in Pxs_df.columns]
    sectors_bare = pd.Series(
        sectors_s.values,
        index=[clean_ticker(t) for t in sectors_s.index]
    )
    universe = [t for t in universe if t in Pxs_bare.columns]

    print(f"  Universe    : {len(universe)} stocks")

    if dates_to_compute:
        print("\n  Pre-loading download dates per ticker (bulk query)...")
        with ENGINE.connect() as conn:
            dl_df = pd.read_sql(text(
                "SELECT DISTINCT ticker, download_date FROM income_data ORDER BY ticker, download_date"
            ), conn)
        dl_df['download_date'] = pd.to_datetime(dl_df['download_date'])
        dl_df['ticker'] = dl_df['ticker'].apply(clean_ticker)
        ticker_dl_dates = {
            t: pd.DatetimeIndex(grp['download_date'].values)
            for t, grp in dl_df.groupby('ticker')
        }
        for t in universe:
            if t not in ticker_dl_dates:
                ticker_dl_dates[t] = pd.DatetimeIndex([])
        print(f"  Done — {len(dl_df)} records across {len(ticker_dl_dates)} tickers")

    # ------------------------------------------------------------------
    # COMPUTE MISSING DATES
    # ------------------------------------------------------------------
    if dates_to_compute:
        print(f"\n  Computing {len(dates_to_compute)} dates...\n")
        n_total = len(dates_to_compute)

        for i, calc_date in enumerate(dates_to_compute, 1):
            print(f"  [{i:>4}/{n_total}] {calc_date.date()}")
            all_sectors = sorted(sectors_bare.unique())

            sector_vals = {}
            for sector in all_sectors:
                sector_tickers = [t for t in universe
                                  if sectors_bare.get(t) == sector]
                if not sector_tickers:
                    continue

                print(f"           └─ {sector} ({len(sector_tickers)} stocks)", end='\r')

                # Get prices at calc_date
                price_dates = Pxs_bare.index[Pxs_bare.index <= calc_date]
                if len(price_dates) == 0:
                    continue
                price_row = Pxs_bare.loc[price_dates[-1]]

                weighted_vals = []
                weights       = []

                for t in sector_tickers:
                    try:
                        price = price_row.get(t)
                        if price is None or np.isnan(price) or price <= 0:
                            continue

                        dl_dates = ticker_dl_dates.get(t, pd.DatetimeIndex([]))
                        feq = resolve_past_feq(t, calc_date, dl_dates)
                        if feq is None:
                            continue

                        shares = _get_shares(t, feq, calc_date)
                        if shares is None or shares <= 0:
                            continue
                        mkt_cap = price * shares
                        if mkt_cap <= 0:
                            continue

                        if metric_type == 'valuation':
                            val = _calc_stock_valuation(t, table, metric, basis,
                                                         feq, calc_date, price)
                        else:
                            val = _calc_stock_growth(t, table, metric, basis,
                                                      feq, calc_date)

                        if val is None or np.isnan(val) or np.isinf(val):
                            continue

                        weighted_vals.append(val * mkt_cap)
                        weights.append(mkt_cap)
                    except Exception:
                        continue   # silently skip stocks with any data issue

                if len(weights) >= MIN_STOCKS:
                    raw = sum(weighted_vals) / sum(weights)
                    sector_vals[sector] = round(raw * 100, 2) if metric_type == 'growth' else raw

            # Compute SPX and QQQ index metrics
            index_vals = {}
            for idx_name, idx_df in [('SPX', spx_df), ('QQQ', qqq_df)]:
                if idx_df is None or idx_df.empty:
                    continue
                # Find most recent constituent list before calc_date
                idx_dates = idx_df.index[idx_df.index <= calc_date]
                if len(idx_dates) == 0:
                    continue
                constituents_raw = idx_df.loc[idx_dates[-1]].dropna().tolist()
                constituents = [str(t).split(' ')[0].strip().upper()
                                for t in constituents_raw]
                constituents = [t for t in constituents if t in Pxs_bare.columns]

                price_dates = Pxs_bare.index[Pxs_bare.index <= calc_date]
                if len(price_dates) == 0:
                    continue
                price_row = Pxs_bare.loc[price_dates[-1]]

                w_vals, w_caps = [], []
                for t in constituents:
                    try:
                        price = price_row.get(t)
                        if price is None or np.isnan(price) or price <= 0:
                            continue
                        dl_dates = ticker_dl_dates.get(t, pd.DatetimeIndex([]))
                        feq = resolve_past_feq(t, calc_date, dl_dates)
                        if feq is None:
                            continue
                        shares = _get_shares(t, feq, calc_date)
                        if shares is None or shares <= 0:
                            continue
                        mkt_cap = price * shares
                        if mkt_cap <= 0:
                            continue
                        if metric_type == 'valuation':
                            val = _calc_stock_valuation(t, table, metric, basis,
                                                         feq, calc_date, price)
                        else:
                            val = _calc_stock_growth(t, table, metric, basis,
                                                      feq, calc_date)
                        if val is None or np.isnan(val) or np.isinf(val):
                            continue
                        w_vals.append(val * mkt_cap)
                        w_caps.append(mkt_cap)
                    except Exception:
                        continue   # silently skip stocks with any data issue

                if len(w_caps) >= MIN_STOCKS:
                    raw = sum(w_vals) / sum(w_caps)
                    index_vals[idx_name] = round(raw * 100, 2) if metric_type == 'growth' else raw

            # Merge sector and index results before saving
            all_vals = {**sector_vals, **index_vals}

            print(f"  [{i:>4}/{n_total}] {calc_date.date()} "
                  f"— {len(sector_vals)} sectors, {len(index_vals)} indexes computed" + " " * 30)

            _save_to_cache(cache_table, calc_date, all_vals)

    # ------------------------------------------------------------------
    # LOAD FULL RESULTS FROM CACHE
    # ------------------------------------------------------------------
    full_df = _load_from_cache(cache_table)

    # Filter to the requested date range and calc_dates only
    if not full_df.empty:
        full_df = full_df[full_df.index.isin(calc_dates)]
        full_df = full_df.sort_index()

    if full_df.empty:
        print("\n  WARNING: no data to display.")
        return pd.DataFrame(), None

    # ------------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------------
    fig = _plot_results(full_df, metric_type, metric, basis)
    plt.show()

    print(f"\n  Done. DataFrame shape: {full_df.shape}")
    sector_cols = [c for c in full_df.columns if c not in ('SPX', 'QQQ')]
    index_cols  = [c for c in ('SPX', 'QQQ') if c in full_df.columns]
    print(f"  Sectors : {sector_cols}")
    if index_cols:
        print(f"  Indexes : {index_cols}")

    return full_df, fig


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    print("Usage:")
    print("    from sector_fundamentals import run")
    print("    df, fig = run(Pxs_df, sectors_s)")
    print("    df, fig = run(Pxs_df, sectors_s, override=True)  # recompute all")
