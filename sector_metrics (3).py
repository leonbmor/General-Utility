#!/usr/bin/env python
# coding: utf-8

"""
sector_metrics.py
=================
Cap-weighted sector and index aggregation of pre-computed valuation metrics
from valuation_metrics_anchors or valuation_consolidated tables.

For each snapshot date in the table, computes the cap-weighted average
of the selected metric across all stocks in each sector and index.

The last calculation date uses updated market caps:
    mkt_cap_updated = mkt_cap_last_snapshot × (Px_today / Px_last_snapshot)

Results cached in DB table: sector_metric_{metric_name}
Re-runs only compute missing dates by default (override=False).

Usage:
    from sector_metrics import run
    df_sectors, df_indexes, fig = run(Pxs_df, sectors_s)
    df_sectors, df_indexes, fig = run(Pxs_df, sectors_s, spx_df=spx_df, qqq_df=qqq_df)
    df_sectors, df_indexes, fig = run(Pxs_df, sectors_s, override=True)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================

ENGINE         = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")
DEFAULT_TABLE  = 'valuation_metrics_anchors'   # switch to 'valuation_consolidated' if needed
MIN_STOCKS     = 3                             # minimum stocks for a valid sector/index value
WINSOR_BOUNDS  = (0.02, 0.98)                  # cross-sectional winsorization percentiles
VALUATION_MULTIPLES = {                        # use denominator-aggregation (Option B) for these
    'P/S', 'P/Ee', 'P/Eo', 'sP/S', 'sP/E', 'sP/GP', 'P/GP'
}
FILTER_TS_OUTLIERS   = True   # apply time-series outlier filter to output DataFrames
TS_JUMP_THRESHOLD    = 0.25   # flag dates where |pct_change| exceeds this

AVAILABLE_METRICS = [
    'P/S', 'P/Ee', 'P/Eo', 'OM-t0', 'OM', 'OMd', 'GS', 'GE', 'r2 S', 'r2 E',
    'GGP', 'r2 GP', 'Size', 'ROI-P', 'ROI', 'ROId', 'ROE-P', 'ROE', 'ROEd',
    'sP/S', 'sP/E', 'sP/GP', 'P/GP', 'S Vol', 'E Vol', 'GP Vol', 'r&d',
    'HSG', 'SGD', 'LastSGD', 'PIG', 'PSG', 'ISGD', 'FCF_PG',
]


# ==============================================================================
# HELPERS
# ==============================================================================

def clean_ticker(t: str) -> str:
    return str(t).split(' ')[0].strip().upper()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def _load_valuation_table(source_table: str) -> pd.DataFrame:
    """
    Load full valuation table from DB.
    Returns DataFrame with columns: date, ticker, + all metric columns.
    Tickers are cleaned (bare, no ' US').
    """
    print(f"  Loading '{source_table}' from DB...")
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {source_table} ORDER BY date, ticker"), conn)

    if df.empty:
        print("  WARNING: table is empty")
        return pd.DataFrame()

    # Standardise date and ticker columns
    date_col   = next((c for c in df.columns if c.lower() == 'date'), None)
    ticker_col = next((c for c in df.columns if c.lower() == 'ticker'), None)

    if date_col is None or ticker_col is None:
        print(f"  ERROR: could not find date/ticker columns. Columns: {df.columns.tolist()}")
        return pd.DataFrame()

    df = df.rename(columns={date_col: 'date', ticker_col: 'ticker'})
    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].apply(clean_ticker)

    # Coerce all metric columns to numeric
    metric_cols = [c for c in df.columns if c not in ('date', 'ticker')]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  Loaded: {len(df['date'].unique())} dates × {len(df['ticker'].unique())} tickers")
    print(f"  Metric columns: {metric_cols}")
    return df


# ==============================================================================
# CAP-WEIGHTED AGGREGATION
# ==============================================================================

def _winsorize(s: pd.Series,
               bounds: tuple = WINSOR_BOUNDS) -> pd.Series:
    """Winsorize a Series at given percentile bounds (cross-sectional)."""
    lo = s.quantile(bounds[0])
    hi = s.quantile(bounds[1])
    return s.clip(lower=lo, upper=hi)


def _cap_weighted_avg(group_df: pd.DataFrame, metric: str) -> object:
    """
    Cap-weighted aggregation of metric within a group (sector or index).
    Weight = Size (market cap in $M).

    For VALUATION_MULTIPLES (P/S, P/E etc.): uses denominator aggregation
    (Option B) to correctly handle negative earnings/sales:
        sector_multiple = Σ(Size_i) / Σ(Size_i / multiple_i)
    Stocks with zero multiple are excluded; negative denominators are kept
    (they reduce aggregate earnings, correctly pushing the multiple higher).

    For all other metrics: cap-weighted arithmetic mean after winsorization.

    Returns None if fewer than MIN_STOCKS valid rows.
    """
    valid = group_df[['Size', metric]].dropna()
    valid = valid[valid['Size'] > 0]
    if len(valid) < MIN_STOCKS:
        return None

    if metric in VALUATION_MULTIPLES:
        # Option B: aggregate in denominator space
        # denominator_i = Size_i / multiple_i  (implied earnings/sales/GP)
        # Exclude zero multiples to avoid division by zero
        valid = valid[valid[metric] != 0]
        if len(valid) < MIN_STOCKS:
            return None
        implied_denom = valid['Size'] / valid[metric]
        total_size    = valid['Size'].sum()
        total_denom   = implied_denom.sum()
        if total_denom == 0 or np.isnan(total_denom):
            return None
        return float(total_size / total_denom)
    else:
        # Arithmetic cap-weighted mean with winsorization
        values = _winsorize(valid[metric])
        weights = valid['Size'].values
        return float(np.dot(weights, values.values) / weights.sum())


def _compute_all_dates(val_df: pd.DataFrame,
                       metric: str,
                       sectors_bare: pd.Series,
                       spx_constituents: dict,
                       qqq_constituents: dict,
                       Pxs_bare: pd.DataFrame) -> tuple:
    """
    For each snapshot date in val_df, compute cap-weighted metric per sector and index.

    For the last date, update market caps using:
        mkt_cap_updated = Size_last_snapshot × (Px_today / Px_last_snapshot)

    Returns:
        sector_records : list of {date, sector, value}
        index_records  : list of {date, index, value}
    """
    dates = sorted(val_df['date'].unique())
    if not dates:
        return [], []

    last_snap_date = dates[-1]

    # Compute updated mkt caps for the last snapshot date using latest prices
    updated_size = None
    if not Pxs_bare.empty:
        # Last price date available
        last_px_date   = Pxs_bare.index[-1]
        # Price on last snapshot date (or nearest before)
        snap_px_dates  = Pxs_bare.index[Pxs_bare.index <= last_snap_date]
        if len(snap_px_dates) > 0:
            snap_px_row  = Pxs_bare.loc[snap_px_dates[-1]]
            last_px_row  = Pxs_bare.loc[last_px_date]
            # Size from last snapshot
            snap_size    = val_df[val_df['date'] == last_snap_date].set_index('ticker')['Size']
            # Price ratio per ticker
            px_ratio     = (last_px_row / snap_px_row).replace([np.inf, -np.inf], np.nan)
            updated_size = (snap_size * px_ratio).dropna()
            updated_size = updated_size[updated_size > 0]

    sector_records = []
    index_records  = []
    n = len(dates)

    for i, dt in enumerate(dates):
        print(f"  [{i+1:>4}/{n}] {pd.Timestamp(dt).date()}", end='\r')
        day_df = val_df[val_df['date'] == dt].copy()

        # For the last snapshot date, replace Size with updated mkt caps
        is_last = (dt == last_snap_date)
        if is_last and updated_size is not None:
            day_df = day_df.set_index('ticker')
            day_df.loc[day_df.index.isin(updated_size.index), 'Size'] = \
                updated_size.reindex(day_df.index)
            day_df = day_df.reset_index()

        # --- Sectors ---
        for sector in sectors_bare.unique():
            tickers = sectors_bare[sectors_bare == sector].index.tolist()
            grp     = day_df[day_df['ticker'].isin(tickers)]
            val     = _cap_weighted_avg(grp, metric)
            if val is not None:
                sector_records.append({'date': dt, 'sector': sector, 'value': val})

        # --- Indexes ---
        for idx_name, constituents_by_date in [('SPX', spx_constituents),
                                                ('QQQ', qqq_constituents)]:
            if not constituents_by_date:
                continue
            # Most recent constituent list on or before this date
            c_dates = [d for d in constituents_by_date if d <= dt]
            if not c_dates:
                continue
            tickers = constituents_by_date[max(c_dates)]
            grp     = day_df[day_df['ticker'].isin(tickers)]
            val     = _cap_weighted_avg(grp, metric)
            if val is not None:
                index_records.append({'date': dt, 'index': idx_name, 'value': val})

    print()  # newline after progress
    return sector_records, index_records


# ==============================================================================
# DB CACHE
# ==============================================================================

def _ensure_cache_tables(metric_tag: str):
    sec_tbl = f"sector_metric_{metric_tag}"
    idx_tbl = f"index_metric_{metric_tag}"
    with ENGINE.connect() as conn:
        for tbl, col in [(sec_tbl, 'sector'), (idx_tbl, 'index_name')]:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                    calc_date  DATE NOT NULL,
                    {col}      TEXT NOT NULL,
                    value      NUMERIC,
                    PRIMARY KEY (calc_date, {col})
                )
            """))
        conn.commit()
    return sec_tbl, idx_tbl


def _load_cached_dates_sm(table_name: str) -> set:
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                f"SELECT DISTINCT calc_date FROM {table_name}"
            )).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _save_cache(sec_tbl: str, idx_tbl: str,
                sector_records: list, index_records: list,
                dates_to_save: set):
    """Save only records belonging to dates_to_save."""
    if sector_records:
        df = pd.DataFrame(sector_records)
        df['calc_date'] = pd.to_datetime(df['date']).dt.date
        df = df[pd.to_datetime(df['date']).isin(dates_to_save)]
        if not df.empty:
            with ENGINE.begin() as conn:
                for dt in df['calc_date'].unique():
                    conn.execute(text(
                        f"DELETE FROM {sec_tbl} WHERE calc_date = :d"
                    ), {"d": dt})
            df[['calc_date', 'sector', 'value']].to_sql(
                sec_tbl, ENGINE, if_exists='append', index=False)

    if index_records:
        df = pd.DataFrame(index_records)
        df['calc_date'] = pd.to_datetime(df['date']).dt.date
        df = df[pd.to_datetime(df['date']).isin(dates_to_save)]
        if not df.empty:
            df = df.rename(columns={'index': 'index_name'})
            with ENGINE.begin() as conn:
                for dt in df['calc_date'].unique():
                    conn.execute(text(
                        f"DELETE FROM {idx_tbl} WHERE calc_date = :d"
                    ), {"d": dt})
            df[['calc_date', 'index_name', 'value']].to_sql(
                idx_tbl, ENGINE, if_exists='append', index=False)


def _load_from_cache(sec_tbl: str, idx_tbl: str) -> tuple:
    """Load full cached results as two pivot DataFrames."""
    def _pivot(tbl, group_col):
        try:
            with ENGINE.connect() as conn:
                df = pd.read_sql(text(
                    f"SELECT calc_date, {group_col}, value FROM {tbl} ORDER BY calc_date"
                ), conn)
            if df.empty:
                return pd.DataFrame()
            df['calc_date'] = pd.to_datetime(df['calc_date'])
            df['value']     = pd.to_numeric(df['value'], errors='coerce')
            return df.pivot(index='calc_date', columns=group_col, values='value')
        except Exception:
            return pd.DataFrame()

    return _pivot(sec_tbl, 'sector'), _pivot(idx_tbl, 'index_name')


# ==============================================================================
# PLOTTING
# ==============================================================================

def _filter_ts_outliers(df: pd.DataFrame,
                        jump_threshold: float = TS_JUMP_THRESHOLD) -> pd.DataFrame:
    """
    For each column, identify dates where |pct_change| > jump_threshold.
    If the jump is at least partially reversed on the next date (i.e. the
    absolute deviation from the pre-jump level shrinks), replace the flagged
    date with the average of its two neighbours.

    Applied iteratively so that consecutive bad dates are each cleaned in turn.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 3:
            continue
        idx = s.index.tolist()
        i = 1
        while i < len(idx) - 1:
            v_prev = s.loc[idx[i - 1]]
            v_curr = s.loc[idx[i]]
            v_next = s.loc[idx[i + 1]]
            if v_prev == 0 or np.isnan(v_prev) or np.isnan(v_curr) or np.isnan(v_next):
                i += 1
                continue
            jump = (v_curr - v_prev) / abs(v_prev)
            if abs(jump) <= jump_threshold:
                i += 1
                continue
            # Check if next date is any closer to v_prev than v_curr is
            if abs(v_next - v_prev) < abs(v_curr - v_prev):
                df.loc[idx[i], col] = (v_prev + v_next) / 2.0
                s = df[col].dropna()   # refresh so next iteration sees corrected value
                idx = s.index.tolist()
                # Don't advance i — re-check from same position with updated value
                continue
            i += 1
    return df


def _plot_results(df_sectors: pd.DataFrame,
                  df_indexes: pd.DataFrame,
                  metric: str) -> plt.Figure:
    """Two-panel figure: all sectors (top), SPX + QQQ (bottom)."""
    colors = plt.cm.tab20.colors

    has_sectors = not df_sectors.empty
    has_indexes = not df_indexes.empty
    n_panels    = int(has_sectors) + int(has_indexes)

    if n_panels == 0:
        print("  No data to plot.")
        return None

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(16, 6 * n_panels),
                             sharex=(n_panels > 1))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"Cap-Weighted Sector & Index Metric: {metric}",
                 fontsize=13, fontweight='bold')

    panel = 0

    # --- Sectors panel ---
    if has_sectors:
        ax = axes[panel]
        for i, col in enumerate(sorted(df_sectors.columns)):
            s = df_sectors[col].dropna()
            if s.empty:
                continue
            ax.plot(s.index.to_numpy(), s.values,
                    label=col,
                    color=colors[i % len(colors)],
                    linewidth=1.5)
        ax.set_title(f"Sectors — {metric}", fontsize=11, fontweight='bold')
        ax.set_ylabel(metric, fontsize=9)
        ax.legend(fontsize=8, ncol=4, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        panel += 1

    # --- Indexes panel ---
    if has_indexes:
        ax = axes[panel]
        INDEX_STYLES = {
            'SPX': {'color': 'black',   'linewidth': 2.2, 'linestyle': '-'},
            'QQQ': {'color': 'dimgrey', 'linewidth': 2.0, 'linestyle': '--'},
        }
        for col in df_indexes.columns:
            s = df_indexes[col].dropna()
            if s.empty:
                continue
            style = INDEX_STYLES.get(col, {'color': 'steelblue', 'linewidth': 1.5})
            ax.plot(s.index.to_numpy(), s.values, label=col, **style)
        ax.set_title(f"Indexes — {metric}", fontsize=11, fontweight='bold')
        ax.set_ylabel(metric, fontsize=9)
        ax.legend(fontsize=9, loc='best')
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
        spx_df: pd.DataFrame = None,
        qqq_df: pd.DataFrame = None,
        override: bool = False,
        source_table: str = DEFAULT_TABLE) -> tuple:
    """
    Cap-weighted sector and index aggregation of valuation metrics.

    Parameters
    ----------
    Pxs_df       : DataFrame (dates x tickers), daily prices, bare tickers
    sectors_s    : Series (ticker -> sector), bare tickers
    spx_df       : DataFrame (dates x positions), SPX constituents (' US' suffix ok)
    qqq_df       : DataFrame (dates x positions), QQQ constituents (' US' suffix ok)
    override     : if True, recompute all dates ignoring DB cache
    source_table : valuation table to use (default: 'valuation_metrics_anchors')

    Returns
    -------
    (df_sectors, df_indexes, fig)
      df_sectors : DataFrame (dates x sectors)
      df_indexes : DataFrame (dates x indexes)
      fig        : matplotlib Figure
    """

    # ------------------------------------------------------------------
    # LOAD VALUATION TABLE AND CHECK COLUMNS
    # ------------------------------------------------------------------
    val_df = _load_valuation_table(source_table)
    if val_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    available = [c for c in AVAILABLE_METRICS if c in val_df.columns]
    extra     = [c for c in val_df.columns if c not in ('date', 'ticker')
                 and c not in AVAILABLE_METRICS]
    if extra:
        available = available + extra   # include any new columns not in hardcoded list

    # ------------------------------------------------------------------
    # METRIC SELECTION PROMPT
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  AVAILABLE METRICS")
    print("="*60)
    for i, m in enumerate(available, 1):
        print(f"  {i:>3}.  {m}")
    print("="*60)

    while True:
        ans = input(f"\n  Select metric (1-{len(available)}): ").strip()
        try:
            idx = int(ans) - 1
            if 0 <= idx < len(available):
                metric = available[idx]
                break
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(available)}.")

    print(f"\n  Selected: {metric}")

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    # Sanitise metric name for use as DB table suffix
    metric_tag = metric.replace('/', '_').replace(' ', '_').replace('-', '_').lower()
    sec_tbl, idx_tbl = _ensure_cache_tables(metric_tag)

    # Bare-ticker versions of inputs
    Pxs_bare     = Pxs_df.copy()
    Pxs_bare.columns = [clean_ticker(c) for c in Pxs_df.columns]
    sectors_bare = pd.Series(sectors_s.values,
                             index=[clean_ticker(t) for t in sectors_s.index])

    # Parse index constituent DataFrames into {date -> [tickers]} dicts
    def _parse_constituents(df):
        if df is None or df.empty:
            return {}
        result = {}
        for dt in df.index:
            tickers = [clean_ticker(str(t)) for t in df.loc[dt].dropna().tolist()]
            result[pd.Timestamp(dt)] = tickers
        return result

    spx_constituents = _parse_constituents(spx_df)
    qqq_constituents = _parse_constituents(qqq_df)

    # ------------------------------------------------------------------
    # DETERMINE DATES TO COMPUTE
    # ------------------------------------------------------------------
    all_snap_dates = set(val_df['date'].unique())

    if override:
        print("\n  override=True: recomputing all dates")
        dates_to_compute = all_snap_dates
    else:
        cached_sec = _load_cached_dates_sm(sec_tbl)
        cached_idx = _load_cached_dates_sm(idx_tbl)
        cached     = cached_sec & cached_idx if (spx_constituents or qqq_constituents) \
                     else cached_sec
        dates_to_compute = all_snap_dates - cached
        print(f"\n  {len(cached)} dates already cached, "
              f"{len(dates_to_compute)} to compute")

    # ------------------------------------------------------------------
    # COMPUTE MISSING DATES
    # ------------------------------------------------------------------
    if dates_to_compute:
        # Filter val_df to only dates that need computing
        val_to_compute = val_df[val_df['date'].isin(dates_to_compute)]

        print(f"  Computing {len(dates_to_compute)} dates for metric '{metric}'...\n")
        sector_records, index_records = _compute_all_dates(
            val_df         = val_to_compute,
            metric         = metric,
            sectors_bare   = sectors_bare,
            spx_constituents = spx_constituents,
            qqq_constituents = qqq_constituents,
            Pxs_bare       = Pxs_bare,
        )

        print(f"  Saving {len(sector_records)} sector rows, "
              f"{len(index_records)} index rows to DB...")
        _save_cache(sec_tbl, idx_tbl, sector_records, index_records,
                    {pd.Timestamp(d) for d in dates_to_compute})

    # ------------------------------------------------------------------
    # LOAD FULL RESULTS
    # ------------------------------------------------------------------
    df_sectors, df_indexes = _load_from_cache(sec_tbl, idx_tbl)
    df_sectors = df_sectors.sort_index() if not df_sectors.empty else df_sectors
    df_indexes = df_indexes.sort_index() if not df_indexes.empty else df_indexes

    if df_sectors.empty and df_indexes.empty:
        print("\n  WARNING: no data to display.")
        return pd.DataFrame(), pd.DataFrame(), None

    # ------------------------------------------------------------------
    # TIME-SERIES OUTLIER FILTER  (controlled by FILTER_TS_OUTLIERS)
    # ------------------------------------------------------------------
    if FILTER_TS_OUTLIERS:
        if not df_sectors.empty:
            df_sectors = _filter_ts_outliers(df_sectors)
        if not df_indexes.empty:
            df_indexes = _filter_ts_outliers(df_indexes)

    # ------------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------------
    fig = _plot_results(df_sectors, df_indexes, metric)
    if fig:
        plt.show()

    # Summary
    print(f"\n  Done.")
    print(f"  Sectors df : {df_sectors.shape} — {list(df_sectors.columns)}")
    if not df_indexes.empty:
        print(f"  Indexes df : {df_indexes.shape} — {list(df_indexes.columns)}")

    return df_sectors, df_indexes, fig


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    print("Usage:")
    print("    from sector_metrics import run")
    print("    df_sectors, df_indexes, fig = run(Pxs_df, sectors_s)")
    print("    df_sectors, df_indexes, fig = run(Pxs_df, sectors_s,")
    print("                                      spx_df=spx_df, qqq_df=qqq_df)")
    print("    df_sectors, df_indexes, fig = run(Pxs_df, sectors_s, override=True)")
    print("    # Switch source table:")
    print("    df_sectors, df_indexes, fig = run(Pxs_df, sectors_s,")
    print("                                      source_table='valuation_consolidated')")
