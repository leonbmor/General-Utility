#!/usr/bin/env python
# coding: utf-8

"""
load_sector_fundamentals.py
============================
Scans the DB for all cached tables produced by sector_fundamentals.py
(sector_valuation_* and sector_growth_*), loads them all, and returns
a library of DataFrames ready for use in other notebook cells.

Table naming convention from sector_fundamentals.py:
    sector_{metric_type}_{metric_short}_{basis}
    e.g. sector_valuation_sales_ltm
         sector_growth_ni_ntm

Usage:
    from load_sector_fundamentals import load_all
    lib = load_all()

    # Access by type / metric / basis:
    lib['valuation']['sales']['ltm']    # DataFrame (dates x sectors)
    lib['growth']['ni']['ntm']          # DataFrame (dates x sectors)

    # Flat access by full table tag:
    lib['flat']['valuation_sales_ltm']  # same DataFrame

    # See what's available:
    lib['available']                    # list of (metric_type, metric, basis) tuples
"""

import pandas as pd
from sqlalchemy import create_engine, text, inspect

ENGINE = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")

METRIC_TYPES = ('valuation', 'growth')
METRICS      = ('sales', 'ni', 'ebitda')
BASES        = ('ltm', 'ntm')


def _get_fundamentals_tables() -> dict:
    """
    Scan DB for all sector_valuation_* and sector_growth_* tables.
    Returns dict: {full_tag -> table_name}
    e.g. {'valuation_sales_ltm': 'sector_valuation_sales_ltm', ...}
    """
    inspector  = inspect(ENGINE)
    all_tables = inspector.get_table_names()

    found = {}
    for tbl in all_tables:
        if tbl.startswith('sector_valuation_') or tbl.startswith('sector_growth_'):
            tag = tbl[len('sector_'):]   # strip 'sector_' prefix → e.g. 'valuation_sales_ltm'
            found[tag] = tbl

    return found


def _load_pivot(table_name: str) -> pd.DataFrame:
    """Load a cached fundamentals table and pivot to (dates x sectors)."""
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(
                f"SELECT calc_date, sector, value FROM {table_name} ORDER BY calc_date"
            ), conn)
        if df.empty:
            return pd.DataFrame()
        df['calc_date'] = pd.to_datetime(df['calc_date'])
        df['value']     = pd.to_numeric(df['value'], errors='coerce')
        return df.pivot(index='calc_date', columns='sector', values='value').sort_index()
    except Exception as e:
        print(f"  WARNING: could not load '{table_name}': {e}")
        return pd.DataFrame()


def load_all(verbose: bool = True) -> dict:
    """
    Load all cached sector fundamentals DataFrames from DB.

    Returns
    -------
    dict with keys:
      'valuation' : dict { metric_short -> { basis -> DataFrame } }
      'growth'    : dict { metric_short -> { basis -> DataFrame } }
      'flat'      : dict { full_tag -> DataFrame }  (e.g. 'valuation_sales_ltm')
      'available' : list of full tags found
    """
    found = _get_fundamentals_tables()

    if not found:
        print("  No cached sector fundamentals tables found in DB.")
        return {'valuation': {}, 'growth': {}, 'flat': {}, 'available': []}

    # Initialise nested structure
    lib = {
        'valuation': {m: {} for m in METRICS},
        'growth':    {m: {} for m in METRICS},
        'flat':      {},
        'available': [],
    }

    if verbose:
        print(f"\n  Found {len(found)} cached table(s) in DB:\n")
        print(f"  {'Tag':<35} {'Table':<45} {'Dates':>6} {'Sectors':>8}")
        print(f"  {'-'*98}")

    for tag in sorted(found):
        tbl = found[tag]
        df  = _load_pivot(tbl)

        lib['flat'][tag] = df
        lib['available'].append(tag)

        # Parse tag into (metric_type, metric_short, basis)
        # tag format: e.g. 'valuation_sales_ltm' or 'growth_ni_ntm'
        parts = tag.split('_')
        if len(parts) >= 3:
            mtype  = parts[0]                    # 'valuation' or 'growth'
            basis  = parts[-1]                   # 'ltm' or 'ntm'
            metric = '_'.join(parts[1:-1])       # 'sales', 'ni', 'ebitda'
            if mtype in lib and metric in lib[mtype]:
                lib[mtype][metric][basis] = df

        if verbose:
            n_dates   = len(df) if not df.empty else 0
            n_sectors = len(df.columns) if not df.empty else 0
            print(f"  {tag:<35} {tbl:<45} {n_dates:>6} {n_sectors:>8}")

    if verbose:
        print(f"\n  Loaded {len(found)} DataFrame(s) total.")
        print(f"\n  Access examples:")
        if lib['available']:
            ex = lib['available'][0]
            parts = ex.split('_')
            if len(parts) >= 3:
                mtype  = parts[0]
                basis  = parts[-1]
                metric = '_'.join(parts[1:-1])
                print(f"    lib['{mtype}']['{metric}']['{basis}']")
        print(f"    lib['flat']['{lib['available'][0]}']" if lib['available'] else "")
        print(f"    lib['available']  →  {lib['available']}")

    return lib
