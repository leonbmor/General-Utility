#!/usr/bin/env python
# coding: utf-8

"""
load_sector_metrics.py
======================
Scans the DB for all cached sector_metric_* and index_metric_* tables
produced by sector_metrics.py, loads them all, and returns a dict of
DataFrames ready for use in other notebook cells.

Usage:
    from load_sector_metrics import load_all
    lib = load_all()

    # Access by metric name:
    lib['sector']['p_s']          # sector P/S DataFrame
    lib['index']['p_s']           # index P/S DataFrame

    # See what's available:
    lib['available']              # list of metric tags found in DB

    # Quick summary:
    load_all(verbose=True)
"""

import pandas as pd
from sqlalchemy import create_engine, text, inspect

ENGINE = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")


def _get_metric_tables() -> dict:
    """
    Scan DB for all sector_metric_* and index_metric_* tables.
    Returns dict: {metric_tag: {'sector': table_name, 'index': table_name}}
    where either key may be None if that table doesn't exist.
    """
    inspector  = inspect(ENGINE)
    all_tables = inspector.get_table_names()

    tags = {}
    for tbl in all_tables:
        if tbl.startswith('sector_metric_'):
            tag = tbl[len('sector_metric_'):]
            tags.setdefault(tag, {})['sector'] = tbl
        elif tbl.startswith('index_metric_'):
            tag = tbl[len('index_metric_'):]
            tags.setdefault(tag, {})['index'] = tbl

    return tags


def _load_pivot(table_name: str, group_col: str) -> pd.DataFrame:
    """Load a cached metric table and pivot to (dates x groups)."""
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(
                f"SELECT calc_date, {group_col}, value FROM {table_name} ORDER BY calc_date"
            ), conn)
        if df.empty:
            return pd.DataFrame()
        df['calc_date'] = pd.to_datetime(df['calc_date'])
        df['value']     = pd.to_numeric(df['value'], errors='coerce')
        return df.pivot(index='calc_date', columns=group_col, values='value').sort_index()
    except Exception as e:
        print(f"  WARNING: could not load '{table_name}': {e}")
        return pd.DataFrame()


def load_all(verbose: bool = True) -> dict:
    """
    Load all cached sector and index metric DataFrames from DB.

    Returns
    -------
    dict with keys:
      'sector'    : dict {metric_tag -> DataFrame (dates x sectors)}
      'index'     : dict {metric_tag -> DataFrame (dates x indexes)}
      'available' : list of metric tags found
    """
    tags = _get_metric_tables()

    if not tags:
        print("  No cached sector/index metric tables found in DB.")
        return {'sector': {}, 'index': {}, 'available': []}

    lib_sector = {}
    lib_index  = {}

    if verbose:
        print(f"\n  Found {len(tags)} metric(s) in DB:\n")
        print(f"  {'Metric tag':<30} {'Sector table':<35} {'Index table'}")
        print(f"  {'-'*90}")

    for tag in sorted(tags):
        entry      = tags[tag]
        sec_tbl    = entry.get('sector')
        idx_tbl    = entry.get('index')

        sec_df = _load_pivot(sec_tbl, 'sector')    if sec_tbl else pd.DataFrame()
        idx_df = _load_pivot(idx_tbl, 'index_name') if idx_tbl else pd.DataFrame()

        lib_sector[tag] = sec_df
        lib_index[tag]  = idx_df

        if verbose:
            sec_info = f"{sec_tbl} ({len(sec_df)} dates)" if not sec_df.empty else f"{sec_tbl or '—'} (empty)"
            idx_info = f"{idx_tbl} ({len(idx_df)} dates)" if not idx_df.empty else f"{idx_tbl or '—'} (empty)"
            print(f"  {tag:<30} {sec_info:<35} {idx_info}")

    if verbose:
        print(f"\n  Loaded {len(lib_sector)} sector DataFrames, {len(lib_index)} index DataFrames.")
        print(f"\n  Access examples:")
        if tags:
            example = sorted(tags.keys())[0]
            print(f"    lib['sector']['{example}']")
            print(f"    lib['index']['{example}']")
        print(f"    lib['available']  →  {sorted(tags.keys())}")

    return {
        'sector':    lib_sector,
        'index':     lib_index,
        'available': sorted(tags.keys()),
    }
