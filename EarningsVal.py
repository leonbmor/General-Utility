#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
Earnings Report Follow-Up
==========================
For a given stock, shows KPIs for:
  - FEQ     : First Estimated Quarter
  - FEQ+1   : Next quarter after FEQ
  - YofI    : Full-year sum (4 quarters) for the Year of Interest
  - YofI Val: Valuation based on YofI sum and current price

Year of Interest logic:
  FEQ = YYYYQ1/Q2/Q3  -> YofI = YYYY  (Q1+Q2+Q3+Q4)
  FEQ = YYYYQ4        -> YofI = YYYY+1 (Q1+Q2+Q3+Q4 of next year)

Usage:
    from earnings_followup import run
    run(Pxs_s)
"""

import pandas as pd
from sqlalchemy import create_engine, text

DB_NAME = "factormodel_db"
ENGINE  = create_engine(f"postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/{DB_NAME}")

INCOME_METRICS = [
    'totalRevenues', 'grossProfit', 'operatingIncome',
    'normalizedNetIncome', 'netIncome', 'eps', 'dilutedEps',
    'dilutedAverageShares', 'ebitda'
]
CASH_METRICS = [
    'freeCashFlow', 'capitalExpenditures', 'operatingCashFlow',
    'cashFromFinancing', 'cashFromInvesting'
]
SUMMARY_METRICS = [
    'ebitda', 'cashAndCashEquivalents', 'debt', 'netDebt'
]

# Metrics that should NOT be divided by 1M (keep raw, just round to 2dp)
NO_SCALE_METRICS = {'eps', 'dilutedEps'}

# Metrics where valuation = price / KPI (not market_cap / KPI)
PRICE_OVER_METRIC = {'eps', 'dilutedEps'}

# Metrics where valuation doesn't make sense (skip)
NO_VALUATION_METRICS = {'dilutedAverageShares', 'cashAndCashEquivalents',
                         'debt', 'capitalExpenditures', 'cashFromFinancing',
                         'cashFromInvesting'}


# ==============================================================================
# HELPERS
# ==============================================================================

def clean_ticker(ticker: str) -> str:
    return ticker.strip().split(' ')[0].upper()


def add_quarters(quarter: str, n: int) -> str:
    year, q = int(quarter[:4]), int(quarter[5])
    q += n
    while q > 4: q -= 4; year += 1
    while q < 1: q += 4; year -= 1
    return f"{year}Q{q}"


def year_of_interest(feq: str) -> int:
    year, q = int(feq[:4]), int(feq[5])
    return year + 1 if q == 4 else year


def yoi_quarters(yoi: int) -> list:
    return [f"{yoi}Q{q}" for q in range(1, 5)]


def get_feq(ticker: str) -> str:
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        row = conn.execute(text("""
            SELECT first_estimated_period FROM estimation_status
            WHERE ticker = :t AND category = 'income'
        """), {"t": t}).fetchone()
    return row[0] if row else None


def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v in ('-', 'nan', 'None'):
        return True
    return False


def fmt(val, metric: str) -> str:
    """Format value: divide by 1M (unless eps), round to 2dp, add commas."""
    if is_missing(val):
        return '-'
    if metric not in NO_SCALE_METRICS:
        val = val / 1_000_000
    return f"{val:,.2f}"


def fmt_val(val) -> str:
    """Format valuation ratio."""
    if is_missing(val):
        return '-'
    return f"{val:,.2f}"


def get_price(ticker: str, Pxs_s: pd.Series) -> float:
    """Look up current price from Pxs_s, trying both 'AAPL' and 'AAPL US'."""
    t    = clean_ticker(ticker)
    full = t + ' US'
    if full in Pxs_s.index:
        return float(Pxs_s[full])
    if t in Pxs_s.index:
        return float(Pxs_s[t])
    return None


def calc_valuation(yoi_sum, metric: str, price: float, shares_yoi: float) -> float:
    """
    Compute valuation for YofI:
      - eps / dilutedEps : price / KPI
      - everything else  : market_cap / KPI  where market_cap = shares * price / 1M
    shares_yoi is the raw (unscaled) YofI sum of dilutedAverageShares.
    """
    if metric in NO_VALUATION_METRICS:
        return None
    if is_missing(yoi_sum) or price is None:
        return None
    if yoi_sum == 0:
        return None

    if metric in PRICE_OVER_METRIC:
        return round(price / yoi_sum, 2)
    else:
        if shares_yoi is None or shares_yoi == 0:
            return None
        market_cap = (shares_yoi * price) / 1_000_000  # in $M
        kpi_m      = yoi_sum / 1_000_000                # in $M
        return round(market_cap / kpi_m, 2)


def fetch_metrics_ffill(ticker: str, table: str, metrics: list, periods: list) -> pd.DataFrame:
    t    = clean_ticker(ticker)
    rows = []
    with ENGINE.connect() as conn:
        for metric in metrics:
            for period in periods:
                row = conn.execute(text(f"""
                    SELECT value FROM {table}
                    WHERE ticker = :t
                      AND metric_name = :m
                      AND period = :p
                      AND value IS NOT NULL
                    ORDER BY download_date DESC
                    LIMIT 1
                """), {"t": t, "m": metric, "p": period}).fetchone()
                val = float(row[0]) if row else None
                rows.append({"metric": metric, "period": period, "value": val})
    return pd.DataFrame(rows)


def build_table(ticker: str, table: str, metrics: list,
                feq: str, feq1: str, yoi_qs: list,
                price: float, shares_yoi_raw: float) -> pd.DataFrame:
    all_periods = list(dict.fromkeys([feq, feq1] + yoi_qs))
    df          = fetch_metrics_ffill(ticker, table, metrics, all_periods)

    results = []
    for metric in metrics:
        mdf    = df[df['metric'] == metric].set_index('period')['value']
        feq_v  = mdf.get(feq)
        feq1_v = mdf.get(feq1)

        yoi_vals = [mdf.get(q) for q in yoi_qs if not is_missing(mdf.get(q))]
        yoi_sum  = sum(yoi_vals) if yoi_vals else None

        val_v = calc_valuation(yoi_sum, metric, price, shares_yoi_raw)

        results.append({
            'Metric':    metric,
            feq:         fmt(feq_v,  metric),
            feq1:        fmt(feq1_v, metric),
            'YofI_sum':  fmt(yoi_sum, metric),
            'YofI_val':  fmt_val(val_v)
        })

    return pd.DataFrame(results).set_index('Metric')


# ==============================================================================
# MAIN
# ==============================================================================

def run(Pxs_s: pd.Series):
    ticker_input = input("Enter ticker (e.g. NVDA or NVDA US): ").strip()
    t            = clean_ticker(ticker_input)

    feq = get_feq(t)
    if not feq:
        print(f"  No estimation_status found for {t}. Is the ticker in the DB?")
        return

    feq1  = add_quarters(feq, 1)
    yoi   = year_of_interest(feq)
    yqs   = yoi_quarters(yoi)
    price = get_price(t, Pxs_s)

    if price is None:
        print(f"  Warning: no price found for {t} in Pxs_s -- valuation column will be empty")

    # Use single quarter shares (latest available) not the 4-quarter sum
    shares_df      = fetch_metrics_ffill(t, 'income_data', ['dilutedAverageShares'], [feq])
    shares_row     = shares_df[shares_df['metric'] == 'dilutedAverageShares']['value']
    shares_yoi_raw = float(shares_row.iloc[0]) if not shares_row.empty else None

    print(f"\n  Ticker          : {t}")
    print(f"  FEQ             : {feq}")
    print(f"  FEQ+1           : {feq1}")
    print(f"  Year of Interest: {yoi} ({', '.join(yqs)})")
    print(f"  Current Price   : {price:,.2f}" if price else "  Current Price   : N/A")
    print(f"  (values in $M except eps/dilutedEps)")
    print()

    col_labels = {
        feq:        f"FEQ ({feq})",
        feq1:       f"FEQ+1 ({feq1})",
        'YofI_sum': f"YofI {yoi} (sum)",
        'YofI_val': f"YofI {yoi} (val)"
    }

    for label, table, metrics in [
        ("INCOME",  "income_data",  INCOME_METRICS),
        ("CASH",    "cash_data",    CASH_METRICS),
        ("SUMMARY", "summary_data", SUMMARY_METRICS),
    ]:
        df = build_table(t, table, metrics, feq, feq1, yqs, price, shares_yoi_raw)
        df = df.rename(columns=col_labels)

        # Drop rows where all 4 values are missing
        df = df[~df.apply(lambda row: all(is_missing(v) for v in row), axis=1)]

        print("=" * 80)
        print(f"  {label}")
        print("=" * 80)
        print(df.to_string())
        print()


if __name__ == "__main__":
    Pxs_df = openF_df('prices_relation')
    Pxs_s = Pxs_df.iloc[-1]
    run(Pxs_s)

