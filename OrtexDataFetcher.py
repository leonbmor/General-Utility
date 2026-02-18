#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Ortex Fundamentals Data Fetcher
Fetches income and cash flow data for stocks and stores in PostgreSQL database
"""

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, date
import time
import logging
from typing import List, Dict, Optional, Tuple

# Configuration
API_BASE_URL = "https://api.ortex.com/api/v1/stock/US/"
API_KEY = "2HQKW1rx.jFO09ZJraZqfZaMz12ruIgaoqunK2c9Q"  # Replace with your actual API key
HEADERS = {
    "accept": "application/json",
    "Ortex-Api-Key": API_KEY
}

# Database configuration
DB_NAME = "factormodel_db"
CONNECTION_STRING = (
    "postgresql+psycopg2://{username}:{pswd}@{host}:{port}/{database}"
)
ENGINE = create_engine(
    CONNECTION_STRING.format(
        username="postgres",
        pswd="akf7a7j5",
        host="localhost",
        port=5432,
        database=DB_NAME
    )
)

# Credit monitoring thresholds
CREDITS_PER_ROUND = 1000
MIN_CREDITS_REMAINING = 5000

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('ortex_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def initialize_database():
    """Create tables if they don't exist"""
    with ENGINE.connect() as conn:
        # Income data table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS income_data (
                ticker TEXT NOT NULL,
                download_date DATE NOT NULL,
                period TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value NUMERIC,
                PRIMARY KEY (ticker, download_date, period, metric_name)
            )
        """))
        
        # Cash data table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cash_data (
                ticker TEXT NOT NULL,
                download_date DATE NOT NULL,
                period TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value NUMERIC,
                PRIMARY KEY (ticker, download_date, period, metric_name)
            )
        """))
        
        # Summary data table (NEW)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS summary_data (
                ticker TEXT NOT NULL,
                download_date DATE NOT NULL,
                period TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value NUMERIC,
                PRIMARY KEY (ticker, download_date, period, metric_name)
            )
        """))
        
        # Add estimated_values column to existing tables if it doesn't exist
        for table_name in ['income_data', 'cash_data', 'summary_data']:
            try:
                conn.execute(text(f"""
                    ALTER TABLE {table_name} 
                    ADD COLUMN IF NOT EXISTS estimated_values BOOLEAN DEFAULT FALSE
                """))
                logger.info(f"Added estimated_values column to {table_name} (if needed)")
            except Exception as e:
                logger.debug(f"Column may already exist in {table_name}: {e}")
        
        # Estimation status tracking table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS estimation_status (
                ticker TEXT NOT NULL,
                category TEXT NOT NULL,
                first_estimated_period TEXT,
                last_checked DATE,
                PRIMARY KEY (ticker, category)
            )
        """))
        
        conn.commit()
    
    logger.info("Database tables initialized")


def generate_periods(start_period: str, end_period: str) -> List[str]:
    """
    Generate list of quarterly periods between start and end
    Example: '2020Q1' to '2025Q4' -> ['2020Q1', '2020Q2', ..., '2025Q4']
    """
    start_year = int(start_period[:4])
    start_quarter = int(start_period[5])
    end_year = int(end_period[:4])
    end_quarter = int(end_period[5])
    
    periods = []
    for year in range(start_year, end_year + 1):
        start_q = start_quarter if year == start_year else 1
        end_q = end_quarter if year == end_year else 4
        
        for quarter in range(start_q, end_q + 1):
            periods.append(f"{year}Q{quarter}")
    
    return periods


def fetch_fundamentals(ticker: str, period: str, category: str) -> Optional[Dict]:
    """
    Fetch fundamental data from Ortex API with retry logic for rate limiting
    
    Args:
        ticker: Stock ticker symbol
        period: Period in format 'YYYYQQ' (e.g., '2025Q4')
        category: Either 'income', 'cash', or 'summary'
    
    Returns:
        Dictionary with API response or None if failed
    """
    url = f"{API_BASE_URL}{ticker}/fundamentals/{category}?period={period}"
    
    max_retries = 3
    base_delay = 2  # Start with 2 seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            
            # Debug logging to see what we're getting
            logger.debug(f"API Response keys: {data.keys()}")
            logger.debug(f"creditsUsed: {data.get('creditsUsed')}, creditsLeft: {data.get('creditsLeft')}")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limit hit
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                    logger.warning(f"{ticker} | {category} | {period} | Rate limit (429) - Retry {attempt+1}/{max_retries} after {delay}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"{ticker} | {category} | {period} | Rate limit (429) - All retries exhausted")
                    return None
            else:
                # Other HTTP error
                logger.error(f"{ticker} | {category} | {period} | API Error: {str(e)}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"{ticker} | {category} | {period} | API Error: {str(e)}")
            return None
    
    return None


def get_already_fetched_periods_today(ticker: str, category: str) -> set:
    """Get set of periods already fetched today for this ticker/category"""
    today = date.today()
    table_name = f"{category}_data"
    
    query = text(f"""
        SELECT DISTINCT period
        FROM {table_name}
        WHERE ticker = :ticker AND download_date = :today
    """)
    
    with ENGINE.connect() as conn:
        result = conn.execute(query, {"ticker": ticker, "today": today}).fetchall()
        return {row[0] for row in result}


def get_all_fetched_periods_ever(ticker: str, category: str) -> set:
    """Get set of all periods ever fetched for this ticker/category (any date)"""
    table_name = f"{category}_data"
    
    query = text(f"""
        SELECT DISTINCT period
        FROM {table_name}
        WHERE ticker = :ticker
    """)
    
    with ENGINE.connect() as conn:
        result = conn.execute(query, {"ticker": ticker}).fetchall()
        return {row[0] for row in result}


def get_estimation_status(ticker: str, category: str) -> Optional[str]:
    """Get the first estimated period for a ticker/category"""
    query = text("""
        SELECT first_estimated_period
        FROM estimation_status
        WHERE ticker = :ticker AND category = :category
    """)
    
    with ENGINE.connect() as conn:
        result = conn.execute(query, {"ticker": ticker, "category": category}).fetchone()
        return result[0] if result else None


def update_estimation_status(ticker: str, category: str, first_estimated_period: str):
    """Update the first estimated period for a ticker/category"""
    today = date.today()
    
    query = text("""
        INSERT INTO estimation_status (ticker, category, first_estimated_period, last_checked)
        VALUES (:ticker, :category, :period, :today)
        ON CONFLICT (ticker, category)
        DO UPDATE SET
            first_estimated_period = :period,
            last_checked = :today
    """)
    
    with ENGINE.connect() as conn:
        conn.execute(query, {
            "ticker": ticker,
            "category": category,
            "period": first_estimated_period,
            "today": today
        })
        conn.commit()


def detect_first_estimated_from_db(ticker: str, category: str) -> Optional[str]:
    """
    Scan the database to find the ACTUAL first estimated period by looking at all data.
    This is more reliable than tracking during fetch since:
    - We might not fetch the earliest estimated period in every run
    - Estimates can change over time
    
    Returns:
        First period with estimated_values = True, or None if no estimates found
    """
    table_name = f"{category}_data"
    
    # Query to get all periods with their estimated status, ordered chronologically
    query = text(f"""
        WITH latest_data AS (
            SELECT period, estimated_values,
                   ROW_NUMBER() OVER (PARTITION BY period ORDER BY download_date DESC) as rn
            FROM {table_name}
            WHERE ticker = :ticker
        )
        SELECT period, estimated_values
        FROM latest_data
        WHERE rn = 1
        ORDER BY period
    """)
    
    with ENGINE.connect() as conn:
        result = conn.execute(query, {"ticker": ticker}).fetchall()
    
    # Find first period where estimated_values = True
    for period, estimated_values in result:
        if estimated_values:
            return period
    
    return None


def get_last_value(ticker: str, category: str, period: str, metric: str) -> Optional[float]:
    """Get the most recent value for a specific metric"""
    table_name = f"{category}_data"
    
    query = text(f"""
        SELECT value
        FROM {table_name}
        WHERE ticker = :ticker AND period = :period AND metric_name = :metric
        ORDER BY download_date DESC
        LIMIT 1
    """)
    
    with ENGINE.connect() as conn:
        result = conn.execute(query, {
            "ticker": ticker,
            "period": period,
            "metric": metric
        }).fetchone()
        return result[0] if result else None


def has_value_changed(ticker: str, category: str, period: str, new_data: Dict) -> bool:
    """Check if any metric value has changed for a given period"""
    for metric, value in new_data.items():
        if metric == 'estimatedValues':
            continue
        
        last_value = get_last_value(ticker, category, period, metric)
        
        # If we have no previous value, consider it changed
        if last_value is None:
            return True
        
        # Check if value changed (handling None values)
        if value != last_value:
            return True
    
    return False


def save_fundamentals_data(ticker: str, period: str, category: str, data: Dict, 
                           estimated: bool, download_date: date):
    """Save fundamental data to database"""
    table_name = f"{category}_data"
    
    # For summary category, only save specific metrics
    if category == 'summary':
        allowed_metrics = {'cashAndCashEquivalents', 'debt', 'netDebt', 'ebitda'}
        filtered_data = {k: v for k, v in data.items() 
                        if k in allowed_metrics and k != 'estimatedValues'}
    else:
        filtered_data = {k: v for k, v in data.items() if k != 'estimatedValues'}
    
    records = []
    for metric, value in filtered_data.items():
        records.append({
            'ticker': ticker,
            'download_date': download_date,
            'period': period,
            'metric_name': metric,
            'value': value,
            'estimated_values': estimated  # Add the estimated flag
        })
    
    if records:
        # First, delete any existing data for this ticker/period/download_date combination
        # This handles the case where we're updating a period that was previously estimated
        delete_query = text(f"""
            DELETE FROM {table_name}
            WHERE ticker = :ticker 
            AND period = :period 
            AND download_date = :download_date
        """)
        
        with ENGINE.begin() as conn:
            conn.execute(delete_query, {
                'ticker': ticker,
                'period': period,
                'download_date': download_date
            })
        
        # Now insert the new data
        df = pd.DataFrame(records)
        df.to_sql(table_name, ENGINE, if_exists='append', index=False)
        logger.info(f"Saved {ticker} | {category} | {period} | Estimated: {estimated} | Metrics: {len(records)}")
    else:
        logger.warning(f"No data to save for {ticker} | {category} | {period}")


def get_next_period(period: str) -> str:
    """Get the next quarterly period"""
    year = int(period[:4])
    quarter = int(period[5])
    
    if quarter == 4:
        return f"{year + 1}Q1"
    else:
        return f"{year}Q{quarter + 1}"


def should_skip_stock(ticker: str, credits_tracker: Dict) -> Tuple[bool, Optional[str]]:
    """
    Check if we should skip this stock entirely by checking totalRevenues in first estimated period.
    
    Returns:
        (should_skip: bool, reason: Optional[str])
    """
    # Get first estimated period for income (use income as the reference)
    first_estimated = get_estimation_status(ticker, 'income')
    
    if first_estimated is None:
        # First run - don't skip
        return False, None
    
    # Check if already fetched today
    fetched_today = get_already_fetched_periods_today(ticker, 'income')
    if first_estimated in fetched_today:
        # Already processed today - skip
        return True, f"already fetched today (period {first_estimated})"
    
    # Fetch the first estimated period to check totalRevenues
    response = fetch_fundamentals(ticker, first_estimated, 'income')
    
    if response is None:
        # API error - don't skip, let the normal process handle it
        return False, None
    
    # Update credits
    update_credits(credits_tracker, 
                  response.get('creditsUsed', 0), 
                  response.get('creditsLeft'))
    
    # Check if we need confirmation
    if not check_credits_and_confirm(credits_tracker):
        # User wants to stop - signal to abort entire run
        return True, "USER_ABORT"
    
    data = response['data']
    estimated = data.get('estimatedValues', False)
    
    # If no longer estimated, update status and don't skip (need to fetch actuals)
    if not estimated:
        new_first_estimated = get_next_period(first_estimated)
        update_estimation_status(ticker, 'income', new_first_estimated)
        logger.info(f"{ticker} - {first_estimated} now finalized, new first estimated: {new_first_estimated}")
        return False, None
    
    # Check if totalRevenues changed
    total_revenues = data.get('totalRevenues')
    if total_revenues is None:
        # No totalRevenues in response - don't skip
        return False, None
    
    last_total_revenues = get_last_value(ticker, 'income', first_estimated, 'totalRevenues')
    
    if last_total_revenues is None:
        # No previous value - don't skip (first time seeing this period)
        return False, None
    
    if total_revenues != last_total_revenues:
        # Value changed - don't skip
        logger.info(f"{ticker} - totalRevenues changed in {first_estimated}: {last_total_revenues} → {total_revenues}")
        return False, None
    
    # totalRevenues unchanged - skip entire stock
    return True, f"totalRevenues unchanged in {first_estimated}"


def update_credits(credits_tracker: Dict, credits_used: float, credits_left: Optional[float]):
    """Update credits tracker with new API response data"""
    credits_tracker['used_this_round'] += credits_used
    credits_tracker['total_used'] += credits_used
    
    # Only update remaining if we got a valid value from API
    if credits_left is not None:
        credits_tracker['remaining'] = credits_left
        logger.debug(f"Updated credits: used={credits_used:.2f}, remaining={credits_left:.2f}")
    else:
        logger.warning(f"creditsLeft not in API response, keeping previous value: {credits_tracker['remaining']}")


def process_ticker(ticker: str, category: str, all_periods: List[str], 
                   credits_tracker: Dict) -> bool:
    """
    Process a single ticker for a specific category (income or cash)
    
    Returns:
        True if should continue processing, False if should stop
    """
    today = date.today()
    
    # Get periods already fetched today
    fetched_periods_today = get_already_fetched_periods_today(ticker, category)
    
    # Get all periods ever fetched (to identify new historical periods)
    fetched_periods_ever = get_all_fetched_periods_ever(ticker, category)
    
    # Filter out already fetched periods from today
    remaining_periods = [p for p in all_periods if p not in fetched_periods_today]
    
    # Identify new periods that have never been fetched before
    new_periods = [p for p in all_periods if p not in fetched_periods_ever]
    
    if not remaining_periods:
        logger.info(f"Skipping {ticker} | {category} - all periods already fetched today")
        return True
    
    if fetched_periods_today:
        logger.info(f"{ticker} | {category} - {len(fetched_periods_today)} periods already fetched today, "
                   f"{len(remaining_periods)} remaining")
    
    if new_periods:
        logger.info(f"{ticker} | {category} - {len(new_periods)} new periods detected (never fetched before)")
    
    # Get estimation status
    first_estimated_period = get_estimation_status(ticker, category)
    
    # Determine which periods to fetch
    if first_estimated_period is None:
        # First run - fetch all remaining periods
        periods_to_fetch = remaining_periods
        logger.info(f"First run for {ticker} | {category} - fetching {len(periods_to_fetch)} periods")
    else:
        # We need to fetch:
        # 1. Any new historical periods (never fetched before)
        # 2. Estimated periods if they've changed
        
        # Start with new periods that are in remaining_periods
        periods_to_fetch = [p for p in new_periods if p in remaining_periods]
        
        # Now handle estimated periods logic
        if first_estimated_period not in remaining_periods:
            # First estimated period already fetched today or is outside our range
            # Add any remaining estimated periods (>= first_estimated_period)
            try:
                estimated_index = all_periods.index(first_estimated_period)
                estimated_periods = [p for p in remaining_periods 
                                   if all_periods.index(p) >= estimated_index]
                # Add to fetch list (avoiding duplicates)
                for p in estimated_periods:
                    if p not in periods_to_fetch:
                        periods_to_fetch.append(p)
            except ValueError:
                # first_estimated_period not in all_periods
                pass
        else:
            # First estimated period is in remaining_periods
            # Only check if estimated values changed if we haven't fetched it today
            if first_estimated_period in fetched_periods_today:
                # Already fetched today - skip the check unless values changed
                logger.info(f"{ticker} | {category} - First estimated period {first_estimated_period} already fetched today, skipping check")
                # Don't add estimated periods to fetch list
                pass
            else:
                # Not fetched today - check if estimated values changed
                response = fetch_fundamentals(ticker, first_estimated_period, category)
                
                if response is None:
                    # On error, still fetch new historical periods if any
                    if not periods_to_fetch:
                        return True
                else:
                    update_credits(credits_tracker, 
                                  response.get('creditsUsed', 0), 
                                  response.get('creditsLeft'))
                    
                    # Check if we need to ask for confirmation
                    if not check_credits_and_confirm(credits_tracker):
                        return False
                    
                    estimated = response['data'].get('estimatedValues', False)
                    
                    # If this period is no longer estimated, update tracking
                    if not estimated:
                        new_first_estimated = get_next_period(first_estimated_period)
                        update_estimation_status(ticker, category, new_first_estimated)
                        logger.info(f"{ticker} | {category} - {first_estimated_period} now finalized, "
                                   f"new first estimated: {new_first_estimated}")
                        # Add new estimated periods that we haven't fetched yet
                        try:
                            estimated_index = all_periods.index(new_first_estimated)
                            estimated_periods = [p for p in remaining_periods 
                                               if all_periods.index(p) >= estimated_index]
                            for p in estimated_periods:
                                if p not in periods_to_fetch:
                                    periods_to_fetch.append(p)
                        except ValueError:
                            pass
                    else:
                        # Check if values changed
                        if has_value_changed(ticker, category, first_estimated_period, response['data']):
                            logger.info(f"{ticker} | {category} - estimated values changed, "
                                       f"fetching all estimated periods")
                            estimated_index = all_periods.index(first_estimated_period)
                            estimated_periods = [p for p in remaining_periods 
                                               if all_periods.index(p) >= estimated_index]
                            for p in estimated_periods:
                                if p not in periods_to_fetch:
                                    periods_to_fetch.append(p)
                        else:
                            logger.info(f"{ticker} | {category} - estimated values unchanged")
                            # Still fetch new historical periods if any
    
    if not periods_to_fetch:
        logger.info(f"{ticker} | {category} - no periods to fetch")
        return True
    
    logger.info(f"{ticker} | {category} - fetching {len(periods_to_fetch)} periods")
    
    # Fetch all required periods
    first_estimated_found = None
    
    for period in periods_to_fetch:
        response = fetch_fundamentals(ticker, period, category)
        
        if response is None:
            continue  # Skip on error
        
        update_credits(credits_tracker, 
                      response.get('creditsUsed', 0), 
                      response.get('creditsLeft'))  # Pass None if missing
        
        # Save data
        estimated = response['data'].get('estimatedValues', False)
        save_fundamentals_data(ticker, period, category, response['data'], estimated, today)
        
        # Track first estimated period
        if estimated and first_estimated_found is None:
            first_estimated_found = period
        
        # Check credits after each fetch
        if not check_credits_and_confirm(credits_tracker):
            return False
        
        # Small delay to be nice to the API
        time.sleep(0.1)
    
    # Update estimation status if we found estimated periods
    if first_estimated_found:
        update_estimation_status(ticker, category, first_estimated_found)
    
    # CRITICAL: After all fetches, scan the database to verify the ACTUAL first estimated period
    # This handles cases where:
    # 1. We didn't fetch the earliest estimated period this run
    # 2. An earlier period became estimated since last run
    # 3. The estimation_status table was never properly initialized
    actual_first_estimated = detect_first_estimated_from_db(ticker, category)
    if actual_first_estimated:
        current_status = get_estimation_status(ticker, category)
        if actual_first_estimated != current_status:
            logger.info(f"{ticker} | {category} - Correcting first_estimated_period: "
                       f"{current_status} → {actual_first_estimated}")
            update_estimation_status(ticker, category, actual_first_estimated)
    
    return True


def check_credits_and_confirm(credits_tracker: Dict) -> bool:
    """
    Check credit usage and prompt for confirmation if thresholds met
    
    Returns:
        True to continue, False to stop
    """
    used_this_round = credits_tracker['used_this_round']
    total_used = credits_tracker['total_used']
    remaining = credits_tracker['remaining']
    
    # Skip check if we haven't gotten credits info yet
    if remaining is None:
        return True
    
    should_prompt = (
        used_this_round >= CREDITS_PER_ROUND or
        remaining < MIN_CREDITS_REMAINING
    )
    
    if should_prompt:
        print("\n" + "="*60)
        print(f"Credits used this round: {used_this_round:.2f}")
        print(f"Total credits used (session): {total_used:.2f}")
        print(f"Credits remaining: {remaining:.2f}")
        print("="*60)
        
        response = input("Continue? (y/n): ").lower().strip()
        
        if response != 'y':
            logger.info("User chose to stop processing")
            return False
        
        # Reset the round counter (but keep total_used)
        credits_tracker['used_this_round'] = 0
        logger.info("User confirmed - continuing processing")
    
    return True


def main(stock_list: List[str], start_period: str, end_period: str, lastEarnings_l: Optional[pd.Series] = None):
    """
    Main function to fetch and store fundamentals data
    
    Args:
        stock_list: List of ticker symbols
        start_period: Starting period (e.g., '2020Q1')
        end_period: Ending period (e.g., '2025Q4')
        lastEarnings_l: Optional pandas Series with earnings dates per ticker (index=ticker, value=date)
    """
    logger.info(f"Starting Ortex fundamentals fetch")
    logger.info(f"Tickers: {len(stock_list)}")
    logger.info(f"Period range: {start_period} to {end_period}")
    
    # Mode selection
    print("\n" + "=" * 70)
    print("SELECT FETCH MODE")
    print("=" * 70)
    print("1. Fetch ALL stocks (regardless of earnings date)")
    print("2. Fetch only stocks with earnings date < today")
    print("3. Fetch only stocks with earnings in last 10 days")
    print("4. Fetch only stocks with earnings in last 5 days (ULTRA FAST)")
    print("=" * 70)
    
    while True:
        mode = input("\nSelect mode (1/2/3/4): ").strip()
        if mode in ['1', '2', '3', '4']:
            break
        print("✗ Invalid selection. Please choose 1, 2, 3, or 4")
    
    mode = int(mode)
    
    # Filter stock list based on mode
    if mode == 1:
        filtered_stocks = stock_list
        print(f"\n✓ Mode 1: Fetching ALL {len(stock_list)} stocks")
    elif mode in [2, 3, 4]:
        if lastEarnings_l is None:
            print("\n✗ Error: lastEarnings_l not provided but required for mode 2/3/4")
            print("   Please provide earnings dates or use mode 1")
            return
        
        today = date.today()
        filtered_stocks = []
        skipped_no_date = []
        skipped_future = []
        skipped_old = []
        
        # Set day threshold based on mode
        if mode == 3:
            day_threshold = 10
        elif mode == 4:
            day_threshold = 5
        else:
            day_threshold = None  # Mode 2 has no threshold
        
        for ticker in stock_list:
            if ticker not in lastEarnings_l.index:
                print(f"⚠️  Stock {ticker} not on the earnings dates list!!!!!!!!!!!!!!!!!!!!")
                skipped_no_date.append(ticker)
                continue
            
            earnings_date = lastEarnings_l[ticker]
            
            # Handle different date types
            if isinstance(earnings_date, str):
                earnings_date = pd.to_datetime(earnings_date).date()
            elif isinstance(earnings_date, pd.Timestamp):
                earnings_date = earnings_date.date()
            
            # Mode 2: earnings date < today
            if mode == 2:
                if earnings_date < today:
                    filtered_stocks.append(ticker)
                else:
                    skipped_future.append(ticker)
            
            # Mode 3 or 4: earnings in last N days
            elif mode in [3, 4]:
                days_ago = (today - earnings_date).days
                if 0 <= days_ago <= day_threshold:
                    filtered_stocks.append(ticker)
                elif earnings_date >= today:
                    skipped_future.append(ticker)
                else:
                    skipped_old.append(ticker)
        
        mode_label = {2: "2", 3: "3 (10 days)", 4: "4 (5 days)"}[mode]
        print(f"\n✓ Mode {mode_label}: Filtered to {len(filtered_stocks)} stocks")
        print(f"  - Stocks with earnings data: {len(filtered_stocks)}")
        print(f"  - Skipped (no earnings date): {len(skipped_no_date)}")
        print(f"  - Skipped (earnings in future): {len(skipped_future)}")
        if mode in [3, 4]:
            print(f"  - Skipped (earnings >{day_threshold} days ago): {len(skipped_old)}")
    
    if not filtered_stocks:
        print("\n⚠️  No stocks to fetch after filtering!")
        return
    
    # Ask for confirmation
    response = input(f"\nProceed with fetching {len(filtered_stocks)} stocks? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled by user")
        return
    
    # Initialize database
    initialize_database()
    
    # Generate all periods
    all_periods = generate_periods(start_period, end_period)
    logger.info(f"Total periods: {len(all_periods)}")
    
    # Credits tracker
    credits_tracker = {
        'used_this_round': 0,
        'total_used': 0,  # Track total credits used across entire run
        'remaining': None  # Will be set from first API response
    }
    
    # Process each ticker
    total_tickers = len(filtered_stocks)
    categories = ['income', 'cash', 'summary']  # Now includes summary
    
    for idx, ticker in enumerate(filtered_stocks, 1):
        print("\n" + "█" * 70)
        print(f"  STOCK {idx}/{total_tickers}: {ticker}")
        print(f"  Progress: {(idx-1)/total_tickers*100:.1f}% complete")
        print("█" * 70)
        logger.info(f"\n[{idx}/{total_tickers}] Processing {ticker}")
        
        # Check if we should skip this stock entirely based on totalRevenues
        should_skip, skip_reason = should_skip_stock(ticker, credits_tracker)
        
        if should_skip:
            if skip_reason == "USER_ABORT":
                logger.info("Processing stopped by user")
                break
            logger.info(f"Skipping {ticker} - {skip_reason}")
            print(f"  ⏭️  SKIPPED: {skip_reason}")
            continue
        
        # Process all three categories
        for cat_idx, category in enumerate(categories, 1):
            print(f"  └─ Category {cat_idx}/3: {category}")
            if not process_ticker(ticker, category, all_periods, credits_tracker):
                logger.info("Processing stopped by user")
                break
        else:
            continue
        break
    
    print("\n" + "█" * 70)
    print(f"  ✅ ALL STOCKS COMPLETE ({total_tickers}/{total_tickers})")
    print("█" * 70)
    
    logger.info("\n" + "="*60)
    logger.info("Processing complete!")
    logger.info(f"Total credits used (session): {credits_tracker['total_used']:.2f}")
    logger.info(f"Credits remaining: {credits_tracker.get('remaining', 'Unknown')}")
    logger.info("="*60)


if __name__ == "__main__":
    # Example usage
    # Replace these with your actual parameters
    joker_dikt = {'ZI US': 'GTM US', 'BIGC US': 'CMRC US'}
    exc_l = ['MOMO US', 'BILI US', 'TME US', 'BABA US', 'JD US', 'BIDU US', 'PDD US', 'NIO US', 'XPEV US', 'ZTO US', 'ZLAB US', 
             'TCEHY US', 'TSM US', 'OCFT US', 'VNET US', 'BGNE US', 'YY US', 'TAL US', 'EDU US', 'GOTU US', 'NTES US', 
             'PAGS US', 'STNE US', 'WB US', 'MPNGF US', 'HZNP US', 'COUP US', 'ZEN US', 'LAW US', 'PLUG US', 'WKME US', 
             'COLD US', 'BRK/A US', 'CBOE US', 'ALL US', 'ICE US', 'PGR US', 'MET US', 'KNSL US', 'SPLK US', 
             'ESS US', 'PSA US', 'AJG US', 'AFL US','AIG US', 'GDS US', 'AVB US', 'COF US', 'CI US', 'CINF US', 
             'DLR US', 'FRT US', 'FRC US', 'HIG US', 'IRM US', 'KIM US', 'LNC US', 'PFG US', 'O US', 'UNM US', 
             'VTR US', 'WELL US', 'XP US', 'RDFN US', 'ESMT US']    

    query_earnings = "SELECT * FROM ed_relation"
    ed_df = pd.read_sql_query(query_earnings, engine)
    ed_df = Set_DF(ed_df)
    ed_df = DD_Index(ed_df)    
    ed_df = ed_df.rename(columns = joker_dikt)
    Pxs_df = openF_df('prices_relation').rename(columns = joker_dikt)
    Pxs_df.index = Pxs_df.index.map(lambda x: datetime(x.year, x.month, x.day))
    vayf_df = openF_df('va_yf')    
    stock_l = [d for d in ed_df.columns if d in vayf_df.index and d in Pxs_df.columns]
    stock_l = pd.Series(stock_l)[pd.Series(stock_l).map(lambda x: x.split(' ')[-1]) == 'US']
    stock_l = Pxs_df[stock_l][Pxs_df[stock_l][-10:].pct_change().rolling(5).sum() != 0].iloc[-1].dropna().index.tolist()
    stock_l = [t for t in stock_l if t not in exc_l]
    stock_l = pd.Series(stock_l).map(lambda x: x.split(' ')[0]).tolist()
    
    # Get earnings dates (assume you have a Series with ticker as index, earnings_date as value)
    # Example: lastEarnings_l = pd.Series({'AAPL': '2026-01-30', 'MSFT': '2026-01-25', ...})
    # For now, set to None to use mode 1 (fetch all)
    lastEarnings_s = ed_df.iloc[-1].map(lambda x: datetime(datetime.today().year, int(x[:2]), int(x[-2:])))
    lastEarnings_s.index = lastEarnings_s.index.map(lambda x: vayf_df.loc[x, 'YF Ticker'])
    
    # stock_l = ['APP', 'AMZN']  # Replace with your ~700 ticker list
    start_period = '2018Q1'
    end_period = '2030Q4'
    
    main(stock_l, start_period, end_period, lastEarnings_s)

