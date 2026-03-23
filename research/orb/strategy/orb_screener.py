"""
Intraday Opening Range Breakout (ORB) Research Engine.

Purpose:
    Identifies institutional momentum breakouts by mapping the first 30 minutes 
    of the trading day (9:30 AM - 10:00 AM EST) and flagging when price crosses 
    these boundaries with a statistically significant Volume Z-Score (> 2.0).

Usage:
    python intraday_orb_research.py --ticker <TICKER>
    python intraday_orb_research.py -t NVDA
    python intraday_orb_research.py --help
"""

import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_atr(df, period=14):
    """Calculates the Average True Range (ATR) for volatility measurement."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def run_orb_screener(ticker):
    print(f"Ingesting 60 days of 5-minute intraday data for {ticker}...")
    
    # 1. Data Ingestion (Max allowed by Yahoo Finance for 5m interval)
    df = yf.download(ticker, period="60d", interval="5m", progress=False)
    
    if df.empty:
        print(f"CRITICAL ERROR: Data pull failed for {ticker}. Check ticker symbol.")
        return
        
    # Flatten MultiIndex if yfinance returns one
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # Standardize Timezone to US/Eastern
    df.index = df.index.tz_convert('America/New_York')
    
    # Extract Date and Time into separate columns for easier grouping
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    
    print("Mapping the 9:30 AM - 10:00 AM Opening Range...")

    # 2. Define the Opening Range (OR)
    or_df = df[(df['Time'] >= pd.to_datetime('09:30').time()) & 
               (df['Time'] < pd.to_datetime('10:00').time())]
    
    or_highs = or_df.groupby('Date')['High'].max().rename('OR_High')
    or_lows = or_df.groupby('Date')['Low'].min().rename('OR_Low')
    
    df = df.join(or_highs, on='Date')
    df = df.join(or_lows, on='Date')

    # 3. Institutional Filters (Volume Z-Score & ATR)
    print("Calculating Institutional Flow Filters (Volume Z-Score & ATR)...")
    
    df['ATR'] = calculate_atr(df, period=14)
    
    df['Vol_Mean'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Std'] = df['Volume'].rolling(window=20).std()
    df['Vol_ZScore'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']

    # 4. Signal Generation (The Breakout)
    trade_window = (df['Time'] >= pd.to_datetime('10:00').time()) & \
                   (df['Time'] <= pd.to_datetime('15:30').time())
                   
    long_condition = trade_window & (df['Close'] > df['OR_High']) & (df['Vol_ZScore'] > 2.0)
    short_condition = trade_window & (df['Close'] < df['OR_Low']) & (df['Vol_ZScore'] > 2.0)

    df['Signal'] = 0
    df.loc[long_condition, 'Signal'] = 1
    df.loc[short_condition, 'Signal'] = -1

    # 5. Extracting the Triggers
    signals_df = df[df['Signal'] != 0].copy()
    first_signals = signals_df.groupby('Date').first()

    print("\n" + "="*70)
    print(f"INSTITUTIONAL ORB TRIGGERS ({ticker.upper()}) - Last 60 Days")
    print("="*70)
    print(f"Total Trading Days Analyzed: {len(df['Date'].unique())}")
    print(f"Total Valid Breakouts Caught: {len(first_signals)}")
    print("-" * 70)
    
    if not first_signals.empty:
        display_cols = ['Time', 'Close', 'OR_High', 'OR_Low', 'Vol_ZScore', 'ATR', 'Signal']
        print(first_signals[display_cols].tail(5).to_string())
    else:
        print("No valid breakouts met the strict volume criteria.")

if __name__ == "__main__":
    # Set up argparse for CLI execution and documentation
    parser = argparse.ArgumentParser(
        description="Scans 60 days of 5-minute data to find Opening Range Breakouts confirmed by volume anomalies.",
        epilog="Example usage: python intraday_orb_research.py -t TSLA"
    )
    
    # Add the ticker argument
    parser.add_argument(
        "-t", "--ticker", 
        type=str, 
        required=True, 
        help="The stock or ETF ticker symbol to analyze (e.g., NVDA, SPY, TSLA)"
    )
    
    # Parse the arguments passed from the terminal
    args = parser.parse_args()
    
    # Pass the ticker to the main function
    run_orb_screener(args.ticker)