import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from alpaca_trade_api.rest import REST, TimeFrame
import math

# --- 1. CONFIGURATION ---
API_KEY = "YOUR_PAPER_API_KEY"
API_SECRET = "YOUR_PAPER_SECRET_KEY"
BASE_URL = "https://paper-api.alpaca.markets"

TICKER_1 = "BAC"
TICKER_2 = "MS"
CAPITAL_ALLOCATION = 10000  # Total capital to deploy for this pair

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_zscore_and_beta():
    print(f"Ingesting market data for {TICKER_1} & {TICKER_2}...")
    df1 = yf.download(TICKER_1, period="2y", progress=False)['Close']
    df2 = yf.download(TICKER_2, period="2y", progress=False)['Close']
    
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = [TICKER_1, TICKER_2]
    
    X = sm.add_constant(df[TICKER_2])
    model = sm.OLS(df[TICKER_1], X).fit()
    beta = model.params.iloc[1]
    
    df['Spread'] = df[TICKER_1] - (beta * df[TICKER_2])
    df['Rolling_Mean'] = df['Spread'].rolling(window=30).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window=30).std()
    df['Z_Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    current_z = df['Z_Score'].iloc[-1]
    price1 = df[TICKER_1].iloc[-1]
    price2 = df[TICKER_2].iloc[-1]
    
    return current_z, beta, price1, price2

def execute_trade():
    print("=" * 40)
    print("ALPACA EXECUTION ENGINE INITIALIZED")
    print("=" * 40)
    
    current_z, beta, price1, price2 = get_zscore_and_beta()
    print(f"Current Z-Score: {current_z:.2f} | Beta: {beta:.4f}")
    
    # Check current positions
    positions = api.list_positions()
    pos_tickers = [p.symbol for p in positions]
    
    in_trade = TICKER_1 in pos_tickers or TICKER_2 in pos_tickers
    
    # Exit Logic: Reversion to Mean (Z crosses 0)
    # If we are in a trade and Z-score reverts between -0.5 and 0.5, we exit.
    if in_trade and abs(current_z) < 0.5:
        print(f"SIGNAL: MEAN REVERSION ACHIEVED. Closing all positions for {TICKER_1} & {TICKER_2}.")
        if TICKER_1 in pos_tickers:
            api.close_position(TICKER_1)
        if TICKER_2 in pos_tickers:
            api.close_position(TICKER_2)
        return

    # If we are already in a trade and it hasn't reverted, do nothing.
    if in_trade:
        print("Status: Currently in active trade. Waiting for mean reversion.")
        return

    # Position Sizing Math (Beta Neutral)
    weight1 = 1 / (1 + abs(beta))
    weight2 = abs(beta) / (1 + abs(beta))
    
    qty1 = math.floor((CAPITAL_ALLOCATION * weight1) / price1)
    qty2 = math.floor((CAPITAL_ALLOCATION * weight2) / price2)
    
    # Entry Logic
    if current_z < -2.0:
        print(f"SIGNAL: LONG SPREAD. Submitting orders...")
        api.submit_order(symbol=TICKER_1, qty=qty1, side='buy', type='market', time_in_force='day')
        api.submit_order(symbol=TICKER_2, qty=qty2, side='sell', type='market', time_in_force='day') # Short
        print(f"Executed: Bought {qty1} {TICKER_1}, Shorted {qty2} {TICKER_2}")
        
    elif current_z > 2.0:
        print(f"SIGNAL: SHORT SPREAD. Submitting orders...")
        api.submit_order(symbol=TICKER_1, qty=qty1, side='sell', type='market', time_in_force='day') # Short
        api.submit_order(symbol=TICKER_2, qty=qty2, side='buy', type='market', time_in_force='day')
        print(f"Executed: Shorted {qty1} {TICKER_1}, Bought {qty2} {TICKER_2}")
        
    else:
        print("SIGNAL: NEUTRAL. No execution required today.")

if __name__ == "__main__":
    execute_trade()