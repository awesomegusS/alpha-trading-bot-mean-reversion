from dotenv import load_dotenv
load_dotenv()
import os
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import math
from alpaca_trade_api.rest import REST
from prefect import task, flow, get_run_logger

# --- SECURE CONFIGURATION ---
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

# Our Validated Portfolio
ACTIVE_PAIRS = [
    {"t1": "BAC", "t2": "MS", "alloc": 10000},
    {"t1": "CI", "t2": "ELV", "alloc": 10000}
]

if not API_KEY or not API_SECRET:
    raise ValueError("CRITICAL: Alpaca API keys not found in environment variables.")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

@task(retries=3, retry_delay_seconds=60)
def process_pair(ticker_1, ticker_2, allocation):
    logger = get_run_logger()
    logger.info(f"--- Processing {ticker_1} & {ticker_2} ---")
    
    # 1. Ingestion & Math
    df1 = yf.download(ticker_1, period="2y", progress=False)['Close']
    df2 = yf.download(ticker_2, period="2y", progress=False)['Close']
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = [ticker_1, ticker_2]
    
    X = sm.add_constant(df[ticker_2])
    beta = sm.OLS(df[ticker_1], X).fit().params.iloc[1]
    
    df['Spread'] = df[ticker_1] - (beta * df[ticker_2])
    df['Rolling_Mean'] = df['Spread'].rolling(window=30).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window=30).std()
    
    current_z = (df['Spread'].iloc[-1] - df['Rolling_Mean'].iloc[-1]) / df['Rolling_Std'].iloc[-1]
    price1, price2 = df[ticker_1].iloc[-1], df[ticker_2].iloc[-1]
    
    logger.info(f"Z-Score: {current_z:.2f} | Beta: {beta:.4f}")
    
    # 2. Execution Logic
    positions = [p.symbol for p in api.list_positions()]
    in_trade = ticker_1 in positions or ticker_2 in positions
    
    if in_trade and abs(current_z) < 0.5:
        logger.info(f"MEAN REVERSION ACHIEVED. Closing {ticker_1}/{ticker_2}.")
        if ticker_1 in positions: api.close_position(ticker_1)
        if ticker_2 in positions: api.close_position(ticker_2)
        return f"{ticker_1}/{ticker_2}: CLOSED"

    if in_trade:
        logger.info(f"Active trade holding for {ticker_1}/{ticker_2}.")
        return f"{ticker_1}/{ticker_2}: HOLDING"

    # Beta-Neutral Sizing
    weight1 = 1 / (1 + abs(beta))
    weight2 = abs(beta) / (1 + abs(beta))
    qty1 = math.floor((allocation * weight1) / price1)
    qty2 = math.floor((allocation * weight2) / price2)
    
    if current_z < -2.0:
        logger.info(f"LONG SIGNAL. Routing orders for {ticker_1}/{ticker_2}.")
        api.submit_order(symbol=ticker_1, qty=qty1, side='buy', type='market', time_in_force='day')
        api.submit_order(symbol=ticker_2, qty=qty2, side='sell', type='market', time_in_force='day')
        return f"{ticker_1}/{ticker_2}: EXECUTED_LONG"
        
    elif current_z > 2.0:
        logger.info(f"SHORT SIGNAL. Routing orders for {ticker_1}/{ticker_2}.")
        api.submit_order(symbol=ticker_1, qty=qty1, side='sell', type='market', time_in_force='day')
        api.submit_order(symbol=ticker_2, qty=qty2, side='buy', type='market', time_in_force='day')
        return f"{ticker_1}/{ticker_2}: EXECUTED_SHORT"
        
    logger.info(f"NEUTRAL. No action for {ticker_1}/{ticker_2}.")
    return f"{ticker_1}/{ticker_2}: NEUTRAL"

@flow(name="Multi-Pair Stat Arb Execution")
def stat_arb_pipeline():
    results = []
    for pair in ACTIVE_PAIRS:
        status = process_pair(pair["t1"], pair["t2"], pair["alloc"])
        results.append(status)
    return results

if __name__ == "__main__":
    stat_arb_pipeline()