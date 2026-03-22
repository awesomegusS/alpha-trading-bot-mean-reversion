import yfinance as yf
import pandas as pd
import numpy as np

def run_sma_crossover_backtest(ticker="BTC-USD", fast_period=20, slow_period=50, fee_rate=0.001):
    print(f"Ingesting daily data for {ticker}...")
    
    # 1. Data Ingestion
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    
    if data.empty:
        print("Failed to pull data. Check your internet connection or ticker symbol.")
        return
    
    # yfinance sometimes returns MultiIndex columns, let's flatten or grab 'Close' safely
    if isinstance(data.columns, pd.MultiIndex):
        df = pd.DataFrame({'Close': data['Close'][ticker]})
    else:
        df = pd.DataFrame({'Close': data['Close']})

    # 2. Signal Generation (The Hypothesis)
    df['SMA_Fast'] = df['Close'].rolling(window=fast_period).mean()
    df['SMA_Slow'] = df['Close'].rolling(window=slow_period).mean()
    
    # 1 if Fast > Slow (Bullish), 0 otherwise (Flat/Neutral)
    # We drop NAs first so our strategy doesn't start until day 50
    df.dropna(inplace=True)
    df['Signal'] = np.where(df['SMA_Fast'] > df['SMA_Slow'], 1, 0)
    
    # 3. Vectorized Returns Calculation
    # Shift the signal by 1 day to avoid look-ahead bias (we trade *after* the signal triggers)
    df['Position'] = df['Signal'].shift(1)
    
    # Calculate daily logarithmic returns of the underlying asset
    df['Market_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Strategy returns = Position * Market Returns
    df['Strategy_Returns'] = df['Position'] * df['Market_Returns']
    
    # 4. Apply Transaction Costs
    # We only pay fees when our position changes (e.g., going from 0 to 1, or 1 to 0)
    df['Trades'] = df['Position'].diff().abs()
    df['Strategy_Returns'] = df['Strategy_Returns'] - (df['Trades'] * fee_rate)
    
    # Drop the first row which will have NaN from shifting
    df.dropna(inplace=True)
    
    # 5. Performance Metrics (The Reality Check)
    cumulative_market = np.exp(df['Market_Returns'].cumsum()) - 1
    cumulative_strategy = np.exp(df['Strategy_Returns'].cumsum()) - 1
    
    # Annualized Sharpe Ratio (Assuming 365 trading days for Crypto, use 252 for traditional equities)
    trading_days = 365 if "USD" in ticker else 252
    
    mean_return = df['Strategy_Returns'].mean()
    std_return = df['Strategy_Returns'].std()
    
    if std_return == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(trading_days)
    
    # Output Results
    print("-" * 30)
    print(f"Backtest Results: {fast_period}/{slow_period} SMA Crossover")
    print("-" * 30)
    print(f"Total Market Return:   {cumulative_market.iloc[-1]:.2%}")
    print(f"Total Strategy Return: {cumulative_strategy.iloc[-1]:.2%}")
    print(f"Total Trades Executed: {int(df['Trades'].sum())}")
    print(f"Annualized Sharpe:     {sharpe_ratio:.3f}")
    print("-" * 30)

if __name__ == "__main__":
    run_sma_crossover_backtest()