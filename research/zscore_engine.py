import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

def generate_zscore_signals(ticker1, ticker2, period="2y", rolling_window=30, entry_z=2.0, exit_z=0.0):
    print(f"Generating Statistical Arbitrage signals for {ticker1} & {ticker2}...")
    
    # 1. Data Ingestion
    df1 = yf.download(ticker1, period=period, progress=False)['Close']
    df2 = yf.download(ticker2, period=period, progress=False)['Close']
    
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = [ticker1, ticker2]
    
    # 2. Calculate the Hedge Ratio (Beta) using Ordinary Least Squares (OLS)
    # We regress Ticker1 against Ticker2
    X = sm.add_constant(df[ticker2])
    model = sm.OLS(df[ticker1], X).fit()
    beta = model.params[ticker2]
    
    print(f"Calculated Hedge Ratio (Beta): {beta:.4f}")
    print(f"For every 1 share of {ticker1}, you hedge with {beta:.4f} shares of {ticker2}")
    
    # 3. Calculate the Spread
    df['Spread'] = df[ticker1] - (beta * df[ticker2])
    
    # 4. Calculate the Rolling Z-Score
    df['Rolling_Mean'] = df['Spread'].rolling(window=rolling_window).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window=rolling_window).std()
    df['Z_Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    # 5. Generate Trading Signals
    df.dropna(inplace=True)
    
    # Logic: 
    # Long the Spread if Z-Score < -2.0 (Ticker 1 is undervalued relative to Ticker 2)
    # Short the Spread if Z-Score > 2.0 (Ticker 1 is overvalued relative to Ticker 2)
    # Exit when Z-Score crosses 0 (Mean Reversion achieved)
    
    df['Signal'] = 0
    # Create Long Signal
    df.loc[df['Z_Score'] < -entry_z, 'Signal'] = 1 
    # Create Short Signal
    df.loc[df['Z_Score'] > entry_z, 'Signal'] = -1 
    
    # Filter to show only days where action is taken
    action_days = df[df['Signal'] != 0].copy()
    
    print("-" * 50)
    print(f"Signal Summary (Rolling Window: {rolling_window} days)")
    print("-" * 50)
    print(f"Total action signals generated: {len(action_days)}")
    print(f"Current Z-Score (Last Close): {df['Z_Score'].iloc[-1]:.2f}")
    
    if abs(df['Z_Score'].iloc[-1]) > entry_z:
        print(">>> ACTIVE TRADE SIGNAL TODAY <<<")
    else:
        print("Status: Waiting for divergence.")
        
    return df

if __name__ == "__main__":
    # Replace these with your passing pair
    ticker1 = input('Enter first ticker: ')
    ticker2 = input('Enter second ticker: ')
    
    generate_zscore_signals(ticker1, ticker2)
