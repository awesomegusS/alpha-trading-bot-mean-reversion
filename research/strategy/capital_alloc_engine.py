import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_capital_allocation_engine(ticker1="BAC", ticker2="MS", period="2y", capital=10000):
    print(f"Igniting Z-Score Engine for {ticker1} & {ticker2}...\n")
    
    # 1. Data Ingestion
    df1 = yf.download(ticker1, period=period, progress=False)['Close']
    df2 = yf.download(ticker2, period=period, progress=False)['Close']
    
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = [ticker1, ticker2]
    
    # 2. Calculate the Hedge Ratio (Beta)
    X = sm.add_constant(df[ticker2])
    model = sm.OLS(df[ticker1], X).fit()
    beta = model.params.iloc[1] # Extract the beta coefficient
    
    print("-" * 50)
    print("POSITION SIZING MATH")
    print("-" * 50)
    print(f"Hedge Ratio (Beta): {beta:.4f}")
    
    # 3. Calculate Spread & Z-Score
    df['Spread'] = df[ticker1] - (beta * df[ticker2])
    
    # Using a 30-day rolling window for dynamic mean and standard deviation
    rolling_window = 30
    df['Rolling_Mean'] = df['Spread'].rolling(window=rolling_window).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window=rolling_window).std()
    df['Z_Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    df.dropna(inplace=True)
    
    # 4. Current State & Capital Allocation
    current_z = df['Z_Score'].iloc[-1]
    price1 = df[ticker1].iloc[-1]
    price2 = df[ticker2].iloc[-1]
    
    print("\n" + "-" * 50)
    print("CURRENT MARKET STATE")
    print("-" * 50)
    print(f"Current Z-Score: {current_z:.2f}")
    print(f"{ticker1} Price:   ${price1:.2f}")
    print(f"{ticker2} Price:   ${price2:.2f}")
    
    print("\n" + "-" * 50)
    print(f"EXECUTION PROTOCOL (Assuming ${capital:,.2f} Capital)")
    print("-" * 50)
    
    # We split capital based on the hedge ratio to remain Beta Neutral
    # Weight of Ticker 1 = 1 / (1 + |Beta|)
    # Weight of Ticker 2 = |Beta| / (1 + |Beta|)
    
    weight1 = 1 / (1 + abs(beta))
    weight2 = abs(beta) / (1 + abs(beta))
    
    alloc1 = capital * weight1
    alloc2 = capital * weight2
    
    shares1 = alloc1 / price1
    shares2 = alloc2 / price2
    
    if current_z < -2.0:
        print(f"SIGNAL: LONG THE SPREAD (Spread is undervalued)")
        print(f"Action 1: BUY  {shares1:.2f} shares of {ticker1} (${alloc1:.2f})")
        print(f"Action 2: SHORT {shares2:.2f} shares of {ticker2} (${alloc2:.2f})")
        print(f"Exit Protocol: Close both positions when Z-Score hits 0.0")
        
    elif current_z > 2.0:
        print(f"SIGNAL: SHORT THE SPREAD (Spread is overvalued)")
        print(f"Action 1: SHORT {shares1:.2f} shares of {ticker1} (${alloc1:.2f})")
        print(f"Action 2: BUY   {shares2:.2f} shares of {ticker2} (${alloc2:.2f})")
        print(f"Exit Protocol: Close both positions when Z-Score hits 0.0")
        
    else:
        print(f"SIGNAL: NEUTRAL (Z-Score is between -2.0 and 2.0)")
        print("Action: Do nothing. Wait for the rubber band to stretch further.")
        print(f"Mock Position if triggered today:")
        print(f"-> {ticker1} Leg: {shares1:.2f} shares")
        print(f"-> {ticker2} Leg: {shares2:.2f} shares")

if __name__ == "__main__":
    ticker1 = input('Enter first ticker: ')
    ticker2 = input('Enter second ticker: ')
    run_capital_allocation_engine(ticker1, ticker2, capital=10000)