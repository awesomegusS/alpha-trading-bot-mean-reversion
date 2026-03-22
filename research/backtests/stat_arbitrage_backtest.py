import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_pairs_backtest(ticker1, ticker2, period="2y", entry_z=2.0, exit_z=0.0, fee_rate=0.001):
    # 1. Ingestion
    df1 = yf.download(ticker1, period=period, progress=False)['Close']
    df2 = yf.download(ticker2, period=period, progress=False)['Close']
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = [ticker1, ticker2]
    
    # Calculate daily percentage returns for the underlying assets
    df['Ret1'] = df[ticker1].pct_change()
    df['Ret2'] = df[ticker2].pct_change()

    # 2. Hedge Ratio & Z-Score
    X = sm.add_constant(df[ticker2])
    model = sm.OLS(df[ticker1], X).fit()
    beta = model.params.iloc[1]
    
    df['Spread'] = df[ticker1] - (beta * df[ticker2])
    df['Rolling_Mean'] = df['Spread'].rolling(window=30).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window=30).std()
    df['Z_Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']
    df.dropna(inplace=True)

    # 3. State Machine for Trading Logic (Avoids Look-Ahead Bias)
    positions = np.zeros(len(df))
    current_pos = 0
    
    for i in range(len(df)):
        z = df['Z_Score'].iloc[i]
        
        # Entry Logic
        if current_pos == 0:
            if z < -entry_z:
                current_pos = 1  # Long Spread
            elif z > entry_z:
                current_pos = -1 # Short Spread
                
        # Exit Logic (Mean Reversion)
        elif current_pos == 1 and z >= exit_z:
            current_pos = 0
        elif current_pos == -1 and z <= exit_z:
            current_pos = 0
            
        positions[i] = current_pos

    df['Position'] = positions
    # Shift position by 1 day because we execute at the close/next open AFTER signal
    df['Position'] = df['Position'].shift(1).fillna(0)
    
    # 4. Strategy Returns Calculation
    # Weighting the portfolio to maintain Beta Neutrality
    w1_base = 1 / (1 + abs(beta))
    w2_base = beta / (1 + abs(beta))
    
    # If Long Spread: Long T1, Short T2. If Short Spread: Short T1, Long T2.
    w1 = df['Position'] * w1_base
    w2 = df['Position'] * (-w2_base) 
    
    df['Strat_Ret'] = (w1 * df['Ret1']) + (w2 * df['Ret2'])
    
    # Apply Fees (0.1% per trade per leg)
    df['Trades'] = df['Position'].diff().abs()
    # We multiply by 2 because entering a pairs trade requires 2 executions (buy A, sell B)
    df['Strat_Ret'] -= (df['Trades'] * fee_rate * 2) 
    
    # 5. Performance Metrics
    cumulative_returns = (1 + df['Strat_Ret']).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    # Annualized Sharpe (252 trading days)
    mean_ret = df['Strat_Ret'].mean()
    std_ret = df['Strat_Ret'].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    # Max Drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    max_dd = drawdown.min()
    
    num_trades = int(df['Trades'].sum())
    
    print(f"--- {ticker1} / {ticker2} ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Total Trades: {num_trades}\n")

if __name__ == "__main__":
    pairs_to_test = [("BAC", "MS"), ("PNC", "USB"), ("BAC", "GS"), ("CI", "ELV")]
    
    print("=" * 40)
    print("STATISTICAL ARBITRAGE BACKTEST RESULTS")
    print("=" * 40)
    for t1, t2 in pairs_to_test:
        run_pairs_backtest(t1, t2)