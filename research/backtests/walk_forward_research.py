import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta

def calculate_metrics(df, ticker1, ticker2, beta, fee_rate=0.001):
    """Core logic to generate signals and calculate returns."""
    temp_df = df.copy()
    
    # 1. Spread and Z-Score using the provided Beta
    temp_df['Spread'] = temp_df[ticker1] - (beta * temp_df[ticker2])
    temp_df['Rolling_Mean'] = temp_df['Spread'].rolling(window=30).mean()
    temp_df['Rolling_Std'] = temp_df['Spread'].rolling(window=30).std()
    temp_df['Z_Score'] = (temp_df['Spread'] - temp_df['Rolling_Mean']) / temp_df['Rolling_Std']
    temp_df.dropna(inplace=True)

    # 2. Market Returns
    temp_df['Ret1'] = temp_df[ticker1].pct_change()
    temp_df['Ret2'] = temp_df[ticker2].pct_change()

    # 3. State Machine (Entry/Exit Logic)
    positions = np.zeros(len(temp_df))
    current_pos = 0
    for i in range(len(temp_df)):
        z = temp_df['Z_Score'].iloc[i]
        if current_pos == 0:
            if z < -2.0: current_pos = 1
            elif z > 2.0: current_pos = -1
        elif current_pos == 1 and z >= 0.0:
            current_pos = 0
        elif current_pos == -1 and z <= 0.0:
            current_pos = 0
        positions[i] = current_pos

    temp_df['Position'] = positions
    temp_df['Position'] = temp_df['Position'].shift(1).fillna(0)
    
    # 4. Beta-Neutral Weighting & Returns
    w1_base = 1 / (1 + abs(beta))
    w2_base = beta / (1 + abs(beta))
    w1 = temp_df['Position'] * w1_base
    w2 = temp_df['Position'] * (-w2_base) 
    
    temp_df['Strat_Ret'] = (w1 * temp_df['Ret1']) + (w2 * temp_df['Ret2'])
    temp_df['Trades'] = temp_df['Position'].diff().abs()
    temp_df['Strat_Ret'] -= (temp_df['Trades'] * fee_rate * 2) 
    temp_df.dropna(inplace=True)

    # 5. Metrics
    if len(temp_df) == 0:
        return 0, 0, 0

    cumulative_returns = (1 + temp_df['Strat_Ret']).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    mean_ret = temp_df['Strat_Ret'].mean()
    std_ret = temp_df['Strat_Ret'].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    max_dd = drawdown.min()
    
    return total_return, max_dd, sharpe

def run_walk_forward(ticker1, ticker2):
    print(f"\nRunning Walk-Forward Optimization for {ticker1} & {ticker2}...")
    
    # Download 3 Years of Data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3*365)
    
    df1 = yf.download(ticker1, start=start_date, end=end_date, progress=False)['Close']
    df2 = yf.download(ticker2, start=start_date, end=end_date, progress=False)['Close']
    df = pd.concat([df1, df2], axis=1).dropna()
    df.columns = [ticker1, ticker2]

    # Time Slicing
    t_minus_1 = end_date - timedelta(days=365)
    t_minus_2 = end_date - timedelta(days=2*365)

    df_is = df[(df.index >= start_date) & (df.index < t_minus_1)] # T-3 to T-1 (2 years IS)
    df_oos = df[(df.index >= t_minus_1)]                          # T-1 to Now (1 year OOS)
    df_bench = df[(df.index >= t_minus_2)]                        # T-2 to Now (2 year Benchmark)

    # --- Phase 1: IN-SAMPLE (Train) ---
    X_is = sm.add_constant(df_is[ticker2])
    beta_is = sm.OLS(df_is[ticker1], X_is).fit().params.iloc[1]
    ret_is, dd_is, sharpe_is = calculate_metrics(df_is, ticker1, ticker2, beta_is)

    # --- Phase 2: OUT-OF-SAMPLE (Test) ---
    # We strictly use the beta calculated in the IS phase. No look-ahead bias allowed.
    ret_oos, dd_oos, sharpe_oos = calculate_metrics(df_oos, ticker1, ticker2, beta_is)

    # --- Phase 3: BENCHMARK (T-2 to Now) ---
    # Standard backtest calculating a fresh beta over the 2 year period
    X_bench = sm.add_constant(df_bench[ticker2])
    beta_bench = sm.OLS(df_bench[ticker1], X_bench).fit().params.iloc[1]
    ret_bench, dd_bench, sharpe_bench = calculate_metrics(df_bench, ticker1, ticker2, beta_bench)

    # Print Matrix
    print("=" * 60)
    print(f"{'Metric':<20} | {'IS (T-3 to T-1)':<15} | {'OOS (T-1 to Now)':<15}")
    print("-" * 60)
    print(f"{'Locked Beta':<20} | {beta_is:<15.4f} | {beta_is:<15.4f} (Locked)")
    print(f"{'Sharpe Ratio':<20} | {sharpe_is:<15.3f} | {sharpe_oos:<15.3f}")
    print(f"{'Max Drawdown':<20} | {dd_is:<15.2%} | {dd_oos:<15.2%}")
    print(f"{'Total Return':<20} | {ret_is:<15.2%} | {ret_oos:<15.2%}")
    print("=" * 60)
    print(f"BENCHMARK (T-2 to Now, Dynamic Beta: {beta_bench:.4f})")
    print(f"Sharpe: {sharpe_bench:.3f} | Max DD: {dd_bench:.2%} | Return: {ret_bench:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    pairs = [
        ("BAC", "MS"),    # Our validated Financials pair
        # ("INTC", "MU"),   # Our new Tech candidate
        ("PSX", "VLO"),   # Our new Energy candidate
        # ("GOOG", "INTC"), # Other Tech candidate
        # ("AVGO", "NVDA"), # Other Tech candidate
        # ("GOOGL", "INTC"), # Other Tech candidate
        ("CI", "ELV"),
        # ("CI", "CNC"),
    ]
    for t1, t2 in pairs:
        run_walk_forward(t1, t2)