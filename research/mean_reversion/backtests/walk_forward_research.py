import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta

def calculate_metrics(df, ticker1, ticker2, beta, fee_rate=0.001):
    """Core logic to generate signals and calculate returns & trade stats."""
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

    if len(temp_df) == 0:
        return 0, 0, 0, 0, 0, 0

    # 5. Core Metrics
    cumulative_returns = (1 + temp_df['Strat_Ret']).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    trading_days = len(temp_df)
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
    
    mean_ret = temp_df['Strat_Ret'].mean()
    std_ret = temp_df['Strat_Ret'].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    max_dd = drawdown.min()
    
    # 6. Trade Level Statistics (Win Rate & Count)
    trades_pnl = []
    current_trade_pnl = 1.0
    in_trade = False
    
    for i in range(len(temp_df)):
        pos = temp_df['Position'].iloc[i]
        ret = temp_df['Strat_Ret'].iloc[i]
        
        if pos != 0 and not in_trade:
            in_trade = True
            current_trade_pnl = 1.0 + ret
        elif pos != 0 and in_trade:
            current_trade_pnl *= (1.0 + ret)
        elif pos == 0 and in_trade:
            current_trade_pnl *= (1.0 + ret) # Capture exit fee
            trades_pnl.append(current_trade_pnl - 1.0)
            in_trade = False
            
    if in_trade:
        trades_pnl.append(current_trade_pnl - 1.0)
        
    num_trades = len(trades_pnl)
    win_rate = sum(1 for p in trades_pnl if p > 0) / num_trades if num_trades > 0 else 0.0
    
    return total_return, max_dd, sharpe, annualized_return, win_rate, num_trades

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
    ret_is, dd_is, sharpe_is, ann_is, win_is, trades_is = calculate_metrics(df_is, ticker1, ticker2, beta_is := sm.OLS(df_is[ticker1], sm.add_constant(df_is[ticker2])).fit().params.iloc[1])

    # --- Phase 2: OUT-OF-SAMPLE (Test) ---
    ret_oos, dd_oos, sharpe_oos, ann_oos, win_oos, trades_oos = calculate_metrics(df_oos, ticker1, ticker2, beta_is)

    # --- Phase 3: BENCHMARK (T-2 to Now) ---
    ret_bench, dd_bench, sharpe_bench, ann_bench, win_bench, trades_bench = calculate_metrics(df_bench, ticker1, ticker2, beta_bench := sm.OLS(df_bench[ticker1], sm.add_constant(df_bench[ticker2])).fit().params.iloc[1])

    # Print Matrix
    print("=" * 85)
    print(f"{'Metric':<20} | {'IS (T-3 to T-1)':<25} | {'OOS (T-1 to Now)':<25}")
    print("-" * 85)
    print(f"{'Locked Beta':<20} | {beta_is:<25.4f} | {beta_is:<25.4f} (Locked)")
    print(f"{'Sharpe Ratio':<20} | {sharpe_is:<25.3f} | {sharpe_oos:<25.3f}")
    print(f"{'Win Rate':<20} | {win_is:<25.2%} | {win_oos:<25.2%}")
    print(f"{'Number of Trades':<20} | {trades_is:<25} | {trades_oos:<25}")
    print(f"{'Max Drawdown':<20} | {dd_is:<25.2%} | {dd_oos:<25.2%}")
    print(f"{'Total Return':<20} | {ret_is:<25.2%} | {ret_oos:<25.2%}")
    print(f"{'Annualized Return':<20} | {ann_is:<25.2%} | {ann_oos:<25.2%}")
    print("=" * 85)
    print(f"BENCHMARK (T-2 to Now, Dynamic Beta: {beta_bench:.4f})")
    print(f"Sharpe: {sharpe_bench:.3f} | Win Rate: {win_bench:.2%} | Trades: {trades_bench} | Max DD: {dd_bench:.2%} | Ann Ret: {ann_bench:.2%}")
    print("=" * 85)

if __name__ == "__main__":
    pairs = [
        ("BAC", "MS"),    
        ("PSX", "VLO"),   
        ("CI", "ELV"),
    ]
    for t1, t2 in pairs:
        run_walk_forward(t1, t2)