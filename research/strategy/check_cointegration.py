import yfinance as yf
import pandas as pd
import statsmodels.tsa.stattools as ts

def run_engle_granger(ticker1="BTC-USD", ticker2="ETH-USD", period="2y"):
    print(f"Fetching daily close prices for {ticker1} and {ticker2}...")
    
    # 1. Data Ingestion
    data1 = yf.download(ticker1, period=period, progress=False)['Close']
    data2 = yf.download(ticker2, period=period, progress=False)['Close']
    
    # Handle yfinance MultiIndex if present
    if isinstance(data1, pd.DataFrame):
        data1 = data1.iloc[:, 0]
    if isinstance(data2, pd.DataFrame):
        data2 = data2.iloc[:, 0]
        
    # 2. Data Alignment
    # Cointegration requires perfectly aligned indices
    df = pd.concat([data1, data2], axis=1).dropna()
    df.columns = [ticker1, ticker2]
    
    print(f"Data aligned. Running Engle-Granger test on {len(df)} data points...")
    
    # 3. The Engle-Granger Test
    # Null Hypothesis (H0): The two time series are NOT cointegrated.
    # We want a p-value < 0.05 to reject the null hypothesis.
    score, pvalue, _ = ts.coint(df[ticker1], df[ticker2])
    
    # 4. Results Output
    print("-" * 45)
    print(f"Cointegration Test Results: {ticker1} vs {ticker2}")
    print("-" * 45)
    print(f"t-statistic: {score:.4f}")
    print(f"p-value:     {pvalue:.4f}")
    print("-" * 45)
    
    if pvalue < 0.05:
        print("Verdict: PASS. The p-value is < 0.05.")
        print("We reject the null hypothesis. These assets are cointegrated.")
    else:
        print("Verdict: FAIL. The p-value is >= 0.05.")
        print("We cannot reject the null hypothesis. Do not trade this pair.")

if __name__ == "__main__":
    ticker1 = input("Enter the first ticker symbol: ")
    ticker2 = input("Enter the second ticker symbol: ")
    run_engle_granger(ticker1, ticker2)