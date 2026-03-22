import yfinance as yf
import pandas as pd
import statsmodels.tsa.stattools as ts
import itertools

def run_cointegration_screener(tickers, period="2y"):
    print(f"Downloading data for {len(tickers)} tickers...")
    
    # 1. Batch Data Ingestion (Much faster than individual calls)
    data = yf.download(tickers, period=period, progress=False)['Close']
    
    # Drop any tickers that have missing data (keeps our matrix clean)
    data = data.dropna(axis=1)
    clean_tickers = data.columns.tolist()
    
    print(f"Usable tickers after cleaning: {len(clean_tickers)}")
    
    # 2. Generate all unique pairs
    pairs = list(itertools.combinations(clean_tickers, 2))
    print(f"Testing {len(pairs)} unique combinations. This might take a minute...")
    
    results = []
    
    # 3. Run the Engle-Granger test on every pair
    for ticker1, ticker2 in pairs:
        score, pvalue, _ = ts.coint(data[ticker1], data[ticker2])
        
        # Only save pairs that pass our strict 0.05 threshold
        if pvalue < 0.05:
            results.append({
                'Pair': f"{ticker1} & {ticker2}",
                'p-value': pvalue,
                't-statistic': score
            })
            
    # 4. Rank and Output Results
    if not results:
        print("No cointegrated pairs found in this universe.")
        return
        
    results_df = pd.DataFrame(results)
    # Sort by lowest p-value (strongest statistical relationship)
    results_df = results_df.sort_values(by='p-value', ascending=True).reset_index(drop=True)
    
    print("\n" + "="*40)
    print("🏆 TOP COINTEGRATED PAIRS 🏆")
    print("="*40)
    print(results_df.to_string())
    print("="*40)
    
    return results_df

if __name__ == "__main__":
    # Let's test a universe of major financial and energy stocks. 
    # Sectors with high regulatory overlap tend to cointegrate better.
    # universe = [
    #     "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", # Financials
    #     "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "VLO"      # Energy
    # ]
    
    # run_cointegration_screener(universe)

    # UNIVERSE 1: Semiconductors & Mega-Tech
    # tech_universe = [
    #     "NVDA", "AMD", "INTC", "TXN", "QCOM", "AVGO", "MU", # Semis
    #     "AAPL", "MSFT", "GOOG", "GOOGL", "META"             # Mega-Cap
    # ]
    
    # # UNIVERSE 2: Energy Exploration & Refining
    # energy_universe = [
    #     "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "VLO", "MPC", "PSX"
    # ]
    
    # print("--- SCANNING TECH UNIVERSE ---")
    # run_cointegration_screener(tech_universe)
    
    # print("\n--- SCANNING ENERGY UNIVERSE ---")
    # run_cointegration_screener(energy_universe)

    universes = {
        "Financials":[
            "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", # Financials
            "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "VLO"      # Energy
            ],

        "Tech": [ 
            "NVDA", "AMD", "INTC", "TXN", "QCOM", "AVGO", "MU", # Semis
            "AAPL", "MSFT", "GOOG", "GOOGL", "META"             # Mega-Cap
            ],

        "Energy": [
            "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "VLO", "MPC", "PSX"
            ]
    }
    
    for sector_name, tickers in universes.items():
        print(f"\n--- SCANNING {sector_name.upper()} UNIVERSE ---")
        run_cointegration_screener(tickers)

    # UNIVERSE 3: Defense & Aerospace
    defense_universe = [
        "LMT",   # Lockheed Martin
        "NOC",   # Northrop Grumman
        "GD",    # General Dynamics
        "RTX",   # RTX Corp (Raytheon)
        "HII",   # Huntington Ingalls
        "LHX"    # L3Harris Technologies
    ]
    
    # UNIVERSE 4: Managed Healthcare & Insurance
    healthcare_universe = [
        "UNH",   # UnitedHealth Group
        "ELV",   # Elevance Health (formerly Anthem)
        "CI",    # Cigna
        "HUM",   # Humana
        "CNC",   # Centene
        "CVS"    # CVS Health (Aetna)
    ]
    
    print("--- SCANNING DEFENSE UNIVERSE ---")
    run_cointegration_screener(defense_universe)
    
    print("\n--- SCANNING HEALTHCARE UNIVERSE ---")
    run_cointegration_screener(healthcare_universe)