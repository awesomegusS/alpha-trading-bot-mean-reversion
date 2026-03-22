# Quantitative Statistical Arbitrage Engine

This repository contains a production-grade algorithmic trading pipeline focused on Market-Neutral Statistical Arbitrage. It is designed to identify, validate, and execute mean-reversion trades on cointegrated asset pairs.

## 1. The Core Strategy: Mean Reversion on Ornstein-Uhlenbeck


Unlike directional trading, this strategy does not attempt to predict if the broader market will go up or down. Instead, it relies on the **Ornstein-Uhlenbeck process**, which models the mathematical elasticity between two fundamentally tethered assets (e.g., two banks, or two healthcare insurers).

When the spread between the two assets deviates from its historical mean due to market noise, we assume it will act like a rubber band and snap back.
* We **Long the Spread** when the primary asset is statistically undervalued relative to the secondary asset.
* We **Short the Spread** when the primary asset is statistically overvalued.
* We remain **Beta-Neutral**, meaning market-wide crashes or rallies affect our long and short positions equally, netting $0 in directional risk.

## 2. The Math: Cointegration, Spread, and Z-Score

We first prove the two assets move together using the **Engle-Granger Two-Step Method**. This runs an Ordinary Least Squares (OLS) regression to find the Hedge Ratio ($\beta$):

$$P_1 = \beta P_2 + \epsilon$$

Once proven, we do not simply subtract the prices. We calculate a Beta-weighted spread to isolate the pure relationship, stripping away overall market volatility:

$$\text{Spread} = P_1 - (\beta \times P_2)$$

To normalize this spread across different price environments, we calculate a **Rolling Z-Score** (using a 30-day window):

$$Z = \frac{\text{Spread} - \mu_{\text{rolling}}}{\sigma_{\text{rolling}}}$$

**Execution Logic:**
* Enter Long Spread: $Z < -2.0$
* Enter Short Spread: $Z > 2.0$
* Exit (Mean Reversion): $Z \approx 0.0$

## 3. Capital Allocation and Position Sizing


To maintain true Beta-Neutrality, we cannot allocate equal dollars to both legs (Dollar Neutrality). If Asset 2 is twice as volatile as Asset 1, a dollar-neutral portfolio would carry hidden directional risk.

We weight the capital inversely to the assets' relative volatility using our calculated Hedge Ratio ($\beta$):

* **Weight of Asset 1:** $W_1 = \frac{1}{1 + |\beta|}$
* **Weight of Asset 2:** $W_2 = \frac{|\beta|}{1 + |\beta|}$

**Calculating Shares:**
Given a fixed `CAPITAL_ALLOCATION` (e.g., $10,000) for the pair, the exact shares to buy/short are calculated by flooring the fractional shares (to comply with broker shorting rules):

$$Q_1 = \lfloor \frac{\text{Capital} \times W_1}{P_1} \rfloor$$
$$Q_2 = \lfloor \frac{\text{Capital} \times W_2}{P_2} \rfloor$$

## 4. Backtesting: Walk-Forward Optimization


To prevent "overfitting" (creating a model that memorizes the past but fails in the future), we use a strict **Walk-Forward** train/test split.

1.  **In-Sample (IS) [T-3 to T-1 Years]:** We calculate the Hedge Ratio ($\beta$) over this two-year training period.
2.  **Out-of-Sample (OOS) [T-1 to Present]:** We apply the locked $\beta$ from the IS period to unseen market data. If the strategy breaks down here, it is rejected.

## 5. Performance Metrics

This engine evaluates pairs strictly on risk-adjusted institutional metrics. Total return is secondary to capital preservation—a crucial framework for managing funded accounts and strict risk evaluations.

* **Sharpe Ratio:** Measures the excess return per unit of volatility. 
    $$S = \frac{R_p - R_f}{\sigma_p}$$
    *Target: $S > 1.0$ (Highly Tradable), $S > 2.0$ (Holy Grail)*
* **Maximum Drawdown:** The largest peak-to-trough drop in the equity curve.
    *Target: Strictly under -10% to ensure survival during black swan events.*

## 6. Repository Structure

* `src/stat_arb_flow.py`: The production Prefect execution pipeline. Pulls data, calculates Z-scores, and routes beta-neutral orders to the Alpaca API.
* `research/cointegration_screener.py`: Scans massive sector-specific equity universes (Tech, Energy, Financials, Healthcare) to find pairs with a p-value $< 0.05$.
* `research/walk_forward_research.py`: The validation gauntlet. Runs the IS/OOS backtests and outputs Sharpe/Drawdown matrices.
* `research/zscore_engine.py`: Local scratchpad for visualizing current spread states without executing trades.
* `Dockerfile`: Containerization instructions for cloud deployment.
* `requirements.txt`: Pinned Python dependencies.

## 7. Deployment Details
This system is deployed as a headless container on Railway. It utilizes a native cron job to wake up, execute the `stat_arb_flow.py` Prefect flow, and shut down every weekday at exactly 3:55 PM EST, capturing closing data and executing before the 4:00 PM bell.