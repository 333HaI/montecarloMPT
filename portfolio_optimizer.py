import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_user_tickers():
    print("Enter stock tickers separated by spaces or commas (e.g., GLXY, NVDA, CRCL, COIN).")
    input_str = input("Tickers: ").strip()
    
    if not input_str:
        print("No tickers entered. Exiting.")
        return None
        
    tickers = [ticker.strip().upper() for ticker in input_str.replace(',', ' ').split() if ticker.strip()]
    print(f"You entered: {tickers}")
    return tickers

def get_simulation_parameters():
    while True:
        try:
            num_str = input("Enter the number of portfolio simulations to run [default: 20000]: ").strip()
            if not num_str:
                num_portfolios = 20000
                break
            num_portfolios = int(num_str)
            if num_portfolios < 500:
                print("For meaningful results, please enter a number of at least 1000.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # Get risk-free rate with validation
    while True:
        try:
            rate_str = input("Enter the risk-free rate as a percentage (e.g., 2 for 2%) [default: 2.0]: ").strip()
            if not rate_str:
                risk_free_rate = 0.02
                break
            risk_free_rate = float(rate_str) / 100.0
            break
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 2.5).")
            
    print(f"\nConfiguration: Running {num_portfolios:,} simulations with a risk-free rate of {risk_free_rate:.2%}.")
    return num_portfolios, risk_free_rate


if __name__ == "__main__":
    TICKERS = get_user_tickers()
    if not TICKERS:
        exit()

    NUM_PORTFOLIOS, RISK_FREE_RATE = get_simulation_parameters()
    
    print("\nFetching historical data from Yahoo Finance...")
    price_data = yf.download(TICKERS, period='5y')['Close']

    if price_data.empty:
        print(f"Error: Could not download any data. Please check ticker symbols.")
        exit()

    missing_tickers = price_data.columns[price_data.isna().all()].tolist()
    if missing_tickers:
        print(f"Warning: Could not fetch data for: {missing_tickers}. They will be excluded.")
        price_data.drop(columns=missing_tickers, inplace=True)
        TICKERS = price_data.columns.tolist()

    if len(TICKERS) < 2:
        print("Error: You need at least two valid stocks for portfolio optimization. Exiting.")
        exit()

    print(f"Successfully downloaded data for: {TICKERS}")

    log_returns = np.log(price_data / price_data.shift(1))





    #Monte Carlo Simulation
    portfolio_returns = []
    portfolio_volatility = []
    portfolio_weights = []

    num_assets = len(TICKERS)
    cov_matrix_annual = log_returns.cov() * 252

    print("\nRunning Monte Carlo Simulation...")
    for i in range(NUM_PORTFOLIOS):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_weights.append(weights)
        
        returns_annual = np.sum(log_returns.mean() * weights) * 252
        portfolio_returns.append(returns_annual)
        
        port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
        port_volatility = np.sqrt(port_variance)
        portfolio_volatility.append(port_volatility)

    portfolio_data = {
        'Return': portfolio_returns,
        'Volatility': portfolio_volatility
    }
    portfolio_data['Sharpe'] = (np.array(portfolio_returns) - RISK_FREE_RATE) / np.array(portfolio_volatility)

    portfolios_df = pd.DataFrame(portfolio_data)

    for i, symbol in enumerate(TICKERS):
        portfolios_df[symbol + "_weight"] = [w[i] for w in portfolio_weights]

    print("Simulation Complete. Identifying optimal portfolios.")

    
    
    
    
    #Identify Optimal Portfolios
    max_sharpe_portfolio = portfolios_df.iloc[portfolios_df['Sharpe'].idxmax()]
    min_vol_portfolio = portfolios_df.iloc[portfolios_df['Volatility'].idxmin()]

    print("\n--- Max Sharpe Ratio Portfolio ---")
    print(max_sharpe_portfolio.round(4))

    print("\n--- Minimum Volatility Portfolio ---")
    print(min_vol_portfolio.round(4))

    #PLOT
    print("\nGenerating plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    portfolios_df.plot.scatter(
        x='Volatility',
        y='Return',
        c='Sharpe',
        cmap='viridis',
        edgecolors='black',
        figsize=(12, 8),
        grid=True,
        label='Simulated Portfolios'
    )

    plt.scatter(
        max_sharpe_portfolio['Volatility'],
        max_sharpe_portfolio['Return'],
        c='red',
        marker='*',
        s=200,
        label='Max Sharpe Ratio Portfolio'
    )
    plt.scatter(
        min_vol_portfolio['Volatility'],
        min_vol_portfolio['Return'],
        c='orange',
        marker='*',
        s=200,
        label='Minimum Volatility Portfolio'
    )

    plt.title(f'Portfolio Optimization - {" & ".join(TICKERS)}')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Return')
    plt.legend(labelspacing=0.8)
    plt.show()

    print("Plot displayed.")