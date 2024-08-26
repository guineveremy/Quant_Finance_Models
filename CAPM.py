"""Portfolio Optimization and Efficient Frontier Analysis with CAPM
    - Calculate expected returns, variances, and covariances of assets.
    - Construct the efficient frontier to identify optimal portfolios.
    - Analyze the impact of adding a risk-free asset.
    - Perform sensitivity analysis on asset weights and risk aversion levels."""

"""1. Calculate Expected Returns, Variances, and Covariances of Assets
Start by calculating the basic statistics for a set of assets. This will involve calculating expected returns based on historical prices, variances, and covariances."""
import numpy as np
import pandas as pd

def calculate_statistics(prices):
    """
    Calculate expected returns, variances, and covariances of the assets.
    prices: DataFrame of asset prices where each column represents an asset
    """
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Expected returns (annualized)
    expected_returns = returns.mean() * 252
    
    # Covariance matrix (annualized)
    cov_matrix = returns.cov() * 252
    
    return expected_returns, cov_matrix

"""2. Construct the Efficient Frontier
The Efficient Frontier represents the set of portfolios that offer the highest expected return for a given level of risk or the lowest risk for a given level of expected return."""
from scipy.optimize import minimize

def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def minimize_volatility(target_return, expected_returns, cov_matrix):
    num_assets = len(expected_returns)
    initial_guess = np.repeat(1/num_assets, num_assets)
    bounds = ((0.0, 1.0),) * num_assets
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights: portfolio_return(weights, expected_returns) - target_return}
    ]
    result = minimize(portfolio_volatility, initial_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(expected_returns, cov_matrix, returns_range):
    efficient_portfolios = []
    for ret in returns_range:
        efficient_portfolios.append(minimize_volatility(ret, expected_returns, cov_matrix))
    return efficient_portfolios

"""3. Analyze the Impact of Adding a Risk-Free Asset
Adding a risk-free asset to the portfolio allows for the creation of a capital allocation line (CAL) which can potentially improve the risk-return profile."""
def capital_allocation_line(risk_free_rate, expected_returns, cov_matrix):
    num_assets = len(expected_returns)
    excess_returns = expected_returns - risk_free_rate
    weights = np.dot(np.linalg.inv(cov_matrix), excess_returns)
    weights /= np.sum(weights)
    return weights

"""4. Perform Sensitivity Analysis on Asset Weights and Risk Aversion Levels
Analyzing how changes in asset weights and investor's risk aversion affect the portfolio's expected return and risk."""
def sensitivity_analysis(base_weights, expected_returns, cov_matrix, perturbation=0.05):
    results = {}
    num_assets = len(base_weights)
    for i in range(num_assets):
        perturbed_weights = base_weights.copy()
        perturbed_weights[i] += perturbation
        perturbed_weights = np.clip(perturbed_weights, 0, 1)
        perturbed_weights /= np.sum(perturbed_weights)
        expected_return = portfolio_return(perturbed_weights, expected_returns)
        volatility = portfolio_volatility(perturbed_weights, cov_matrix)
        results[f'Asset {i+1} Sensitivity'] = {'Return': expected_return, 'Volatility': volatility}
    return results

"""Integration and Usage
Integrate these functions into your financial analysis pipeline. To use these functions, you'll need historical price data for the assets you want to analyze:
    1. Feed price data into calculate_statistics to get expected returns and the covariance matrix.
    2. Use these outputs to calculate the efficient frontier and perform other analyses.
    3. Run the sensitivity analysis to understand how changes in assumptions impact portfolio outcomes.

This setup requires historical financial data which you can obtain from financial data APIs or financial market datasets. The accuracy of predictions and optimizations heavily depends on the quality and granularity of the data used."""
