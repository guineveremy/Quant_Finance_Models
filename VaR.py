"""
Value at Risk (VaR) and Expected Shortfall Modeling with Backtesting and Stress Testing
    - Calculate VaR using Variance-Covariance, Historical Simulation, and Monte Carlo Simulation methods.
    - Implement Expected Shortfall as a more comprehensive risk measure.
    - Perform backtesting using the Kupiec test and Basel III traffic light approach.
    - Conduct stress testing to evaluate portfolio resilience under adverse conditions.

Key Areas covered
1. Calculation of VaR using three methods: Variance-Covariance, Historical Simulation, and Monte Carlo Simulation.
2. Implementation of Expected Shortfall (ES) as a measure of tail risk.
3. Backtesting of VaR models using the Kupiec test and Basel III traffic light approach.
4. Stress testing to evaluate the resilience of the portfolio under extreme market conditions.
"""

#1. Calculate VaR
import numpy as np
import pandas as pd
from scipy.stats import norm

def var_covar(portfolio_returns, alpha=0.05):
    # Variance-Covariance method
    mean = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns)
    var = norm.ppf(alpha, mean, sigma)
    return var

def historical_var(portfolio_returns, alpha=0.05):
    # Historical Simulation method
    var = np.percentile(portfolio_returns, alpha * 100)
    return var

def monte_carlo_var(S, mu, sigma, T, alpha=0.05, iterations=1000):
    # Monte Carlo Simulation
    results = []
    for _ in range(iterations):
        daily_returns = np.random.normal(mu, sigma, T)
        future_price = S * np.prod(1 + daily_returns)
        results.append(future_price)
    var = np.percentile(results, alpha * 100)
    return var

"""
2. Implement Expected Shortfall (ES)
Expected Shortfall (also known as Conditional VaR) measures the average loss assuming that the loss is beyond the VaR threshold.
"""
def expected_shortfall(portfolio_returns, alpha=0.05):
    var = historical_var(portfolio_returns, alpha)
    es = np.mean([x for x in portfolio_returns if x <= var])
    return es

"""
3. Backtesting VaR Models
Implement the Kupiec test and Basel III traffic light approach for backtesting the accuracy of the VaR models.
"""
def kupiec_test(portfolio_returns, var, confidence_level=0.95):
    # Number of exceedances
    exceedances = np.sum(portfolio_returns < var)
    N = len(portfolio_returns)
    p = 1 - confidence_level
    LR = -2 * np.log(((1 - p) ** (N - exceedances)) * (p ** exceedances) / ((1 - exceedances / N) ** (N - exceedances) * (exceedances / N) ** exceedances))
    return LR

def basel_traffic_light(test_statistic):
    # Basel III traffic light zones
    if test_statistic < 3.8415:
        return 'Green Zone'
    elif test_statistic < 6.6349:
        return 'Yellow Zone'
    else:
        return 'Red Zone'

"""
4. Stress Testing
Stress testing involves evaluating how portfolios might perform under extreme market conditions. Here, we apply hypothetical or historical worst-case scenarios.
"""
def stress_test(portfolio_returns, stress_factor):
    stressed_returns = portfolio_returns * stress_factor
    stressed_var = historical_var(stressed_returns)
    stressed_es = expected_shortfall(stressed_returns)
    return stressed_var, stressed_es

"""
Integration and Example Usage
"""
# Example Data
np.random.seed(42)
portfolio_returns = np.random.normal(-0.01, 0.1, 1000)

# Calculate VaR and ES
var = historical_var(portfolio_returns, alpha=0.05)
es = expected_shortfall(portfolio_returns, alpha=0.05)

# Backtesting
kupiec_stat = kupiec_test(portfolio_returns, var)
traffic_light = basel_traffic_light(kupiec_stat)

# Stress Testing
stress_var, stress_es = stress_test(portfolio_returns, 1.5)

print(f"VaR: {var}, ES: {es}")
print(f"Kupiec Test Statistic: {kupiec_stat}, Basel III Traffic Light: {traffic_light}")
print(f"Stressed VaR: {stress_var}, Stressed ES: {stress_es}")
