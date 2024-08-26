"""Modeling and Pricing Derivatives Using Black-Scholes and Binomial Tree Methods
    - Implement the Black-Scholes model for European options.
    - Construct Binomial Tree models for pricing American options.
    - Analyze the assumptions, limitations, and differences between the models.
    - Calculate Greeks (Delta, Gamma, Theta, Vega, Rho) for sensitivity analysis."""

## 1. Black-Scholes Model for European Options
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S: stock price
    K: strike price
    T: time to maturity (in years)
    r: risk-free interest rate
    sigma: volatility of the underlying asset
    option_type: 'call' or 'put'
    """
    # Calculations
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    
    return price

## 2. Binomial Tree Model for American Options
def binomial_tree(S, K, T, r, sigma, N, option_type='call'):
    """
    S: stock price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    sigma: volatility
    N: number of time steps
    option_type: 'call' or 'put'
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize the stock price tree
    stock_price = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            stock_price[j, i] = S * (u**(i-j)) * (d**j)
    
    # Initialize the option value tree
    option_value = np.zeros_like(stock_price)
    # Set up the last column for payoff
    if option_type == 'call':
        option_value[:, N] = np.maximum(stock_price[:, N] - K, 0)
    else:
        option_value[:, N] = np.maximum(K - stock_price[:, N], 0)
    
    # Backward induction for option price
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            option_value[j, i] = max((stock_price[j, i] - K if option_type == 'call' else K - stock_price[j, i]), 
                                     np.exp(-r * dt) * (p * option_value[j, i+1] + (1 - p) * option_value[j+1, i+1]))
    
    return option_value[0, 0]

## 3. Calculate Greeks for Sensitivity Analysis
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Greeks
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - (r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

"""4. Analysis of Assumptions, Limitations, and Differences
Black-Scholes Model: Assumes a log-normal distribution of stock prices, constant volatility, and no dividends. It is not suitable for American options that can be exercised early.

Binomial Tree Model: More flexible than Black-Scholes; can handle varying volatilities, dividends, and early exercise features of American options. More computationally intensive."""
