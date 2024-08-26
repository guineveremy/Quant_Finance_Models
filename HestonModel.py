"""
Stochastic Volatility Modeling with the Heston Model
    - Implement the Heston model for pricing derivatives with stochastic volatility.
    - Calibrate model parameters using historical market data.
    - Analyze the model’s ability to capture the volatility smile.
    - Compare with the Black-Scholes model to assess improvements in accuracy.
"""

"""
1. Implement the Heston Model for Pricing Derivatives
The Heston model assumes that volatility is a stochastic process. It uses two stochastic differential equations (SDEs): one for the asset price and another for the variance.
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def heston_model_pricing(S, K, T, r, v0, kappa, theta, sigma, rho):
    """
    Price European options using the Heston model.
    S: Initial stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    v0: Initial variance
    kappa: Rate of mean reversion
    theta: Long-term variance
    sigma: Volatility of variance
    rho: Correlation between stock and variance
    """
    def integrand(phi):
        d1 = np.sqrt((kappa - 1j*rho*sigma*phi)**2 + (phi**2 + 1j*phi)*sigma**2)
        g = (kappa - 1j*rho*sigma*phi - d1) / (kappa - 1j*rho*sigma*phi + d1)
        C = r*1j*phi*T + (kappa*theta/sigma**2)*((kappa - 1j*rho*sigma*phi - d1)*T - 2*np.log((1 - g*np.exp(-d1*T))/(1 - g)))
        D = ((kappa - 1j*rho*sigma*phi - d1) / sigma**2) * ((1 - np.exp(-d1*T)) / (1 - g*np.exp(-d1*T)))
        f = np.exp(C + D*v0 + 1j*phi*np.log(S))
        return np.real(np.exp(-1j*phi*np.log(K))*f / (1j*phi))
    
    P1 = 0.5 + (1/np.pi)*quad(integrand, 0, np.inf)[0]
    P2 = 0.5 + (1/np.pi)*quad(integrand, 0, np.inf)[0] - np.exp(-r*T) * S / K
    
    return S*P1 - K*np.exp(-r*T)*P2

"""
2. Calibrate Model Parameters Using Historical Market Data
Calibration involves optimizing the Heston model parameters to fit market data, typically done using a least squares approach or other optimization techniques to minimize the difference between model and market prices."""
from scipy.optimize import minimize

def calibrate_heston(market_prices, strikes, maturities, S, r, v0):
    """
    Calibrate the Heston model parameters to market data.
    """
    def objective(params):
        kappa, theta, sigma, rho = params
        model_prices = [heston_model_pricing(S, K, T, r, v0, kappa, theta, sigma, rho) for K, T in zip(strikes, maturities)]
        return np.sum((model_prices - market_prices) ** 2)
    
    initial_guess = [0.5, 0.2, 0.1, -0.1]
    bounds = [(0.01, 3), (0.01, 0.5), (0.01, 0.5), (-0.99, 0.99)]
    result = minimize(objective, initial_guess, bounds=bounds)
    return result.x

"""
3. Analyze the Model’s Ability to Capture the Volatility Smile
Volatility smile can be analyzed by plotting implied volatilities across strike prices at a fixed maturity.
"""
import matplotlib.pyplot as plt

def plot_volatility_smile(strikes, implied_vols):
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, implied_vols, marker='o')
    plt.title('Volatility Smile')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.grid(True)
    plt.show()

"""
4. Compare with the Black-Scholes Model
Assess improvements by comparing market prices, model prices from Black-Scholes, and Heston models for the same derivatives.
"""
def compare_models(strikes, maturities, market_prices, S, r, v0, calibrated_params):
    kappa, theta, sigma, rho = calibrated_params
    bs_prices = [black_scholes(S, K, T, r, np.sqrt(v0), 'call') for K, T in zip(strikes, maturities)]
    heston_prices = [heston_model_pricing(S, K, T, r, v0, kappa, theta, sigma, rho) for K, T in zip(strikes, maturities)]
    
    for i, K in enumerate(strikes):
        print(f"Strike: {K}, Market: {market_prices[i]}, Black-Scholes: {bs_prices[i]}, Heston: {heston_prices[i]}")

"""
Integration and Testing
To test these functions, we will need to gather historical option data (market prices, strikes, maturities) and the current stock price. Then we would calibrate the Heston model to this data, evaluate the volatility smile, and compare it against the predictions from the Black-Scholes model to check for improvements in pricing accuracy.

This comprehensive implementation should provide a robust framework for stochastic volatility modeling in a financial context. 

Freely to adjust the code to fit your specific data and optimization needs.
"""
