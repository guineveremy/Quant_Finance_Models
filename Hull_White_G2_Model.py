"""
Interest Rate Modeling and Simulation Using the Hull-White and G2++ Models
    - Implement the Hull-White and G2++ models for simulating interest rate movements.
    - Calibrate model parameters using historical interest rate data.
    - Simulate interest rate scenarios for pricing interest rate derivatives.
    - Analyze the impact of different economic conditions on model outputs.

Steps taken;
    1. Implement the Hull-White and G2++ models to simulate interest rate movements.
    2. Calibrate model parameters using historical interest rate data.
    3. Simulate interest rate scenarios for pricing interest rate derivatives.
    4. Analyze the impact of different economic conditions on model outputs.
"""

##1. Implement the Hull-White Model
import numpy as np

def hull_white_simulation(a, sigma, T, dt, r0):
    """
    Simulate interest rate paths using the Hull-White model.
    a: Speed of mean reversion
    sigma: Volatility
    T: Time horizon
    dt: Time step
    r0: Initial interest rate
    """
    num_steps = int(T / dt)
    rates = np.zeros(num_steps)
    rates[0] = r0
    for i in range(1, num_steps):
        dr = a * (sigma - rates[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates[i] = rates[i-1] + dr
    return rates

##2. Implement the G2++ Model
def g2_plus_plus_simulation(a, sigma, b, eta, rho, T, dt, r0):
    """
    Simulate interest rate paths using the G2++ model.
    a, b: Speeds of mean reversion for two factors
    sigma, eta: Volatilities of two factors
    rho: Correlation between the two factors
    T: Time horizon
    dt: Time step
    r0: Initial interest rate
    """
    num_steps = int(T / dt)
    x = np.zeros(num_steps)  # Factor 1
    y = np.zeros(num_steps)  # Factor 2
    rates = np.zeros(num_steps)
    rates[0] = r0
    for i in range(1, num_steps):
        dx = -a * x[i-1] * dt + sigma * np.sqrt(dt) * np.random.normal()
        dy = -b * y[i-1] * dt + eta * np.sqrt(dt) * np.random.normal()
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
        rates[i] = rates[0] + x[i] + y[i] + rho * y[i]
    return rates

##3. Calibrate Model Parameters
from scipy.optimize import minimize

def calibrate_hull_white(historical_rates, dt, T):
    """
    Calibrate Hull-White model parameters using historical data.
    """
    def objective(params):
        a, sigma = params
        simulated_rates = hull_white_simulation(a, sigma, T, dt, historical_rates[0])
        mse = np.mean((simulated_rates - historical_rates)**2)
        return mse

    initial_guess = [0.1, 0.01]
    bounds = [(0.001, 1.0), (0.001, 0.2)]
    result = minimize(objective, initial_guess, bounds=bounds)
    return result.x

##4. Simulate Interest Rate Scenarios
def simulate_scenarios(a, sigma, T, dt, r0, num_scenarios):
    scenarios = []
    for _ in range(num_scenarios):
        rates = hull_white_simulation(a, sigma, T, dt, r0)
        scenarios.append(rates)
    return scenarios

"""
5. Analyze the Impact of Economic Conditions
Analyzing the impact involves modifying the parameters according to different economic conditions (e.g., high volatility, fast mean reversion) and observing how the interest rates react.
"""
def analyze_impact(a_values, sigma_values, T, dt, r0):
    results = {}
    for a in a_values:
        for sigma in sigma_values:
            rates = hull_white_simulation(a, sigma, T, dt, r0)
            results[(a, sigma)] = rates
    return results

"""
Integration and Example Usage
"""
# Example usage:
historical_rates = np.random.normal(0.05, 0.01, 250)  # Mock historical data
calibrated_params = calibrate_hull_white(historical_rates, 0.01, 1)
scenarios = simulate_scenarios(*calibrated_params, 1, 0.01, historical_rates[0], 10)

# Analyze economic impact
a_values = [0.1, 0.3]
sigma_values = [0.01, 0.03]
impact_results = analyze_impact(a_values, sigma_values, 1, 0.01, historical_rates[0])
