"""Yield Curve Construction and Analysis Using Nelson-Siegel and B-Spline Methods 
    - Implement the Nelson-Siegel and Nelson-Siegel-Svensson models for yield curve fitting.
    - Apply B-Spline methods for smooth curve fitting.
    - Conduct sensitivity analysis to interest rate changes.
    - Evaluate the accuracy of yield curve predictions using historical data."""

"""1. Implementing the Nelson-Siegel and Nelson-Siegel-Svensson Models
The Nelson-Siegel model is a way to fit yield curves using a formula that describes interest rates given time to maturity. The Nelson-Siegel-Svensson model extends this to better fit the long end of the curve."""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline

def nelson_siegel(t, beta0, beta1, beta2, tau):
    """
    Nelson-Siegel model function.
    t: time to maturity
    beta0, beta1, beta2, tau: parameters of the Nelson-Siegel model
    """
    return beta0 + (beta1 + beta2) * (tau / t) * (1 - np.exp(-t / tau)) - beta2 * np.exp(-t / tau)

def nelson_siegel_svensson(t, beta0, beta1, beta2, beta3, tau1, tau2):
    """
    Nelson-Siegel-Svensson model function.
    t: time to maturity
    beta0, beta1, beta2, beta3, tau1, tau2: parameters of the Nelson-Siegel-Svensson model
    """
    return (beta0 + (beta1 + beta2) * (tau1 / t) * (1 - np.exp(-t / tau1)) - beta2 * np.exp(-t / tau1) + 
            beta3 * ((tau2 / t) * (1 - np.exp(-t / tau2)) - np.exp(-t / tau2)))

def fit_nelson_siegel_svensson(t, y):
    """
    Fit the Nelson-Siegel-Svensson model using nonlinear least squares.
    t: array of times to maturity
    y: array of yields
    """
    objective = lambda params: np.sum((nelson_siegel_svensson(t, *params) - y)**2)
    initial_params = [0.01, -0.01, 0.01, 0.01, 1, 1]
    result = minimize(objective, initial_params)
    return result.x

"""2. Applying B-Spline Methods for Smooth Curve Fitting
B-Splines are useful for creating smooth, flexible curves and are widely used in finance for curve fitting."""
def fit_b_spline(t, y, k=3, s=None):
    """
    Fit a B-Spline to the given data.
    t: array of times to maturity
    y: array of yields
    k: degree of the spline
    s: smoothing factor
    """
    t, c, k = interpolate.splrep(t, y, k=k, s=s)
    spline = BSpline(t, c, k)
    return spline

def evaluate_spline(spline, t):
    """
    Evaluate the fitted B-Spline at specific times to maturity.
    spline: B-Spline object
    t: times to maturity
    """
    return spline(t)

"""3. Conducting Sensitivity Analysis
Here, we consider how the yield curves react to changes in interest rates. This involves slightly perturbing the rates and observing the changes in the curve."""
def sensitivity_analysis(fit_params, delta=0.01):
    """
    Conduct sensitivity analysis on Nelson-Siegel-Svensson parameters.
    fit_params: fitted parameters
    delta: small change to apply to each parameter
    """
    sensitivities = {}
    for i, param in enumerate(fit_params):
        params_up = fit_params.copy()
        params_up[i] += delta
        params_down = fit_params.copy()
        params_down[i] -= delta
        curve_up = nelson_siegel_svensson(t, *params_up)
        curve_down = nelson_siegel_svensson(t, *params_down)
        sensitivities[f'param_{i}'] = (curve_up - curve_down) / (2 * delta)
    return sensitivities

"""4. Evaluating the Accuracy of Yield Curve Predictions
For this, we need historical data. Fit the model to part of the data and test on the rest, comparing predicted yields to actual yields."""
def evaluate_model_accuracy(t_train, y_train, t_test, y_test):
    """
    Evaluate the model by fitting it on training data and testing on test data.
    """
    params = fit_nelson_siegel_svensson(t_train, y_train)
    predicted_y_test = nelson_siegel_svensson(t_test, *params)
    mse = np.mean((predicted_y_test - y_test)**2)
    return mse

"""Integration and Testing
Integrate these functions into your codebase, and test them with actual yield data. Make sure to fine-tune parameters and test thoroughly with historical data for accurate prediction and analysis.

These Python scripts provide a framework for your tasks. They need real yield data for proper testing and deployment, and you might need to adjust parameters based on the data specifics and desired smoothness of the fit."""