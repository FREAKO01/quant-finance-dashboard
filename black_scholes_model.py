"""
Black-Scholes Option Pricing Model Implementation
Author: AI Assistant
Date: 2025

This implementation provides a complete Black-Scholes model with Greeks calculations.
"""

import numpy as np
from scipy.stats import norm
import time

class BlackScholesModel:
    """
    Black-Scholes Model for European Option Pricing

    The Black-Scholes model assumes:
    - Constant volatility and risk-free rate
    - No dividends
    - European-style exercise
    - Geometric Brownian Motion for stock price

    Mathematical Foundation:
    The Black-Scholes PDE: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

    Call Option Formula: C = S₀N(d₁) - Ke^(-rT)N(d₂)
    Where:
    d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
    """

    def __init__(self, S, K, T, r, sigma):
        """
        Initialize Black-Scholes model parameters

        Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free rate
        sigma (float): Volatility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        """
        Calculate d1 parameter
        d1 measures how far the current stock price is from the strike price
        in units of standard deviation of the stock return
        """
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        """
        Calculate d2 parameter
        d2 = d1 - σ√T
        d2 represents the risk-adjusted probability of exercise
        """
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        """
        Calculate European call option price using Black-Scholes formula

        The formula represents:
        - S×N(d1): Expected value of stock at expiration if option finishes in-the-money
        - K×e^(-rT)×N(d2): Present value of strike price weighted by probability of exercise
        """
        d1_val = self.d1()
        d2_val = self.d2()

        call_price = (self.S * norm.cdf(d1_val) - 
                     self.K * np.exp(-self.r * self.T) * norm.cdf(d2_val))
        return call_price

    def put_price(self):
        """
        Calculate European put option price using put-call parity
        Put-call parity: P = C - S + Ke^(-rT)
        """
        call_price = self.call_price()
        put_price = call_price - self.S + self.K * np.exp(-self.r * self.T)
        return put_price

    def delta(self):
        """
        Calculate Delta: ∂V/∂S
        Delta measures the rate of change of option price with respect to stock price
        For calls: Δ = N(d1)
        For puts: Δ = N(d1) - 1
        """
        return norm.cdf(self.d1())

    def gamma(self):
        """
        Calculate Gamma: ∂²V/∂S²
        Gamma measures the rate of change of delta with respect to stock price
        Γ = φ(d1) / (S × σ × √T)
        where φ is the standard normal probability density function
        """
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def theta(self):
        """
        Calculate Theta: -∂V/∂t
        Theta measures the rate of change of option price with respect to time
        Often called "time decay"
        """
        d1_val = self.d1()
        d2_val = self.d2()

        theta = (-(self.S * norm.pdf(d1_val) * self.sigma) / (2 * np.sqrt(self.T)) -
                self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2_val))
        return theta / 365  # Per day

    def vega(self):
        """
        Calculate Vega: ∂V/∂σ
        Vega measures the rate of change of option price with respect to volatility
        ν = S × φ(d1) × √T
        """
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T) / 100  # Per 1% volatility change

    def rho(self):
        """
        Calculate Rho: ∂V/∂r
        Rho measures the rate of change of option price with respect to risk-free rate
        """
        d2_val = self.d2()
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2_val) / 100  # Per 1% rate change

def main():
    """Example usage of Black-Scholes model"""
    print("Black-Scholes Option Pricing Model")
    print("=" * 50)

    # Parameters
    S0 = 100.0    # Current stock price
    K = 100.0     # Strike price
    T = 1.0       # Time to maturity (1 year)
    r = 0.05      # Risk-free rate (5%)
    sigma = 0.2   # Volatility (20%)

    # Create model
    bs_model = BlackScholesModel(S0, K, T, r, sigma)

    # Calculate prices and Greeks
    start_time = time.time()
    call_price = bs_model.call_price()
    put_price = bs_model.put_price()
    calculation_time = time.time() - start_time

    print(f"Parameters:")
    print(f"Stock Price (S₀): ${S0}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Maturity (T): {T} year")
    print(f"Risk-free Rate (r): {r:.1%}")
    print(f"Volatility (σ): {sigma:.1%}")

    print(f"\nResults:")
    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price: ${put_price:.4f}")

    print(f"\nGreeks:")
    print(f"Delta (Δ): {bs_model.delta():.4f}")
    print(f"Gamma (Γ): {bs_model.gamma():.4f}")
    print(f"Theta (Θ): ${bs_model.theta():.4f} per day")
    print(f"Vega (ν): ${bs_model.vega():.4f} per 1% volatility")
    print(f"Rho (ρ): ${bs_model.rho():.4f} per 1% rate change")

    print(f"\nTechnical Details:")
    print(f"d₁ = {bs_model.d1():.4f}")
    print(f"d₂ = {bs_model.d2():.4f}")
    print(f"N(d₁) = {norm.cdf(bs_model.d1()):.4f}")
    print(f"N(d₂) = {norm.cdf(bs_model.d2()):.4f}")
    print(f"Calculation Time: {calculation_time:.6f} seconds")

if __name__ == "__main__":
    main()
