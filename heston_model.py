
"""
Heston Stochastic Volatility Model Implementation
Author: Arnav Sharma
Date: 2025

This implementation provides the Heston model for option pricing with stochastic volatility.
Includes both Monte Carlo simulation and semi-analytical pricing via characteristic functions.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

class HestonModel:
    """
    Heston Stochastic Volatility Model

    The Heston model extends Black-Scholes by making volatility stochastic:

    dS_t = rS_t dt + √v_t S_t dW₁_t
    dv_t = κ(θ - v_t) dt + ξ√v_t dW₂_t

    where:
    - S_t: Stock price at time t
    - v_t: Variance at time t (volatility squared)
    - r: Risk-free rate
    - κ: Rate of mean reversion of variance
    - θ: Long-term variance level
    - ξ: Volatility of volatility (vol-of-vol)
    - dW₁, dW₂: Correlated Wiener processes with correlation ρ

    The Feller condition 2κθ > ξ² ensures variance stays positive.
    """

    def __init__(self, S0, K, T, r, v0, kappa, theta, xi, rho):
        """
        Initialize Heston model parameters

        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free rate
        v0 (float): Initial variance
        kappa (float): Mean reversion speed
        theta (float): Long-term variance
        xi (float): Volatility of volatility
        rho (float): Correlation between stock and volatility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

        # Check Feller condition
        self.feller_condition = 2 * kappa * theta > xi**2
        if not self.feller_condition:
            print(f"Warning: Feller condition violated! 2κθ = {2*kappa*theta:.4f} ≤ ξ² = {xi**2:.4f}")

    def characteristic_function(self, phi, option_type='call'):
        """
        Heston characteristic function for semi-analytical pricing

        This is the fundamental building block for pricing using Fourier transforms.
        The characteristic function captures all the statistical properties of the
        log-stock price distribution under the Heston model.

        Parameters:
        phi (complex): Frequency parameter
        option_type (str): 'call' or 'put'

        Returns:
        complex: Characteristic function value
        """
        # Model parameters
        kappa, theta, xi, rho, v0, r, T = self.kappa, self.theta, self.xi, self.rho, self.v0, self.r, self.T

        # Choose integration parameters based on option type
        if option_type == 'call':
            u = 0.5
            a = 1
        else:  # put
            u = -0.5
            a = 0

        # Complex parameters
        d = np.sqrt((rho * xi * phi * 1j - kappa)**2 - xi**2 * (2 * u * phi * 1j - phi**2))
        g = (kappa - rho * xi * phi * 1j + d) / (kappa - rho * xi * phi * 1j - d)

        # Avoid numerical issues
        if np.abs(g) > 1:
            g = 1 / g
            d = -d

        # Calculate A and B functions
        A = (r * phi * 1j * T + 
             (kappa * theta / xi**2) * 
             (T * (kappa - rho * xi * phi * 1j + d) - 
              2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))

        B = ((kappa - rho * xi * phi * 1j + d) / xi**2) *             ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

        # Characteristic function
        char_func = np.exp(A + B * v0 + 1j * phi * np.log(self.S0))

        return char_func

    def european_call_price_analytical(self):
        """
        Price European call option using semi-analytical Heston formula

        This method uses the characteristic function and numerical integration
        to compute option prices. It's much faster than Monte Carlo for
        European options and provides exact solutions (up to numerical precision).

        Returns:
        float: Call option price
        """
        def integrand1(phi):
            """Integrand for probability P1"""
            char_func = self.characteristic_function(phi - 1j, 'call')
            numerator = np.exp(-1j * phi * np.log(self.K)) * char_func
            denominator = 1j * phi * self.S0
            return np.real(numerator / denominator)

        def integrand2(phi):
            """Integrand for probability P2"""
            char_func = self.characteristic_function(phi, 'call')
            numerator = np.exp(-1j * phi * np.log(self.K)) * char_func
            denominator = 1j * phi
            return np.real(numerator / denominator)

        # Numerical integration
        try:
            P1 = 0.5 + (1/np.pi) * quad(integrand1, 0, 100, limit=1000)[0]
            P2 = 0.5 + (1/np.pi) * quad(integrand2, 0, 100, limit=1000)[0]
        except:
            # If integration fails, return NaN
            return np.nan

        # Call price formula
        call_price = self.S0 * P1 - self.K * np.exp(-self.r * self.T) * P2

        return max(call_price, 0)  # Ensure non-negative price

    def monte_carlo_simulation(self, n_simulations=100000, n_steps=252, scheme='euler'):
        """
        Price options using Monte Carlo simulation with various discretization schemes

        Parameters:
        n_simulations (int): Number of Monte Carlo paths
        n_steps (int): Number of time steps
        scheme (str): Discretization scheme ('euler', 'milstein', 'full_truncation')

        Returns:
        tuple: (call_price, put_price, paths, variance_paths)
        """
        dt = self.T / n_steps

        # Initialize arrays
        S = np.zeros((n_simulations, n_steps + 1))
        v = np.zeros((n_simulations, n_steps + 1))

        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # Generate correlated random numbers
        for t in range(n_steps):
            # Independent standard normal random variables
            Z1 = np.random.normal(0, 1, n_simulations)
            Z2 = np.random.normal(0, 1, n_simulations)

            # Create correlated random variables
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            if scheme == 'euler':
                # Simple Euler-Maruyama scheme
                v_next = v[:, t] + self.kappa * (self.theta - v[:, t]) * dt +                          self.xi * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * W2
                v[:, t + 1] = np.maximum(v_next, 0)  # Reflection at zero

            elif scheme == 'full_truncation':
                # Full Truncation scheme (more accurate for CIR process)
                v_next = v[:, t] + self.kappa * (self.theta - np.maximum(v[:, t], 0)) * dt +                          self.xi * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * W2
                v[:, t + 1] = np.maximum(v_next, 0)

            elif scheme == 'milstein':
                # Milstein scheme (higher order accuracy)
                sqrt_v = np.sqrt(np.maximum(v[:, t], 0))
                v_next = (v[:, t] + self.kappa * (self.theta - v[:, t]) * dt + 
                         self.xi * sqrt_v * np.sqrt(dt) * W2 +
                         0.25 * self.xi**2 * dt * (W2**2 - 1))
                v[:, t + 1] = np.maximum(v_next, 0)

            # Stock price evolution
            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - 0.5 * v[:, t]) * dt + 
                np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * W1
            )

        # Calculate option prices
        final_prices = S[:, -1]
        call_payoffs = np.maximum(final_prices - self.K, 0)
        put_payoffs = np.maximum(self.K - final_prices, 0)

        call_price = np.exp(-self.r * self.T) * np.mean(call_payoffs)
        put_price = np.exp(-self.r * self.T) * np.mean(put_payoffs)

        return call_price, put_price, S, v

    def implied_volatility_smile(self, strikes, market_prices=None):
        """
        Generate implied volatility smile using Heston model

        Parameters:
        strikes (array): Array of strike prices
        market_prices (array): Market option prices (if available for calibration)

        Returns:
        tuple: (strikes, heston_prices, implied_vols)
        """
        heston_prices = []
        implied_vols = []

        original_K = self.K

        for K in strikes:
            self.K = K
            try:
                price = self.european_call_price_analytical()
                if not np.isnan(price) and price > 0:
                    heston_prices.append(price)

                    # Calculate implied volatility using Newton-Raphson
                    implied_vol = self._implied_volatility_newton(price)
                    implied_vols.append(implied_vol)
                else:
                    heston_prices.append(np.nan)
                    implied_vols.append(np.nan)
            except:
                heston_prices.append(np.nan)
                implied_vols.append(np.nan)

        self.K = original_K  # Restore original strike

        return strikes, np.array(heston_prices), np.array(implied_vols)

    def _implied_volatility_newton(self, market_price, max_iterations=100, tolerance=1e-6):
        """
        Calculate implied volatility using Newton-Raphson method
        """
        from scipy.stats import norm

        # Initial guess
        sigma = 0.2

        for i in range(max_iterations):
            d1 = (np.log(self.S0/self.K) + (self.r + 0.5*sigma**2)*self.T) / (sigma*np.sqrt(self.T))
            d2 = d1 - sigma*np.sqrt(self.T)

            # Black-Scholes price and vega
            bs_price = self.S0*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
            vega = self.S0 * norm.pdf(d1) * np.sqrt(self.T)

            if abs(vega) < 1e-10:  # Avoid division by zero
                break

            # Newton-Raphson update
            sigma_new = sigma - (bs_price - market_price) / vega

            if abs(sigma_new - sigma) < tolerance:
                return sigma_new

            sigma = max(sigma_new, 0.001)  # Keep sigma positive

        return sigma

def main():
    """Example usage of Heston model"""
    print("Heston Stochastic Volatility Model")
    print("=" * 50)

    # Model parameters
    S0 = 100.0      # Initial stock price
    K = 100.0       # Strike price
    T = 1.0         # Time to maturity
    r = 0.05        # Risk-free rate
    v0 = 0.04       # Initial variance (20% volatility)
    kappa = 2.0     # Mean reversion speed
    theta = 0.04    # Long-term variance (20% long-term volatility)
    xi = 0.3        # Volatility of volatility
    rho = -0.7      # Correlation (negative for leverage effect)

    # Create Heston model
    heston = HestonModel(S0, K, T, r, v0, kappa, theta, xi, rho)

    print(f"Parameters:")
    print(f"Stock Price (S₀): ${S0}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Maturity (T): {T} year")
    print(f"Risk-free Rate (r): {r:.1%}")
    print(f"Initial Variance (v₀): {v0:.4f} (σ₀ = {np.sqrt(v0):.1%})")
    print(f"Long-term Variance (θ): {theta:.4f} (σ∞ = {np.sqrt(theta):.1%})")
    print(f"Mean Reversion (κ): {kappa:.4f}")
    print(f"Vol-of-vol (ξ): {xi:.4f}")
    print(f"Correlation (ρ): {rho:.4f}")
    print(f"Feller Condition: {heston.feller_condition} (2κθ = {2*kappa*theta:.4f} vs ξ² = {xi**2:.4f})")

    # Semi-analytical pricing
    print(f"\nSemi-Analytical Pricing:")
    start_time = time.time()
    analytical_call = heston.european_call_price_analytical()
    analytical_time = time.time() - start_time

    if not np.isnan(analytical_call):
        print(f"Call Price: ${analytical_call:.4f}")
        print(f"Computation Time: {analytical_time:.4f} seconds")
    else:
        print("Semi-analytical pricing failed (numerical integration issues)")

    # Monte Carlo pricing
    print(f"\nMonte Carlo Pricing:")
    schemes = ['euler', 'full_truncation', 'milstein']
    n_sims = 50000

    for scheme in schemes:
        start_time = time.time()
        mc_call, mc_put, S_paths, v_paths = heston.monte_carlo_simulation(
            n_sims, n_steps=100, scheme=scheme
        )
        mc_time = time.time() - start_time

        print(f"{scheme.title()} Scheme:")
        print(f"  Call Price: ${mc_call:.4f}")
        print(f"  Put Price: ${mc_put:.4f}")
        print(f"  Computation Time: {mc_time:.2f} seconds")

        # Statistics from simulation
        final_vols = np.sqrt(v_paths[:, -1])
        print(f"  Final Volatility - Mean: {np.mean(final_vols):.1%}")
        print(f"  Final Volatility - Std: {np.std(final_vols):.1%}")

    # Implied volatility smile
    print(f"\nImplied Volatility Smile:")
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    _, heston_prices, implied_vols = heston.implied_volatility_smile(strikes)

    print(f"{'Strike':<8} {'Price':<10} {'Impl Vol':<10} {'Moneyness':<10}")
    print("-" * 40)
    for i, K in enumerate(strikes):
        if not np.isnan(heston_prices[i]):
            moneyness = K / S0
            print(f"{K:<8.0f} ${heston_prices[i]:<9.4f} {implied_vols[i]:<9.1%} {moneyness:<9.3f}")

    # Compare with Black-Scholes
    from scipy.stats import norm
    bs_vol = np.sqrt(v0)  # Use initial volatility
    d1 = (np.log(S0/K) + (r + 0.5*bs_vol**2)*T) / (bs_vol*np.sqrt(T))
    d2 = d1 - bs_vol*np.sqrt(T)
    bs_call = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    print(f"\nComparison with Black-Scholes:")
    print(f"Black-Scholes Call: ${bs_call:.4f}")
    if not np.isnan(analytical_call):
        print(f"Heston Call: ${analytical_call:.4f}")
        print(f"Difference: ${analytical_call - bs_call:.4f} ({(analytical_call - bs_call)/bs_call*100:+.2f}%)")

if __name__ == "__main__":
    main()
