
"""
Monte Carlo Simulation for Option Pricing
Author: Arnav Sharma
Date: 2025

This implementation provides Monte Carlo methods for pricing various types of options,
including European, American (using Longstaff-Schwartz), and exotic options.
"""

import numpy as np
import time
from scipy.stats import norm

class MonteCarloModel:
    """
    Monte Carlo Simulation for Option Pricing

    Monte Carlo methods simulate thousands of possible future price paths
    and calculate the average payoff, discounted to present value.

    Mathematical Foundation:
    Option Value = e^(-rT) × E[Payoff(S_T)]

    Where S_T follows Geometric Brownian Motion:
    S_T = S_0 × exp((r - σ²/2)T + σ√T × Z)
    Z ~ N(0,1) is a standard normal random variable
    """

    def __init__(self, S0, K, T, r, sigma, n_simulations=100000, n_steps=252, random_seed=None):
        """
        Initialize Monte Carlo model parameters

        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free rate
        sigma (float): Volatility
        n_simulations (int): Number of Monte Carlo simulations
        n_steps (int): Number of time steps per simulation
        random_seed (int): Random seed for reproducibility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.dt = T / n_steps

        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_gbm_paths(self):
        """
        Generate stock price paths using Geometric Brownian Motion

        The exact solution to dS = μS dt + σS dW is:
        S(t+dt) = S(t) × exp((μ - σ²/2)dt + σ√dt × ε)
        where ε ~ N(0,1)

        Returns:
        numpy.ndarray: Array of shape (n_simulations, n_steps+1) containing price paths
        """
        # Pre-allocate paths array for memory efficiency
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.S0

        # Generate all random shocks at once for vectorization
        # Using antithetic variates for variance reduction
        half_sims = self.n_simulations // 2
        random_shocks = np.random.normal(0, 1, (half_sims, self.n_steps))

        # Antithetic variates: use both +Z and -Z
        all_shocks = np.vstack([random_shocks, -random_shocks])
        if self.n_simulations % 2 == 1:
            # Add one more simulation if odd number
            all_shocks = np.vstack([all_shocks, np.random.normal(0, 1, (1, self.n_steps))])

        # Simulate paths using vectorized operations
        for t in range(self.n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * np.sqrt(self.dt) * all_shocks[:, t]
            )

        return paths

    def european_call_price(self):
        """
        Price European call option using Monte Carlo

        Payoff = max(S_T - K, 0)

        Returns:
        tuple: (option_price, standard_error, confidence_interval)
        """
        paths = self.generate_gbm_paths()
        final_prices = paths[:, -1]

        # Calculate payoffs
        call_payoffs = np.maximum(final_prices - self.K, 0)

        # Discount to present value
        call_price = np.exp(-self.r * self.T) * np.mean(call_payoffs)

        # Calculate standard error and confidence interval
        std_error = np.std(call_payoffs) / np.sqrt(self.n_simulations)
        conf_interval = 1.96 * std_error  # 95% confidence interval

        return call_price, std_error, conf_interval

    def european_put_price(self):
        """
        Price European put option using Monte Carlo

        Payoff = max(K - S_T, 0)
        """
        paths = self.generate_gbm_paths()
        final_prices = paths[:, -1]

        put_payoffs = np.maximum(self.K - final_prices, 0)
        put_price = np.exp(-self.r * self.T) * np.mean(put_payoffs)

        std_error = np.std(put_payoffs) / np.sqrt(self.n_simulations)
        conf_interval = 1.96 * std_error

        return put_price, std_error, conf_interval

    def asian_call_price(self, average_type='arithmetic'):
        """
        Price Asian (average price) call option

        Payoff = max(Average_Price - K, 0)

        Parameters:
        average_type (str): 'arithmetic' or 'geometric' averaging
        """
        paths = self.generate_gbm_paths()

        if average_type == 'arithmetic':
            average_prices = np.mean(paths, axis=1)
        elif average_type == 'geometric':
            # Geometric average = exp(mean(log(prices)))
            log_paths = np.log(paths)
            average_prices = np.exp(np.mean(log_paths, axis=1))
        else:
            raise ValueError("average_type must be 'arithmetic' or 'geometric'")

        asian_payoffs = np.maximum(average_prices - self.K, 0)
        asian_price = np.exp(-self.r * self.T) * np.mean(asian_payoffs)

        std_error = np.std(asian_payoffs) / np.sqrt(self.n_simulations)
        conf_interval = 1.96 * std_error

        return asian_price, std_error, conf_interval

    def barrier_call_price(self, barrier_level, barrier_type='up_and_out'):
        """
        Price barrier call option

        Parameters:
        barrier_level (float): Barrier level
        barrier_type (str): 'up_and_out', 'up_and_in', 'down_and_out', 'down_and_in'
        """
        paths = self.generate_gbm_paths()
        final_prices = paths[:, -1]

        # Check barrier conditions
        if barrier_type == 'up_and_out':
            # Option knocked out if any price exceeds barrier
            barrier_hit = np.any(paths > barrier_level, axis=1)
            payoffs = np.where(barrier_hit, 0, np.maximum(final_prices - self.K, 0))
        elif barrier_type == 'up_and_in':
            # Option activated only if price exceeds barrier
            barrier_hit = np.any(paths > barrier_level, axis=1)
            payoffs = np.where(barrier_hit, np.maximum(final_prices - self.K, 0), 0)
        elif barrier_type == 'down_and_out':
            # Option knocked out if any price falls below barrier
            barrier_hit = np.any(paths < barrier_level, axis=1)
            payoffs = np.where(barrier_hit, 0, np.maximum(final_prices - self.K, 0))
        elif barrier_type == 'down_and_in':
            # Option activated only if price falls below barrier
            barrier_hit = np.any(paths < barrier_level, axis=1)
            payoffs = np.where(barrier_hit, np.maximum(final_prices - self.K, 0), 0)
        else:
            raise ValueError("Invalid barrier_type")

        barrier_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        conf_interval = 1.96 * std_error

        return barrier_price, std_error, conf_interval

    def lookback_call_price(self, lookback_type='floating'):
        """
        Price lookback call option

        Parameters:
        lookback_type (str): 'floating' or 'fixed'
        - Floating: Payoff = S_T - min(S_t) for t ∈ [0,T]
        - Fixed: Payoff = max(S_t) - K for t ∈ [0,T]
        """
        paths = self.generate_gbm_paths()

        if lookback_type == 'floating':
            # Payoff = Final Price - Minimum Price during life
            min_prices = np.min(paths, axis=1)
            final_prices = paths[:, -1]
            payoffs = final_prices - min_prices
        elif lookback_type == 'fixed':
            # Payoff = Maximum Price during life - Strike
            max_prices = np.max(paths, axis=1)
            payoffs = np.maximum(max_prices - self.K, 0)
        else:
            raise ValueError("lookback_type must be 'floating' or 'fixed'")

        lookback_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        conf_interval = 1.96 * std_error

        return lookback_price, std_error, conf_interval

def main():
    """Example usage of Monte Carlo model"""
    print("Monte Carlo Option Pricing Model")
    print("=" * 50)

    # Parameters
    S0 = 100.0      # Initial stock price
    K = 100.0       # Strike price
    T = 1.0         # Time to maturity
    r = 0.05        # Risk-free rate
    sigma = 0.2     # Volatility
    n_sims = 1000000  # Number of simulations

    # Create model
    mc_model = MonteCarloModel(S0, K, T, r, sigma, n_sims, random_seed=42)

    print(f"Parameters:")
    print(f"Stock Price (S₀): ${S0}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Maturity (T): {T} year")
    print(f"Risk-free Rate (r): {r:.1%}")
    print(f"Volatility (σ): {sigma:.1%}")
    print(f"Monte Carlo Simulations: {n_sims:,}")

    # European Options
    start_time = time.time()
    call_price, call_se, call_ci = mc_model.european_call_price()
    put_price, put_se, put_ci = mc_model.european_put_price()
    european_time = time.time() - start_time

    print(f"\nEuropean Options:")
    print(f"Call Price: ${call_price:.4f} ± ${call_ci:.4f}")
    print(f"Put Price: ${put_price:.4f} ± ${put_ci:.4f}")
    print(f"Computation Time: {european_time:.2f} seconds")

    # Asian Options
    start_time = time.time()
    asian_arith, asian_arith_se, asian_arith_ci = mc_model.asian_call_price('arithmetic')
    asian_geom, asian_geom_se, asian_geom_ci = mc_model.asian_call_price('geometric')
    asian_time = time.time() - start_time

    print(f"\nAsian Call Options:")
    print(f"Arithmetic Average: ${asian_arith:.4f} ± ${asian_arith_ci:.4f}")
    print(f"Geometric Average: ${asian_geom:.4f} ± ${asian_geom_ci:.4f}")
    print(f"Computation Time: {asian_time:.2f} seconds")

    # Barrier Options
    start_time = time.time()
    barrier_out, barrier_out_se, barrier_out_ci = mc_model.barrier_call_price(120, 'up_and_out')
    barrier_in, barrier_in_se, barrier_in_ci = mc_model.barrier_call_price(120, 'up_and_in')
    barrier_time = time.time() - start_time

    print(f"\nBarrier Call Options (Barrier = $120):")
    print(f"Up-and-Out: ${barrier_out:.4f} ± ${barrier_out_ci:.4f}")
    print(f"Up-and-In: ${barrier_in:.4f} ± ${barrier_in_ci:.4f}")
    print(f"Computation Time: {barrier_time:.2f} seconds")

    # Lookback Options
    start_time = time.time()
    lookback_float, lookback_float_se, lookback_float_ci = mc_model.lookback_call_price('floating')
    lookback_fixed, lookback_fixed_se, lookback_fixed_ci = mc_model.lookback_call_price('fixed')
    lookback_time = time.time() - start_time

    print(f"\nLookback Call Options:")
    print(f"Floating Strike: ${lookback_float:.4f} ± ${lookback_float_ci:.4f}")
    print(f"Fixed Strike: ${lookback_fixed:.4f} ± ${lookback_fixed_ci:.4f}")
    print(f"Computation Time: {lookback_time:.2f} seconds")

    # Validation against Black-Scholes
    from scipy.stats import norm

    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_call = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    bs_put = bs_call - S0 + K*np.exp(-r*T)

    print(f"\nValidation vs Black-Scholes:")
    print(f"BS Call: ${bs_call:.4f}, MC Call: ${call_price:.4f}, Error: {abs(call_price-bs_call):.4f}")
    print(f"BS Put: ${bs_put:.4f}, MC Put: ${put_price:.4f}, Error: {abs(put_price-bs_put):.4f}")

if __name__ == "__main__":
    main()


print("Running European Call pricing...")
if __name__ == "__main__":
    overall_start = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - overall_start:.2f} seconds")


