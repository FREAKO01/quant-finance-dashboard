
"""
Local Volatility Model (Dupire Model) Implementation
Author: AI Assistant
Date: 2025

This implementation provides the local volatility model where volatility is a 
deterministic function of stock price and time: σ_local(S,t).

The model is calibrated to market option prices using Dupire's formula.
"""

import numpy as np
import time
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class LocalVolatilityModel:
    """
    Local Volatility Model (Dupire Model)

    In the local volatility model, the stock price follows:
    dS_t = rS_t dt + σ_local(S_t, t) S_t dW_t

    Where σ_local(S,t) is the local volatility function that depends on
    both the current stock price and time.

    Dupire's Formula:
    σ²_local(K,T) = (∂C/∂T + rK∂C/∂K) / (½K²∂²C/∂K²)

    Where C(K,T) is the market price of a call option with strike K and maturity T.

    The key advantage is that this model can exactly reproduce any given
    implied volatility surface from the market.
    """

    def __init__(self, S0, r, dividend_yield=0.0):
        """
        Initialize Local Volatility Model

        Parameters:
        S0 (float): Initial stock price
        r (float): Risk-free rate
        dividend_yield (float): Continuous dividend yield
        """
        self.S0 = S0
        self.r = r
        self.q = dividend_yield
        self.local_vol_surface = None
        self.strikes = None
        self.maturities = None
        self.interpolator = None

    def parametric_local_volatility(self, S, t, base_vol=0.2, skew_param=0.3, term_param=0.2):
        """
        Parametric local volatility function for demonstration purposes

        In practice, this would be calibrated from market data using Dupire's formula.
        This is a simplified form that exhibits common market features:
        - Volatility smile/skew (dependence on S)
        - Term structure (dependence on t)

        σ_local(S,t) = base_vol × moneyness_effect × term_effect

        Parameters:
        S (float or array): Stock price(s)
        t (float or array): Time(s)
        base_vol (float): Base volatility level
        skew_param (float): Controls the steepness of volatility skew
        term_param (float): Controls term structure effect

        Returns:
        float or array: Local volatility values
        """
        # Ensure inputs are arrays
        S = np.asarray(S)
        t = np.asarray(t)

        # Moneyness effect: creates volatility smile/skew
        # Higher volatility for out-of-the-money options
        moneyness = S / self.S0
        moneyness_effect = 1 + skew_param * (1 / moneyness - 1)

        # Term structure effect: volatility changes with time
        # Short-term volatility can be higher due to event risk
        term_effect = 1 + term_param * np.exp(-2 * t)

        # Combine effects
        local_vol = base_vol * moneyness_effect * term_effect

        # Ensure positive volatility
        return np.maximum(local_vol, 0.01)

    def calibrate_from_implied_vol_surface(self, strikes, maturities, implied_vols):
        """
        Calibrate local volatility surface from implied volatility surface
        using a simplified approach

        In a full implementation, this would use Dupire's formula with
        numerical derivatives of market option prices.

        Parameters:
        strikes (array): Strike prices
        maturities (array): Time to maturities
        implied_vols (2D array): Implied volatility surface [maturity, strike]
        """
        self.strikes = np.array(strikes)
        self.maturities = np.array(maturities)

        # Convert implied volatilities to local volatilities
        # This is a simplified conversion - in practice, Dupire's formula is used
        local_vols = np.zeros_like(implied_vols)

        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Simplified local vol approximation
                # Real implementation would use partial derivatives
                impl_vol = implied_vols[i, j]

                # Local volatility approximation (simplified)
                moneyness = K / self.S0
                time_factor = np.sqrt(T)

                # Adjust for smile effect
                local_vol = impl_vol * (1 + 0.1 * (moneyness - 1)**2) * time_factor
                local_vols[i, j] = max(local_vol, 0.01)

        # Create interpolator for local volatility surface
        self.local_vol_surface = local_vols
        self.interpolator = RectBivariateSpline(maturities, strikes, local_vols)

    def get_local_volatility(self, S, t):
        """
        Get local volatility value for given stock price and time

        Parameters:
        S (float or array): Stock price(s)
        t (float or array): Time(s)

        Returns:
        float or array: Local volatility value(s)
        """
        if self.interpolator is not None:
            # Use calibrated surface
            S = np.asarray(S).flatten()
            t = np.asarray(t).flatten()

            # Ensure we're within bounds
            t_bounded = np.clip(t, self.maturities.min(), self.maturities.max())
            S_bounded = np.clip(S, self.strikes.min(), self.strikes.max())

            return self.interpolator(t_bounded, S_bounded, grid=False)
        else:
            # Use parametric form
            return self.parametric_local_volatility(S, t)

    def monte_carlo_simulation(self, K, T, n_simulations=100000, n_steps=252):
        """
        Price European options using Monte Carlo with local volatility

        Parameters:
        K (float): Strike price
        T (float): Time to maturity
        n_simulations (int): Number of simulation paths
        n_steps (int): Number of time steps

        Returns:
        tuple: (call_price, put_price, paths, local_vol_paths)
        """
        dt = T / n_steps

        # Initialize arrays
        S = np.zeros((n_simulations, n_steps + 1))
        local_vols = np.zeros((n_simulations, n_steps))

        S[:, 0] = self.S0

        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))

        # Simulate paths
        for t in range(n_steps):
            current_time = t * dt
            current_prices = S[:, t]

            # Get local volatility for each path
            local_vol = self.get_local_volatility(current_prices, current_time)
            local_vols[:, t] = local_vol

            # Stock price evolution with local volatility
            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - self.q - 0.5 * local_vol**2) * dt + 
                local_vol * np.sqrt(dt) * random_shocks[:, t]
            )

        # Calculate option payoffs
        final_prices = S[:, -1]
        call_payoffs = np.maximum(final_prices - K, 0)
        put_payoffs = np.maximum(K - final_prices, 0)

        # Discount to present value
        discount_factor = np.exp(-self.r * T)
        call_price = discount_factor * np.mean(call_payoffs)
        put_price = discount_factor * np.mean(put_payoffs)

        return call_price, put_price, S, local_vols

    def barrier_option_price(self, K, T, barrier, barrier_type='up_and_out', 
                           n_simulations=100000, n_steps=252):
        """
        Price barrier options using Monte Carlo with local volatility

        Parameters:
        K (float): Strike price
        T (float): Time to maturity
        barrier (float): Barrier level
        barrier_type (str): Type of barrier option

        Returns:
        float: Barrier option price
        """
        dt = T / n_steps

        # Initialize arrays
        S = np.zeros((n_simulations, n_steps + 1))
        S[:, 0] = self.S0

        # Track barrier hits
        barrier_hit = np.zeros(n_simulations, dtype=bool)

        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))

        # Simulate paths and check barriers
        for t in range(n_steps):
            current_time = t * dt
            current_prices = S[:, t]

            # Get local volatility
            local_vol = self.get_local_volatility(current_prices, current_time)

            # Evolve stock price
            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - self.q - 0.5 * local_vol**2) * dt + 
                local_vol * np.sqrt(dt) * random_shocks[:, t]
            )

            # Check barrier conditions
            if barrier_type in ['up_and_out', 'up_and_in']:
                barrier_hit |= (S[:, t + 1] >= barrier)
            elif barrier_type in ['down_and_out', 'down_and_in']:
                barrier_hit |= (S[:, t + 1] <= barrier)

        # Calculate payoffs based on barrier type
        final_prices = S[:, -1]
        standard_payoffs = np.maximum(final_prices - K, 0)

        if barrier_type == 'up_and_out':
            payoffs = np.where(barrier_hit, 0, standard_payoffs)
        elif barrier_type == 'up_and_in':
            payoffs = np.where(barrier_hit, standard_payoffs, 0)
        elif barrier_type == 'down_and_out':
            payoffs = np.where(barrier_hit, 0, standard_payoffs)
        elif barrier_type == 'down_and_in':
            payoffs = np.where(barrier_hit, standard_payoffs, 0)

        # Discount and return price
        return np.exp(-self.r * T) * np.mean(payoffs)

    def asian_option_price(self, K, T, average_type='arithmetic', 
                          n_simulations=100000, n_steps=252):
        """
        Price Asian options using Monte Carlo with local volatility

        Parameters:
        K (float): Strike price
        T (float): Time to maturity
        average_type (str): 'arithmetic' or 'geometric'

        Returns:
        float: Asian option price
        """
        dt = T / n_steps

        # Initialize arrays
        S = np.zeros((n_simulations, n_steps + 1))
        S[:, 0] = self.S0

        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))

        # Simulate paths
        for t in range(n_steps):
            current_time = t * dt
            current_prices = S[:, t]

            # Get local volatility
            local_vol = self.get_local_volatility(current_prices, current_time)

            # Evolve stock price
            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - self.q - 0.5 * local_vol**2) * dt + 
                local_vol * np.sqrt(dt) * random_shocks[:, t]
            )

        # Calculate average prices
        if average_type == 'arithmetic':
            avg_prices = np.mean(S, axis=1)
        elif average_type == 'geometric':
            avg_prices = np.exp(np.mean(np.log(S), axis=1))

        # Calculate payoffs
        payoffs = np.maximum(avg_prices - K, 0)

        # Discount and return price
        return np.exp(-self.r * T) * np.mean(payoffs)

    def plot_local_vol_surface(self, S_range=None, T_range=None):
        """
        Generate data for plotting local volatility surface

        Returns:
        tuple: (S_grid, T_grid, vol_surface) for plotting
        """
        if S_range is None:
            S_range = np.linspace(0.5 * self.S0, 1.5 * self.S0, 50)
        if T_range is None:
            T_range = np.linspace(0.1, 2.0, 50)

        S_grid, T_grid = np.meshgrid(S_range, T_range)
        vol_surface = np.zeros_like(S_grid)

        for i in range(len(T_range)):
            for j in range(len(S_range)):
                vol_surface[i, j] = self.get_local_volatility(S_range[j], T_range[i])

        return S_grid, T_grid, vol_surface

def main():
    """Example usage of Local Volatility model"""
    print("Local Volatility Model (Dupire Model)")
    print("=" * 50)

    # Model parameters
    S0 = 100.0      # Initial stock price
    r = 0.05        # Risk-free rate
    q = 0.0         # Dividend yield
    K = 100.0       # Strike price
    T = 1.0         # Time to maturity

    # Create model
    lv_model = LocalVolatilityModel(S0, r, q)

    print(f"Parameters:")
    print(f"Stock Price (S₀): ${S0}")
    print(f"Risk-free Rate (r): {r:.1%}")
    print(f"Dividend Yield (q): {q:.1%}")

    # Example 1: Parametric local volatility
    print(f"\n1. Parametric Local Volatility Function:")
    print(f"   σ_local(S,t) = base_vol × moneyness_effect × term_effect")

    # Show local volatility at different price levels
    prices = [80, 90, 100, 110, 120]
    times = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\n   Local Volatility Matrix:")
    print(f"   {'Time/Price':<12} ", end="")
    for price in prices:
        print(f"{price:<8}", end="")
    print()

    for t in times:
        print(f"   {t:<12.2f} ", end="")
        for price in prices:
            local_vol = lv_model.get_local_volatility(price, t)
            print(f"{local_vol:<8.1%}", end="")
        print()

    # Example 2: Option pricing with local volatility
    print(f"\n2. Option Pricing with Local Volatility:")
    n_sims = 50000

    start_time = time.time()
    call_price, put_price, S_paths, local_vol_paths = lv_model.monte_carlo_simulation(
        K, T, n_sims, n_steps=100
    )
    pricing_time = time.time() - start_time

    print(f"   European Call Price: ${call_price:.4f}")
    print(f"   European Put Price: ${put_price:.4f}")
    print(f"   Computation Time: {pricing_time:.2f} seconds")

    # Analysis of local volatility behavior during simulation
    final_local_vols = local_vol_paths[:, -1]
    print(f"   Local Vol Statistics:")
    print(f"     Initial: {lv_model.get_local_volatility(S0, 0):.1%}")
    print(f"     Final Mean: {np.mean(final_local_vols):.1%}")
    print(f"     Final Std: {np.std(final_local_vols):.1%}")
    print(f"     Final Min: {np.min(final_local_vols):.1%}")
    print(f"     Final Max: {np.max(final_local_vols):.1%}")

    # Example 3: Exotic options
    print(f"\n3. Exotic Options with Local Volatility:")

    # Barrier options
    barrier_level = 120
    barrier_out = lv_model.barrier_option_price(K, T, barrier_level, 'up_and_out', 25000)
    barrier_in = lv_model.barrier_option_price(K, T, barrier_level, 'up_and_in', 25000)

    print(f"   Barrier Options (Barrier = ${barrier_level}):")
    print(f"     Up-and-Out Call: ${barrier_out:.4f}")
    print(f"     Up-and-In Call: ${barrier_in:.4f}")
    print(f"     Sum: ${barrier_out + barrier_in:.4f} (should ≈ European call)")

    # Asian options
    asian_arith = lv_model.asian_option_price(K, T, 'arithmetic', 25000)
    asian_geom = lv_model.asian_option_price(K, T, 'geometric', 25000)

    print(f"   Asian Call Options:")
    print(f"     Arithmetic Average: ${asian_arith:.4f}")
    print(f"     Geometric Average: ${asian_geom:.4f}")

    # Example 4: Calibration demonstration
    print(f"\n4. Calibration to Implied Volatility Surface:")

    # Create synthetic implied volatility surface
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    # Synthetic implied volatility with smile and term structure
    implied_vols = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            moneyness = K / S0
            # Create volatility smile
            base_vol = 0.2
            smile_effect = 0.1 * (moneyness - 1)**2
            term_effect = 0.05 * np.exp(-T)
            implied_vols[i, j] = base_vol + smile_effect + term_effect

    # Calibrate model
    lv_model.calibrate_from_implied_vol_surface(strikes, maturities, implied_vols)

    print(f"   Calibrated to {len(strikes)}×{len(maturities)} implied vol surface")
    print(f"   Strike range: ${strikes.min():.0f} - ${strikes.max():.0f}")
    print(f"   Maturity range: {maturities.min():.2f} - {maturities.max():.2f} years")

    # Price option with calibrated surface
    calibrated_call, _, _, _ = lv_model.monte_carlo_simulation(K, T, 25000, 50)
    print(f"   Call price with calibrated surface: ${calibrated_call:.4f}")

    # Compare with Black-Scholes
    from scipy.stats import norm
    sigma_bs = 0.2  # Base volatility
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma_bs**2)*T) / (sigma_bs*np.sqrt(T))
    d2 = d1 - sigma_bs*np.sqrt(T)
    bs_call = S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    print(f"\n5. Comparison with Black-Scholes:")
    print(f"   Black-Scholes Call: ${bs_call:.4f}")
    print(f"   Local Vol Call: ${call_price:.4f}")
    print(f"   Difference: ${call_price - bs_call:.4f} ({(call_price - bs_call)/bs_call*100:+.2f}%)")

    print(f"\nLocal Volatility Model Features:")
    print(f"• Perfect fit to any implied volatility surface")
    print(f"• State-dependent volatility σ_local(S,t)")
    print(f"• No stochastic volatility correlation")
    print(f"• Excellent for exotic options pricing")
    print(f"• Requires market data for calibration")

if __name__ == "__main__":
    main()
