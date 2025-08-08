
"""
Option Pricing Models Comparison and Analysis
Author: Arnav Sharma
Date: 2025

This script compares all four option pricing models:
1. Black-Scholes
2. Monte Carlo
3. Heston Stochastic Volatility
4. Local Volatility (Dupire)

It provides comprehensive analysis, performance comparison, and visualization.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Import our model implementations
from black_scholes_model import BlackScholesModel
from monte_carlo_model import MonteCarloModel  
from heston_model import HestonModel
from local_volatility_model import LocalVolatilityModel

class ModelComparison:
    """
    Comprehensive comparison of option pricing models
    """

    def __init__(self, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2):
        """
        Initialize comparison with common parameters

        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free rate
        sigma (float): Base volatility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

        # Heston-specific parameters
        self.v0 = sigma**2      # Initial variance
        self.kappa = 2.0        # Mean reversion speed
        self.theta = sigma**2   # Long-term variance
        self.xi = 0.3           # Vol of vol
        self.rho = -0.7         # Correlation

        print(f"Model Comparison Initialized")
        print(f"Common Parameters: S₀=${S0}, K=${K}, T={T}, r={r:.1%}, σ={sigma:.1%}")
        print(f"Heston Parameters: κ={self.kappa}, θ={self.theta:.4f}, ξ={self.xi}, ρ={self.rho}")

    def run_all_models(self, n_simulations=50000):
        """
        Run all models and collect results

        Parameters:
        n_simulations (int): Number of Monte Carlo simulations

        Returns:
        dict: Results from all models
        """
        results = {}

        print(f"\nRunning all models with {n_simulations:,} simulations...")
        print("=" * 60)

        # 1. Black-Scholes Model
        print("1. Black-Scholes Model...")
        start_time = time.time()

        bs_model = BlackScholesModel(self.S0, self.K, self.T, self.r, self.sigma)
        bs_call = bs_model.call_price()
        bs_put = bs_model.put_price()
        bs_greeks = {
            'delta': bs_model.delta(),
            'gamma': bs_model.gamma(),
            'theta': bs_model.theta(),
            'vega': bs_model.vega(),
            'rho': bs_model.rho()
        }

        bs_time = time.time() - start_time

        results['black_scholes'] = {
            'call_price': bs_call,
            'put_price': bs_put,
            'greeks': bs_greeks,
            'computation_time': bs_time,
            'std_error': 0.0  # Analytical solution
        }

        print(f"   Call: ${bs_call:.4f}, Put: ${bs_put:.4f}, Time: {bs_time:.4f}s")

        # 2. Monte Carlo Model
        print("2. Monte Carlo Model...")
        start_time = time.time()

        mc_model = MonteCarloModel(self.S0, self.K, self.T, self.r, self.sigma, 
                                  n_simulations, random_seed=42)
        mc_call, mc_call_se, mc_call_ci = mc_model.european_call_price()
        mc_put, mc_put_se, mc_put_ci = mc_model.european_put_price()

        mc_time = time.time() - start_time

        results['monte_carlo'] = {
            'call_price': mc_call,
            'put_price': mc_put,
            'call_std_error': mc_call_se,
            'put_std_error': mc_put_se,
            'computation_time': mc_time
        }

        print(f"   Call: ${mc_call:.4f}±{mc_call_se:.4f}, Put: ${mc_put:.4f}±{mc_put_se:.4f}, Time: {mc_time:.2f}s")

        # 3. Heston Model
        print("3. Heston Model...")
        start_time = time.time()

        heston_model = HestonModel(self.S0, self.K, self.T, self.r, 
                                  self.v0, self.kappa, self.theta, self.xi, self.rho)

        # Try analytical first
        try:
            heston_call_analytical = heston_model.european_call_price_analytical()
            analytical_available = not np.isnan(heston_call_analytical)
        except:
            analytical_available = False
            heston_call_analytical = np.nan

        # Monte Carlo
        heston_call_mc, heston_put_mc, _, _ = heston_model.monte_carlo_simulation(
            n_simulations//2, n_steps=100, scheme='full_truncation'
        )

        heston_time = time.time() - start_time

        results['heston'] = {
            'call_price_analytical': heston_call_analytical if analytical_available else None,
            'call_price_mc': heston_call_mc,
            'put_price_mc': heston_put_mc,
            'computation_time': heston_time,
            'analytical_available': analytical_available
        }

        if analytical_available:
            print(f"   Call (Analytical): ${heston_call_analytical:.4f}")
        print(f"   Call (MC): ${heston_call_mc:.4f}, Put (MC): ${heston_put_mc:.4f}, Time: {heston_time:.2f}s")

        # 4. Local Volatility Model
        print("4. Local Volatility Model...")
        start_time = time.time()

        lv_model = LocalVolatilityModel(self.S0, self.r)
        lv_call, lv_put, _, _ = lv_model.monte_carlo_simulation(
            self.K, self.T, n_simulations//2, n_steps=100
        )

        lv_time = time.time() - start_time

        results['local_volatility'] = {
            'call_price': lv_call,
            'put_price': lv_put,
            'computation_time': lv_time
        }

        print(f"   Call: ${lv_call:.4f}, Put: ${lv_put:.4f}, Time: {lv_time:.2f}s")

        return results

    def analyze_convergence(self, simulation_sizes=[1000, 5000, 10000, 25000, 50000, 100000]):
        """
        Analyze Monte Carlo convergence for different models

        Parameters:
        simulation_sizes (list): Different numbers of simulations to test

        Returns:
        dict: Convergence analysis results
        """
        print(f"\nAnalyzing Monte Carlo Convergence...")
        print("=" * 40)

        bs_model = BlackScholesModel(self.S0, self.K, self.T, self.r, self.sigma)
        true_call_price = bs_model.call_price()

        convergence_results = {
            'simulation_sizes': simulation_sizes,
            'monte_carlo_errors': [],
            'monte_carlo_times': [],
            'heston_errors': [],
            'heston_times': [],
            'local_vol_errors': [],
            'local_vol_times': []
        }

        for n_sims in simulation_sizes:
            print(f"Testing with {n_sims:,} simulations...")

            # Monte Carlo (GBM)
            start_time = time.time()
            mc_model = MonteCarloModel(self.S0, self.K, self.T, self.r, self.sigma, 
                                      n_sims, random_seed=42)
            mc_call, _, _ = mc_model.european_call_price()
            mc_time = time.time() - start_time
            mc_error = abs(mc_call - true_call_price)

            convergence_results['monte_carlo_errors'].append(mc_error)
            convergence_results['monte_carlo_times'].append(mc_time)

            # Heston Monte Carlo
            start_time = time.time()
            heston_model = HestonModel(self.S0, self.K, self.T, self.r, 
                                      self.v0, self.kappa, self.theta, self.xi, self.rho)
            heston_call, _, _, _ = heston_model.monte_carlo_simulation(n_sims, n_steps=50)
            heston_time = time.time() - start_time
            heston_error = abs(heston_call - true_call_price)

            convergence_results['heston_errors'].append(heston_error)
            convergence_results['heston_times'].append(heston_time)

            # Local Volatility Monte Carlo
            start_time = time.time()
            lv_model = LocalVolatilityModel(self.S0, self.r)
            lv_call, _, _, _ = lv_model.monte_carlo_simulation(self.K, self.T, n_sims, n_steps=50)
            lv_time = time.time() - start_time
            lv_error = abs(lv_call - true_call_price)

            convergence_results['local_vol_errors'].append(lv_error)
            convergence_results['local_vol_times'].append(lv_time)

            print(f"   MC Error: {mc_error:.4f}, Heston Error: {heston_error:.4f}, LV Error: {lv_error:.4f}")

        return convergence_results

    def volatility_smile_analysis(self):
        """
        Analyze volatility smiles generated by different models

        Returns:
        dict: Volatility smile data
        """
        print(f"\nVolatility Smile Analysis...")
        print("=" * 30)

        strikes = np.linspace(80, 120, 21)

        # Black-Scholes (flat volatility)
        bs_prices = []
        bs_impl_vols = []

        for K in strikes:
            bs_model = BlackScholesModel(self.S0, K, self.T, self.r, self.sigma)
            price = bs_model.call_price()
            bs_prices.append(price)
            bs_impl_vols.append(self.sigma)  # Constant by definition

        # Heston model smile
        heston_model = HestonModel(self.S0, self.K, self.T, self.r, 
                                  self.v0, self.kappa, self.theta, self.xi, self.rho)

        try:
            _, heston_prices, heston_impl_vols = heston_model.implied_volatility_smile(strikes)
        except:
            print("   Heston smile calculation failed, using simplified approach")
            heston_prices = bs_prices.copy()
            heston_impl_vols = [self.sigma] * len(strikes)

        smile_data = {
            'strikes': strikes,
            'black_scholes': {
                'prices': np.array(bs_prices),
                'implied_vols': np.array(bs_impl_vols)
            },
            'heston': {
                'prices': np.array(heston_prices),
                'implied_vols': np.array(heston_impl_vols)
            }
        }

        return smile_data

    def generate_summary_report(self, results):
        """
        Generate comprehensive summary report

        Parameters:
        results (dict): Results from run_all_models()
        """
        print(f"\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)

        # Extract prices for comparison
        bs_call = results['black_scholes']['call_price']
        mc_call = results['monte_carlo']['call_price']
        heston_call = results['heston']['call_price_mc']
        lv_call = results['local_volatility']['call_price']

        bs_put = results['black_scholes']['put_price']
        mc_put = results['monte_carlo']['put_price']
        heston_put = results['heston']['put_price_mc']
        lv_put = results['local_volatility']['put_price']

        # Summary table
        print(f"\nPRICING RESULTS SUMMARY")
        print("-" * 80)
        print(f"{'Model':<20} {'Call Price':<12} {'Put Price':<12} {'vs BS Call':<12} {'Time (s)':<10}")
        print("-" * 80)

        print(f"{'Black-Scholes':<20} ${bs_call:<11.4f} ${bs_put:<11.4f} {'0.00%':<12} {results['black_scholes']['computation_time']:<10.4f}")

        mc_diff = (mc_call - bs_call) / bs_call * 100
        print(f"{'Monte Carlo':<20} ${mc_call:<11.4f} ${mc_put:<11.4f} {mc_diff:<+11.2f}% {results['monte_carlo']['computation_time']:<10.2f}")

        heston_diff = (heston_call - bs_call) / bs_call * 100
        print(f"{'Heston':<20} ${heston_call:<11.4f} ${heston_put:<11.4f} {heston_diff:<+11.2f}% {results['heston']['computation_time']:<10.2f}")

        lv_diff = (lv_call - bs_call) / bs_call * 100
        print(f"{'Local Volatility':<20} ${lv_call:<11.4f} ${lv_put:<11.4f} {lv_diff:<+11.2f}% {results['local_volatility']['computation_time']:<10.2f}")

        # Greeks comparison (Black-Scholes only)
        print(f"\nGREEKS (Black-Scholes)")
        print("-" * 40)
        greeks = results['black_scholes']['greeks']
        print(f"Delta (Δ): {greeks['delta']:.4f}")
        print(f"Gamma (Γ): {greeks['gamma']:.4f}")
        print(f"Theta (Θ): ${greeks['theta']:.4f} per day")
        print(f"Vega (ν): ${greeks['vega']:.4f} per 1% vol")
        print(f"Rho (ρ): ${greeks['rho']:.4f} per 1% rate")

        # Model characteristics
        print(f"\nMODEL CHARACTERISTICS")
        print("-" * 40)

        characteristics = {
            'Black-Scholes': {
                'Pros': ['Analytical solution', 'Very fast', 'Well understood', 'Greeks available'],
                'Cons': ['Constant volatility', 'No volatility smile', 'Unrealistic assumptions'],
                'Best for': ['Quick estimates', 'Simple derivatives', 'Benchmarking']
            },
            'Monte Carlo': {
                'Pros': ['Extremely flexible', 'Path-dependent options', 'Multi-asset derivatives'],
                'Cons': ['Computationally intensive', 'Statistical error', 'Slow convergence'],
                'Best for': ['Exotic options', 'American options', 'Complex payoffs']
            },
            'Heston': {
                'Pros': ['Stochastic volatility', 'Volatility smile', 'Leverage effect', 'Analytical solutions available'],
                'Cons': ['More parameters', 'Calibration complexity', 'Computational cost'],
                'Best for': ['Volatility derivatives', 'Long-term options', 'Risk management']
            },
            'Local Volatility': {
                'Pros': ['Perfect market fit', 'State-dependent vol', 'Exotic option pricing'],
                'Cons': ['No vol correlation', 'Forward-looking', 'Calibration intensive'],
                'Best for': ['Barrier options', 'Volatility trading', 'Market making']
            }
        }

        for model, chars in characteristics.items():
            print(f"\n{model}:")
            print(f"  ✓ Pros: {', '.join(chars['Pros'])}")
            print(f"  ✗ Cons: {', '.join(chars['Cons'])}")
            print(f"  📊 Best for: {', '.join(chars['Best for'])}")

        # Statistical analysis
        print(f"\nSTATISTICAL ANALYSIS")
        print("-" * 40)

        if 'call_std_error' in results['monte_carlo']:
            mc_se = results['monte_carlo']['call_std_error']
            print(f"Monte Carlo Standard Error: ±${mc_se:.4f}")
            print(f"95% Confidence Interval: ±${1.96 * mc_se:.4f}")
            print(f"Coefficient of Variation: {mc_se / mc_call * 100:.2f}%")

        # Performance analysis
        total_time = sum([results[model]['computation_time'] for model in results])
        print(f"\nPERFORMANCE ANALYSIS")
        print("-" * 40)
        print(f"Total Computation Time: {total_time:.2f} seconds")
        print(f"Fastest Model: Black-Scholes ({results['black_scholes']['computation_time']:.4f}s)")

        slowest_model = max(results.keys(), key=lambda x: results[x]['computation_time'])
        slowest_time = results[slowest_model]['computation_time']
        print(f"Slowest Model: {slowest_model.replace('_', ' ').title()} ({slowest_time:.2f}s)")
        print(f"Speed Ratio: {slowest_time / results['black_scholes']['computation_time']:.0f}:1")

def main():
    """
    Main function demonstrating comprehensive model comparison
    """
    print("Option Pricing Models - Comprehensive Comparison")
    print("=" * 60)
    print("This script compares Black-Scholes, Monte Carlo, Heston, and Local Volatility models")

    # Initialize comparison
    comparison = ModelComparison(
        S0=100.0,    # Current stock price
        K=100.0,     # Strike price  
        T=1.0,       # 1 year to maturity
        r=0.05,      # 5% risk-free rate
        sigma=0.2    # 20% volatility
    )

    # Run all models
    results = comparison.run_all_models(n_simulations=50000)

    # Generate comprehensive report
    comparison.generate_summary_report(results)

    # Convergence analysis
    print(f"\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)

    convergence_data = comparison.analyze_convergence([1000, 5000, 10000, 25000])

    print(f"\nConvergence Summary:")
    for i, n_sims in enumerate(convergence_data['simulation_sizes']):
        mc_error = convergence_data['monte_carlo_errors'][i]
        heston_error = convergence_data['heston_errors'][i]
        lv_error = convergence_data['local_vol_errors'][i]

        print(f"{n_sims:>6,} sims: MC={mc_error:.4f}, Heston={heston_error:.4f}, LV={lv_error:.4f}")

    # Volatility smile analysis
    print(f"\n" + "="*60)
    print("VOLATILITY SMILE ANALYSIS")
    print("="*60)

    smile_data = comparison.volatility_smile_analysis()

    print(f"Implied Volatility by Strike:")
    print(f"{'Strike':<8} {'BS Impl Vol':<12} {'Heston Impl Vol':<15} {'Difference':<10}")
    print("-" * 50)

    for i, strike in enumerate(smile_data['strikes'][::4]):  # Show every 4th strike
        bs_vol = smile_data['black_scholes']['implied_vols'][i*4]
        heston_vol = smile_data['heston']['implied_vols'][i*4]

        if not np.isnan(heston_vol):
            diff = heston_vol - bs_vol
            print(f"{strike:<8.0f} {bs_vol:<12.1%} {heston_vol:<15.1%} {diff:<+10.1%}")
        else:
            print(f"{strike:<8.0f} {bs_vol:<12.1%} {'N/A':<15} {'N/A':<10}")

    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("All model files have been created and are ready for local execution:")
    print("• black_scholes_model.py")
    print("• monte_carlo_model.py") 
    print("• heston_model.py")
    print("• local_volatility_model.py")
    print("• model_comparison.py (this file)")
    print("\nRun any individual model file or this comparison script to see detailed results!")

if __name__ == "__main__":
    main()
