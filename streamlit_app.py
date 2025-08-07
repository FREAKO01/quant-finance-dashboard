"""
Quantitative Finance Models - Interactive Dashboard
Author: Arnav Sharma
Live demonstration of Black-Scholes, Monte Carlo, Heston, and Local Volatility models
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# Import your model classes
from black_scholes_model import BlackScholesModel
from monte_carlo_model import MonteCarloModel
from heston_model import HestonModel
from local_volatility_model import LocalVolatilityModel

# Page configuration
st.set_page_config(
    page_title="Quant Models - Arnav Sharma",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme matching your portfolio
st.markdown("""
<style>
    .main-header {
        color: #00FFFF;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #00FFFF, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .model-card {
        background: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.1);
    }
    
    .metric-card {
        background: #2A2A2A;
        border: 1px solid #00FFFF;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .stButton > button {
        background: #00FFFF;
        color: #000;
        font-weight: 600;
        border: none;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #00E5FF;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">Quantitative Finance Models Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive demonstration of advanced option pricing models")
    
    # Sidebar for model selection
    st.sidebar.markdown("## 🎛️ Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a Model",
        ["Black-Scholes", "Monte Carlo", "Heston Stochastic Volatility", "Local Volatility", "Model Comparison"]
    )
    
    # Common parameters sidebar
    st.sidebar.markdown("## 📊 Market Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        S0 = st.number_input("Stock Price ($)", min_value=1.0, value=100.0, step=1.0)
        T = st.number_input("Time to Maturity (years)", min_value=0.01, value=1.0, step=0.25)
    
    with col2:
        K = st.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=1.0)
        r = st.number_input("Risk-free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
    
    sigma = st.sidebar.slider("Volatility (%)", min_value=1, max_value=100, value=20, step=1) / 100
    
    # Model-specific sections
    if model_choice == "Black-Scholes":
        black_scholes_section(S0, K, T, r, sigma)
    elif model_choice == "Monte Carlo":
        monte_carlo_section(S0, K, T, r, sigma)
    elif model_choice == "Heston Stochastic Volatility":
        heston_section(S0, K, T, r, sigma)
    elif model_choice == "Local Volatility":
        local_volatility_section(S0, K, T, r, sigma)
    elif model_choice == "Model Comparison":
        model_comparison_section(S0, K, T, r, sigma)

def black_scholes_section(S0, K, T, r, sigma):
    """Black-Scholes model section"""
    st.markdown("## 🎯 Black-Scholes Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        The Black-Scholes model provides analytical solutions for European options using constant volatility assumptions.
        
        **Formula**: C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)
        """)
    
    # Calculate results
    bs_model = BlackScholesModel(S0, K, T, r, sigma)
    
    with col2:
        if st.button("Calculate", key="bs_calc"):
            with st.spinner("Calculating..."):
                start_time = time.time()
                call_price = bs_model.call_price()
                put_price = bs_model.put_price()
                calc_time = time.time() - start_time
                
                st.success(f"✅ Calculated in {calc_time:.4f}s")
    
    # Results
    col1, col2, col3, col4 = st.columns(4)
    
    call_price = bs_model.call_price()
    put_price = bs_model.put_price()
    delta = bs_model.delta()
    gamma = bs_model.gamma()
    
    with col1:
        st.metric("Call Price", f"${call_price:.4f}")
    with col2:
        st.metric("Put Price", f"${put_price:.4f}")
    with col3:
        st.metric("Delta", f"{delta:.4f}")
    with col4:
        st.metric("Gamma", f"{gamma:.4f}")
    
    # Greeks section
    st.markdown("### Greeks Analysis")
    
    greeks_col1, greeks_col2 = st.columns(2)
    
    with greeks_col1:
        theta = bs_model.theta()
        vega = bs_model.vega()
        rho = bs_model.rho()
        
        greeks_df = pd.DataFrame({
            'Greek': ['Theta (per day)', 'Vega (per 1%)', 'Rho (per 1%)'],
            'Value': [f"${theta:.4f}", f"${vega:.4f}", f"${rho:.4f}"]
        })
        st.table(greeks_df)
    
    with greeks_col2:
        # Volatility sensitivity chart
        vol_range = np.linspace(0.1, 0.5, 50)
        call_prices = []
        
        for vol in vol_range:
            temp_model = BlackScholesModel(S0, K, T, r, vol)
            call_prices.append(temp_model.call_price())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vol_range * 100,
            y=call_prices,
            mode='lines',
            name='Call Price',
            line=dict(color='#00FFFF', width=3)
        ))
        
        fig.update_layout(
            title="Option Price vs Volatility",
            xaxis_title="Volatility (%)",
            yaxis_title="Call Price ($)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def monte_carlo_section(S0, K, T, r, sigma):
    """Monte Carlo model section"""
    st.markdown("## 🎲 Monte Carlo Simulation")
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        n_simulations = st.selectbox(
            "Number of Simulations",
            [1000, 5000, 10000, 25000, 50000, 100000],
            index=3
        )
    
    with col2:
        option_type = st.selectbox(
            "Option Type",
            ["European Call", "European Put", "Asian Call", "Barrier Call", "Lookback Call"]
        )
    
    if st.button("Run Simulation", key="mc_run"):
        with st.spinner("Running Monte Carlo simulation..."):
            mc_model = MonteCarloModel(S0, K, T, r, sigma, n_simulations, random_seed=42)
            
            start_time = time.time()
            
            if option_type == "European Call":
                price, std_error, conf_interval = mc_model.european_call_price()
                st.success(f"European Call Price: ${price:.4f} ± ${conf_interval:.4f}")
                
            elif option_type == "European Put":
                price, std_error, conf_interval = mc_model.european_put_price()
                st.success(f"European Put Price: ${price:.4f} ± ${conf_interval:.4f}")
                
            elif option_type == "Asian Call":
                price, std_error, conf_interval = mc_model.asian_call_price('arithmetic')
                st.success(f"Asian Call Price: ${price:.4f} ± ${conf_interval:.4f}")
                
            elif option_type == "Barrier Call":
                barrier_level = st.sidebar.number_input("Barrier Level", value=120.0)
                price, std_error, conf_interval = mc_model.barrier_call_price(barrier_level, 'up_and_out')
                st.success(f"Barrier Call Price: ${price:.4f} ± ${conf_interval:.4f}")
                
            elif option_type == "Lookback Call":
                price, std_error, conf_interval = mc_model.lookback_call_price('floating')
                st.success(f"Lookback Call Price: ${price:.4f} ± ${conf_interval:.4f}")
            
            calc_time = time.time() - start_time
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price", f"${price:.4f}")
            with col2:
                st.metric("Standard Error", f"${std_error:.4f}")
            with col3:
                st.metric("Computation Time", f"{calc_time:.2f}s")
            
            # Generate and display sample paths
            paths = mc_model.generate_gbm_paths()
            
            fig = go.Figure()
            
            # Plot sample paths (first 100)
            for i in range(min(100, n_simulations)):
                fig.add_trace(go.Scatter(
                    x=np.linspace(0, T, mc_model.n_steps + 1),
                    y=paths[i, :],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(0, 255, 255, 0.3)'),
                    showlegend=False
                ))
            
            # Add mean path
            mean_path = np.mean(paths, axis=0)
            fig.add_trace(go.Scatter(
                x=np.linspace(0, T, mc_model.n_steps + 1),
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(color='#FF6B6B', width=3)
            ))
            
            fig.update_layout(
                title=f"Monte Carlo Simulation - {n_simulations:,} Paths",
                xaxis_title="Time (years)",
                yaxis_title="Stock Price ($)",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def heston_section(S0, K, T, r, sigma):
    """Heston model section"""
    st.markdown("## 🌊 Heston Stochastic Volatility Model")
    
    # Heston parameters
    st.markdown("### Model Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        kappa = st.number_input("Mean Reversion Speed (κ)", min_value=0.1, value=2.0, step=0.1)
        theta = st.number_input("Long-term Variance (θ)", min_value=0.01, value=sigma**2, step=0.01, format="%.4f")
    
    with col2:
        xi = st.number_input("Volatility of Volatility (ξ)", min_value=0.1, value=0.3, step=0.1)
        rho = st.number_input("Correlation (ρ)", min_value=-1.0, max_value=1.0, value=-0.7, step=0.1)
    
    with col3:
        v0 = st.number_input("Initial Variance (v₀)", min_value=0.001, value=sigma**2, step=0.001, format="%.4f")
        method = st.selectbox("Pricing Method", ["Monte Carlo", "Semi-Analytical"])
    
    if st.button("Calculate Heston Price", key="heston_calc"):
        with st.spinner("Calculating Heston price..."):
            heston_model = HestonModel(S0, K, T, r, v0, kappa, theta, xi, rho)
            
            start_time = time.time()
            
            if method == "Semi-Analytical":
                try:
                    call_price = heston_model.european_call_price_analytical()
                    if not np.isnan(call_price):
                        st.success(f"Semi-Analytical Call Price: ${call_price:.4f}")
                        calc_time = time.time() - start_time
                        st.info(f"Calculation time: {calc_time:.4f} seconds")
                    else:
                        st.error("Semi-analytical pricing failed. Using Monte Carlo instead.")
                        call_price, put_price, _, _ = heston_model.monte_carlo_simulation(25000, 100)
                        st.success(f"Monte Carlo Call Price: ${call_price:.4f}")
                except:
                    st.error("Semi-analytical pricing failed. Using Monte Carlo instead.")
                    call_price, put_price, _, _ = heston_model.monte_carlo_simulation(25000, 100)
                    st.success(f"Monte Carlo Call Price: ${call_price:.4f}")
            
            else:  # Monte Carlo
                call_price, put_price, S_paths, v_paths = heston_model.monte_carlo_simulation(25000, 100)
                calc_time = time.time() - start_time
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Call Price", f"${call_price:.4f}")
                with col2:
                    st.metric("Put Price", f"${put_price:.4f}")
                with col3:
                    st.metric("Time", f"{calc_time:.2f}s")
                
                # Plot volatility paths
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Stock Price Paths', 'Volatility Paths'],
                    vertical_spacing=0.1
                )
                
                # Stock price paths (sample)
                for i in range(0, min(50, len(S_paths)), 5):
                    fig.add_trace(
                        go.Scatter(
                            x=np.linspace(0, T, S_paths.shape[1]),
                            y=S_paths[i, :],
                            mode='lines',
                            line=dict(width=1, color='rgba(0, 255, 255, 0.5)'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                
                # Volatility paths (sample)
                for i in range(0, min(50, len(v_paths)), 5):
                    fig.add_trace(
                        go.Scatter(
                            x=np.linspace(0, T, v_paths.shape[1]),
                            y=np.sqrt(v_paths[i, :]) * 100,  # Convert to percentage
                            mode='lines',
                            line=dict(width=1, color='rgba(255, 107, 107, 0.5)'),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                
                fig.update_xaxes(title_text="Time (years)", row=2, col=1)
                fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
                
                fig.update_layout(
                    title="Heston Model - Stochastic Paths",
                    template="plotly_dark",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Feller condition check
    feller_value = 2 * kappa * theta
    feller_condition = feller_value > xi**2
    
    st.markdown("### Model Diagnostics")
    
    if feller_condition:
        st.success(f"✅ Feller Condition Satisfied: 2κθ = {feller_value:.4f} > ξ² = {xi**2:.4f}")
    else:
        st.warning(f"⚠️ Feller Condition Violated: 2κθ = {feller_value:.4f} ≤ ξ² = {xi**2:.4f}")

def local_volatility_section(S0, K, T, r, sigma):
    """Local Volatility model section"""
    st.markdown("## 📈 Local Volatility (Dupire) Model")
    
    st.markdown("""
    The Local Volatility model uses a deterministic volatility function σ_local(S,t) 
    that depends on both stock price and time.
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        base_vol = st.slider("Base Volatility (%)", min_value=5, max_value=50, value=20) / 100
        skew_param = st.slider("Skew Parameter", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    with col2:
        term_param = st.slider("Term Structure Parameter", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        n_sims = st.selectbox("Simulations", [10000, 25000, 50000], index=1)
    
    if st.button("Run Local Vol Simulation", key="lv_run"):
        with st.spinner("Running Local Volatility simulation..."):
            lv_model = LocalVolatilityModel(S0, r)
            
            start_time = time.time()
            call_price, put_price, S_paths, local_vol_paths = lv_model.monte_carlo_simulation(
                K, T, n_sims, 100
            )
            calc_time = time.time() - start_time
            
            # Results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Call Price", f"${call_price:.4f}")
            with col2:
                st.metric("Put Price", f"${put_price:.4f}")
            with col3:
                st.metric("Time", f"{calc_time:.2f}s")
            
            # Local volatility surface
            S_range = np.linspace(0.7 * S0, 1.3 * S0, 30)
            T_range = np.linspace(0.1, T, 20)
            S_grid, T_grid = np.meshgrid(S_range, T_range)
            
            vol_surface = np.zeros_like(S_grid)
            for i in range(len(T_range)):
                for j in range(len(S_range)):
                    vol_surface[i, j] = lv_model.parametric_local_volatility(
                        S_range[j], T_range[i], base_vol, skew_param, term_param
                    )
            
            fig = go.Figure(data=[go.Surface(
                x=S_grid,
                y=T_grid,
                z=vol_surface * 100,  # Convert to percentage
                colorscale='Viridis',
                showscale=True
            )])
            
            fig.update_layout(
                title='Local Volatility Surface',
                scene=dict(
                    xaxis_title='Stock Price ($)',
                    yaxis_title='Time (years)',
                    zaxis_title='Local Volatility (%)'
                ),
                template="plotly_dark",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

def model_comparison_section(S0, K, T, r, sigma):
    """Model comparison section"""
    st.markdown("## ⚖️ Model Comparison")
    
    n_sims = st.sidebar.number_input("Monte Carlo Simulations", min_value=1000, value=25000, step=1000)
    
    if st.button("Run All Models", key="comparison_run"):
        with st.spinner("Running all models..."):
            results = {}
            
            # Black-Scholes
            start_time = time.time()
            bs_model = BlackScholesModel(S0, K, T, r, sigma)
            bs_call = bs_model.call_price()
            bs_time = time.time() - start_time
            results['Black-Scholes'] = {'price': bs_call, 'time': bs_time}
            
            # Monte Carlo
            start_time = time.time()
            mc_model = MonteCarloModel(S0, K, T, r, sigma, n_sims, random_seed=42)
            mc_call, _, _ = mc_model.european_call_price()
            mc_time = time.time() - start_time
            results['Monte Carlo'] = {'price': mc_call, 'time': mc_time}
            
            # Heston
            start_time = time.time()
            heston_model = HestonModel(S0, K, T, r, sigma**2, 2.0, sigma**2, 0.3, -0.7)
            heston_call, _, _, _ = heston_model.monte_carlo_simulation(n_sims//2, 50)
            heston_time = time.time() - start_time
            results['Heston'] = {'price': heston_call, 'time': heston_time}
            
            # Local Volatility
            start_time = time.time()
            lv_model = LocalVolatilityModel(S0, r)
            lv_call, _, _, _ = lv_model.monte_carlo_simulation(K, T, n_sims//2, 50)
            lv_time = time.time() - start_time
            results['Local Vol'] = {'price': lv_call, 'time': lv_time}
        
        # Display comparison table
        df = pd.DataFrame({
            'Model': list(results.keys()),
            'Call Price': [f"${results[model]['price']:.4f}" for model in results],
            'vs BS Diff': [f"{((results[model]['price'] - bs_call) / bs_call * 100):+.2f}%" 
                          for model in results],
            'Time (s)': [f"{results[model]['time']:.4f}" for model in results]
        })
        
        st.table(df)
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Call Prices', 'Computation Times'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = list(results.keys())
        prices = [results[model]['price'] for model in models]
        times = [results[model]['time'] for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=prices, name='Call Price', marker_color='#00FFFF'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=times, name='Time (s)', marker_color='#FF6B6B'),
            row=1, col=2
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Created by Arnav Sharma**")
st.sidebar.markdown("📧 contact@arnavsharma.me")
st.sidebar.markdown("🐙 [GitHub](https://github.com/FREAKO01)")

if __name__ == "__main__":
    main()
