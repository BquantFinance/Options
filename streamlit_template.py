import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Import the OptionsAnalyzer class
# Make sure to have the options_analyzer.py file in the same directory
from options_analyzer import OptionsAnalyzer

def create_streamlit_app():
    # Set page config
    st.set_page_config(
        page_title="Options Strategy Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Page title
    st.title("🚀 Options Strategy Analyzer")
    st.markdown("Analyze options strategies with customizable parameters using yahooquery data")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = OptionsAnalyzer()
    
    if 'tickers_loaded' not in st.session_state:
        st.session_state.tickers_loaded = False
        
    if 'available_strikes' not in st.session_state:
        st.session_state.available_strikes = {"call": [], "put": []}
    
    # Sidebar for controls
    st.sidebar.header("Settings")
    
    # Ticker input
    ticker_input = st.sidebar.text_input(
        "Ticker Symbol(s)", 
        value="SPY,QQQ,AAPL", 
        help="Enter one or more tickers separated by commas"
    )
    
    fetch_button = st.sidebar.button("Fetch Data", key="fetch_data")
    
    # Fetch data when button is clicked
    if fetch_button:
        # Parse ticker symbols
        tickers = [t.strip() for t in ticker_input.split(',')]
        
        with st.spinner(f"Fetching data for {', '.join(tickers)}..."):
            try:
                expiry_dates = st.session_state.analyzer.fetch_ticker_data(tickers)
                
                if expiry_dates:
                    st.session_state.tickers_loaded = True
                    
                    # Display current price
                    current_ticker = st.session_state.analyzer.current_ticker
                    st.sidebar.success(f"Loaded data for {len(tickers)} ticker(s)")
                    
                    # Update available strikes for the current ticker and expiry
                    update_available_strikes()
                else:
                    st.sidebar.error(f"Failed to load option data")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    # Only show these controls if we have data
    if st.session_state.tickers_loaded and st.session_state.analyzer.underlying_price is not None:
        # Current ticker selection
        available_tickers = st.session_state.analyzer.ticker_symbols
        current_ticker = st.session_state.analyzer.current_ticker
        
        selected_ticker = st.sidebar.selectbox(
            "Select Ticker",
            options=available_tickers,
            index=available_tickers.index(current_ticker) if current_ticker in available_tickers else 0,
            key="ticker_selector"
        )
        
        # Update when ticker is changed
        if selected_ticker != st.session_state.analyzer.current_ticker:
            st.session_state.analyzer.set_current_ticker(selected_ticker)
            update_available_strikes()
            
        # Show current price
        st.sidebar.metric(
            "Current Price", 
            f"${st.session_state.analyzer.underlying_price:.2f}"
        )
        
        # Expiry selection
        current_ticker = st.session_state.analyzer.current_ticker
        if current_ticker in st.session_state.analyzer.expiration_dates:
            expiry_dates = st.session_state.analyzer.expiration_dates[current_ticker]
            
            if expiry_dates:
                current_expiry = st.session_state.analyzer.current_expiry
                expiry_index = expiry_dates.index(current_expiry) if current_expiry in expiry_dates else 0
                
                selected_expiry = st.sidebar.selectbox(
                    "Expiration Date",
                    options=expiry_dates,
                    index=expiry_index,
                    key="expiry_selector"
                )
                
                # Update when expiry is changed
                if selected_expiry != st.session_state.analyzer.current_expiry:
                    st.session_state.analyzer.set_expiry(selected_expiry)
                    update_available_strikes()
        
        # IV adjustment slider
        vol_adjustment = st.sidebar.slider(
            "Implied Volatility Adjustment",
            min_value=-0.5,
            max_value=0.5,
            value=0.0,
            step=0.05,
            format="%.0f%%",
            help="Adjust implied volatility by this percentage"
        )
        st.session_state.analyzer.set_vol_adjustment(vol_adjustment)
        
        # Days to expiry display
        days = st.session_state.analyzer.calculate_days_to_expiry()
        st.sidebar.metric("Days to Expiration", days)
        
        # Strategy builder section
        st.sidebar.header("Strategy Builder")
        
        # Position type selector
        position_type = st.sidebar.radio(
            "Position Type",
            options=["Call", "Put", "Stock"],
            index=0
        )
        
        # Controls based on position type
        if position_type in ["Call", "Put"]:
            # Option controls
            strikes = st.session_state.available_strikes[position_type.lower()]
            
            if strikes:
                # Find ATM strike
                underlying_price = st.session_state.analyzer.underlying_price
                atm_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
                
                selected_strike = st.sidebar.selectbox(
                    "Strike Price",
                    options=strikes,
                    index=atm_index
                )
                
                # Option quantity
                quantity = st.sidebar.number_input(
                    "Quantity",
                    min_value=-100,  # Allow negative for selling options
                    max_value=100,
                    value=1
                )
                
                # Add option button
                option_action = "Buy" if quantity > 0 else "Sell"
                if st.sidebar.button(f"{option_action} {position_type} Option"):
                    if st.session_state.analyzer.add_option(selected_strike, position_type.lower(), quantity):
                        st.sidebar.success(f"Added {abs(quantity)} {position_type} option(s) at strike ${selected_strike:.2f} ({option_action})")
                    else:
                        st.sidebar.error("Failed to add option")
            else:
                st.sidebar.warning(f"No {position_type}s available for selected expiration")
        else:
            # Stock controls
            quantity = st.sidebar.number_input(
                "Number of Shares",
                min_value=-10000,  # Allow negative for short positions
                max_value=10000,
                value=100
            )
            
            # Add stock button
            stock_action = "Buy" if quantity > 0 else "Short"
            if st.sidebar.button(f"{stock_action} Stock Position"):
                if st.session_state.analyzer.add_stock(quantity):
                    st.sidebar.success(f"Added {abs(quantity)} shares at ${st.session_state.analyzer.underlying_price:.2f} ({stock_action})")
                else:
                    st.sidebar.error("Failed to add stock position")
        
        # Clear strategy button
        if st.sidebar.button("Clear Strategy", type="secondary"):
            st.session_state.analyzer.clear_strategy()
            st.sidebar.info("Strategy cleared")
        
        # Main content area
        if st.session_state.analyzer.strategies:
            # Display strategy summary
            st.subheader("Strategy Summary")
            
            # Create strategy table
            strategy_data = []
            total_cost = 0
            
            for position in st.session_state.analyzer.strategies:
                if position['type'] == 'stock':
                    action = "Long" if position['quantity'] > 0 else "Short"
                    position_desc = f"{action} Stock: {abs(position['quantity'])} shares"
                    position_price = f"${position['price']:.2f}/share"
                    position_cost = position['quantity'] * position['price']
                else:
                    action = "Long" if position['quantity'] > 0 else "Short"
                    position_desc = f"{action} {position['type'].upper()}: {abs(position['quantity'])} contract(s) @ ${position['strike']:.2f}"
                    position_price = f"${position['price']:.2f}/share"
                    position_cost = position['quantity'] * position['price'] * 100  # 100 shares per contract
                
                strategy_data.append({
                    "Position": position_desc,
                    "Price": position_price,
                    "Cost": f"${position_cost:.2f}"
                })
                total_cost += position_cost
            
            # Show strategy table
            st.table(pd.DataFrame(strategy_data))
            
            # Show total cost
            st.metric("Total Position Cost", f"${total_cost:.2f}")
            
            # Calculate and display key metrics
            max_profit, max_loss = st.session_state.analyzer.get_max_profit_loss()
            prob_profit = st.session_state.analyzer.calculate_probability_profit()
            
            # Create metrics columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Max Profit", f"${max_profit:.2f}" if max_profit is not None else "N/A")
            
            with col2:
                st.metric("Max Loss", f"${max_loss:.2f}" if max_loss is not None else "N/A")
            
            with col3:
                st.metric("Probability of Profit", f"{prob_profit:.1f}%" if prob_profit is not None else "N/A")
            
            # Generate and display charts
            payoff_chart, greeks_chart = st.session_state.analyzer.generate_plotly_charts()
            
            if payoff_chart is not None and greeks_chart is not None:
                st.plotly_chart(payoff_chart, use_container_width=True)
                st.plotly_chart(greeks_chart, use_container_width=True)
            else:
                st.error("Failed to generate charts")
                
            # Display breakeven points
            price_min = st.session_state.analyzer.underlying_price * 0.8
            price_max = st.session_state.analyzer.underlying_price * 1.2
            price_range = np.linspace(price_min, price_max, 1000)
            breakeven_points = st.session_state.analyzer.get_breakeven_points(price_range)
            
            if breakeven_points:
                st.subheader("Breakeven Points")
                be_text = ", ".join([f"${point:.2f}" for point in breakeven_points])
                st.write(f"This strategy breaks even at: {be_text}")
        else:
            st.info("No positions added to strategy yet. Use the sidebar to build your strategy.")
            
            # Show option chain data in table
            if st.checkbox("Show Option Chain Data"):
                option_chain = st.session_state.analyzer.get_current_option_chain()
                if option_chain:
                    tab1, tab2 = st.tabs(["Calls", "Puts"])
                    
                    with tab1:
                        calls_df = option_chain['calls'].sort_values(by='strike')
                        # Add a column for moneyness 
                        current_price = st.session_state.analyzer.underlying_price
                        calls_df['moneyness'] = ((calls_df['strike'] - current_price) / current_price * 100).round(2)
                        st.dataframe(calls_df)
                    
                    with tab2:
                        puts_df = option_chain['puts'].sort_values(by='strike')
                        # Add a column for moneyness
                        puts_df['moneyness'] = ((puts_df['strike'] - current_price) / current_price * 100).round(2)
                        st.dataframe(puts_df)
                else:
                    st.warning("No option chain data available")
    else:
        st.info("Enter ticker symbol(s) and click 'Fetch Data' to start")

def update_available_strikes():
    """Update the available strikes for the current ticker and expiry."""
    if not st.session_state.analyzer.current_ticker or not st.session_state.analyzer.current_expiry:
        return
    
    option_chain = st.session_state.analyzer.get_current_option_chain()
    if not option_chain:
        st.session_state.available_strikes = {"call": [], "put": []}
        return
    
    # Extract unique strikes
    calls_strikes = sorted(option_chain['calls']['strike'].unique())
    puts_strikes = sorted(option_chain['puts']['strike'].unique())
    
    st.session_state.available_strikes = {
        "call": calls_strikes,
        "put": puts_strikes
    }

if __name__ == '__main__':
    create_streamlit_app()
