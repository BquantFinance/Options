import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class OptionsAnalyzer:
    """A class to analyze options strategies with customizable parameters."""
    
    def __init__(self):
        self.strategies = []
        self.underlying_price = None
        self.ticker = None
        self.expiration_dates = None
        self.current_expiry = None
        self.option_chain = None
        self.stock_data = None
        self.vol_adjustment = 0.0  # Adjustment to implied volatility in percentage points
    
    def fetch_ticker_data(self, ticker_symbol):
        """Fetch ticker data and available option expiration dates."""
        self.ticker = yf.Ticker(ticker_symbol)
        self.stock_data = self.ticker.history(period="1d")
        if len(self.stock_data) == 0:
            st.error(f"No data found for ticker {ticker_symbol}. Please check the symbol and try again.")
            return []
            
        self.underlying_price = self.stock_data['Close'].iloc[-1]
        self.expiration_dates = self.ticker.options
        if len(self.expiration_dates) > 0:
            self.current_expiry = self.expiration_dates[0]
            self.option_chain = self.ticker.option_chain(self.current_expiry)
        return self.expiration_dates
    
    def set_expiry(self, expiry_date):
        """Set the current expiration date for analysis."""
        if expiry_date in self.expiration_dates:
            self.current_expiry = expiry_date
            self.option_chain = self.ticker.option_chain(self.current_expiry)
            return True
        return False
    
    def set_vol_adjustment(self, adjustment):
        """Set the volatility adjustment factor."""
        self.vol_adjustment = adjustment
    
    def calculate_days_to_expiry(self):
        """Calculate days to expiration."""
        if self.current_expiry:
            expiry_date = datetime.strptime(self.current_expiry, '%Y-%m-%d')
            days = (expiry_date - datetime.now()).days
            return max(1, days)  # Ensure at least 1 day
        return 30  # Default
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option price using Black-Scholes model.
        
        Parameters:
        S: underlying price
        K: strike price
        T: time to expiration in years
        r: risk-free interest rate
        sigma: volatility
        option_type: 'call' or 'put'
        """
        # Adjust volatility by the vol_adjustment factor
        adjusted_sigma = max(0.01, sigma * (1 + self.vol_adjustment))
        
        d1 = (np.log(S / K) + (r + 0.5 * adjusted_sigma**2) * T) / (adjusted_sigma * np.sqrt(T))
        d2 = d1 - adjusted_sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        return price
    
    def calculate_option_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks."""
        # Adjust volatility by the vol_adjustment factor
        adjusted_sigma = max(0.01, sigma * (1 + self.vol_adjustment))
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * adjusted_sigma**2) * T) / (adjusted_sigma * np.sqrt(T))
        d2 = d1 - adjusted_sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = -norm.cdf(-d1)
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * adjusted_sigma * np.sqrt(T))
        
        # Theta
        if option_type.lower() == 'call':
            theta = -S * norm.pdf(d1) * adjusted_sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            theta = -S * norm.pdf(d1) * adjusted_sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        # Vega (same for call and put)
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01  # 1% change in volatility
        
        # Rho
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01  # 1% change in interest rate
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
            
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Daily theta
            'vega': vega,
            'rho': rho
        }
    
    def add_option(self, strike, option_type, quantity=1, custom_price=None):
        """Add an option to the strategy."""
        if not self.option_chain:
            st.error("No option chain loaded. Please fetch ticker data first.")
            return False
        
        # Find the option in the chain
        if option_type.lower() == 'call':
            options_df = self.option_chain.calls
        else:
            options_df = self.option_chain.puts
            
        option_row = options_df[options_df['strike'] == strike]
        if option_row.empty:
            st.error(f"No {option_type} option found with strike {strike}")
            return False
        
        # Get option details
        option_data = option_row.iloc[0]
        
        # Use custom price if provided, otherwise use market price
        price = custom_price if custom_price is not None else option_data['lastPrice']
        implied_vol = option_data['impliedVolatility']
        
        # Create option dictionary
        option = {
            'strike': strike,
            'type': option_type.lower(),
            'quantity': quantity,
            'price': price,
            'impliedVolatility': implied_vol
        }
        
        self.strategies.append(option)
        return True
    
    def add_stock(self, quantity):
        """Add stock position to the strategy."""
        if not self.underlying_price:
            st.error("No underlying price. Please fetch ticker data first.")
            return False
            
        stock = {
            'type': 'stock',
            'quantity': quantity,
            'price': self.underlying_price
        }
        
        self.strategies.append(stock)
        return True
    
    def clear_strategy(self):
        """Clear all positions in the current strategy."""
        self.strategies = []
    
    def calculate_payoff(self, price_range=None):
        """Calculate strategy payoff across a range of prices."""
        if not self.underlying_price:
            st.error("No underlying price. Please fetch ticker data first.")
            return None
        
        # Set price range if not provided
        if price_range is None:
            # Default to ±20% of current price
            price_min = self.underlying_price * 0.8
            price_max = self.underlying_price * 1.2
            price_range = np.linspace(price_min, price_max, 100)
        
        days_to_expiry = self.calculate_days_to_expiry()
        T = days_to_expiry / 365.0  # Time in years
        r = 0.05  # Risk-free rate (example value)
        
        payoffs = []
        for price in price_range:
            payoff = 0
            for position in self.strategies:
                if position['type'] == 'stock':
                    # Stock payoff is simply the change in price times quantity
                    payoff += position['quantity'] * (price - position['price'])
                else:
                    # Option payoff
                    strike = position['strike']
                    option_type = position['type']
                    quantity = position['quantity']
                    option_price = position['price']
                    
                    if T <= 0:  # At expiration
                        # Calculate intrinsic value at expiration
                        if option_type == 'call':
                            intrinsic = max(0, price - strike)
                        else:  # put
                            intrinsic = max(0, strike - price)
                        payoff += quantity * (intrinsic - option_price)
                    else:
                        # Before expiration - use Black-Scholes
                        implied_vol = position['impliedVolatility']
                        new_price = self.black_scholes_price(price, strike, T, r, implied_vol, option_type)
                        payoff += quantity * (new_price - option_price)
            
            payoffs.append(payoff)
        
        return np.array(payoffs)
    
    def calculate_greeks_profile(self, price_range=None):
        """Calculate greeks across a range of prices."""
        if not self.underlying_price:
            st.error("No underlying price. Please fetch ticker data first.")
            return None
            
        # Set price range if not provided
        if price_range is None:
            # Default to ±20% of current price
            price_min = self.underlying_price * 0.8
            price_max = self.underlying_price * 1.2
            price_range = np.linspace(price_min, price_max, 100)
        
        days_to_expiry = self.calculate_days_to_expiry()
        T = days_to_expiry / 365.0  # Time in years
        r = 0.05  # Risk-free rate (example value)
        
        # Initialize greek profiles
        delta_profile = np.zeros_like(price_range)
        gamma_profile = np.zeros_like(price_range)
        theta_profile = np.zeros_like(price_range)
        vega_profile = np.zeros_like(price_range)
        
        # Calculate greeks for each position at each price
        for i, price in enumerate(price_range):
            for position in self.strategies:
                if position['type'] == 'stock':
                    # Stock greeks
                    delta_profile[i] += position['quantity']  # Delta for stock is always 1
                    # Other greeks are 0 for stock
                else:
                    # Option greeks
                    strike = position['strike']
                    option_type = position['type']
                    quantity = position['quantity']
                    implied_vol = position['impliedVolatility']
                    
                    greeks = self.calculate_option_greeks(price, strike, T, r, implied_vol, option_type)
                    
                    delta_profile[i] += quantity * greeks['delta']
                    gamma_profile[i] += quantity * greeks['gamma']
                    theta_profile[i] += quantity * greeks['theta']
                    vega_profile[i] += quantity * greeks['vega']
        
        return {
            'delta': delta_profile,
            'gamma': gamma_profile,
            'theta': theta_profile,
            'vega': vega_profile
        }
    
    def get_max_profit_loss(self, price_range=None):
        """Calculate maximum profit and loss over price range."""
        payoffs = self.calculate_payoff(price_range)
        
        if payoffs is None:
            return None, None
            
        max_profit = np.max(payoffs)
        max_loss = np.min(payoffs)
        
        return max_profit, max_loss
    
    def get_breakeven_points(self, price_range=None):
        """Find breakeven points where payoff crosses zero."""
        if price_range is None:
            # Default to ±20% of current price
            price_min = self.underlying_price * 0.8
            price_max = self.underlying_price * 1.2
            price_range = np.linspace(price_min, price_max, 1000)  # More points for better accuracy
            
        payoffs = self.calculate_payoff(price_range)
        
        if payoffs is None:
            return []
            
        breakeven_indices = []
        for i in range(1, len(payoffs)):
            if (payoffs[i-1] <= 0 and payoffs[i] > 0) or (payoffs[i-1] >= 0 and payoffs[i] < 0):
                breakeven_indices.append(i)
        
        breakeven_points = [price_range[i] for i in breakeven_indices]
        return breakeven_points
    
    def calculate_probability_profit(self):
        """Calculate approximate probability of profit based on implied volatility."""
        if not self.strategies or not self.underlying_price:
            return None
        
        # This is a simplified model - would need to be refined for accurate results
        days_to_expiry = self.calculate_days_to_expiry()
        T = days_to_expiry / 365.0  # Time in years
        
        # Calculate breakeven points
        # Default to ±30% of current price for a broader range
        price_min = self.underlying_price * 0.7
        price_max = self.underlying_price * 1.3
        price_range = np.linspace(price_min, price_max, 1000)
        
        breakeven_points = self.get_breakeven_points(price_range)
        
        if not breakeven_points:
            # Return None if there are no breakeven points
            return None
        
        # Get average implied volatility from options in the strategy
        total_iv = 0
        option_count = 0
        
        for position in self.strategies:
            if position['type'] != 'stock':
                adjusted_iv = position['impliedVolatility'] * (1 + self.vol_adjustment)
                total_iv += adjusted_iv
                option_count += 1
        
        if option_count == 0:
            # No options in the strategy, use overall implied volatility
            if self.option_chain is not None:
                # Average IV from calls and puts at the money
                calls_iv = self.option_chain.calls['impliedVolatility']
                puts_iv = self.option_chain.puts['impliedVolatility']
                nearest_strike_call = self.option_chain.calls.iloc[(self.option_chain.calls['strike'] - self.underlying_price).abs().argsort()[0]]
                nearest_strike_put = self.option_chain.puts.iloc[(self.option_chain.puts['strike'] - self.underlying_price).abs().argsort()[0]]
                avg_iv = (nearest_strike_call['impliedVolatility'] + nearest_strike_put['impliedVolatility']) / 2
                avg_iv *= (1 + self.vol_adjustment)  # Apply volatility adjustment
            else:
                # If no option chain, use a default value
                avg_iv = 0.3  # Example value
        else:
            avg_iv = total_iv / option_count
        
        # Calculate expected move
        expected_move = self.underlying_price * avg_iv * np.sqrt(T)
        
        # Use normal distribution to approximate probability
        # This is simplified and would need refinement for a real tool
        payoffs = self.calculate_payoff(price_range)
        
        # Find where payoff is positive
        positive_payoff_indices = np.where(payoffs > 0)[0]
        
        if len(positive_payoff_indices) == 0:
            return 0.0  # No probability of profit
        
        # Calculate probability that price will end up in profit region
        total_prob = 0.0
        
        # Group consecutive positive payoff indices
        from itertools import groupby
        from operator import itemgetter
        
        for k, g in groupby(enumerate(positive_payoff_indices), lambda ix: ix[0] - ix[1]):
            profit_region = list(map(itemgetter(1), g))
            
            # Calculate probability for this region
            lower_price = price_range[profit_region[0]]
            upper_price = price_range[profit_region[-1]]
            
            # Calculate z-scores
            z_lower = (lower_price - self.underlying_price) / expected_move
            z_upper = (upper_price - self.underlying_price) / expected_move
            
            # Probability is the area under the normal curve between these z-scores
            region_prob = norm.cdf(z_upper) - norm.cdf(z_lower)
            total_prob += region_prob
        
        return total_prob * 100  # Return as percentage

    def generate_plotly_charts(self):
        """Generate Plotly charts for strategy analysis."""
        if not self.strategies:
            st.error("No strategy defined. Please add positions first.")
            return None, None
            
        # Set up price range for plotting
        price_min = self.underlying_price * 0.8
        price_max = self.underlying_price * 1.2
        price_range = np.linspace(price_min, price_max, 100)
        
        # Calculate payoff and greeks
        payoffs = self.calculate_payoff(price_range)
        greeks = self.calculate_greeks_profile(price_range)
        
        if payoffs is None or greeks is None:
            return None, None
            
        # Create payoff chart
        payoff_fig = go.Figure()
        
        # Add payoff line
        payoff_fig.add_trace(go.Scatter(
            x=price_range,
            y=payoffs,
            mode='lines',
            name='Payoff',
            line=dict(color='blue', width=3)
        ))
        
        # Add zero line
        payoff_fig.add_trace(go.Scatter(
            x=[price_min, price_max],
            y=[0, 0],
            mode='lines',
            name='Breakeven',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        # Add current price line
        payoff_fig.add_trace(go.Scatter(
            x=[self.underlying_price, self.underlying_price],
            y=[min(payoffs), max(payoffs)],
            mode='lines',
            name='Current Price',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        # Highlight breakeven points
        breakeven_points = self.get_breakeven_points(price_range)
        for i, point in enumerate(breakeven_points):
            payoff_fig.add_trace(go.Scatter(
                x=[point, point],
                y=[min(payoffs), max(payoffs)],
                mode='lines',
                name=f'Breakeven {i+1}: ${point:.2f}',
                line=dict(color='purple', width=1, dash='dot')
            ))
        
        # Update layout for payoff chart
        payoff_fig.update_layout(
            title=f'Strategy Payoff - {self.ticker.ticker} (Current: ${self.underlying_price:.2f})',
            xaxis_title='Underlying Price',
            yaxis_title='Profit/Loss ($)',
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Create greeks chart
        greeks_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta', 'Gamma', 'Theta (Daily)', 'Vega'),
            shared_xaxes=True
        )
        
        # Add delta trace
        greeks_fig.add_trace(
            go.Scatter(x=price_range, y=greeks['delta'], name='Delta', line=dict(color='green')),
            row=1, col=1
        )
        
        # Add gamma trace
        greeks_fig.add_trace(
            go.Scatter(x=price_range, y=greeks['gamma'], name='Gamma', line=dict(color='magenta')),
            row=1, col=2
        )
        
        # Add theta trace
        greeks_fig.add_trace(
            go.Scatter(x=price_range, y=greeks['theta'], name='Theta', line=dict(color='red')),
            row=2, col=1
        )
        
        # Add vega trace
        greeks_fig.add_trace(
            go.Scatter(x=price_range, y=greeks['vega'], name='Vega', line=dict(color='cyan')),
            row=2, col=2
        )
        
        # Add current price lines to each subplot
        for row in [1, 2]:
            for col in [1, 2]:
                greeks_fig.add_vline(
                    x=self.underlying_price,
                    line_dash="dash",
                    line_color="green",
                    row=row,
                    col=col
                )
        
        # Update layout for greeks chart
        greeks_fig.update_layout(
            title='Option Greeks Analysis',
            height=600,
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Update x-axis titles
        greeks_fig.update_xaxes(title_text='Underlying Price', row=2, col=1)
        greeks_fig.update_xaxes(title_text='Underlying Price', row=2, col=2)
        
        return payoff_fig, greeks_fig


def create_streamlit_app():
    # Set page config
    st.set_page_config(
        page_title="Options Strategy Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create analyzer instance
    analyzer = OptionsAnalyzer()
    
    # Page title
    st.title("🚀 Options Strategy Analyzer")
    st.markdown("Analyze options strategies with customizable parameters, similar to OptionStrat")
    
    # Sidebar for controls
    st.sidebar.header("Settings")
    
    # Ticker input
    ticker_input = st.sidebar.text_input("Ticker Symbol", value="SPY")
    fetch_button = st.sidebar.button("Fetch Data", key="fetch_data")
    
    # Store analyzer in session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = analyzer
    
    if 'expiry_dates' not in st.session_state:
        st.session_state.expiry_dates = []
        
    if 'current_strikes' not in st.session_state:
        st.session_state.current_strikes = {"call": [], "put": []}
    
    # Fetch data when button is clicked
    if fetch_button:
        with st.spinner(f"Fetching data for {ticker_input}..."):
            expiry_dates = st.session_state.analyzer.fetch_ticker_data(ticker_input)
            
            if expiry_dates:
                st.session_state.expiry_dates = expiry_dates
                st.session_state.analyzer.set_expiry(expiry_dates[0])  # Select first expiry by default
                
                # Get available strikes
                calls = st.session_state.analyzer.option_chain.calls['strike'].unique()
                puts = st.session_state.analyzer.option_chain.puts['strike'].unique()
                st.session_state.current_strikes = {"call": sorted(calls), "put": sorted(puts)}
                
                st.sidebar.success(f"Loaded data for {ticker_input}")
            else:
                st.sidebar.error(f"Failed to load data for {ticker_input}")
    
    # Only show these controls if we have data
    if st.session_state.analyzer.underlying_price is not None:
        st.sidebar.metric("Current Price", f"${st.session_state.analyzer.underlying_price:.2f}")
        
        # Expiry selection
        if len(st.session_state.expiry_dates) > 0:
            selected_expiry = st.sidebar.selectbox(
                "Expiration Date",
                options=st.session_state.expiry_dates,
                index=0
            )
            
            # Update expiry when changed
            if selected_expiry != st.session_state.analyzer.current_expiry:
                st.session_state.analyzer.set_expiry(selected_expiry)
                
                # Update available strikes
                calls = st.session_state.analyzer.option_chain.calls['strike'].unique()
                puts = st.session_state.analyzer.option_chain.puts['strike'].unique()
                st.session_state.current_strikes = {"call": sorted(calls), "put": sorted(puts)}
        
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
            
            # Show available strikes
            strikes = st.session_state.current_strikes[position_type.lower()]
            
            if len(strikes) > 0:
                # Find ATM strike
                atm_index = (np.abs(np.array(strikes) - st.session_state.analyzer.underlying_price)).argmin()
                
                selected_strike = st.sidebar.selectbox(
                    "Strike Price",
                    options=strikes,
                    index=atm_index
                )
                
                # Option quantity
                quantity = st.sidebar.number_input(
                    "Quantity",
                    min_value=1,
                    max_value=100,
                    value=1
                )
                
                # Add option button
                if st.sidebar.button(f"Add {position_type} Option"):
                    if st.session_state.analyzer.add_option(selected_strike, position_type.lower(), quantity):
                        st.sidebar.success(f"Added {quantity} {position_type} option(s) at strike ${selected_strike:.2f}")
                    else:
                        st.sidebar.error("Failed to add option")
            else:
                st.sidebar.warning(f"No {position_type}s available for selected expiration")
        else:
            # Stock controls
            quantity = st.sidebar.number_input(
                "Number of Shares",
                min_value=1,
                max_value=10000,
                value=100
            )
            
            # Add stock button
            if st.sidebar.button("Add Stock Position"):
                if st.session_state.analyzer.add_stock(quantity):
                    st.sidebar.success(f"Added {quantity} shares at ${st.session_state.analyzer.underlying_price:.2f}")
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
                    position_desc = f"Stock: {position['quantity']} shares"
                    position_price = f"${position['price']:.2f}/share"
                    position_cost = position['quantity'] * position['price']
                else:
                    position_desc = f"{position['type'].upper()}: {position['quantity']} contract(s) @ ${position['strike']:.2f}"
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
        else:
            st.info("No positions added to strategy yet. Use the sidebar to build your strategy.")

# Create a function to run the app
if __name__ == '__main__':
    create_streamlit_app()
