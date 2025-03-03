import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from datetime import datetime, timedelta
from yahooquery import Ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class OptionsAnalyzer:
    """A class to analyze options strategies with customizable parameters."""
    
    def __init__(self):
        self.strategies = []
        self.underlying_price = None
        self.ticker_symbols = []
        self.ticker_data = None
        self.expiration_dates = {}  # Dictionary mapping ticker to expiry dates
        self.option_chains = {}  # Dictionary mapping (ticker, expiry) to option chain
        self.current_ticker = None
        self.current_expiry = None
        self.vol_adjustment = 0.0  # Adjustment to implied volatility in percentage points
    
    def fetch_ticker_data(self, ticker_symbols):
        """Fetch ticker data and available option expiration dates for multiple tickers."""
        if isinstance(ticker_symbols, str):
            ticker_symbols = [ticker_symbols]
        
        self.ticker_symbols = ticker_symbols
        self.ticker_data = Ticker(ticker_symbols)
        
        # Get price data
        price_data = self.ticker_data.price
        
        # Handle single vs. multiple tickers
        if len(ticker_symbols) == 1:
            symbol = ticker_symbols[0]
            if symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
                self.underlying_price = price_data[symbol]['regularMarketPrice']
                self.current_ticker = symbol
            else:
                print(f"Error: Could not get price data for {symbol}")
                return {}
        else:
            # For multiple tickers, just populate the price data
            self.current_ticker = ticker_symbols[0]  # Default to first ticker
            if self.current_ticker in price_data and 'regularMarketPrice' in price_data[self.current_ticker]:
                self.underlying_price = price_data[self.current_ticker]['regularMarketPrice']
            else:
                print(f"Error: Could not get price data for {self.current_ticker}")
                return {}
        
        # Get option expiration dates for all tickers
        self.expiration_dates = {}
        
        # Fetch all option chains
        all_options = self.ticker_data.option_chain
        
        for symbol in ticker_symbols:
            if symbol in all_options and isinstance(all_options[symbol], pd.DataFrame):
                # Extract unique expiration dates
                expiry_dates = all_options[symbol]['expiration'].unique()
                self.expiration_dates[symbol] = sorted([str(expiry.date()) for expiry in pd.to_datetime(expiry_dates, unit='s')])
                
                # Process option chains for each expiry
                for expiry in expiry_dates:
                    expiry_date = str(pd.Timestamp(expiry, unit='s').date())
                    mask = all_options[symbol]['expiration'] == expiry
                    expiry_chain = all_options[symbol][mask].copy()
                    
                    # Separate calls and puts
                    calls = expiry_chain[expiry_chain['optionType'] == 'call'].reset_index(drop=True)
                    puts = expiry_chain[expiry_chain['optionType'] == 'put'].reset_index(drop=True)
                    
                    # Store in dictionary
                    self.option_chains[(symbol, expiry_date)] = {'calls': calls, 'puts': puts}
            else:
                print(f"No option data found for {symbol}")
                self.expiration_dates[symbol] = []
        
        # Set current expiry date for the current ticker if available
        if self.current_ticker in self.expiration_dates and len(self.expiration_dates[self.current_ticker]) > 0:
            self.current_expiry = self.expiration_dates[self.current_ticker][0]
        
        return self.expiration_dates
    
    def set_current_ticker(self, ticker):
        """Set the current ticker for analysis."""
        if ticker in self.ticker_symbols:
            self.current_ticker = ticker
            
            # Update current price
            price_data = self.ticker_data.price
            if ticker in price_data and 'regularMarketPrice' in price_data[ticker]:
                self.underlying_price = price_data[ticker]['regularMarketPrice']
            
            # Set current expiry if available
            if ticker in self.expiration_dates and len(self.expiration_dates[ticker]) > 0:
                self.current_expiry = self.expiration_dates[ticker][0]
            else:
                self.current_expiry = None
            
            return True
        return False
    
    def set_expiry(self, expiry_date):
        """Set the current expiration date for analysis."""
        if self.current_ticker in self.expiration_dates and expiry_date in self.expiration_dates[self.current_ticker]:
            self.current_expiry = expiry_date
            return True
        return False
    
    def get_current_option_chain(self):
        """Get the option chain for the current ticker and expiry."""
        if self.current_ticker and self.current_expiry:
            key = (self.current_ticker, self.current_expiry)
            if key in self.option_chains:
                return self.option_chains[key]
        return None
    
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
        option_chain = self.get_current_option_chain()
        if not option_chain:
            print("Error: No option chain loaded. Please fetch ticker data first.")
            return False
        
        # Find the option in the chain
        if option_type.lower() == 'call':
            options_df = option_chain['calls']
        else:
            options_df = option_chain['puts']
            
        option_row = options_df[options_df['strike'] == strike]
        
        if option_row.empty:
            print(f"Error: No {option_type} option found with strike {strike}")
            return False
        
        # Get option details
        option_data = option_row.iloc[0]
        
        # Use custom price if provided, otherwise use market price
        price = custom_price if custom_price is not None else option_data['lastPrice']
        
        # Get implied volatility, with fallback if it's not available directly
        if 'impliedVolatility' in option_data:
            implied_vol = option_data['impliedVolatility']
        else:
            # Calculate implied vol from bid/ask if available
            mid_price = (option_data['bid'] + option_data['ask']) / 2 if 'bid' in option_data and 'ask' in option_data else price
            days = self.calculate_days_to_expiry()
            T = days / 365.0
            r = 0.05  # Risk-free rate
            
            # Simple approximation of implied vol (would need an iterative solver for accuracy)
            implied_vol = 0.3  # Default value
        
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
            print("Error: No underlying price. Please fetch ticker data first.")
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
            print("Error: No underlying price. Please fetch ticker data first.")
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
                        payoff += quantity * (intrinsic - option_price) * 100  # 100 shares per contract
                    else:
                        # Before expiration - use Black-Scholes
                        implied_vol = position['impliedVolatility']
                        new_price = self.black_scholes_price(price, strike, T, r, implied_vol, option_type)
                        payoff += quantity * (new_price - option_price) * 100  # 100 shares per contract
            
            payoffs.append(payoff)
        
        return np.array(payoffs)
    
    def calculate_greeks_profile(self, price_range=None):
        """Calculate greeks across a range of prices."""
        if not self.underlying_price:
            print("Error: No underlying price. Please fetch ticker data first.")
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
                    
                    delta_profile[i] += quantity * greeks['delta'] * 100  # 100 shares per contract
                    gamma_profile[i] += quantity * greeks['gamma'] * 100
                    theta_profile[i] += quantity * greeks['theta'] * 100
                    vega_profile[i] += quantity * greeks['vega'] * 100
        
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
            # No options in the strategy, use a default value
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
    
    def plot_strategy(self):
        """Plot the strategy payoff and greeks."""
        if not self.strategies:
            print("No strategy defined. Please add positions first.")
            return
            
        # Set up price range for plotting
        price_min = self.underlying_price * 0.8
        price_max = self.underlying_price * 1.2
        price_range = np.linspace(price_min, price_max, 100)
        
        # Calculate payoff and greeks
        payoffs = self.calculate_payoff(price_range)
        greeks = self.calculate_greeks_profile(price_range)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # Plot payoff
        ax_payoff = plt.subplot(gs[0, :])
        ax_payoff.plot(price_range, payoffs, 'b-', linewidth=2)
        ax_payoff.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax_payoff.axvline(x=self.underlying_price, color='g', linestyle='--', alpha=0.5)
        ax_payoff.set_title(f'Strategy Payoff - {self.current_ticker} (Current: ${self.underlying_price:.2f})')
        ax_payoff.set_xlabel('Underlying Price')
        ax_payoff.set_ylabel('Profit/Loss ($)')
        ax_payoff.grid(True)
        
        # Highlight breakeven points
        breakeven_points = self.get_breakeven_points(price_range)
        for point in breakeven_points:
            ax_payoff.axvline(x=point, color='purple', linestyle=':', alpha=0.7)
            ax_payoff.text(point, 0, f'BE: ${point:.2f}', rotation=90, verticalalignment='bottom')
        
        # Calculate max profit/loss
        max_profit, max_loss = self.get_max_profit_loss(price_range)
        
        # Add annotations for key metrics
        metrics_text = f"Max Profit: ${max_profit:.2f}\n"
        metrics_text += f"Max Loss: ${max_loss:.2f}\n"
        
        # Add probability of profit if available
        prob_profit = self.calculate_probability_profit()
        if prob_profit is not None:
            metrics_text += f"Probability of Profit: {prob_profit:.1f}%\n"
        
        metrics_text += f"Days to Expiry: {self.calculate_days_to_expiry()}\n"
        metrics_text += f"IV Adjustment: {self.vol_adjustment * 100:.1f}%"
        
        # Add text box with metrics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax_payoff.text(0.02, 0.98, metrics_text, transform=ax_payoff.transAxes, 
                       fontsize=10, verticalalignment='top', bbox=props)
        
        # Plot delta
        ax_delta = plt.subplot(gs[1, 0])
        ax_delta.plot(price_range, greeks['delta'], 'g-')
        ax_delta.axvline(x=self.underlying_price, color='g', linestyle='--', alpha=0.5)
        ax_delta.set_title('Delta')
        ax_delta.set_xlabel('Underlying Price')
        ax_delta.set_ylabel('Delta')
        ax_delta.grid(True)
        
        # Plot gamma
        ax_gamma = plt.subplot(gs[1, 1])
        ax_gamma.plot(price_range, greeks['gamma'], 'm-')
        ax_gamma.axvline(x=self.underlying_price, color='g', linestyle='--', alpha=0.5)
        ax_gamma.set_title('Gamma')
        ax_gamma.set_xlabel('Underlying Price')
        ax_gamma.set_ylabel('Gamma')
        ax_gamma.grid(True)
        
        # Plot theta
        ax_theta = plt.subplot(gs[2, 0])
        ax_theta.plot(price_range, greeks['theta'], 'r-')
        ax_theta.axvline(x=self.underlying_price, color='g', linestyle='--', alpha=0.5)
        ax_theta.set_title('Theta (Daily)')
        ax_theta.set_xlabel('Underlying Price')
        ax_theta.set_ylabel('Theta')
        ax_theta.grid(True)
        
        # Plot vega
        ax_vega = plt.subplot(gs[2, 1])
        ax_vega.plot(price_range, greeks['vega'], 'c-')
        ax_vega.axvline(x=self.underlying_price, color='g', linestyle='--', alpha=0.5)
        ax_vega.set_title('Vega')
        ax_vega.set_xlabel('Underlying Price')
        ax_vega.set_ylabel('Vega')
        ax_vega.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print strategy summary
        print("\nStrategy Summary:")
        total_cost = 0
        for position in self.strategies:
            if position['type'] == 'stock':
                print(f"Stock: {position['quantity']} shares at ${position['price']:.2f}")
                total_cost += position['quantity'] * position['price']
            else:
                print(f"{position['type'].capitalize()}: {position['quantity']} contracts at strike ${position['strike']:.2f}, price ${position['price']:.2f}")
                total_cost += position['quantity'] * position['price'] * 100  # Assuming standard 100 shares per contract
        
        print(f"\nTotal Position Cost: ${total_cost:.2f}")
        print(f"Max Profit: ${max_profit:.2f}")
        print(f"Max Loss: ${max_loss:.2f}")
        if prob_profit is not None:
            print(f"Probability of Profit: {prob_profit:.1f}%")
    
    def generate_plotly_charts(self):
        """Generate Plotly charts for strategy analysis."""
        if not self.strategies:
            print("No strategy defined. Please add positions first.")
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
            title=f'Strategy Payoff - {self.current_ticker} (Current: ${self.underlying_price:.2f})',
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
