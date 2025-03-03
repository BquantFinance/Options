import streamlit as st
from Main import OptionsAnalyzer

def add_common_strategies(analyzer):
    """
    Add common options strategy templates to the analyzer.
    
    Parameters:
    analyzer (OptionsAnalyzer): The options analyzer instance
    """
    
    st.header("Strategy Templates")
    strategy_type = st.selectbox(
        "Select Strategy Type",
        options=[
            "Long Call", 
            "Long Put", 
            "Bull Call Spread",
            "Bear Put Spread",
            "Iron Condor",
            "Butterfly",
            "Covered Call",
            "Cash-Secured Put",
            "Long Straddle",
            "Long Strangle"
        ]
    )
    
    # Get option chain for current ticker and expiry
    option_chain = analyzer.get_current_option_chain()
    if not option_chain:
        st.error("No option chain available. Please fetch data first.")
        return
    
    # Get current price
    current_price = analyzer.underlying_price
    if not current_price:
        st.error("No price data available.")
        return
    
    # Get calls and puts
    calls = option_chain['calls'].sort_values(by='strike')
    puts = option_chain['puts'].sort_values(by='strike')
    
    # Find ATM and nearby strikes
    calls_strikes = sorted(calls['strike'].unique())
    puts_strikes = sorted(puts['strike'].unique())
    
    # Find closest ATM strike
    atm_call_idx = min(range(len(calls_strikes)), key=lambda i: abs(calls_strikes[i] - current_price))
    atm_put_idx = min(range(len(puts_strikes)), key=lambda i: abs(puts_strikes[i] - current_price))
    
    atm_call_strike = calls_strikes[atm_call_idx]
    atm_put_strike = puts_strikes[atm_put_idx]
    
    # Try to get strikes above and below for spreads
    call_strike_above = calls_strikes[atm_call_idx + 1] if atm_call_idx + 1 < len(calls_strikes) else atm_call_strike * 1.05
    call_strike_below = calls_strikes[atm_call_idx - 1] if atm_call_idx > 0 else atm_call_strike * 0.95
    
    put_strike_above = puts_strikes[atm_put_idx + 1] if atm_put_idx + 1 < len(puts_strikes) else atm_put_strike * 1.05
    put_strike_below = puts_strikes[atm_put_idx - 1] if atm_put_idx > 0 else atm_put_strike * 0.95
    
    # Strategy parameters container
    strategy_params = st.expander("Strategy Parameters", expanded=True)
    
    with strategy_params:
        if strategy_type == "Long Call":
            strike = st.selectbox("Call Strike", options=calls_strikes, index=atm_call_idx)
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Long Call Strategy"):
                analyzer.clear_strategy()
                analyzer.add_option(strike, 'call', quantity)
                st.success(f"Added Long Call strategy with {quantity} call(s) at strike ${strike}")
                
        elif strategy_type == "Long Put":
            strike = st.selectbox("Put Strike", options=puts_strikes, index=atm_put_idx)
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Long Put Strategy"):
                analyzer.clear_strategy()
                analyzer.add_option(strike, 'put', quantity)
                st.success(f"Added Long Put strategy with {quantity} put(s) at strike ${strike}")
                
        elif strategy_type == "Bull Call Spread":
            lower_strike = st.selectbox("Lower Strike (Buy)", options=calls_strikes, index=max(0, atm_call_idx - 1))
            upper_strike = st.selectbox("Upper Strike (Sell)", options=calls_strikes, 
                                      index=min(len(calls_strikes)-1, atm_call_idx + 1))
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Bull Call Spread"):
                analyzer.clear_strategy()
                analyzer.add_option(lower_strike, 'call', quantity)  # Buy lower strike
                analyzer.add_option(upper_strike, 'call', -quantity)  # Sell higher strike
                st.success(f"Added Bull Call Spread: Buy {quantity} call(s) at ${lower_strike}, Sell {quantity} call(s) at ${upper_strike}")
                
        elif strategy_type == "Bear Put Spread":
            upper_strike = st.selectbox("Upper Strike (Buy)", options=puts_strikes, 
                                      index=min(len(puts_strikes)-1, atm_put_idx + 1))
            lower_strike = st.selectbox("Lower Strike (Sell)", options=puts_strikes, index=max(0, atm_put_idx - 1))
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Bear Put Spread"):
                analyzer.clear_strategy()
                analyzer.add_option(upper_strike, 'put', quantity)  # Buy higher strike
                analyzer.add_option(lower_strike, 'put', -quantity)  # Sell lower strike
                st.success(f"Added Bear Put Spread: Buy {quantity} put(s) at ${upper_strike}, Sell {quantity} put(s) at ${lower_strike}")
                
        elif strategy_type == "Iron Condor":
            # Find strike indices for approximately 10% OTM positions
            otm_call_idx = min(range(len(calls_strikes)), 
                             key=lambda i: abs(calls_strikes[i] - current_price * 1.1))
            otm_put_idx = min(range(len(puts_strikes)), 
                            key=lambda i: abs(puts_strikes[i] - current_price * 0.9))
            
            # Puts (lower) side
            put_sell_strike = st.selectbox("Put Sell Strike", options=puts_strikes, index=otm_put_idx)
            put_buy_strike = st.selectbox("Put Buy Strike (Wing)", 
                                        options=[s for s in puts_strikes if s < put_sell_strike],
                                        index=0)
            
            # Calls (upper) side
            call_sell_strike = st.selectbox("Call Sell Strike", options=calls_strikes, index=otm_call_idx)
            call_buy_strike = st.selectbox("Call Buy Strike (Wing)", 
                                         options=[s for s in calls_strikes if s > call_sell_strike],
                                         index=0)
            
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Iron Condor"):
                analyzer.clear_strategy()
                # Puts side
                analyzer.add_option(put_sell_strike, 'put', -quantity)  # Sell put
                analyzer.add_option(put_buy_strike, 'put', quantity)    # Buy put (wing)
                # Calls side
                analyzer.add_option(call_sell_strike, 'call', -quantity)  # Sell call
                analyzer.add_option(call_buy_strike, 'call', quantity)    # Buy call (wing)
                
                st.success("Added Iron Condor Strategy")
                
        elif strategy_type == "Butterfly":
            center_strike = st.selectbox("Center Strike (Sell 2x)", options=calls_strikes, index=atm_call_idx)
            wing_width = st.slider("Wing Width", min_value=1, max_value=10, value=2)
            
            # Find wing strikes
            center_idx = calls_strikes.index(center_strike)
            lower_idx = max(0, center_idx - wing_width)
            upper_idx = min(len(calls_strikes) - 1, center_idx + wing_width)
            
            lower_strike = calls_strikes[lower_idx]
            upper_strike = calls_strikes[upper_idx]
            
            st.write(f"Lower Wing: ${lower_strike}")
            st.write(f"Upper Wing: ${upper_strike}")
            
            option_type = st.radio("Option Type", ["Call", "Put"])
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Butterfly"):
                analyzer.clear_strategy()
                if option_type == "Call":
                    analyzer.add_option(lower_strike, 'call', quantity)   # Buy lower wing
                    analyzer.add_option(center_strike, 'call', -2 * quantity)  # Sell 2x center
                    analyzer.add_option(upper_strike, 'call', quantity)   # Buy upper wing
                else:
                    analyzer.add_option(lower_strike, 'put', quantity)    # Buy lower wing
                    analyzer.add_option(center_strike, 'put', -2 * quantity)   # Sell 2x center
                    analyzer.add_option(upper_strike, 'put', quantity)    # Buy upper wing
                    
                st.success(f"Added {option_type} Butterfly Strategy")
                
        elif strategy_type == "Covered Call":
            call_strike = st.selectbox("Call Strike (Sell)", options=calls_strikes, 
                                    index=min(len(calls_strikes)-1, atm_call_idx + 1))
            shares = st.number_input("Number of Shares", min_value=100, max_value=1000, value=100, step=100)
            calls = st.number_input("Number of Calls to Sell", min_value=1, max_value=10, value=int(shares/100))
            
            if st.button("Create Covered Call"):
                analyzer.clear_strategy()
                analyzer.add_stock(shares)  # Buy shares
                analyzer.add_option(call_strike, 'call', -calls)  # Sell calls
                st.success(f"Added Covered Call: Long {shares} shares, Short {calls} call(s) at ${call_strike}")
                
        elif strategy_type == "Cash-Secured Put":
            put_strike = st.selectbox("Put Strike (Sell)", options=puts_strikes, index=atm_put_idx)
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Cash-Secured Put"):
                analyzer.clear_strategy()
                analyzer.add_option(put_strike, 'put', -quantity)  # Sell put
                st.success(f"Added Cash-Secured Put: Short {quantity} put(s) at ${put_strike}")
                
        elif strategy_type == "Long Straddle":
            strike = st.selectbox("Strike (ATM)", options=calls_strikes, index=atm_call_idx)
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Long Straddle"):
                analyzer.clear_strategy()
                analyzer.add_option(strike, 'call', quantity)  # Buy call
                analyzer.add_option(strike, 'put', quantity)   # Buy put
                st.success(f"Added Long Straddle: Long {quantity} call(s) and {quantity} put(s) at ${strike}")
                
        elif strategy_type == "Long Strangle":
            call_strike = st.selectbox("Call Strike (OTM)", 
                                    options=[s for s in calls_strikes if s > current_price],
                                    index=0)
            put_strike = st.selectbox("Put Strike (OTM)", 
                                   options=[s for s in puts_strikes if s < current_price],
                                   index=len([s for s in puts_strikes if s < current_price])-1)
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
            
            if st.button("Create Long Strangle"):
                analyzer.clear_strategy()
                analyzer.add_option(call_strike, 'call', quantity)  # Buy OTM call
                analyzer.add_option(put_strike, 'put', quantity)    # Buy OTM put
                st.success(f"Added Long Strangle: Long {quantity} call(s) at ${call_strike} and {quantity} put(s) at ${put_strike}")

def insert_strategy_templates_to_app():
    """
    Function to insert the strategy templates section in the Streamlit app.
    """
    # This should be called in the main app file
    if 'analyzer' in st.session_state and st.session_state.analyzer.underlying_price is not None:
        add_common_strategies(st.session_state.analyzer)
