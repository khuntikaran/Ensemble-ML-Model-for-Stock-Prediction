import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands

logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        
    def calculate_position_size(self, price, volatility, confidence, max_risk=0.02):
        """Calculate optimal position size based on volatility and confidence"""
        # Kelly Criterion with safety factor
        kelly_fraction = (confidence - 0.5) / volatility
        safe_kelly = kelly_fraction * 0.5  # Using half Kelly for safety
        
        # Risk-based position sizing
        max_position_size = self.capital * max_risk / volatility
        
        # Combine both methods
        position_size = min(max_position_size, self.capital * safe_kelly)
        
        # Convert to number of shares
        shares = int(position_size / price)
        
        return shares

    def generate_signals(self, data, predictions, confidence_intervals):
        """Generate trading signals based on predictions and technical indicators"""
        # Use only the last portion of data matching predictions length
        last_n_days = len(predictions)
        signals = pd.DataFrame(index=data.index[-last_n_days:])
        
        # Basic prediction signals
        current_prices = data['Close/Last'].iloc[-last_n_days:]
        signals['pred_return'] = (np.array(predictions) / current_prices.values) - 1
        signals['confidence_range'] = np.array([high - low for low, high in confidence_intervals])
        
        # Technical indicators for last n days
        rsi = RSIIndicator(data['Close/Last']).rsi().iloc[-last_n_days:]
        
        macd = MACD(data['Close/Last'])
        signals['MACD'] = macd.macd_diff().iloc[-last_n_days:]
        
        bb = BollingerBands(data['Close/Last'])
        signals['BB_position'] = ((data['Close/Last'] - bb.bollinger_mavg()) / 
                                (bb.bollinger_hband() - bb.bollinger_lband())).iloc[-last_n_days:]
        
        adx = ADXIndicator(data['High'], data['Low'], data['Close/Last'])
        signals['ADX'] = adx.adx().iloc[-last_n_days:]
        
        # Generate trading signals
        signals['signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
        
        # Long signals with improved criteria
        signals.loc[(signals['pred_return'] > 0.01) &  # Predicted return > 1%
                   (rsi < 70) &  # Not overbought
                   (signals['MACD'] > 0) &  # MACD positive
                   (signals['ADX'] > 25), 'signal'] = 1  # Strong trend
        
        # Short signals with improved criteria
        signals.loc[(signals['pred_return'] < -0.01) &  # Predicted return < -1%
                   (rsi > 30) &  # Not oversold
                   (signals['MACD'] < 0) &  # MACD negative
                   (signals['ADX'] > 25), 'signal'] = -1  # Strong trend
        
        return signals

    def calculate_stop_loss(self, price, volatility, confidence_interval):
        """Calculate dynamic stop loss levels"""
        atr_multiple = 2
        stop_loss = price - (confidence_interval[1] - confidence_interval[0]) * atr_multiple
        return max(stop_loss, price * 0.95)  # Maximum 5% loss

    def calculate_take_profit(self, price, volatility, confidence_interval):
        """Calculate dynamic take profit levels"""
        atr_multiple = 3
        take_profit = price + (confidence_interval[1] - confidence_interval[0]) * atr_multiple
        return take_profit

    def backtest(self, data, predictions, confidence_intervals):
        """Run backtest of the trading strategy"""
        # Use only the last portion of data matching predictions length
        last_n_days = len(predictions)
        recent_data = data.iloc[-last_n_days:].copy()
        
        signals = self.generate_signals(data, predictions, confidence_intervals)
        
        portfolio_value = []
        trades = []
        self.capital = self.initial_capital
        self.position = 0
        
        for i in range(len(recent_data)):
            current_price = recent_data['Close/Last'].iloc[i]
            
            # Calculate position size if we're going to trade
            if signals['signal'].iloc[i] != 0 and self.position == 0:
                volatility = signals['confidence_range'].iloc[i] / current_price
                confidence = 0.5 + abs(signals['pred_return'].iloc[i])
                position_size = self.calculate_position_size(current_price, volatility, confidence)
                
                # Execute trade
                if signals['signal'].iloc[i] == 1:  # Buy
                    cost = position_size * current_price
                    if cost <= self.capital:
                        self.position = position_size
                        self.capital -= cost
                        stop_loss = self.calculate_stop_loss(
                            current_price, 
                            volatility,
                            confidence_intervals[i]
                        )
                        take_profit = self.calculate_take_profit(
                            current_price,
                            volatility,
                            confidence_intervals[i]
                        )
                        trades.append({
                            'date': recent_data.index[i],
                            'type': 'buy',
                            'price': current_price,
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
                
                elif signals['signal'].iloc[i] == -1:  # Sell
                    cost = position_size * current_price
                    if cost <= self.capital:
                        self.position = -position_size
                        self.capital -= cost
                        trades.append({
                            'date': recent_data.index[i],
                            'type': 'sell',
                            'price': current_price,
                            'size': position_size
                        })
            
            # Check for exit conditions
            elif self.position != 0:
                last_trade = trades[-1]
                if last_trade['type'] == 'buy':
                    if (current_price <= last_trade['stop_loss'] or 
                        current_price >= last_trade['take_profit']):
                        # Close position
                        self.capital += self.position * current_price
                        trades.append({
                            'date': recent_data.index[i],
                            'type': 'close',
                            'price': current_price,
                            'size': self.position
                        })
                        self.position = 0
            
            # Calculate portfolio value
            portfolio_value.append(self.capital + (self.position * current_price))
        
        # Calculate performance metrics
        returns = pd.Series(portfolio_value).pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        max_drawdown = (pd.Series(portfolio_value).cummax() - portfolio_value).max() / pd.Series(portfolio_value).cummax()
        
        return {
            'final_value': portfolio_value[-1],
            'returns': (portfolio_value[-1] - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'portfolio_values': portfolio_value
        }

def generate_trading_report(backtest_results, company):
    """Generate a comprehensive trading report"""
    report = f"\nTrading Report for {company}\n"
    report += "=" * 50 + "\n"
    
    # Performance metrics
    returns_pct = backtest_results['returns'] * 100
    report += f"Performance Metrics:\n"
    report += f"Initial Capital: ${backtest_results['portfolio_values'][0]:,.2f}\n"
    report += f"Final Capital: ${backtest_results['final_value']:,.2f}\n"
    report += f"Total Return: {returns_pct:.2f}%\n"
    report += f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}\n"
    report += f"Maximum Drawdown: {backtest_results['max_drawdown']*100:.2f}%\n"
    
    # Trade analysis
    trades = backtest_results['trades']
    winning_trades = sum(1 for i in range(len(trades)-1) 
                        if trades[i]['type'] in ['buy', 'sell'] 
                        and trades[i+1]['price'] * (1 if trades[i]['type']=='buy' else -1) 
                        > trades[i]['price'] * (1 if trades[i]['type']=='buy' else -1))
    total_trades = sum(1 for t in trades if t['type'] in ['buy', 'sell'])
    
    report += f"\nTrade Analysis:\n"
    report += f"Total Trades: {total_trades}\n"
    report += f"Win Rate: {(winning_trades/total_trades*100):.2f}%\n"
    
    return report 