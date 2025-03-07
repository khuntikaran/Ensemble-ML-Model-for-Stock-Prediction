import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, stocks, initial_capital=100000):
        self.stocks = stocks
        self.initial_capital = initial_capital
        
    def calculate_portfolio_metrics(self, returns, weights):
        """Calculate portfolio return and risk metrics"""
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_std
        
        return {
            'return': portfolio_return,
            'risk': portfolio_std,
            'sharpe': sharpe_ratio
        }
    
    def optimize_portfolio(self, stock_data, risk_free_rate=0.02):
        """Optimize portfolio weights using Modern Portfolio Theory"""
        returns = pd.DataFrame()
        
        # Calculate returns for each stock
        for stock in self.stocks:
            stock_returns = stock_data[stock]['Close/Last'].pct_change()
            returns[stock] = stock_returns
        
        returns = returns.dropna()
        
        # Number of stocks
        n = len(self.stocks)
        
        # Initial guess (equal weights)
        weights = np.array([1/n] * n)
        
        # Constraints
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        )
        
        # Bounds for weights (0 to 1)
        bounds = tuple((0, 1) for _ in range(n))
        
        # Objective function to maximize Sharpe Ratio
        def objective(weights):
            portfolio_metrics = self.calculate_portfolio_metrics(returns, weights)
            sharpe = (portfolio_metrics['return'] - risk_free_rate) / portfolio_metrics['risk']
            return -sharpe  # Minimize negative Sharpe Ratio
        
        # Optimize
        result = minimize(objective, weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        # Calculate metrics with optimal weights
        metrics = self.calculate_portfolio_metrics(returns, optimal_weights)
        
        portfolio_allocation = {}
        for stock, weight in zip(self.stocks, optimal_weights):
            allocation = self.initial_capital * weight
            shares = int(allocation / stock_data[stock]['Close/Last'].iloc[-1])
            portfolio_allocation[stock] = {
                'weight': weight,
                'allocation': allocation,
                'shares': shares
            }
        
        return {
            'weights': optimal_weights,
            'metrics': metrics,
            'allocation': portfolio_allocation
        }
    
    def calculate_var(self, stock_data, confidence_level=0.95):
        """Calculate Value at Risk for the portfolio"""
        returns = pd.DataFrame()
        weights = []
        
        # Calculate weighted returns
        for stock in self.stocks:
            stock_returns = stock_data[stock]['Close/Last'].pct_change()
            returns[stock] = stock_returns
            price = stock_data[stock]['Close/Last'].iloc[-1]
            weight = price / sum(stock_data[s]['Close/Last'].iloc[-1] for s in self.stocks)
            weights.append(weight)
        
        weights = np.array(weights)
        portfolio_returns = returns.dot(weights)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return {
            'VaR': -var * self.initial_capital,
            'CVaR': -cvar * self.initial_capital
        }
    
    def generate_rebalancing_trades(self, current_positions, optimal_allocation):
        """Generate trades needed to rebalance portfolio"""
        trades = []
        
        for stock in self.stocks:
            current = current_positions.get(stock, 0)
            target = optimal_allocation[stock]['shares']
            
            if current != target:
                trades.append({
                    'stock': stock,
                    'action': 'BUY' if target > current else 'SELL',
                    'shares': abs(target - current)
                })
        
        return trades

def generate_portfolio_report(optimizer, optimization_result, risk_metrics):
    """Generate comprehensive portfolio report"""
    report = "\nPortfolio Optimization Report\n"
    report += "=" * 50 + "\n\n"
    
    # Portfolio metrics
    report += "Portfolio Metrics:\n"
    report += f"Expected Annual Return: {optimization_result['metrics']['return']*100:.2f}%\n"
    report += f"Portfolio Risk (std): {optimization_result['metrics']['risk']*100:.2f}%\n"
    report += f"Sharpe Ratio: {optimization_result['metrics']['sharpe']:.2f}\n"
    report += f"Value at Risk (95%): ${risk_metrics['VaR']:,.2f}\n"
    report += f"Conditional VaR: ${risk_metrics['CVaR']:,.2f}\n\n"
    
    # Asset allocation
    report += "Optimal Asset Allocation:\n"
    for stock, alloc in optimization_result['allocation'].items():
        report += f"{stock}:\n"
        report += f"  Weight: {alloc['weight']*100:.2f}%\n"
        report += f"  Allocation: ${alloc['allocation']:,.2f}\n"
        report += f"  Shares: {alloc['shares']}\n"
    
    return report 