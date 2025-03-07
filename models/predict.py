import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from src.models.train import StockPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Add the models directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(company):
    """Load the latest model for a company"""
    try:
        model_path = f'models/{company}_model_latest.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for {company}")
        
        model_info = joblib.load(model_path)
        logger.info(f"Loaded model for {company}")
        return model_info
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def calculate_volatility(prices, window=20):
    """Calculate historical volatility"""
    returns = np.log(prices / prices.shift(1))
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def calculate_confidence_interval(predictions, std_dev, confidence=0.95):
    """Calculate confidence intervals for predictions"""
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * std_dev
    return predictions - margin, predictions + margin

def calculate_risk_metrics(actual_values, predictions, current_price):
    """Calculate various risk and performance metrics"""
    # Ensure we're using numpy arrays and handle Series objects
    if isinstance(actual_values, pd.Series):
        actual_values = actual_values.values
    if isinstance(predictions, pd.Series):
        predictions = predictions.values
    if isinstance(current_price, pd.Series):
        current_price = float(current_price.iloc[-1])
    
    actual_values = np.array(actual_values[-len(predictions):])
    predictions = np.array(predictions)
    
    # Calculate returns
    returns = np.diff(actual_values) / actual_values[:-1]
    
    # Calculate metrics
    volatility = float(np.std(returns) * np.sqrt(252))
    sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    var_95 = float(np.percentile(returns, 5) * current_price)
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))
    
    return {
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'max_drawdown': max_drawdown
    }

def analyze_trend(prices, short_window=10, long_window=50):
    """Analyze price trend using multiple indicators and time periods"""
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Convert to pandas Series for easier calculation
    price_series = pd.Series(prices)
    
    # Calculate multiple moving averages
    sma_10 = price_series.rolling(window=10).mean()
    sma_20 = price_series.rolling(window=20).mean()
    sma_50 = price_series.rolling(window=50).mean()
    
    # Calculate price momentum over different periods
    momentum_5 = (price_series - price_series.shift(5)) / price_series.shift(5)
    momentum_10 = (price_series - price_series.shift(10)) / price_series.shift(10)
    momentum_20 = (price_series - price_series.shift(20)) / price_series.shift(20)
    
    # Get recent values
    current_price = price_series.iloc[-1]
    prev_price_5 = price_series.iloc[-6] if len(price_series) > 5 else price_series.iloc[0]
    prev_price_10 = price_series.iloc[-11] if len(price_series) > 10 else price_series.iloc[0]
    
    # Calculate trend indicators
    trend_indicators = {
        'sma_alignment': (
            sma_10.iloc[-1] > sma_20.iloc[-1] and 
            sma_20.iloc[-1] > sma_50.iloc[-1]
        ),
        'price_above_sma': current_price > sma_50.iloc[-1],
        'recent_momentum': all(
            m > 0 for m in [
                momentum_5.iloc[-1],
                momentum_10.iloc[-1],
                momentum_20.iloc[-1]
            ]
        ),
        'higher_lows': all(
            price_series.iloc[i-3:i].min() > price_series.iloc[i-6:i-3].min()
            for i in range(len(price_series)-3, len(price_series), 3)
            if i >= 6
        )
    }
    
    # Determine trend based on multiple factors
    upward_signals = sum(trend_indicators.values())
    trend_strength = upward_signals / len(trend_indicators)
    
    # More stringent trend determination
    if trend_strength >= 0.7:
        current_trend = 'Upward'
    elif trend_strength <= 0.3:
        current_trend = 'Downward'
    else:
        # Check additional conditions for sideways/uncertain trend
        price_volatility = price_series.pct_change().std()
        if price_volatility > 0.02:  # High volatility
            current_trend = 'Volatile'
        else:
            current_trend = 'Sideways'
    
    # Calculate support and resistance using more data points
    recent_prices = prices[-30:]  # Use 30 days of data
    
    # Dynamic support/resistance calculation
    price_clusters = pd.qcut(recent_prices, q=10, duplicates='drop')
    support = float(price_clusters.categories[1].left)  # 10th percentile
    resistance = float(price_clusters.categories[-2].right)  # 90th percentile
    
    return {
        'trend': current_trend,
        'strength': float(trend_strength),
        'support': support,
        'resistance': resistance,
        'indicators': {
            'sma_alignment': trend_indicators['sma_alignment'],
            'price_above_sma': trend_indicators['price_above_sma'],
            'recent_momentum': trend_indicators['recent_momentum'],
            'higher_lows': trend_indicators['higher_lows']
        }
    }

def predict_multiple_days(predictor, data, days=5):
    """Predict prices for multiple days ahead"""
    predictions = []
    confidence_intervals = []
    last_sequence = data.iloc[-predictor.sequence_length:].copy()
    current_price = float(last_sequence['Close/Last'].iloc[-1])
    
    # Calculate price momentum
    price_momentum = (current_price - float(last_sequence['Close/Last'].iloc[-5])) / float(last_sequence['Close/Last'].iloc[-5])
    
    for i in range(days):
        # Prepare sequence
        X = last_sequence[predictor.features].values
        X = X.reshape(1, predictor.sequence_length * len(predictor.features))
        
        # Make prediction
        base_pred = predictor.predict(X)[0]
        
        # Calculate prediction std using both models
        X_scaled = predictor.scaler.transform(X)
        X_selected = predictor.feature_selector.transform(X_scaled)
        gb_pred = predictor.gb_model.predict(X_selected)[0]
        rf_pred = predictor.rf_model.predict(X_selected)[0]
        
        # Ensemble prediction with momentum adjustment
        pred = (gb_pred * predictor.gb_weight + rf_pred * predictor.rf_weight)
        pred_return = (pred - current_price) / current_price
        
        # Adjust prediction based on momentum
        momentum_factor = 1 + (price_momentum * (1 - i/days))  # Decay momentum impact
        scaled_pred = current_price * (1 + pred_return * momentum_factor)
        
        # Calculate confidence intervals
        pred_std = np.std([gb_pred, rf_pred])
        volatility = calculate_volatility(last_sequence['Close/Last'])
        if isinstance(volatility, pd.Series):
            volatility = volatility.iloc[-1]
        
        # Adjust confidence interval based on volatility
        pred_std_adjusted = pred_std * (1 + volatility)
        lower, upper = calculate_confidence_interval(scaled_pred, pred_std_adjusted)
        
        predictions.append(float(scaled_pred))
        confidence_intervals.append((float(lower), float(upper)))
        
        # Update sequence for next prediction
        new_row = last_sequence.iloc[-1:].copy()
        new_row['Close/Last'] = scaled_pred
        last_sequence = pd.concat([last_sequence[1:], new_row])
        
        # Update momentum
        if i > 0:
            price_momentum = (scaled_pred - predictions[-2]) / predictions[-2]
    
    return predictions, confidence_intervals

def generate_trading_signal(current_price, prediction, trend_info, risk_metrics, technical_indicators):
    """Generate trading signals based on multiple factors"""
    signal = "HOLD"
    confidence = 0.0
    
    # Extract key metrics
    trend = trend_info['trend']
    trend_strength = trend_info['strength']
    support = trend_info['support']
    resistance = trend_info['resistance']
    volatility = risk_metrics['volatility']
    rsi = technical_indicators.get('RSI', 50)  # Default to neutral if not available
    
    # Calculate price position relative to support/resistance
    price_to_resistance = (resistance - current_price) / resistance
    price_to_support = (current_price - support) / support
    predicted_return = (prediction - current_price) / current_price
    
    # BUY signals
    if (
        # Strong upward trend with room to grow
        (trend == 'Upward' and trend_strength > 0.3 and price_to_resistance > 0.02) or
        # Near support with positive prediction
        (price_to_support < 0.01 and predicted_return > 0.01) or
        # Oversold condition with positive momentum
        (rsi < 30 and predicted_return > 0.02 and trend == 'Upward')
    ):
        signal = "BUY"
        confidence = min(0.8, trend_strength + abs(predicted_return))
    
    # SELL signals
    elif (
        # Strong downward trend
        (trend == 'Downward' and trend_strength > 0.3 and price_to_support > 0.02) or
        # Near resistance with negative prediction
        (price_to_resistance < 0.01 and predicted_return < -0.01) or
        # Overbought condition with negative momentum
        (rsi > 70 and predicted_return < -0.02 and trend == 'Downward')
    ):
        signal = "SELL"
        confidence = min(0.8, trend_strength + abs(predicted_return))
    
    # Adjust confidence based on volatility
    confidence = confidence * (1 - min(volatility, 0.5))
    
    return {
        'signal': signal,
        'confidence': confidence,
        'factors': {
            'trend': trend,
            'trend_strength': trend_strength,
            'price_to_resistance': price_to_resistance,
            'price_to_support': price_to_support,
            'predicted_return': predicted_return,
            'rsi': rsi,
            'volatility': volatility
        }
    }

def predict_next_price(company, data_file='data/u_dataset.csv'):
    """Predict the next price and provide comprehensive analysis"""
    try:
        # Load the model info
        model_info = load_model(company)
        predictor = StockPredictor()
        
        # Load model components
        for key, value in model_info.items():
            setattr(predictor, key, value)
        
        # Prepare the data
        data = predictor.prepare_data(data_file, company)
        
        # Convert price columns to float
        for col in ['Close/Last', 'High', 'Low', 'Open']:
            if isinstance(data[col], pd.Series):
                data[col] = data[col].astype(float)
        
        current_price = float(data['Close/Last'].iloc[-1])
        
        # Multi-day predictions
        future_predictions, confidence_intervals = predict_multiple_days(predictor, data)
        
        # Calculate historical performance metrics
        X = data[predictor.features].copy()
        for col in X.columns:
            X[col] = X[col].astype(float)
            
        sequences, actual_values = predictor.create_sequences(X, predictor.sequence_length)
        X_reshaped = sequences.reshape(sequences.shape[0], sequences.shape[1] * sequences.shape[2])
        historical_predictions = predictor.predict(X_reshaped)
        
        # Calculate metrics
        performance_metrics = {
            'mse': float(mean_squared_error(actual_values, historical_predictions)),
            'mae': float(mean_absolute_error(actual_values, historical_predictions)),
            'r2': float(r2_score(actual_values, historical_predictions))
        }
        
        # Risk metrics
        risk_metrics = calculate_risk_metrics(
            actual_values, 
            historical_predictions,
            current_price
        )
        
        # Trend analysis
        trend_info = analyze_trend(data['Close/Last'].astype(float))
        
        # Technical indicators for signal generation
        technical_indicators = {
            'RSI': float(data['RSI'].iloc[-1]),
            'MACD': float(data['MACD'].iloc[-1]),
            'Signal_Line': float(data['Signal_Line'].iloc[-1]),
            'BB_Upper': float(data['BB_Upper'].iloc[-1]),
            'BB_Lower': float(data['BB_Lower'].iloc[-1])
        }
        
        # Generate trading signal
        signal_info = generate_trading_signal(
            current_price,
            future_predictions[0],  # Next day prediction
            trend_info,
            risk_metrics,
            technical_indicators
        )
        
        return {
            'current_price': current_price,
            'predictions': {
                'next_day': float(future_predictions[0]),
                'five_day': [float(p) for p in future_predictions],
                'confidence_intervals': [(float(l), float(u)) for l, u in confidence_intervals]
            },
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'trend_analysis': trend_info,
            'trading_signal': signal_info['signal'],
            'signal_confidence': signal_info['confidence'],
            'signal_factors': signal_info['factors']
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def main():
    try:
        logging.basicConfig(level=logging.INFO)
        companies = ['AAPL', 'MSFT', 'META', 'AMZN', 'NFLX']
        
        for company in companies:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analysis for {company}")
            logger.info(f"{'='*50}")
            
            try:
                # Get predictions and analysis
                result = predict_next_price(company)
                current_price = float(result['current_price'])
                
                # Current price and predictions
                logger.info(f"Current Price: ${current_price:.2f}")
                logger.info(f"\nPredictions:")
                logger.info(f"Next Day: ${result['predictions']['next_day']:.2f}")
                logger.info("\nNext 5 Days Forecast:")
                for day, (pred, (lower, upper)) in enumerate(zip(
                    result['predictions']['five_day'],
                    result['predictions']['confidence_intervals']
                ), 1):
                    logger.info(f"Day {day}: ${pred:.2f} (95% CI: ${lower:.2f} - ${upper:.2f})")
                
                # Performance Metrics
                logger.info(f"\nModel Performance:")
                logger.info(f"R² Score: {result['performance_metrics']['r2']:.4f}")
                logger.info(f"MAE: ${result['performance_metrics']['mae']:.2f}")
                logger.info(f"MSE: ${result['performance_metrics']['mse']:.2f}")
                
                # Risk Metrics
                logger.info(f"\nRisk Analysis:")
                logger.info(f"Volatility (Annualized): {result['risk_metrics']['volatility']*100:.2f}%")
                logger.info(f"Sharpe Ratio: {result['risk_metrics']['sharpe_ratio']:.2f}")
                logger.info(f"Value at Risk (95%): ${abs(result['risk_metrics']['var_95']):.2f}")
                logger.info(f"Maximum Drawdown: {result['risk_metrics']['max_drawdown']*100:.2f}%")
                
                # Trend Analysis
                logger.info(f"\nTrend Analysis:")
                logger.info(f"Current Trend: {result['trend_analysis']['trend']}")
                logger.info(f"Trend Strength: {result['trend_analysis']['strength']*100:.2f}%")
                logger.info(f"Support Level: ${float(result['trend_analysis']['support']):.2f}")
                logger.info(f"Resistance Level: ${float(result['trend_analysis']['resistance']):.2f}")
                
                # Trading Signal
                logger.info(f"\nTrading Signal: {result['trading_signal']}")
                logger.info(f"Signal Confidence: {result['signal_confidence']:.2f}")
                logger.info(f"Signal Factors: {result['signal_factors']}")
                
            except Exception as e:
                logger.error(f"Error analyzing {company}: {str(e)}")
                continue
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()