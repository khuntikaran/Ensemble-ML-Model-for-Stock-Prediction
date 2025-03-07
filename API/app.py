import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import logging
import traceback
import numpy as np

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import using the full path from project root
from src.models.predict import predict_next_price

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response for favicon

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Stock Prediction API is running',
        'endpoints': {
            'health_check': '/api/health',
            'stock_prediction': '/api/stock/<symbol>'
        }
    })

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        logger.info(f"Fetching data for {symbol}")
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid symbol provided")
            
        result = predict_next_price(symbol)
        
        # Validate result structure
        required_keys = ['current_price', 'predictions', 'performance_metrics', 
                        'risk_metrics', 'trend_analysis']
        if not all(key in result for key in required_keys):
            raise ValueError("Invalid prediction result structure")
        
        # Format the response
        response = {
            'current_price': float(result['current_price']),
            'predictions': {
                'next_day': float(result['predictions']['next_day']),
                'five_day': [float(x) for x in result['predictions']['five_day']],
                'confidence_intervals': [[float(x) for x in interval] 
                                      for interval in result['predictions']['confidence_intervals']]
            },
            'performance_metrics': {
                'r2': float(result['performance_metrics']['r2']),
                'mae': float(result['performance_metrics']['mae']),
                'mse': float(result['performance_metrics']['mse'])
            },
            'risk_metrics': {
                'volatility': float(result['risk_metrics']['volatility']),
                'sharpe_ratio': float(result['risk_metrics']['sharpe_ratio']),
                'var_95': float(result['risk_metrics']['var_95']),
                'max_drawdown': float(result['risk_metrics']['max_drawdown'])
            },
            'trend_analysis': {
                'trend': str(result['trend_analysis']['trend']),
                'strength': float(result['trend_analysis']['strength']),
                'support': float(result['trend_analysis']['support']),
                'resistance': float(result['trend_analysis']['resistance'])
            },
            'trading_signal': str(result.get('trading_signal', 'HOLD'))
        }
        
        logger.info(f"Successfully processed data for {symbol}")
        return jsonify(response)
        
    except ValueError as ve:
        logger.error(f"Validation error for {symbol}: {str(ve)}")
        return jsonify({
            'error': str(ve),
            'status': 'error'
        }), 400
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'python_path': sys.path,
            'dependencies': {
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'flask': Flask.__version__
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    try:
        logger.info(f"Project root: {project_root}")
        logger.info(f"Python path: {sys.path}")
        
        # Start the server
        app.run(debug=True, port=5000)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)