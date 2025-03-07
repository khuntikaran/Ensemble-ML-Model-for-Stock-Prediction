import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from datetime import datetime
import joblib
import os
import logging
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.gb_model = None
        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = None
        self.feature_selector = None
        self.sequence_length = 30
        self.prediction_horizon = 1
        self.model_weights = None
        
        # Core features for prediction
        self.features = [
            # Price features
            'Close/Last', 'Volume', 'VWAP',
            'High_Low_Ratio', 'Close_Open_Ratio',
            
            # Moving Averages
            'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_10', 'EMA_20', 'EMA_50',
            
            # Momentum Indicators
            'RSI', 'Stoch_K', 'Stoch_D',
            'ROC_5', 'ROC_10', 'ROC_20',
            
            # Trend Indicators
            'MACD', 'Signal_Line', 'MACD_Hist',
            
            # Volatility Indicators
            'BB_Upper', 'BB_Lower', 'BB_Middle',
            'ATR', 'ATR_Percent',
            
            # Volume Indicators
            'OBV', 'Volume_Change',
            'Volume_MA_Ratio', 'Volume_VWAP_Ratio'
        ]

    def prepare_data(self, file_path, company):
        try:
            data = pd.read_csv(file_path)
            company_data = data[data['Company'] == company].copy()
            
            if company_data.empty:
                raise ValueError(f"No data found for company: {company}")
            
            # Convert price columns
            price_cols = ['Close/Last', 'High', 'Low', 'Open']
            for col in price_cols:
                if company_data[col].dtype == 'object':
                    company_data[col] = company_data[col].str.replace('$', '').astype(float)
            
            # Sort by date
            company_data['Date'] = pd.to_datetime(company_data[['Year', 'Month', 'Day']])
            company_data = company_data.sort_values('Date')
            
            # Basic features
            company_data['High_Low_Ratio'] = company_data['High'] / company_data['Low']
            company_data['Close_Open_Ratio'] = company_data['Close/Last'] / company_data['Open']
            company_data['Price_Change'] = company_data['Close/Last'].pct_change()
            
            # Calculate ATR
            atr_indicator = AverageTrueRange(
                high=company_data['High'],
                low=company_data['Low'],
                close=company_data['Close/Last'],
                window=14
            )
            company_data['ATR'] = atr_indicator.average_true_range()
            company_data['ATR_Percent'] = company_data['ATR'] / company_data['Close/Last'] * 100
            
            # Technical indicators
            bb = BollingerBands(company_data['Close/Last'])
            company_data['BB_Upper'] = bb.bollinger_hband()
            company_data['BB_Lower'] = bb.bollinger_lband()
            company_data['BB_Middle'] = bb.bollinger_mavg()
            
            company_data['RSI'] = RSIIndicator(company_data['Close/Last']).rsi()
            
            macd = MACD(company_data['Close/Last'])
            company_data['MACD'] = macd.macd()
            company_data['Signal_Line'] = macd.macd_signal()
            company_data['MACD_Hist'] = macd.macd_diff()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(
                high=company_data['High'],
                low=company_data['Low'],
                close=company_data['Close/Last']
            )
            company_data['Stoch_K'] = stoch.stoch()
            company_data['Stoch_D'] = stoch.stoch_signal()
            
            # Rate of Change (ROC) calculations
            company_data['ROC_5'] = company_data['Close/Last'].pct_change(periods=5)
            company_data['ROC_10'] = company_data['Close/Last'].pct_change(periods=10)
            company_data['ROC_20'] = company_data['Close/Last'].pct_change(periods=20)
            
            # Moving Averages
            company_data['SMA_10'] = SMAIndicator(close=company_data['Close/Last'], window=10).sma_indicator()
            company_data['SMA_20'] = SMAIndicator(close=company_data['Close/Last'], window=20).sma_indicator()
            company_data['SMA_50'] = SMAIndicator(close=company_data['Close/Last'], window=50).sma_indicator()
            
            company_data['EMA_10'] = EMAIndicator(close=company_data['Close/Last'], window=10).ema_indicator()
            company_data['EMA_20'] = EMAIndicator(close=company_data['Close/Last'], window=20).ema_indicator()
            company_data['EMA_50'] = EMAIndicator(close=company_data['Close/Last'], window=50).ema_indicator()
            
            # Volume indicators
            company_data['Volume_Change'] = company_data['Volume'].pct_change()
            company_data['VWAP'] = VolumeWeightedAveragePrice(
                high=company_data['High'],
                low=company_data['Low'],
                close=company_data['Close/Last'],
                volume=company_data['Volume']
            ).volume_weighted_average_price()
            
            company_data['OBV'] = OnBalanceVolumeIndicator(
                close=company_data['Close/Last'],
                volume=company_data['Volume']
            ).on_balance_volume()
            
            company_data['Volume_MA_Ratio'] = company_data['Volume'] / company_data['Volume'].rolling(20).mean()
            company_data['Volume_VWAP_Ratio'] = company_data['Volume'] / company_data['VWAP']
            
            # Handle missing values
            company_data = company_data.fillna(method='ffill').fillna(method='bfill')
            
            return company_data
            
        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise

    def create_sequences(self, data, seq_length):
        """Create sequences for time series prediction"""
        sequences = []
        targets = []
        
        # Calculate returns for normalization
        returns = data['Close/Last'].pct_change()
        volatility = returns.rolling(window=21).std()
        
        for i in range(len(data) - seq_length - self.prediction_horizon + 1):
            seq = data.iloc[i:(i + seq_length)]
            current_price = data.iloc[i + seq_length - 1]['Close/Last']
            future_price = data.iloc[i + seq_length + self.prediction_horizon - 1]['Close/Last']
            target_return = (future_price - current_price) / current_price
            
            if not volatility.iloc[i + seq_length - 1] == 0:
                target_return = target_return / volatility.iloc[i + seq_length - 1]
            
            sequences.append(seq)
            targets.append(target_return)
        
        return np.array(sequences), np.array(targets)

    def save_model(self, company):
        """Save the model and its components"""
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_info = {
            'gb_model': self.gb_model,
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'lgb_model': self.lgb_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'sequence_length': self.sequence_length,
            'features': self.features,
            'model_weights': self.model_weights
        }
        
        joblib.dump(model_info, f'models/{company}_model_{timestamp}.joblib')
        joblib.dump(model_info, f'models/{company}_model_latest.joblib')
        
        logger.info(f"Saved model for {company}")

    def predict(self, X):
        """Make predictions using the ensemble model"""
        if not all([self.gb_model, self.rf_model, self.xgb_model, self.lgb_model, self.scaler, self.feature_selector]):
            raise ValueError("Model components not properly initialized")
        
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        gb_pred = self.gb_model.best_estimator_.predict(X_selected)
        rf_pred = self.rf_model.best_estimator_.predict(X_selected)
        xgb_pred = self.xgb_model.best_estimator_.predict(X_selected)
        lgb_pred = self.lgb_model.best_estimator_.predict(X_selected)
        
        y_pred = (
            self.model_weights['gb'] * gb_pred +
            self.model_weights['rf'] * rf_pred +
            self.model_weights['xgb'] * xgb_pred +
            self.model_weights['lgb'] * lgb_pred
        )
        
        return y_pred

    def train(self, file_path, company):
        try:
            data = self.prepare_data(file_path, company)
            X = data[self.features]
            y = data['Close/Last']
            
            target_scaler = RobustScaler()
            y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
            
            sequences, targets = self.create_sequences(X, self.sequence_length)
            
            # Log the number of sequences for debugging
            logger.info(f"Number of sequences created: {len(sequences)}")
            
            # Calculate proper test size and number of splits
            total_samples = len(sequences)
            test_size = min(int(total_samples * 0.15), total_samples // 6)  # Reduced to 15% or 1/6
            n_splits = min(3, max(2, (total_samples - test_size) // (test_size * 2)))  # Ensure enough samples per split
            
            logger.info(f"Using {n_splits} splits with test_size {test_size}")
            
            # Initialize TimeSeriesSplit with gap parameter
            tscv = TimeSeriesSplit(
                n_splits=n_splits,
                test_size=test_size,
                gap=5  # Add gap between train and test sets
            )
            
            # Optimized parameter grids
            gb_param_grid = {
                'n_estimators': [500, 1000],
                'learning_rate': [0.01, 0.005],
                'max_depth': [3, 5, 7],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10],
                'subsample': [0.8, 0.9]
            }
            
            rf_param_grid = {
                'n_estimators': [500, 1000],
                'max_depth': [5, 8, 10],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10],
                'max_features': ['sqrt', 'log2']
            }
            
            xgb_param_grid = {
                'n_estimators': [500, 1000],
                'learning_rate': [0.01, 0.005],
                'max_depth': [4, 6, 8],
                'min_child_weight': [3, 5],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'gamma': [0.1, 0.2]
            }
            
            lgb_param_grid = {
                'n_estimators': [500, 1000],
                'learning_rate': [0.01, 0.005],
                'max_depth': [4, 6, 8],
                'num_leaves': [31, 63],
                'min_child_samples': [10, 20],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
            
            # Initialize models with GridSearchCV
            self.gb_model = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid=gb_param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            self.rf_model = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid=rf_param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            self.xgb_model = GridSearchCV(
                XGBRegressor(random_state=42),
                param_grid=xgb_param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            self.lgb_model = GridSearchCV(
                LGBMRegressor(random_state=42, verbose=-1),
                param_grid=lgb_param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            val_predictions = []
            val_targets = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(sequences)):
                X_train, X_val = sequences[train_idx], sequences[val_idx]
                y_train, y_val = targets[train_idx], targets[val_idx]
                
                self.scaler = RobustScaler()
                X_train_scaled = self.scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
                X_val_scaled = self.scaler.transform(X_val.reshape(X_val.shape[0], -1))
                
                if fold == 0:
                    selector = SelectFromModel(XGBRegressor(random_state=42), threshold='median')
                    selector.fit(X_train_scaled, y_train)
                    self.feature_selector = selector
                
                X_train_selected = self.feature_selector.transform(X_train_scaled)
                X_val_selected = self.feature_selector.transform(X_val_scaled)
                
                self.gb_model.fit(X_train_selected, y_train)
                self.rf_model.fit(X_train_selected, y_train)
                self.xgb_model.fit(X_train_selected, y_train)
                self.lgb_model.fit(X_train_selected, y_train)
                
                if fold == 0:
                    logger.info(f"Best parameters for {company}:")
                    logger.info(f"GB: {self.gb_model.best_params_}")
                    logger.info(f"RF: {self.rf_model.best_params_}")
                    logger.info(f"XGB: {self.xgb_model.best_params_}")
                    logger.info(f"LGB: {self.lgb_model.best_params_}")
                
                gb_pred = self.gb_model.best_estimator_.predict(X_val_selected)
                rf_pred = self.rf_model.best_estimator_.predict(X_val_selected)
                xgb_pred = self.xgb_model.best_estimator_.predict(X_val_selected)
                lgb_pred = self.lgb_model.best_estimator_.predict(X_val_selected)
                
                gb_mse = mean_squared_error(y_val, gb_pred)
                rf_mse = mean_squared_error(y_val, rf_pred)
                xgb_mse = mean_squared_error(y_val, xgb_pred)
                lgb_mse = mean_squared_error(y_val, lgb_pred)
                
                total_error = np.exp(-np.array([gb_mse, rf_mse, xgb_mse, lgb_mse]))
                weights = total_error / np.sum(total_error)
                
                self.model_weights = {
                    'gb': weights[0],
                    'rf': weights[1],
                    'xgb': weights[2],
                    'lgb': weights[3]
                }
                
                fold_pred = (
                    weights[0] * gb_pred +
                    weights[1] * rf_pred +
                    weights[2] * xgb_pred +
                    weights[3] * lgb_pred
                )
                
                val_predictions.extend(fold_pred)
                val_targets.extend(y_val)
            
            val_predictions = target_scaler.inverse_transform(np.array(val_predictions).reshape(-1, 1)).ravel()
            val_targets = target_scaler.inverse_transform(np.array(val_targets).reshape(-1, 1)).ravel()
            
            metrics = {
                'mse': mean_squared_error(val_targets, val_predictions),
                'mae': mean_absolute_error(val_targets, val_predictions),
                'r2': r2_score(val_targets, val_predictions)
            }
            
            self.save_model(company)
            return metrics
            
        except Exception as e:
            logger.error(f"Error in train: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        predictor = StockPredictor()
        companies = ['AAPL', 'MSFT', 'META', 'AMZN', 'NFLX']
        
        for company in companies:
            logger.info(f"\nTraining model for {company}")
            metrics = predictor.train('data/u_dataset.csv', company)
            logger.info(f"Metrics for {company}:")
            logger.info(f"MSE: {metrics['mse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"R2 Score: {metrics['r2']:.4f}")
            
    except Exception as e:
        logger.error(f"Error running training: {str(e)}")