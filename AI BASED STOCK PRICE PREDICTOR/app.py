from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

app = Flask(__name__)

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lr_model = LinearRegression()
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            return None
    
    def create_features(self, data):
        """Create technical indicators as features"""
        df = data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        return df
    
    def prepare_data(self, data, lookback_days=10):
        """Prepare data for machine learning"""
        df = self.create_features(data)
        df = df.dropna()
        
        features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'MA_20', 
                   'RSI', 'BB_upper', 'BB_lower', 'Volume_MA', 'Price_Change', 'High_Low_Ratio']
        
        X = []
        y = []
        
        for i in range(lookback_days, len(df)):
            X.append(df[features].iloc[i-lookback_days:i].values.flatten())
            y.append(df['Close'].iloc[i])
        
        return np.array(X), np.array(y), features
    
    def train_models(self, symbol):
        """Train prediction models"""
        data = self.fetch_stock_data(symbol, period="2y")
        if data is None or len(data) < 50:
            return None, "Error fetching data or insufficient data"
        
        X, y, features = self.prepare_data(data)
        
        if len(X) == 0:
            return None, "Insufficient data for training"
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.rf_model.fit(X_train_scaled, y_train)
        self.lr_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        rf_pred = self.rf_model.predict(X_test_scaled)
        lr_pred = self.lr_model.predict(X_test_scaled)
        
        # Calculate metrics
        rf_r2 = r2_score(y_test, rf_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        return {
            'rf_r2': rf_r2,
            'lr_r2': lr_r2,
            'rf_rmse': rf_rmse,
            'lr_rmse': lr_rmse,
            'current_price': data['Close'].iloc[-1],
            'data_points': len(data)
        }, None
    
    def predict_future_prices(self, symbol, days=7):
        """Predict future stock prices"""
        data = self.fetch_stock_data(symbol, period="2y")
        if data is None:
            return None, "Error fetching data"
        
        # Train models first
        result, error = self.train_models(symbol)
        if error:
            return None, error
        
        X, y, features = self.prepare_data(data)
        X_scaled = self.scaler.transform(X)
        
        # Get last sequence for prediction
        last_sequence = X_scaled[-1].reshape(1, -1)
        
        predictions = []
        current_price = data['Close'].iloc[-1]
        
        for day in range(days):
            # Predict with both models
            rf_pred = self.rf_model.predict(last_sequence)[0]
            lr_pred = self.lr_model.predict(last_sequence)[0]
            
            # Ensemble prediction (weighted average)
            ensemble_pred = (rf_pred * 0.6) + (lr_pred * 0.4)
            
            predictions.append({
                'day': day + 1,
                'date': (datetime.now() + timedelta(days=day+1)).strftime('%Y-%m-%d'),
                'predicted_price': round(ensemble_pred, 2),
                'rf_prediction': round(rf_pred, 2),
                'lr_prediction': round(lr_pred, 2)
            })
        
        return {
            'predictions': predictions,
            'current_price': round(current_price, 2),
            'model_performance': result
        }, None

predictor = StockPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        days = int(data.get('days', 7))
        
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        if days < 1 or days > 30:
            return jsonify({'error': 'Days must be between 1 and 30'}), 400
        
        result, error = predictor.predict_future_prices(symbol, days)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stock_info/<symbol>')
def stock_info(symbol):
    try:
        stock = yf.Ticker(symbol.upper())
        info = stock.info
        hist = stock.history(period="5d")
        
        return jsonify({
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'current_price': round(hist['Close'].iloc[-1], 2) if len(hist) > 0 else 'N/A',
            'volume': int(hist['Volume'].iloc[-1]) if len(hist) > 0 else 'N/A'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
