import datetime
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math

# --- Module 1: Data Fetching ---
def fetch_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# --- Module 2: Technical Indicators ---
def calculate_ema(data, period=60):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_parabolic_sar(data, af=0.02, af_max=0.2):
    high = data['High'].values
    low = data['Low'].values
    n = len(data)
    psar = np.zeros(n, dtype=np.float64)
    trend = np.ones(n)
    ep = np.zeros(n, dtype=np.float64)
    af_current = af
    psar[0] = low[0]
    ep[0] = high[0]
    for i in range(1, n):
        psar[i] = psar[i-1] + af_current * (ep[i-1] - psar[i-1])
        if trend[i-1] == 1:
            psar[i] = min(psar[i], low[i-1], low[max(i-2, 0)])
        else:
            psar[i] = max(psar[i], high[i-1], high[max(i-2, 0)])
        if trend[i-1] == 1:
            if low[i] < psar[i]:
                trend[i] = -1
                psar[i] = ep[i-1]
                ep[i] = low[i]
                af_current = af
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af_current = min(af_current + af, af_max)
                else:
                    ep[i] = ep[i-1]
        else:
            if high[i] > psar[i]:
                trend[i] = 1
                psar[i] = ep[i-1]
                ep[i] = high[i]
                af_current = af
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af_current = min(af_current + af, af_max)
                else:
                    ep[i] = ep[i-1]
    return pd.DataFrame({'PSAR': psar}, index=data.index)

# --- Module 3: Construct Signals ---
def construct_signals(data, ema_period=60, psar_af=0.02, psar_af_max=0.2):
    data = data.copy()
    ema_df = calculate_ema(data, ema_period)
    psar_df = calculate_parabolic_sar(data, psar_af, psar_af_max)
    data = data.join(ema_df.rename(f'EMA_{ema_period}'))
    data = data.join(psar_df)
    data['Trend'] = (data['Open'] - data[f'EMA_{ema_period}']) * 100
    data['Return'] = data['Close'].pct_change() * 100
    data['Direction'] = np.where(data['Close'] > data['Open'], 1, -1)
    features = ['Trend', 'Volume', f'EMA_{ema_period}', 'PSAR', 'Return']
    data = data.dropna()
    return data[features + ['Direction']]

# --- Module 4: Create Sequences for LSTM ---
def create_sequences_multifeature(data, feature_cols, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data[feature_cols].iloc[i:i+sequence_length].values
        label = data['Direction'].iloc[i+sequence_length]
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y).reshape(-1, 1)

# --- Module 5: LSTM Model Training ---
def build_lstm_model(input_shape, units1=50, units2=30, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    if X_train.ndim != 3:
        raise ValueError("Training data must have 3 dimensions.")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)
    return model, history

# --- Helper: Safe Conversion for JSON ---
def safe_convert(df):
    df_obj = df.copy().astype(object)
    df_clean = df_obj.applymap(lambda x: None if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)) else x)
    return df_clean.to_dict(orient="records")

# --- Module 6: Web API Integration: Run Analysis ---
def run_analysis(symbol, start_date_str, end_date_str):
    # Convert date strings to datetime objects
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date   = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Fetch raw data from yfinance
    raw_data = fetch_data(symbol, start_date, end_date)
    print("Raw data rows:", len(raw_data))
    if raw_data.empty:
        raise ValueError("No data found for the given parameters.")
    
    # Reset index so that Date becomes a column and convert to string
    raw_data_reset = raw_data.reset_index()
    raw_data_reset["Date"] = raw_data_reset["Date"].astype(str)
    
    # Compute EMA and Trend (for plotting)
    ema_60 = calculate_ema(raw_data_reset, period=60)
    raw_data_reset["EMA_60"] = ema_60
    raw_data_reset["Trend"] = (raw_data_reset["Open"] - raw_data_reset["EMA_60"]) * 100
    
    # Prepare plotting data (sample: first 50 rows)
    plot_data = raw_data_reset.copy()
    plot_data["MovingAvg"] = plot_data["Close"].rolling(window=5).mean()
    plot_data["Return"] = plot_data["Close"].pct_change()
    data_head = safe_convert(plot_data.head(50))
    print("Data head rows (for plotting):", len(data_head))
    
    # Construct signals for ML training
    signals_data = construct_signals(raw_data, ema_period=60, psar_af=0.02, psar_af_max=0.2)
    print("Signals data rows:", len(signals_data))
    signals_data_clean = signals_data.dropna()
    print("Signals data clean rows:", len(signals_data_clean))
    if signals_data_clean.empty:
        signals_data_clean = raw_data.fillna(method='bfill').fillna(method='ffill')
        signals_data_clean = construct_signals(signals_data_clean, ema_period=60, psar_af=0.02, psar_af_max=0.2)
    
    # Compute correlation matrix on signals data
    correlation_matrix = []
    correlation_labels = []
    if not signals_data_clean.empty:
        corr_df = signals_data_clean.corr()
        correlation_matrix = corr_df.values.tolist()
        correlation_labels = list(corr_df.columns)
    print("Correlation matrix size:", len(correlation_matrix))
    
    # Prepare data for LSTM training:
    feature_cols = ['Trend', 'EMA_60', 'PSAR', 'Volume', 'Return']
    signals_data_clean['Direction'] = np.where(signals_data_clean['Direction'] == 1, 1, 0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(signals_data_clean[feature_cols])
    scaled_df = pd.DataFrame(scaled_features, index=signals_data_clean.index, columns=feature_cols)
    scaled_df['Direction'] = signals_data_clean['Direction']
    print("Scaled data rows:", len(scaled_df))
    
    # Create sequences for LSTM
    sequence_length = 10
    if len(scaled_df) <= sequence_length:
        raise ValueError("Not enough data points to create sequences. Increase your date range.")
    X, y = create_sequences_multifeature(scaled_df, feature_cols, sequence_length=sequence_length)
    print("X shape:", X.shape, "y shape:", y.shape)
    
    # Split into training and validation sets (80/20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print("Training set size:", X_train.shape, "Validation set size:", X_val.shape)
    
    # Train the LSTM model
    model, history = train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)
    
    # Check training history arrays
    trainAccuracy = history.history.get('accuracy', [])
    valAccuracy = history.history.get('val_accuracy', [])
    trainLoss = history.history.get('loss', [])
    valLoss = history.history.get('val_loss', [])
    print("Train accuracy:", trainAccuracy)
    
    # Evaluate model on validation set and compute confusion matrix
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_val, y_pred).tolist()
    print("Confusion matrix:", cm)
    
    return {
        "data_head": data_head,
        "trainAccuracy": trainAccuracy or [],
        "valAccuracy": valAccuracy or [],
        "trainLoss": trainLoss or [],
        "valLoss": valLoss or [],
        "confusion_matrix": cm or [],
        "correlation_matrix": correlation_matrix or [],
        "correlation_labels": correlation_labels or []
    }
