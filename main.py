

# =========================
# INSTALL DEPENDENCIES
# =========================
#!pip install python-binance pykalman websockets nest_asyncio river scipy --quiet

# =========================
# IMPORTS
# =========================
import requests
import asyncio
import websockets
import json
from collections import deque
import numpy as np
from datetime import datetime
from pykalman import KalmanFilter
from binance.client import Client
import nest_asyncio
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers, models

nest_asyncio.apply()

# =========================
# PARAMETERS
# =========================
symbol = "BTCUSDT"
interval = 1  # seconds for price fetch
window_size = 30  # trend model window
cache_window = 300  # last 5 minutes for River ML
price_history = []
SYMBOL = "BTCUSDT"
ROWS = 90  # top rows to sum
# Binance client
client = Client(tld="us", api_key="", api_secret="")

# WebSocket
ws_symbol = symbol.lower()
WS_URL = f"wss://stream.binance.us:9443/ws/{ws_symbol}@depth"

# Correct symbol mapping per exchange
EXCHANGE_SYMBOL = {
    "binance": lambda s: s,          # BTCUSDT
    "kraken": lambda s: "XBTUSDT",   # Kraken uses XBT
    "kucoin": lambda s: s.replace("USDT","-USDT"),
    "huobi": lambda s: s.lower(),
    "bybit": lambda s: s,            # BTCUSDT
    "okx": lambda s: s.replace("USDT","-USDT")
}

def safe_json_get(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except:
        return None

# ====== Exchange functions ======
def get_binance(symbol):
    r = safe_json_get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={ROWS}")
    if r:
        return r.get("bids", []), r.get("asks", [])
    return [], []

def get_kraken(symbol):
    r = safe_json_get(f"https://api.kraken.com/0/public/Depth?pair={symbol}&count={ROWS}")
    if r and "result" in r:
        key = list(r["result"].keys())[0]
        return r["result"][key]["bids"], r["result"][key]["asks"]
    return [], []

def get_kucoin(symbol):
    r = safe_json_get(f"https://api.kucoin.com/api/v1/market/orderbook/level2_100?symbol={symbol}")
    if r and "data" in r:
        return r["data"]["bids"], r["data"]["asks"]
    return [], []

def get_huobi(symbol):
    r = safe_json_get(f"https://api.huobi.pro/market/depth?symbol={symbol}&type=step0")
    if r and "tick" in r:
        return r["tick"]["bids"][:ROWS], r["tick"]["asks"][:ROWS]
    return [], []

def get_bybit(symbol):
    r = safe_json_get(f"https://api.bybit.com/v5/market/books?instId={symbol}&sz={ROWS}")
    if r and "data" in r and len(r["data"]) > 0:
        data = r["data"][0]
        return data["bids"], data["asks"]
    return [], []

def get_okx(symbol):
    r = safe_json_get(f"https://www.okx.com/api/v5/market/books?instId={symbol}&sz={ROWS}")
    if r and "data" in r and len(r["data"]) > 0:
        data = r["data"][0]
        return data["bids"], data["asks"]
    return [], []

EXCHANGES = {
    "Binance": get_binance,
    "Kraken": get_kraken,
    "Kucoin": get_kucoin,
    "Huobi": get_huobi,
    "Bybit": get_bybit,
    "OKX": get_okx
}

# ====== Aggregate bids and asks ======
total_bid_amount = 0.0
total_ask_amount = 0.0

for name, func in EXCHANGES.items():
    try:
        mapped_symbol = EXCHANGE_SYMBOL[name.lower()](SYMBOL)
        bids, asks = func(mapped_symbol)
        if not bids:
            print(f"{name} returned empty bids.")
        if not asks:
            print(f"{name} returned empty asks.")
        total_bid_amount += sum(float(b[1]) for b in bids[:ROWS])
        total_ask_amount += sum(float(a[1]) for a in asks[:ROWS])
    except Exception as e:
        print(f"{name} error: {e}")



# Orderbook
bids, asks = {}, {}
last_best_bid, last_best_ask = None, None
vpin_window, vol_window, ofi_window, micro_window, cancel_window = (
    deque(maxlen=50),
    deque(maxlen=100),
    deque(maxlen=20),
    deque(maxlen=10),
    deque(maxlen=50)
)
snapshot_cache = deque(maxlen=cache_window)

# =========================
# MULTI-TRADING SYSTEM
# =========================
balance = 1000.0
positions = {}  # key: strategy_name, value: {"type": "LONG"/"SHORT", "entry": price}
trade_log = []

# =========================
# UTILITY FUNCTIONS
# =========================
def invert_signal(sig):
    if isinstance(sig, str):
        if "BUY" in sig: return "SELL 🔴"
        if "SELL" in sig: return "BUY 🟢"
    return sig

def trend_signal(pred, last_price):
    if pred > last_price: return "BUY 🟢"
    elif pred < last_price: return "SELL 🔴"
    return "NEUTRAL ➖"

def fetch_last_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

def merge_signals(preds, last_price):
    signals = [1 if p>last_price else -1 if p<last_price else 0 for p in preds]
    score = sum(signals)
    if score>0: return "BUY 🟢"
    elif score<0: return "SELL 🔴"
    return "NEUTRAL ➖"

# =========================
# TREND PREDICTION MODELS
# =========================
def predict_lr(prices):
    if len(prices)<2: return prices[-1]
    x = np.arange(len(prices))
    y = np.array(prices)
    A = np.vstack([x, np.ones(len(x))]).T
    m,c = np.linalg.lstsq(A,y,rcond=None)[0]
    return float(m*len(prices)+c)

def predict_hma(prices, period=16):
    if len(prices)<period: return prices[-1]
    def wma(arr,n):
        if len(arr)<n: return arr[-1]
        weights = np.arange(1,n+1)
        return np.sum(arr[-n:]*weights)/weights.sum()
    half = period//2
    sqrt_len = int(np.sqrt(period))
    wma_half = wma(np.array(prices), half)
    wma_full = wma(np.array(prices), period)
    raw_hma = 2*wma_half - wma_full
    return float(wma(np.array([raw_hma]), sqrt_len))

def predict_kalman(prices):
    if len(prices)<2: return prices[-1]
    kf = KalmanFilter(initial_state_mean=prices[0], n_dim_obs=1)
    state_means,_ = kf.smooth(np.array(prices))
    return float(state_means[-1])

def predict_cwma(prices):
    if len(prices)<2: return prices[-1]
    returns = np.diff(prices)
    cov = np.cov(returns) if len(returns)>1 else 1.0
    weight = 1/(1+cov)
    return float(np.average(prices, weights=np.full(len(prices), weight)))

def predict_dma(prices, displacement=3):
    if len(prices)<=displacement: return prices[-1]
    return np.mean(prices[-displacement:])

def predict_ema(prices, period=10):
    if len(prices)<period: return prices[-1]
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return float(np.convolve(prices[-period:], weights, mode='valid')[0])

def predict_tema(prices, period=10):
    if len(prices)<period*3: return prices[-1]
    ema1 = predict_ema(prices, period)
    ema2 = predict_ema([predict_ema(prices[:i+1], period) for i in range(len(prices))], period)
    ema3 = predict_ema([predict_ema([predict_ema(prices[:i+1], period) for i in range(j+1)], period) for j in range(len(prices))], period)
    return float(3*ema1 - 3*ema2 + ema3)

def predict_wma(prices, period=10):
    if len(prices) < period: return prices[-1]
    weights = np.arange(1, period+1)
    return float(np.dot(prices[-period:], weights)/weights.sum())

def predict_smma(prices, period=10):
    if len(prices) < period: return prices[-1]
    smma = np.mean(prices[:period])
    for p in prices[period:]:
        smma = (smma*(period-1)+p)/period
    return float(smma)

def predict_momentum(prices, period=5):
    if len(prices) < period+1: return prices[-1]
    return prices[-1] - prices[-period-1]

# =========================
# EXOTIC MODELS
# =========================
def predict_hzlog(prices, period=14):
    if len(prices)<period: return prices[-1]
    log_prices = np.log(prices[-period:])
    analytic_signal = scipy.signal.hilbert(log_prices)
    hz_signal = np.real(analytic_signal[-1])
    return float(np.exp(hz_signal))

def predict_vydia(prices, period=10):
    if len(prices)<period: return prices[-1]
    vol = np.std(prices[-period:])
    weights = np.exp(np.linspace(-vol, 0., period))
    weights /= weights.sum()
    return float(np.convolve(prices[-period:], weights, mode='valid')[0])

def predict_parma(prices, period=14):
    if len(prices)<period: return prices[-1]
    highs = np.array(prices[-period:])
    lows = np.array(prices[-period:])
    weights = highs - lows + 1e-9
    return float(np.average(prices[-period:], weights=weights))

def predict_junx(prices, period=5):
    if len(prices)<period+1: return prices[-1]
    diffs = np.diff(prices[-period-1:])
    jump_adj = np.sum(diffs[diffs>0]) - np.sum(diffs[diffs<0])
    return float(prices[-1] + jump_adj/period)

def predict_t3(prices, period=10, vfactor=0.7):
    if len(prices)<period*3: return prices[-1]
    def ema(arr, p): return float(np.convolve(arr[-p:], np.exp(np.linspace(-1.,0.,p))/np.exp(np.linspace(-1.,0.,p)).sum(), mode='valid')[0])
    e1 = ema(prices, period)
    e2 = ema([ema(prices[:i+1], period) for i in range(len(prices))], period)
    e3 = ema([ema([ema(prices[:i+1], period) for i in range(j+1)], period) for j in range(len(prices))], period)
    return float(e1*(1+vfactor) - e2*vfactor + e3*(vfactor**2))

def predict_ichimoku(prices, short=9, long=26):
    if len(prices)<long: return prices[-1]
    tenkan = (max(prices[-short:]) + min(prices[-short:]))/2
    kijun  = (max(prices[-long:]) + min(prices[-long:]))/2
    cloud_top = max(tenkan,kijun)
    cloud_bot = min(tenkan,kijun)
    if prices[-1]>cloud_top: return prices[-1]*1.001
    elif prices[-1]<cloud_bot: return prices[-1]*0.999
    else: return prices[-1]

# =========================
# ADDITIONAL EXOTIC MODELS
# =========================

def predict_ar(prices, lags=8):
    """Autoregressive-like predictor using ordinary least squares on lagged values."""
    if len(prices) <= lags:
        return prices[-1]
    y = np.array(prices[lags:])
    X = np.column_stack([np.array(prices[lags - i: -i]) for i in range(1, lags + 1)])
    # add intercept
    A = np.vstack([X.T, np.ones(X.shape[0])]).T
    try:
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        last_row = np.array(prices[-lags:])[::-1]  # most recent lags (in same order used)
        pred = last_row.dot(coef[:-1]) + coef[-1]
        return float(pred)
    except Exception:
        return float(prices[-1])

def predict_fft(prices, keep=6, horizon=1):
    """FFT-based seasonal extrapolation: keep top `keep` spectral components and extrapolate."""
    n = len(prices)
    if n < 6:
        return prices[-1]
    x = np.array(prices) - np.mean(prices)
    freqs = np.fft.rfft(x)
    mags = np.abs(freqs)
    # zero out small components
    idx = np.argsort(mags)[-keep:]
    mask = np.zeros_like(freqs, dtype=bool)
    mask[idx] = True
    filtered = freqs * mask
    # reconstruct and linearly extrapolate the next step by assuming same phase progression
    recon = np.fft.irfft(filtered, n=n)
    # fall back to last reconstructed value + slope from last two reconstructed
    if len(recon) >= 2:
        slope = recon[-1] - recon[-2]
        return float((recon[-1] + slope * horizon) + np.mean(prices))
    else:
        return float(prices[-1])

def predict_macd_midprice(prices, fast=12, slow=26, signal=9):
    """Return a projection based on MACD momentum added to last price (small-step projection)."""
    if len(prices) < slow:
        return prices[-1]
    def ema(arr, n):
        weights = np.exp(np.linspace(-1., 0., n))
        weights /= weights.sum()
        return np.convolve(arr, weights, mode='valid')
    fast_series = ema(prices, fast)
    slow_series = ema(prices, slow)
    if len(fast_series) < 1 or len(slow_series) < 1:
        return prices[-1]
    # align ends
    macd = fast_series[-len(slow_series):] - slow_series
    if len(macd) < signal:
        macd_signal = np.mean(macd)
    else:
        macd_signal = ema(macd, signal)[-1]
    macd_hist = macd[-1] - macd_signal
    # project a small step proportional to macd_hist
    return float(prices[-1] + 0.5 * macd_hist)

def predict_median_filter(prices, period=7):
    """Robust median-filter prediction (resistant to spikes)."""
    if len(prices) < period:
        return prices[-1]
    return float(np.median(prices[-period:]))

def predict_entropy_weighted(prices, period=20):
    """Weights recent prices by inverse normalised return entropy (lower entropy -> higher weight)."""
    if len(prices) < 6:
        return prices[-1]
    window = np.array(prices[-period:])
    ret = np.diff(window)
    # small probabilistic entropy proxy: use normalized absolute returns distribution
    if len(ret) < 2:
        return prices[-1]
    probs, _ = np.histogram(np.abs(ret), bins=6, density=True)
    probs = probs + 1e-9
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    # produce a weight inversely proportional to entropy; apply to recent values
    win_len = min(period, len(window))
    weights = 1.0 / (1.0 + np.linspace(entropy, entropy * 0.5, win_len))
    weights /= weights.sum()
    return float(np.dot(window[-win_len:], weights))

def predict_rls_trend(prices):
    """Simple batch Recursive Least Squares-like trend projection (forgetting via exponential window)."""
    n = len(prices)
    if n < 3:
        return prices[-1]
    lam = 0.97  # forgetting factor
    # fit line with exponential weights
    x = np.arange(n)
    w = lam ** (n - 1 - x)  # recent points higher weight
    W = np.diag(w)
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        beta = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ np.array(prices), rcond=None)[0]
        next_x = n
        return float(beta[0] * next_x + beta[1])
    except Exception:
        return float(prices[-1])

# =========================
# MORE EXOTIC MODELS (paste under your ADDITIONAL EXOTIC MODELS)
# =========================

import math
from scipy.signal import savgol_filter

def predict_savgol(prices, window=9, polyorder=3, horizon=1):
    """S-G denoise + last-slope extrapolation. Fast and robust to spikes."""
    n = len(prices)
    if n < 3:
        return prices[-1]
    # ensure odd window and <= n
    w = min(window, n if n % 2 == 1 else n-1)
    if w < 3:
        return prices[-1]
    try:
        filt = savgol_filter(prices[-w:], w, min(polyorder, w-1))
        slope = filt[-1] - filt[-2]
        return float(filt[-1] + slope * horizon)
    except Exception:
        return float(prices[-1])

def predict_local_poly(prices, window=12, degree=2, horizon=1):
    """Fit low-degree polynomial to last `window` points and extrapolate."""
    n = len(prices)
    if n < degree + 2:
        return prices[-1]
    w = min(window, n)
    y = np.array(prices[-w:])
    x = np.arange(w).astype(float)
    # shift x so poly is numerically stable (predict next index = w)
    x_shift = x - x.mean()
    coeffs = np.polyfit(x_shift, y, min(degree, w-1))
    poly = np.poly1d(coeffs)
    next_x = (w) - x.mean()
    return float(poly(next_x))

def predict_holt(prices, alpha=0.4, beta=0.2, horizon=1):
    """Simple Holt's linear trend (double exp smoothing) 1-step forecast."""
    n = len(prices)
    if n < 3:
        return prices[-1]
    s = prices[0]
    b = prices[1] - prices[0]
    for i in range(1, n):
        last = s
        s = alpha * prices[i] + (1 - alpha) * (s + b)
        b = beta * (s - last) + (1 - beta) * b
    return float(s + b * horizon)

def predict_lms_online(prices, taps=6, mu=0.01):
    """Simple batch LMS/SGD linear predictor using last `taps` values.
       Very cheap and adaptive: fits weights to minimize last-window squared error then forecasts 1-step."""
    n = len(prices)
    if n <= taps:
        return prices[-1]
    # prepare X matrix of lagged values and y vector
    X = []
    y = []
    for i in range(taps, n):
        X.append(prices[i-taps:i][::-1])  # most recent first
        y.append(prices[i])
    X = np.array(X)
    y = np.array(y)
    # initialize weights as zeros and do a few LMS passes (cheap)
    w = np.zeros(taps)
    for xi, yi in zip(X, y):
        pred = w.dot(xi)
        e = yi - pred
        w = w + mu * e * xi
    last_row = np.array(prices[-taps:][::-1])
    return float(w.dot(last_row))

def predict_burg_ar(prices, order=6):
    """Burg algorithm to estimate AR coefficients then 1-step forecast.
       Order should be small for 1s timeframe (3..8)."""
    x = np.array(prices)
    n = len(x)
    if n <= 2 or order < 1:
        return float(prices[-1])
    m = min(order, n-1)
    # initialize
    ef = x.copy()
    eb = x.copy()
    a = np.zeros(m+1)
    a[0] = 1.0
    den = np.dot(ef, ef)
    # reflection coefficients
    coeffs = np.zeros(m)
    for k in range(m):
        # compute numerator and denominator
        num = -2.0 * np.dot(ef[k+1:], eb[k:n-1])
        den = np.dot(ef[k+1:], ef[k+1:]) + np.dot(eb[k:n-1], eb[k:n-1])
        if abs(den) < 1e-12:
            break
        gamma = num / den
        coeffs[k] = gamma
        # update forward/backward errors
        ef_next = ef.copy()
        eb_next = eb.copy()
        ef_next[k+1:n] = ef[k+1:n] + gamma * eb[k:n-1]
        eb_next[k+1:n] = eb[k+1:n] + gamma * ef[k+1:n]
        ef = ef_next
        eb = eb_next
    # convert reflection coefficients to AR coefs (Levinson-Durbin style)
    ar = np.array([0.0]*m)
    for i in range(m):
        k = coeffs[i]
        ar_prev = ar[:i].copy()
        ar[:i] = ar_prev - k * ar_prev[::-1]
        ar[i] = k
    # forecast using last m samples: x_hat = -sum(ar * x_{t - i - 1})
    last_vals = x[-m:][::-1] if m>0 else np.array([x[-1]])
    forecast = -np.dot(ar, last_vals) + x[-1]  # add last to reduce bias
    return float(forecast)

def predict_median_trend(prices, window=10, horizon=1):
    """Take median of local slopes inside window and extrapolate."""
    n = len(prices)
    if n < 3:
        return prices[-1]
    w = min(window, n-1)
    slopes = []
    arr = np.array(prices[-(w+1):])
    for i in range(len(arr)-1):
        slopes.append(arr[i+1] - arr[i])
    med_slope = float(np.median(slopes))
    return float(prices[-1] + med_slope * horizon)

def predict_envelope(prices, window=12):
    """Use local min/max envelope bias to create a tiny projected movement.
       Good when price is trapped in micro-range and you want envelope push."""
    n = len(prices)
    if n < 6:
        return prices[-1]
    w = min(window, n)
    seg = np.array(prices[-w:])
    local_max = np.max(seg)
    local_min = np.min(seg)
    center = (local_max + local_min) / 2.0
    bias = (seg[-1] - center) / (local_max - local_min + 1e-9)
    # bias in [-1,1] scaled to a small step
    step = (local_max - local_min) * 0.15
    return float(seg[-1] + bias * step)

# ========================
# TensorFlow Models
# ========================
def predict_tf_lstm(prices, window=20):
    if len(prices) < window:
        return prices[-1]

    seq = np.array(prices[-window:], dtype=np.float32).reshape(1, window, 1)

    model = models.Sequential([
        layers.LSTM(16, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # quick training (only 1 epoch for speed)
    X = np.array([prices[i-window:i] for i in range(window, len(prices))]).reshape(-1, window, 1)
    y = np.array(prices[window:])
    if len(X) < 2:
        return prices[-1]

    model.fit(X, y, batch_size=8, epochs=1, verbose=0)

    pred = model.predict(seq, verbose=0)[0][0]
    return float(pred)

def predict_tf_cnn(prices, window=32):
    if len(prices) < window:
        return prices[-1]

    seq = np.array(prices[-window:], dtype=np.float32).reshape(1, window, 1)

    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu'),
        layers.Conv1D(16, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # Prepare training data
    X = np.array([prices[i-window:i] for i in range(window, len(prices))]).reshape(-1, window, 1)
    y = np.array(prices[window:])

    if len(X) < 5:
        return prices[-1]

    model.fit(X, y, batch_size=8, epochs=1, verbose=0)

    pred = model.predict(seq, verbose=0)[0][0]
    return float(pred)


def predict_tf_cnn_lstm(prices, window=40):
    if len(prices) < window:
        return prices[-1]

    seq = np.array(prices[-window:], dtype=np.float32).reshape(1, window, 1)

    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(32),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    X = np.array([prices[i-window:i] for i in range(window, len(prices))]).reshape(-1, window, 1)
    y = np.array(prices[window:])

    if len(X) < 10:
        return prices[-1]

    model.fit(X, y, epochs=1, batch_size=8, verbose=0)

    return float(model.predict(seq, verbose=0)[0][0])

def predict_tf_transformer(prices, window=32):
    if len(prices) < window:
        return prices[-1]

    seq = np.array(prices[-window:], dtype=np.float32).reshape(1, window, 1)

    inp = layers.Input(shape=(window, 1))
    x = layers.LayerNormalization()(inp)
    att = layers.MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
    x = layers.Add()([x, att])
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    X = np.array([prices[i-window:i] for i in range(window, len(prices))]).reshape(-1, window, 1)
    y = np.array(prices[window:])

    if len(X) < 20:
        return prices[-1]

    model.fit(X, y, epochs=1, batch_size=8, verbose=0)

    return float(model.predict(seq, verbose=0)[0][0])

def predict_tf_rescnn(prices, window=48):
    if len(prices) < window:
        return prices[-1]

    seq = np.array(prices[-window:], dtype=np.float32).reshape(1, window, 1)

    inp = layers.Input(shape=(window,1))

    x = layers.Conv1D(32,3,padding="same",activation='relu')(inp)
    y = layers.Conv1D(32,3,padding="same",activation='relu')(x)
    x = layers.Add()([x,y])  # residual block

    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    X = np.array([prices[i-window:i] for i in range(window, len(prices))]).reshape(-1,window,1)
    y = np.array(prices[window:])

    if len(X) < 15:
        return prices[-1]

    model.fit(X, y, epochs=1, batch_size=8, verbose=0)
    return float(model.predict(seq, verbose=0)[0][0])

def predict_tf_vol_lstm(prices, window=30):
    if len(prices) < window:
        return prices[-1]

    returns = np.diff(prices)
    vol = np.std(returns[-window:]) if len(returns)>window else 0
    seq = np.array(prices[-window:], dtype=np.float32).reshape(1, window, 1)

    model = models.Sequential([
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    X = np.array([prices[i-window:i] for i in range(window, len(prices))]).reshape(-1,window,1)
    y = np.array(prices[window:])

    if len(X) < 15:
        return prices[-1]

    model.fit(X, y, batch_size=8, epochs=1, verbose=0)

    pred = model.predict(seq, verbose=0)[0][0]
    return float(pred + vol * 0.05)  # volatility bias



# =========================
# HFT INDICATORS
# =========================
def microprice_indicator():
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    w = bids[best_bid]+asks[best_ask]
    return (best_bid*asks[best_ask]+best_ask*bids[best_bid])/w

def spread_indicator():
    return min(asks.keys()) - max(bids.keys())

def order_flow_imbalance():
    global last_best_bid,last_best_ask
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    ofi = 0
    if last_best_bid is not None: ofi += best_bid-last_best_bid
    if last_best_ask is not None: ofi += last_best_ask-best_ask
    last_best_bid,last_best_ask = best_bid,best_ask
    ofi_window.append(ofi)
    return ofi

def pressure_indicator(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth,len(top_bids),len(top_asks))
    if available_levels==0: return 0,0,None
    bid_pressure=sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_pressure=sum([asks[top_asks[i]] for i in range(available_levels)])
    ratio=bid_pressure/ask_pressure if ask_pressure>0 else None
    return bid_pressure,ask_pressure,ratio

def orderbook_slope(depth=10):
    prices = sorted(list(bids.keys())+list(asks.keys()))
    quantities=[bids.get(p,asks.get(p,0)) for p in prices]
    if len(prices)<3: return 0
    return np.polyfit(prices,quantities,1)[0]

def inventory_imbalance(depth=5):
    top_bids=sorted(bids.keys(),reverse=True)
    top_asks=sorted(asks.keys())
    available_levels=min(depth,len(top_bids),len(top_asks))
    if available_levels==0: return 0
    B=sum([bids[top_bids[i]] for i in range(available_levels)])
    A=sum([asks[top_asks[i]] for i in range(available_levels)])
    return (B-A)/(B+A+1e-9)

def vpin_indicator(price):
    vpin_window.append(price)
    if len(vpin_window)<vpin_window.maxlen: return None
    returns=np.diff(vpin_window)
    buy_volume=np.sum(returns>0)
    sell_volume=np.sum(returns<0)
    return abs(buy_volume-sell_volume)/(buy_volume+sell_volume+1e-9)

def short_term_volatility(price):
    vol_window.append(price)
    if len(vol_window)<vol_window.maxlen: return None
    return np.std(np.diff(vol_window))

def liquidity_shock():
    spread = spread_indicator()
    return spread > 1.5*np.mean([abs(x) for x in vol_window]) if len(vol_window)>10 else None

# =========================
# FPGA-STYLE STREAMING FEATURES
# =========================
def weighted_imbalance(levels=5):
    top_bids=sorted(bids.keys(),reverse=True)
    top_asks=sorted(asks.keys())
    available_levels=min(levels,len(top_bids),len(top_asks))
    if available_levels==0: return 0
    imbalance=0
    weight_sum=0
    for i in range(available_levels):
        w=1/(i+1)
        b_qty=bids.get(top_bids[i],0)
        a_qty=asks.get(top_asks[i],0)
        imbalance+=w*(b_qty-a_qty)
        weight_sum+=w*(b_qty+a_qty)
    return imbalance/weight_sum if weight_sum!=0 else 0

def rolling_ofi_sum():
    return sum(ofi_window)

def micro_momentum(price):
    micro_window.append(price)
    if len(micro_window)<2: return 0
    return micro_window[-1]-micro_window[0]

def cancellation_ratio(msg):
    cancels=sum(1 for p,q in msg.get("b",[]) if q==0)+sum(1 for p,q in msg.get("a",[]) if q==0)
    cancel_window.append(cancels)
    return np.mean(cancel_window)

def price_skew(depth=5):
    top_bids=sorted(bids.keys(),reverse=True)
    top_asks=sorted(asks.keys())
    available_levels=min(depth,len(top_bids),len(top_asks))
    if available_levels==0: return 0
    bid_vol=sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_vol=sum([asks[top_asks[i]] for i in range(available_levels)])
    return (bid_vol-ask_vol)/(bid_vol+ask_vol+1e-9)

# =========================
# RIVER ONLINE MODELS
# =========================
price_scaler = preprocessing.StandardScaler()
online_lr = linear_model.LinearRegression()
online_log = linear_model.LogisticRegression()

def update_river_models(midprice, features_dict):
    x = {**features_dict, "midprice": midprice}
    price_scaler.learn_one(x)
    x_scaled = price_scaler.transform_one(x)
    y_pred = online_lr.predict_one(x_scaled) or midprice
    online_lr.learn_one(x_scaled, midprice)
    trend = 1 if midprice > y_pred else -1 if midprice < y_pred else 0
    y_class = {1:"BULLISH 📈", -1:"BEARISH 📉", 0:"NEUTRAL ➖"}
    online_log.learn_one(x_scaled, trend)
    return y_pred, y_class[trend]

# =========================
# HMA + T3 CROSSING STRATEGY
# =========================
hma_values = deque(maxlen=100)
t3_values  = deque(maxlen=100)

def average_cross_signal(hma_vals, t3_vals):
    if len(hma_vals) < 2 or len(t3_vals) < 2:
        return "NEUTRAL ➖"
    hma_prev, hma_curr = hma_vals[-2], hma_vals[-1]
    t3_prev, t3_curr = t3_vals[-2], t3_vals[-1]
    if hma_prev < t3_prev and hma_curr > t3_curr:
        return "BUY 🟢"
    elif hma_prev > t3_prev and hma_curr < t3_curr:
        return "SELL 🔴"
    return "NEUTRAL ➖"

# =========================
# FIBONACCI TREND MODULE
# =========================
def fibonacci_levels(prices, lookback=30):
    if len(prices) < lookback:
        return None
    recent_prices = prices[-lookback:]
    high = max(recent_prices)
    low = min(recent_prices)
    diff = high - low
    levels = {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0": low
    }
    return levels

def fibonacci_signal(midprice, levels):
    if levels is None: return "NEUTRAL ➖"
    support_levels = [levels["0.618"], levels["0.786"]]
    resistance_levels = [levels["0.236"], levels["0.382"]]
    if any(abs(midprice - s)/s < 0.002 for s in support_levels):
        return "BUY 🟢"
    elif any(abs(midprice - r)/r < 0.002 for r in resistance_levels):
        return "SELL 🔴"
    elif midprice > levels["0.236"]:
        return "BULLISH 📈"
    elif midprice < levels["0.786"]:
        return "BEARISH 📉"
    else:
        return "NEUTRAL ➖"
# =========================
# SIGNAL INVERTER
# =========================
def invert_signal(signal):
    """
    Invert BUY/SELL/BULLISH/BEARISH signals.
    """
    if signal in ["BUY 🟢", "BULLISH 📈"]:
        return "SELL 🔴"
    elif signal in ["SELL 🔴", "BEARISH 📉"]:
        return "BUY 🟢"
    return "NEUTRAL ➖"

# =========================
# (Place the rest of your code above unchanged: imports, models, indicators, etc.)
# =========================

# =========================
# AUTO-CLOSE PARAMETERS (improved)
# =========================
# You can set thresholds as absolute PnL values (same units as price)
stop_loss_threshold_abs = -0.1    # example: -0.1 USD (or -0.1 in pricing units)
take_profit_threshold_abs = 0.2   # example: +0.2 USD

# Or set thresholds as percent of entry (more common). Use None to disable.
stop_loss_threshold_pct = -0.001   # -0.1% default example (negative means loss)
take_profit_threshold_pct = 0.002  # +0.2% default example

# When True, if a position hits close condition we will close immediately and
# skip opening a new position for that strategy during the same tick (aggressive)
aggressive_exit = True

# =========================
# Helper: unified check-and-close function (fast/instant)
# =========================
def compute_unrealized(pos, midprice):
    """Return unrealized PnL (absolute) and percent change relative to entry.
       For LONG: pnl = mid - entry, pct = (mid-entry)/entry
       For SHORT: pnl = entry - mid, pct = (entry-mid)/entry
    """
    if pos["type"] == "LONG":
        pnl = midprice - pos["entry"]
        pct = pnl / (pos["entry"] + 1e-12)
    else:  # SHORT
        pnl = pos["entry"] - midprice
        pct = pnl / (pos["entry"] + 1e-12)
    return float(pnl), float(pct)

def should_close(pos, midprice):
    """Decide whether to close based on absolute or percent thresholds."""
    pnl_abs, pct = compute_unrealized(pos, midprice)

    # check absolute thresholds first (if set)
    if stop_loss_threshold_abs is not None and pnl_abs <= stop_loss_threshold_abs:
        return True, pnl_abs
    if take_profit_threshold_abs is not None and pnl_abs >= take_profit_threshold_abs:
        return True, pnl_abs

    # check percent thresholds next (if set)
    if stop_loss_threshold_pct is not None and pct <= stop_loss_threshold_pct:
        # convert pct back to price units for logging consistency
        pnl_price_equiv = pct * pos["entry"]
        return True, pnl_price_equiv
    if take_profit_threshold_pct is not None and pct >= take_profit_threshold_pct:
        pnl_price_equiv = pct * pos["entry"]
        return True, pnl_price_equiv

    return False, 0.0

def close_position_immediate(strategy_name, pos, midprice):
    """Close the position immediately, update balance and logs."""
    global balance
    pnl_abs, _ = compute_unrealized(pos, midprice)
    balance += pnl_abs
    side = "CLOSE LONG" if pos["type"] == "LONG" else "CLOSE SHORT"
    trade_log.append({"strategy": strategy_name, "side": side, "price": midprice, "pnl": pnl_abs, "balance": balance})
    positions.pop(strategy_name, None)
    return pnl_abs

# =========================
# LIVE STREAM LOOP WITH FASTER AUTO-CLOSE
# =========================
async def depth_stream():
    global balance, positions, trade_log
    print("🔵 Inverted High-Frequency Multi-Strategy Engine with Immediate Auto-Close 📊\n")

    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)

            # --- update orderbook (unchanged)
            for price, qty in msg["b"]:
                p, q = float(price), float(qty)
                if q == 0: bids.pop(p, None)
                else: bids[p] = q
            for price, qty in msg["a"]:
                p, q = float(price), float(qty)
                if q == 0: asks.pop(p, None)
                else: asks[p] = q
            if not bids or not asks: continue

            best_bid, best_ask = max(bids.keys()), min(asks.keys())
            midprice = (best_bid + best_ask) / 2

            # --- update price history (unchanged)
            price_history.append(midprice)
            if len(price_history) > window_size:
                price_history[:] = price_history[-window_size:]

            # =========================
            # IMMEDIATE AUTO-CLOSE CHECK (runs BEFORE opening new positions)
            # This ensures positions that already reached thresholds close as soon as possible
            # =========================
            #strategies_closed_this_tick = set()
            #for strat, pos in list(positions.items()):
            #    close_flag, pnl_val = should_close(pos, midprice)
            #    if close_flag:
            #        pnl_closed = close_position_immediate(strat, pos, midprice)
            #        strategies_closed_this_tick.add(strat)
            #        # print immediate closure
            #        trade_log.append({"strategy": strat, "side": "AUTO-CLOSE", "price": midprice, "pnl": pnl_closed, "balance": balance})

            # =========================
            # trend model predictions (unchanged)
            # =========================
            preds = [
               predict_lr(price_history),
                predict_hma(price_history),
                predict_kalman(price_history),
                predict_cwma(price_history),
                predict_dma(price_history),
                predict_ema(price_history),
                predict_tema(price_history),
                predict_wma(price_history),
                predict_smma(price_history),
                predict_momentum(price_history),
                predict_hzlog(price_history),
                predict_vydia(price_history),
                predict_parma(price_history),
                predict_junx(price_history),
                predict_t3(price_history),
                predict_ichimoku(price_history),
                predict_tf_lstm(price_history),
                predict_tf_cnn(price_history),
                predict_tf_cnn_lstm(price_history),
                predict_tf_transformer(price_history),
                predict_tf_rescnn(price_history),
                predict_tf_vol_lstm(price_history)
            ]

            preds2 = [
                 #predict_ar(price_history, lags=8),
                 #predict_fft(price_history, keep=6, horizon=1),
                 #predict_macd_midprice(price_history),
                 #predict_median_filter(price_history, period=7),
                 #predict_entropy_weighted(price_history, period=20),
                 #predict_rls_trend(price_history),
                 #predict_savgol(price_history),
                 #predict_local_poly(price_history),
                 #predict_holt(price_history),
                 #predict_lms_online(price_history),
                 #predict_burg_ar(price_history, order=6),
                 #predict_median_trend(price_history),
                 #predict_envelope(price_history)
            ]

            # trend summary uses first group (preds)
            trend_final = merge_signals(preds, midprice)

            # FIRST SET → INVERTED SIGNALS (M1..)
            signals_dict = {
                f"M{i}": invert_signal(trend_signal(p, midprice))
                for i, p in enumerate(preds, 1)
            }

            # SECOND SET → NATURAL SIGNALS (N1..)
            signals_dict_normal = {
                f"N{i}": trend_signal(p, midprice)
                for i, p in enumerate(preds2, 1)
            }

            # --- HFT & FPGA features (unchanged)
            ofi = order_flow_imbalance()
            bid_p, ask_p, ratio = pressure_indicator()
            hft_features = {
                "microprice": microprice_indicator(),
                "ofi": ofi,
                "spread": spread_indicator(),
                "bid_pressure": bid_p,
                "ask_pressure": ask_p,
                "pressure_ratio": ratio,
                "orderbook_slope": orderbook_slope(),
                "imbalance": inventory_imbalance(),
                "vpin": vpin_indicator(midprice),
                "volatility": short_term_volatility(midprice),
                "liquidity_shock": liquidity_shock()
            }

            w_imb = weighted_imbalance()
            r_ofi = rolling_ofi_sum()
            micro_mom = micro_momentum(midprice)
            cancel_r = cancellation_ratio(msg)
            p_skew = price_skew()
            fpga_features = {
                "weighted_imbalance": (w_imb, trend_signal(w_imb, 0)),
                "rolling_ofi": (r_ofi, trend_signal(r_ofi, 0)),
                "micro_momentum": (micro_mom, trend_signal(micro_mom, 0)),
                "cancel_ratio": (cancel_r, trend_signal(cancel_r, 0)),
                "price_skew": (p_skew, trend_signal(p_skew, 0))
            }

            # --- River online prediction (unchanged)
            next_pred, next_trend = update_river_models(midprice, {k: v[0] for k, v in fpga_features.items()})

            # --- Cache snapshot (unchanged)
            snapshot_cache.append({
                "midprice": midprice,
                **hft_features,
                **{k: v[0] for k, v in fpga_features.items()}
            })

            # --- HMA + T3 crossing (unchanged)
            hma_val = predict_hma(price_history)
            t3_val = predict_t3(price_history)
            hma_values.append(hma_val)
            t3_values.append(t3_val)
            cross_signal = invert_signal(average_cross_signal(hma_values, t3_values))

            # --- Fibonacci trend (unchanged)
            fib_levels = fibonacci_levels(price_history, lookback=30)
            fib_signal = invert_signal(fibonacci_signal(midprice, fib_levels))

            # =========================
            # MULTI-STRATEGY TRADING LOGIC (OPEN/CLOSE) with immediate-close protection
            # =========================

            # A) Trend Models M1.. (INVERTED signals_dict)
            for name, signal in signals_dict.items():
                # if this strategy was auto-closed this tick and aggressive_exit is on, skip reopening it
                if aggressive_exit and name in strategies_closed_this_tick:
                    continue

                if signal == "SELL 🔴":
                    # open LONG becomes SHORT (inverted behavior)
                    curr_type = positions.get(name, {}).get("type")
                    # If there is an existing position check immediate-close again (safety)
                    if curr_type is not None:
                        close_flag, _ = should_close(positions[name], midprice)
                        if close_flag:
                            close_position_immediate(name, positions[name], midprice)
                            strategies_closed_this_tick.add(name)
                            if aggressive_exit:
                                continue  # skip reopening this tick

                    if positions.get(name, {}).get("type") != "SHORT":
                        if positions.get(name, {}).get("type") == "LONG":
                            pnl = midprice - positions[name]["entry"]
                            balance += pnl
                            trade_log.append({"strategy": name, "side": "CLOSE LONG", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[name] = {"type": "SHORT", "entry": midprice}
                        trade_log.append({"strategy": name, "side": "OPEN SHORT", "price": midprice, "balance": balance})

                elif signal == "BUY 🟢":
                    # open SHORT becomes LONG
                    curr_type = positions.get(name, {}).get("type")
                    if curr_type is not None:
                        close_flag, _ = should_close(positions[name], midprice)
                        if close_flag:
                            close_position_immediate(name, positions[name], midprice)
                            strategies_closed_this_tick.add(name)
                            if aggressive_exit:
                                continue

                    if positions.get(name, {}).get("type") != "LONG":
                        if positions.get(name, {}).get("type") == "SHORT":
                            pnl = positions[name]["entry"] - midprice
                            balance += pnl
                            trade_log.append({"strategy": name, "side": "CLOSE SHORT", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[name] = {"type": "LONG", "entry": midprice}
                        trade_log.append({"strategy": name, "side": "OPEN LONG", "price": midprice, "balance": balance})

            # B) Trend Models N1.. (NON-INVERTED signals_dict_normal) — same protection
            for name, signal in signals_dict_normal.items():
                if aggressive_exit and name in strategies_closed_this_tick:
                    continue

                if signal == "BUY 🟢":
                    curr_type = positions.get(name, {}).get("type")
                    if curr_type is not None:
                        close_flag, _ = should_close(positions[name], midprice)
                        if close_flag:
                            close_position_immediate(name, positions[name], midprice)
                            strategies_closed_this_tick.add(name)
                            if aggressive_exit:
                                continue

                    if positions.get(name, {}).get("type") != "LONG":
                        if positions.get(name, {}).get("type") == "SHORT":
                            pnl = positions[name]["entry"] - midprice
                            balance += pnl
                            trade_log.append({"strategy": name, "side": "CLOSE SHORT", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[name] = {"type": "LONG", "entry": midprice}
                        trade_log.append({"strategy": name, "side": "OPEN LONG", "price": midprice, "balance": balance})

                elif signal == "SELL 🔴":
                    curr_type = positions.get(name, {}).get("type")
                    if curr_type is not None:
                        close_flag, _ = should_close(positions[name], midprice)
                        if close_flag:
                            close_position_immediate(name, positions[name], midprice)
                            strategies_closed_this_tick.add(name)
                            if aggressive_exit:
                                continue

                    if positions.get(name, {}).get("type") != "SHORT":
                        if positions.get(name, {}).get("type") == "LONG":
                            pnl = midprice - positions[name]["entry"]
                            balance += pnl
                            trade_log.append({"strategy": name, "side": "CLOSE LONG", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[name] = {"type": "SHORT", "entry": midprice}
                        trade_log.append({"strategy": name, "side": "OPEN SHORT", "price": midprice, "balance": balance})

            # HMA+T3 crossing (keeps inverted behaviour) — same immediate-close protection
            hma_name = "HMA+T3"
            if aggressive_exit and hma_name in strategies_closed_this_tick:
                pass
            else:
                if cross_signal == "BUY 🟢":
                    if positions.get(hma_name, {}).get("type") != "SHORT":
                        if positions.get(hma_name, {}).get("type") == "LONG":
                            pnl = midprice - positions[hma_name]["entry"]
                            balance += pnl
                            trade_log.append({"strategy": hma_name, "side": "CLOSE LONG", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[hma_name] = {"type": "SHORT", "entry": midprice}
                        trade_log.append({"strategy": hma_name, "side": "OPEN SHORT", "price": midprice, "balance": balance})
                elif cross_signal == "SELL 🔴":
                    if positions.get(hma_name, {}).get("type") != "LONG":
                        if positions.get(hma_name, {}).get("type") == "SHORT":
                            pnl = positions[hma_name]["entry"] - midprice
                            balance += pnl
                            trade_log.append({"strategy": hma_name, "side": "CLOSE SHORT", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[hma_name] = {"type": "LONG", "entry": midprice}
                        trade_log.append({"strategy": hma_name, "side": "OPEN LONG", "price": midprice, "balance": balance})

            # Fibonacci strategy (keeps inverted behaviour and protection)
            fib_name = "FIB"
            if aggressive_exit and fib_name in strategies_closed_this_tick:
                pass
            else:
                if fib_signal in ["BUY 🟢", "BULLISH 📈"]:
                    if positions.get(fib_name, {}).get("type") != "SHORT":
                        if positions.get(fib_name, {}).get("type") == "LONG":
                            pnl = midprice - positions[fib_name]["entry"]
                            balance += pnl
                            trade_log.append({"strategy": fib_name, "side": "CLOSE LONG", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[fib_name] = {"type": "SHORT", "entry": midprice}
                        trade_log.append({"strategy": fib_name, "side": "OPEN SHORT", "price": midprice, "balance": balance})
                elif fib_signal in ["SELL 🔴", "BEARISH 📉"]:
                    if positions.get(fib_name, {}).get("type") != "LONG":
                        if positions.get(fib_name, {}).get("type") == "SHORT":
                            pnl = positions[fib_name]["entry"] - midprice
                            balance += pnl
                            trade_log.append({"strategy": fib_name, "side": "CLOSE SHORT", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[fib_name] = {"type": "LONG", "entry": midprice}
                        trade_log.append({"strategy": fib_name, "side": "OPEN LONG", "price": midprice, "balance": balance})

            # NOTE: Because we run should_close at the top of each tick and also
            # re-check before opening a new position, auto-close will be as instantaneous
            # as the incoming websocket tick allows.

            # =========================
            # PRINT OUTPUT (concise) — same style as before
            # =========================
            now = datetime.utcnow()
            print("\n⏱", now, "UTC")
            print("⭐ Trend Models Signals (inverted):", trend_final)
            for k, v in signals_dict.items(): print(f"   {k:7}: {v}")
            print("------------------------------------------------------------")
            print("⭐ Trend Models Signals (Normal):", trend_final)
            for k, v in signals_dict_normal.items(): print(f"   {k:7}: {v}")
            print("------------------------------------------------------------")
            print("⭐ HMA + T3 Crossing Signal:", cross_signal)
            print("⭐ Fibonacci Signal:", fib_signal)
            print("--------------------------------")
            print(f"⭐ Balance: {balance:.2f}")
            print(f"\nBTC (bids, top {ROWS} rows): {total_bid_amount:.6f} BTC")
            print(f"BTC (asks, top {ROWS} rows): {total_ask_amount:.6f} BTC")
            print("--------------------------------")
            print("⭐ Current Positions:")
            for strat, pos in positions.items():
                unrealized, pct = compute_unrealized(pos, midprice)
                print(f"   {strat:10}: {pos['type']} @ {pos['entry']:.2f} | Unrealized PnL: {unrealized:.6f} | {pct*100:.3f}%")
            print("⭐ Last Trades:")
            for t in trade_log[-8:]:
                print(f"   {t['strategy']:10} {t['side']:15} @ {t['price']:.2f} | PnL: {t.get('pnl',0):.6f} | Balance: {t['balance']:.2f}")
            print("------------------------------------------------------------")

# =========================
# RUN
# =========================
await depth_stream()
