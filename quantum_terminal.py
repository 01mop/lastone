"""
╔══════════════════════════════════════════════════════════════════════╗
║   QUANTUM TERMINAL  v3.0  —  All-in-One Professional Quant Dashboard ║
║   Market Data · Statistical Analysis · Backtesting                   ║
║   Risk Management · Factor Models · ML/AI · Correlation              ║
╚══════════════════════════════════════════════════════════════════════╝
Run:  streamlit run quantum_terminal.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import requests
import warnings
warnings.filterwarnings("ignore")

# ── optional imports ───────────────────────────────────────────────────
try:
    import ccxt; CCXT_OK = True
except ImportError:
    CCXT_OK = False

try:
    from fredapi import Fred; FRED_OK = True
except ImportError:
    FRED_OK = False

try:
    from arch import arch_model; ARCH_OK = True
except ImportError:
    ARCH_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer(); VADER_OK = True
except ImportError:
    VADER_OK = False

try:
    import xgboost as xgb; XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.tsa.arima.model import ARIMA
    import statsmodels.api as sm; SM_OK = True
except ImportError:
    SM_OK = False

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QUANTUM TERMINAL v3",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&family=Orbitron:wght@700;900&display=swap');

html, body, [class*="css"] { background-color:#070b10!important; color:#c8d8e8!important; font-family:'IBM Plex Sans',sans-serif; }
.stApp { background-color:#070b10; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0a1520,#080f18)!important; border-right:1px solid #1a2f45; }
section[data-testid="stSidebar"] * { color:#7aafc8!important; }
section[data-testid="stSidebar"] label { color:#4da8da!important; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:1.5px; text-transform:uppercase; }
.stTextInput>div>div input, .stSelectbox>div>div, .stMultiSelect>div>div { background-color:#0d1e2e!important; border:1px solid #1e3a55!important; border-radius:2px!important; color:#4da8da!important; font-family:'Space Mono',monospace!important; }
.stTabs [data-baseweb="tab-list"] { background:#0a1520; border-bottom:1px solid #1a3a55; gap:0; }
.stTabs [data-baseweb="tab"] { color:#4a7a99; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:1px; padding:10px 16px; border-bottom:2px solid transparent; }
.stTabs [aria-selected="true"] { color:#00d4ff!important; border-bottom:2px solid #00d4ff!important; background:transparent!important; }
[data-testid="metric-container"] { background:linear-gradient(135deg,#0d1f30,#0a1520); border:1px solid #1a3a55; border-left:3px solid #00d4ff; border-radius:2px; padding:10px 14px; }
[data-testid="metric-container"] label { color:#4da8da!important; font-family:'Space Mono',monospace; font-size:9px; letter-spacing:1.5px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e8f4fd!important; font-family:'Space Mono',monospace; font-size:16px; }
.blabel { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; color:#2a6a8a; text-transform:uppercase; border-bottom:1px solid #1a3a55; padding-bottom:4px; margin-bottom:10px; }
.tag-free  { background:#0d2a1a; border:1px solid #00ff88; color:#00ff88; font-family:'Space Mono',monospace; font-size:9px; padding:2px 7px; border-radius:2px; display:inline-block; margin:2px; }
.tag-key   { background:#2a1a0d; border:1px solid #ffaa00; color:#ffaa00; font-family:'Space Mono',monospace; font-size:9px; padding:2px 7px; border-radius:2px; display:inline-block; margin:2px; }
.tag-limit { background:#2a0d0d; border:1px solid #ff4466; color:#ff4466; font-family:'Space Mono',monospace; font-size:9px; padding:2px 7px; border-radius:2px; display:inline-block; margin:2px; }
.news-card { background:#0d1f30; border:1px solid #1a3a55; border-left:3px solid #00d4ff; border-radius:2px; padding:10px 14px; margin-bottom:8px; }
.news-title { font-family:'IBM Plex Sans',sans-serif; font-size:13px; color:#e8f4fd; font-weight:600; }
.news-meta  { font-family:'Space Mono',monospace; font-size:9px; color:#4a7a99; margin-top:4px; }
.status-dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:#00ff88; box-shadow:0 0 6px #00ff88; margin-right:5px; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.3} }
hr { border-color:#1a3a55!important; }
::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-thumb { background:#1e3a55; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════
COLORS = ["#00d4ff","#00ff88","#ffaa00","#ff4466","#aa88ff","#ff88aa","#44ffdd","#ffdd44"]

PLOTLY_BASE = dict(
    paper_bgcolor="#070b10", plot_bgcolor="#070b10",
    font=dict(family="Space Mono, monospace", color="#8aafc8", size=10),
    xaxis=dict(gridcolor="#0d1e2e", zerolinecolor="#0d1e2e"),
    yaxis=dict(gridcolor="#0d1e2e", zerolinecolor="#0d1e2e"),
    margin=dict(l=50,r=20,t=36,b=36),
    legend=dict(bgcolor="#0a1520", bordercolor="#1a3a55", borderwidth=1),
)

def blabel(txt):
    st.markdown(f"<div class='blabel'>▸ {txt}</div>", unsafe_allow_html=True)

def fmt_large(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    if abs(v)>=1e12: return f"${v/1e12:.2f}T"
    if abs(v)>=1e9:  return f"${v/1e9:.2f}B"
    if abs(v)>=1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def pct_fmt(v): return f"{v*100:.2f}%" if isinstance(v,(int,float)) and not np.isnan(v) else "N/A"

# ── cached data fetchers ───────────────────────────────────────────────
import time

def _retry(fn, retries=3, delay=2):
    """Call fn() up to `retries` times with exponential backoff."""
    for attempt in range(retries):
        try:
            result = fn()
            if result is not None:
                return result
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    return None

@st.cache_data(ttl=1800)
def get_ohlcv(ticker, period, interval):
    def _fetch():
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna() if not df.empty else None
    result = _retry(_fetch, retries=3, delay=2)
    return result if result is not None else pd.DataFrame()

@st.cache_data(ttl=3600)
def get_info(ticker):
    def _fetch():
        t = yf.Ticker(ticker)
        info = t.info
        # info is valid only if it has more than just a quoteType key
        if info and len(info) > 5:
            return info
        return None
    result = _retry(_fetch, retries=3, delay=2)
    return result if result is not None else {}

@st.cache_data(ttl=3600)
def get_option_expirations(ticker):
    def _fetch():
        exps = yf.Ticker(ticker).options
        return list(exps) if exps else None
    result = _retry(_fetch, retries=3, delay=2)
    return result if result is not None else []

def get_options(ticker):
    exps = get_option_expirations(ticker)
    if not exps: return None, None, []
    def _fetch():
        t = yf.Ticker(ticker)
        # verify ticker is valid
        _ = t.options
        return t
    t_obj = _retry(_fetch, retries=3, delay=2)
    if t_obj is None: return None, None, []
    return t_obj, exps[0], exps

@st.cache_data(ttl=120)
def get_ccxt_ob(exchange_id, symbol, limit=20):
    if not CCXT_OK: return None, "pip install ccxt"
    try:
        ex = getattr(ccxt, exchange_id)()
        ex.load_markets()
        ob = ex.fetch_order_book(symbol, limit=limit)
        return ob, None
    except Exception as e: return None, str(e)

@st.cache_data(ttl=120)
def get_ccxt_ohlcv(exchange_id, symbol, timeframe="1h", limit=300):
    if not CCXT_OK: return pd.DataFrame(), "pip install ccxt"
    try:
        ex = getattr(ccxt, exchange_id)()
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts","Open","High","Low","Close","Volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms"); df.set_index("ts", inplace=True)
        return df, None
    except Exception as e: return pd.DataFrame(), str(e)

@st.cache_data(ttl=1800)
def get_fred(series_id, key):
    if not FRED_OK: return pd.Series(dtype=float)
    try: return Fred(api_key=key).get_series(series_id).dropna()
    except: return pd.Series(dtype=float)

@st.cache_data(ttl=60)
def get_forex_rate(base, target, key):
    try:
        r = requests.get(f"https://v6.exchangerate-api.com/v6/{key}/pair/{base}/{target}", timeout=8).json()
        return (r["conversion_rate"], None) if r.get("result")=="success" else (None, r.get("error-type"))
    except Exception as e: return None, str(e)

@st.cache_data(ttl=600)
def get_news_api(query, key, days=7):
    try:
        params = dict(q=query, from_=(datetime.now()-timedelta(days=days)).strftime("%Y-%m-%d"),
                      sortBy="publishedAt", language="en", pageSize=20, apiKey=key)
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10).json()
        return (r.get("articles",[]), None) if r.get("status")=="ok" else ([], r.get("message","Erreur"))
    except Exception as e: return [], str(e)

# ── technical indicators ───────────────────────────────────────────────
def add_indicators(df, sma1=20, sma2=50, ema=21, bb=20, rsi=14):
    c = df["Close"].squeeze(); df = df.copy()
    df[f"SMA{sma1}"] = c.rolling(sma1).mean()
    df[f"SMA{sma2}"] = c.rolling(sma2).mean()
    df[f"EMA{ema}"]  = c.ewm(span=ema, adjust=False).mean()
    df["BB_mid"] = c.rolling(bb).mean()
    bb_std = c.rolling(bb).std()
    df["BB_up"] = df["BB_mid"] + 2*bb_std
    df["BB_dn"] = df["BB_mid"] - 2*bb_std
    delta = c.diff(); gain = delta.clip(lower=0).rolling(rsi).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi).mean()
    df["RSI"] = 100 - 100/(1 + gain/loss.replace(0,np.nan))
    ema12 = c.ewm(span=12,adjust=False).mean(); ema26 = c.ewm(span=26,adjust=False).mean()
    df["MACD"] = ema12-ema26; df["MACD_sig"] = df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_hist"] = df["MACD"]-df["MACD_sig"]
    df["ATR"] = pd.concat([df["High"].squeeze()-df["Low"].squeeze(),
                            (df["High"].squeeze()-c.shift()).abs(),
                            (df["Low"].squeeze()-c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    return df

# ── risk metrics ───────────────────────────────────────────────────────
def compute_var(returns, confidence=0.95, method="historical"):
    r = returns.dropna()
    if method == "historical":
        return np.percentile(r, (1-confidence)*100)
    elif method == "parametric":
        mu, sigma = r.mean(), r.std()
        return stats.norm.ppf(1-confidence, mu, sigma)
    elif method == "cornish_fisher":
        mu, sigma = r.mean(), r.std()
        s = stats.skew(r); k = stats.kurtosis(r)
        z = stats.norm.ppf(1-confidence)
        z_cf = z + (z**2-1)*s/6 + (z**3-3*z)*k/24 - (2*z**3-5*z)*s**2/36
        return mu + z_cf*sigma

def compute_cvar(returns, confidence=0.95):
    r = returns.dropna(); var = compute_var(r, confidence)
    return r[r <= var].mean()

def sharpe(returns, rf=0.0): 
    r = returns.dropna()
    return (r.mean()-rf/252)/(r.std()+1e-9)*np.sqrt(252)

def sortino(returns, rf=0.0):
    r = returns.dropna(); downside = r[r<0].std()
    return (r.mean()-rf/252)/(downside+1e-9)*np.sqrt(252)

def max_drawdown(prices):
    p = prices.dropna(); roll_max = p.cummax()
    dd = (p - roll_max)/roll_max; return dd.min()

def calmar(returns, prices):
    ann = returns.dropna().mean()*252; mdd = abs(max_drawdown(prices))
    return ann/(mdd+1e-9)

# ── portfolio optimization ─────────────────────────────────────────────
def optimize_portfolio(returns_df, method="sharpe"):
    n = returns_df.shape[1]; mu = returns_df.mean()*252
    cov = returns_df.cov()*252; w0 = np.ones(n)/n
    bounds = [(0,1)]*n; cons = [{"type":"eq","fun":lambda w: w.sum()-1}]
    if method == "sharpe":
        def neg_sharpe(w):
            p_ret = w@mu; p_vol = np.sqrt(w@cov@w)
            return -p_ret/(p_vol+1e-9)
        res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    elif method == "min_vol":
        def port_vol(w): return np.sqrt(w@cov@w)
        res = minimize(port_vol, w0, bounds=bounds, constraints=cons)
    elif method == "equal": return w0
    return res.x if res.success else w0

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-family:Orbitron,sans-serif;font-size:15px;font-weight:900;
                  letter-spacing:4px;color:#00d4ff;text-shadow:0 0 12px rgba(0,212,255,.4);'>⬡ QUANTUM</div>
      <div style='font-family:Space Mono,monospace;font-size:8px;letter-spacing:2px;color:#2a5a7a;margin-top:2px;'>TERMINAL v3.0 · ALL-IN-ONE</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='blabel'>▸ Universe</div>", unsafe_allow_html=True)

    st.markdown("""<div style='font-family:Space Mono,monospace;font-size:9px;color:#4a7a99;line-height:1.8;'>
    Entrez vos tickers séparés par des virgules.<br>
    Actions: AAPL, MSFT, LVMH.PA, AIR.PA<br>
    ETF: SPY, QQQ, IWDA.AS<br>
    Crypto: BTC-USD, ETH-USD<br>
    Forex: EURUSD=X, GBPUSD=X<br>
    Futures: GC=F (Gold), CL=F (Oil)<br>
    Indices: ^GSPC, ^IXIC, ^FCHI
    </div>""", unsafe_allow_html=True)

    raw_input = st.text_area(
        "Tickers (séparés par virgules)",
        value="AAPL, MSFT, NVDA, TSLA",
        height=80,
        help="Exemples: AAPL, LVMH.PA, BTC-USD, ^GSPC, GC=F, EURUSD=X"
    )

    # Parse tickers from text input
    selected = [t.strip().upper() for t in raw_input.replace("\n", ",").split(",") if t.strip()]
    selected = list(dict.fromkeys(selected))[:8]  # deduplicate, max 8

    if not selected:
        selected = ["AAPL"]

    # Validate tickers and show status
    if st.button("✓ Valider les tickers", use_container_width=True):
        st.session_state["validated_tickers"] = selected

    primary  = st.selectbox("Ticker principal (graphique)", selected)

    st.markdown(f"""<div style='font-family:Space Mono,monospace;font-size:9px;color:#2a6a8a;margin-top:4px;'>
    {len(selected)} ticker(s) chargé(s) : {' · '.join(selected)}
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='blabel'>▸ Fenêtre temporelle</div>", unsafe_allow_html=True)
    PERIODS = {
        "1 jour":   ("1d",  "5m"),
        "2 jours":  ("2d",  "15m"),
        "5 jours":  ("5d",  "30m"),
        "1 sem.":   ("7d",  "1h"),
        "2 sem.":   ("14d", "1h"),
        "1 mois":   ("1mo", "1d"),
        "2 mois":   ("2mo", "1d"),
        "3 mois":   ("3mo", "1d"),
        "6 mois":   ("6mo", "1d"),
        "9 mois":   ("1y",  "1d"),
        "1 an":     ("1y",  "1d"),
        "18 mois":  ("2y",  "1wk"),
        "2 ans":    ("2y",  "1wk"),
        "3 ans":    ("3y",  "1wk"),
        "5 ans":    ("5y",  "1wk"),
        "10 ans":   ("10y", "1mo"),
        "20 ans":   ("20y", "1mo"),
        "Max":      ("max", "1mo"),
    }

    INTERVALS = {
        "1 min":"1m","2 min":"2m","5 min":"5m","15 min":"15m",
        "30 min":"30m","1 heure":"1h","1 jour":"1d","1 sem.":"1wk","1 mois":"1mo",
    }

    period_l = st.selectbox("Période", list(PERIODS.keys()), index=9)
    period, default_interval = PERIODS[period_l]

    override_interval = st.checkbox("Forcer l'intervalle", value=False)
    if override_interval:
        inv_keys = list(INTERVALS.keys())
        inv_vals = list(INTERVALS.values())
        def_idx  = inv_vals.index(default_interval) if default_interval in inv_vals else 6
        interval_l = st.selectbox("Intervalle", inv_keys, index=def_idx)
        interval   = INTERVALS[interval_l]
    else:
        interval = default_interval
        inv_map  = {v:k for k,v in INTERVALS.items()}
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:9px;color:#2a6a8a;'>"
                    f"Intervalle auto : {inv_map.get(interval, interval)}</div>",
                    unsafe_allow_html=True)

    chart_type = st.selectbox("Type de graphique", ["Candlestick","OHLC","Line"])

    st.divider()
    st.markdown("<div class='blabel'>▸ Indicateurs</div>", unsafe_allow_html=True)

    ALL_INDICATORS = [
        "SMA","EMA","WMA","DEMA","TEMA",
        "Bollinger Bands","Keltner Channel","Donchian Channel",
        "RSI","Stochastique","CCI","Williams %R","MFI",
        "MACD","PPO","DPO",
        "Volume","OBV","VWAP",
        "ATR","Volatilité historique",
        "Ichimoku","Parabolic SAR","ADX","Aroon",
        "Pivot Points","Support/Résistance",
    ]

    selected_indicators = st.multiselect(
        "Choisir les indicateurs",
        ALL_INDICATORS,
        default=["SMA","Bollinger Bands","RSI","MACD","Volume"],
        help="Sélectionnez autant d'indicateurs que vous voulez"
    )

    with st.expander("⚙ Paramètres indicateurs"):
        sma1   = st.slider("SMA Rapide",       5,  50,  20)
        sma2   = st.slider("SMA Lente",       20, 200,  50)
        ema_w  = st.slider("EMA période",      5, 100,  21)
        wma_w  = st.slider("WMA période",      5, 100,  20)
        bb_w   = st.slider("BB fenêtre",      10,  50,  20)
        bb_std = st.slider("BB écart-type",  1.0, 3.0, 2.0, step=0.5)
        kc_w   = st.slider("Keltner fenêtre", 5,  50,  20)
        kc_m   = st.slider("Keltner mult.",  1.0, 3.0, 1.5, step=0.5)
        rsi_w  = st.slider("RSI période",     5,  30,  14)
        sto_k  = st.slider("Stoch %K",        5,  30,  14)
        sto_d  = st.slider("Stoch %D",        1,  10,   3)
        cci_w  = st.slider("CCI période",    10,  50,  20)
        wpr_w  = st.slider("Williams %R",     5,  30,  14)
        mfi_w  = st.slider("MFI période",     5,  30,  14)
        macd_f = st.slider("MACD Fast",       5,  30,  12)
        macd_s = st.slider("MACD Slow",      15,  50,  26)
        macd_sig=st.slider("MACD Signal",     3,  15,   9)
        adx_w  = st.slider("ADX période",     5,  30,  14)
        aroon_w= st.slider("Aroon période",   5,  50,  25)
        dc_w   = st.slider("Donchian fenêtre",5,  50,  20)
        psar_af= st.slider("PSAR AF step",  0.01,0.1,0.02,step=0.01)
        psar_mx= st.slider("PSAR AF max",   0.1, 0.5, 0.2, step=0.05)

    # backward-compat booleans used in chart renderer
    show_sma  = "SMA"              in selected_indicators
    show_ema  = "EMA"              in selected_indicators
    show_bb   = "Bollinger Bands"  in selected_indicators
    show_vol  = "Volume"           in selected_indicators
    show_rsi  = "RSI"              in selected_indicators
    show_macd = "MACD"             in selected_indicators

    st.divider()
    st.markdown("<div class='blabel'>▸ Clés API (optionnelles)</div>", unsafe_allow_html=True)
    fred_key    = st.text_input("FRED API Key",       type="password", placeholder="fred.stlouisfed.org")
    av_key      = st.text_input("Alpha Vantage Key",  type="password", placeholder="alphavantage.co")
    news_key    = st.text_input("NewsAPI Key",        type="password", placeholder="newsapi.org")
    forex_key   = st.text_input("ExchangeRate Key",   type="password", placeholder="exchangerate-api.com")
    st.divider()
    st.markdown("<div class='blabel'>▸ Binance API (optionnel)</div>", unsafe_allow_html=True)
    binance_key    = st.text_input("Binance API Key",    type="password", placeholder="Pour données privées")
    binance_secret = st.text_input("Binance API Secret", type="password", placeholder="Pour données privées")
    st.markdown("""<div style='font-family:Space Mono,monospace;font-size:9px;color:#2a6a8a;line-height:1.8;'>
    Sans clé : prix, orderbook, OHLCV, trades<br>
    Avec clé : portfolio, positions, P&L
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace;font-size:9px;color:#1e3a55;text-align:center;'>
      <span class='status-dot'></span>LIVE · {datetime.now().strftime('%H:%M:%S')}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:4px;'>
  <span style='font-family:Orbitron,sans-serif;font-size:24px;font-weight:900;letter-spacing:5px;
               color:#00d4ff;text-shadow:0 0 18px rgba(0,212,255,.35);'>QUANTUM TERMINAL</span>
  <span style='font-family:Space Mono,monospace;font-size:9px;color:#2a5a7a;letter-spacing:2px;'>
    <span class='status-dot'></span>{datetime.now().strftime('%d %b %Y · %H:%M UTC')}
  </span>
</div>
<div style='font-family:Space Mono,monospace;font-size:9px;color:#1e3a55;letter-spacing:1px;margin-bottom:12px;'>
  Market Data · Statistical Analysis · Backtesting · Risk Management · Factor Models · ML/AI
</div>""", unsafe_allow_html=True)

if not selected:
    st.warning("Sélectionnez au moins un ticker dans la barre latérale.")
    st.stop()

# Global reload button (top right)
col_reload = st.columns([6,1])[1]
with col_reload:
    if st.button("↺ Actualiser", help="Vider le cache et recharger toutes les données"):
        st.cache_data.clear()
        st.rerun()

# ── live price strip ──────────────────────────────────────────────────
if selected:
    pcols = st.columns(len(selected))
    for i, tkr in enumerate(selected):
        try:
            info  = get_info(tkr)
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            prev  = info.get("previousClose")
            chg   = (price-prev)/prev*100 if price and prev else None
            with pcols[i]:
                st.metric(f"**{tkr}**", f"${price:,.2f}" if price else "N/A",
                          f"{chg:+.2f}%" if chg is not None else None,
                          delta_color="normal" if (chg or 0)>=0 else "inverse")
        except Exception:
            with pcols[i]:
                st.metric(f"**{tkr}**", "N/A")
st.divider()

# ══════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈  Chart",
    "🔗  Options",
    "₿   Crypto L2",
    "📊  Statistiques",
    "🎯  Backtesting",
    "⚠️  Risk",
    "🏗️  Portefeuille",
    "🌐  Macro",
    "💱  Forex",
    "📰  News",
    "🏢  Fondamentaux",
    "🔥  Corrélation",
    "🗺️  Heatmaps",
    "📆  Earnings",
    "💧  Liquidité",
    "🏥  Santé Financière",
    "🧠  Régimes & IA",
    "🟡  Binance API",
])

# ══════════════════════════════════════════════════════════════════════
#  TAB 0 — CHART & INDICATORS
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    blabel("yfinance · OHLCV · Indicateurs Techniques")
    df_raw = get_ohlcv(primary, period, interval)
    if df_raw.empty:
        st.error(f"Aucune donnée pour {primary}.")
    else:
        df = add_indicators(df_raw, sma1, sma2, ema_w, bb_w, rsi_w)
        close = df["Close"].squeeze(); open_ = df["Open"].squeeze()
        high  = df["High"].squeeze();  low   = df["Low"].squeeze()
        vol   = df["Volume"].squeeze()
        SI    = selected_indicators  # shorthand

        # ── compute extra indicators on demand ────────────────────────
        # WMA
        if "WMA" in SI:
            weights_wma = np.arange(1, wma_w+1)
            df["WMA"] = close.rolling(wma_w).apply(
                lambda x: np.dot(x, weights_wma)/weights_wma.sum(), raw=True)
        # DEMA
        if "DEMA" in SI:
            e1 = close.ewm(span=ema_w, adjust=False).mean()
            df["DEMA"] = 2*e1 - e1.ewm(span=ema_w, adjust=False).mean()
        # TEMA
        if "TEMA" in SI:
            e1 = close.ewm(span=ema_w, adjust=False).mean()
            e2 = e1.ewm(span=ema_w, adjust=False).mean()
            e3 = e2.ewm(span=ema_w, adjust=False).mean()
            df["TEMA"] = 3*e1 - 3*e2 + e3
        # Keltner Channel
        if "Keltner Channel" in SI:
            kc_mid = close.ewm(span=kc_w, adjust=False).mean()
            kc_atr = (high-low).rolling(kc_w).mean()
            df["KC_mid"] = kc_mid
            df["KC_up"]  = kc_mid + kc_m*kc_atr
            df["KC_dn"]  = kc_mid - kc_m*kc_atr
        # Donchian Channel
        if "Donchian Channel" in SI:
            df["DC_up"] = high.rolling(dc_w).max()
            df["DC_dn"] = low.rolling(dc_w).min()
            df["DC_mid"]= (df["DC_up"]+df["DC_dn"])/2
        # Stochastic
        if "Stochastique" in SI:
            lo_k = low.rolling(sto_k).min(); hi_k = high.rolling(sto_k).max()
            df["STOCH_K"] = 100*(close-lo_k)/(hi_k-lo_k+1e-9)
            df["STOCH_D"] = df["STOCH_K"].rolling(sto_d).mean()
        # CCI
        if "CCI" in SI:
            tp = (high+low+close)/3
            df["CCI"] = (tp - tp.rolling(cci_w).mean()) / (0.015*tp.rolling(cci_w).std())
        # Williams %R
        if "Williams %R" in SI:
            hi_r = high.rolling(wpr_w).max(); lo_r = low.rolling(wpr_w).min()
            df["WPR"] = -100*(hi_r-close)/(hi_r-lo_r+1e-9)
        # MFI
        if "MFI" in SI:
            tp_mfi = (high+low+close)/3
            rmf     = tp_mfi * vol
            pos_mf  = rmf.where(tp_mfi > tp_mfi.shift(1), 0).rolling(mfi_w).sum()
            neg_mf  = rmf.where(tp_mfi < tp_mfi.shift(1), 0).rolling(mfi_w).sum()
            df["MFI"] = 100 - 100/(1 + pos_mf/(neg_mf+1e-9))
        # PPO
        if "PPO" in SI:
            ema_f = close.ewm(span=macd_f, adjust=False).mean()
            ema_s = close.ewm(span=macd_s, adjust=False).mean()
            df["PPO"] = (ema_f - ema_s) / ema_s * 100
            df["PPO_sig"] = df["PPO"].ewm(span=macd_sig, adjust=False).mean()
        # DPO
        if "DPO" in SI:
            shift_n = int(sma1/2)+1
            df["DPO"] = close.shift(shift_n) - close.rolling(sma1).mean().shift(shift_n)
        # OBV
        if "OBV" in SI:
            obv = [0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]: obv.append(obv[-1]+vol.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]: obv.append(obv[-1]-vol.iloc[i])
                else: obv.append(obv[-1])
            df["OBV"] = obv
        # VWAP
        if "VWAP" in SI:
            tp_v = (high+low+close)/3
            df["VWAP"] = (tp_v*vol).cumsum() / vol.cumsum()
        # Volatilité historique
        if "Volatilité historique" in SI:
            df["HV"] = close.pct_change().rolling(21).std() * np.sqrt(252) * 100
        # ADX
        if "ADX" in SI:
            tr   = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
            plus = (high-high.shift()).clip(lower=0)
            minus= (low.shift()-low).clip(lower=0)
            tr_s = tr.rolling(adx_w).mean()
            df["ADX_plus"]  = 100*plus.rolling(adx_w).mean()/(tr_s+1e-9)
            df["ADX_minus"] = 100*minus.rolling(adx_w).mean()/(tr_s+1e-9)
            dx = (df["ADX_plus"]-df["ADX_minus"]).abs()/(df["ADX_plus"]+df["ADX_minus"]+1e-9)*100
            df["ADX"] = dx.rolling(adx_w).mean()
        # Aroon
        if "Aroon" in SI:
            df["AROON_up"] = high.rolling(aroon_w+1).apply(
                lambda x: (aroon_w - x[::-1].argmax()) / aroon_w * 100, raw=True)
            df["AROON_dn"] = low.rolling(aroon_w+1).apply(
                lambda x: (aroon_w - x[::-1].argmin()) / aroon_w * 100, raw=True)
        # Parabolic SAR (simplified)
        if "Parabolic SAR" in SI:
            psar = close.copy(); bull = True; af = psar_af; ep = float(low.iloc[0]); hp = float(high.iloc[0])
            psar_vals = []
            for i in range(len(close)):
                if i==0: psar_vals.append(float(close.iloc[0])); continue
                prev_psar = psar_vals[-1]
                if bull:
                    new_psar = prev_psar + af*(hp - prev_psar)
                    new_psar = min(new_psar, float(low.iloc[i-1]), float(low.iloc[max(0,i-2)]))
                    if float(low.iloc[i]) < new_psar:
                        bull=False; new_psar=hp; ep=float(low.iloc[i]); af=psar_af
                    else:
                        if float(high.iloc[i]) > hp: hp=float(high.iloc[i]); af=min(af+psar_af,psar_mx)
                else:
                    new_psar = prev_psar - af*(prev_psar - ep)
                    new_psar = max(new_psar, float(high.iloc[i-1]), float(high.iloc[max(0,i-2)]))
                    if float(high.iloc[i]) > new_psar:
                        bull=True; new_psar=ep; ep=float(high.iloc[i]); af=psar_af
                    else:
                        if float(low.iloc[i]) < ep: ep=float(low.iloc[i]); af=min(af+psar_af,psar_mx)
                psar_vals.append(new_psar)
            df["PSAR"] = psar_vals
        # Ichimoku
        if "Ichimoku" in SI:
            high9  = high.rolling(9).max();   low9  = low.rolling(9).min()
            high26 = high.rolling(26).max();   low26 = low.rolling(26).min()
            high52 = high.rolling(52).max();   low52 = low.rolling(52).min()
            df["ICH_conv"]  = (high9+low9)/2
            df["ICH_base"]  = (high26+low26)/2
            df["ICH_spanA"] = ((df["ICH_conv"]+df["ICH_base"])/2).shift(26)
            df["ICH_spanB"] = ((high52+low52)/2).shift(26)
            df["ICH_chikou"]= close.shift(-26)
        # Pivot Points
        if "Pivot Points" in SI:
            pp = (float(high.iloc[-1])+float(low.iloc[-1])+float(close.iloc[-1]))/3
            df["PP"]  = pp
            df["R1"]  = 2*pp - float(low.iloc[-1])
            df["S1"]  = 2*pp - float(high.iloc[-1])
            df["R2"]  = pp + (float(high.iloc[-1])-float(low.iloc[-1]))
            df["S2"]  = pp - (float(high.iloc[-1])-float(low.iloc[-1]))
        # Support/Résistance (local min/max)
        if "Support/Résistance" in SI:
            from scipy.signal import argrelextrema
            order = max(5, len(close)//50)
            sr_hi_idx = argrelextrema(close.values, np.greater, order=order)[0]
            sr_lo_idx = argrelextrema(close.values, np.less,    order=order)[0]
            df["SR_hi"] = np.nan; df["SR_lo"] = np.nan
            df.iloc[sr_hi_idx, df.columns.get_loc("SR_hi")] = close.iloc[sr_hi_idx]
            df.iloc[sr_lo_idx, df.columns.get_loc("SR_lo")] = close.iloc[sr_lo_idx]
        # MACD custom windows
        if "MACD" in SI:
            ema_f2 = close.ewm(span=macd_f, adjust=False).mean()
            ema_s2 = close.ewm(span=macd_s, adjust=False).mean()
            df["MACD"]     = ema_f2 - ema_s2
            df["MACD_sig"] = df["MACD"].ewm(span=macd_sig, adjust=False).mean()
            df["MACD_hist"]= df["MACD"] - df["MACD_sig"]
        # Bollinger with custom std
        if "Bollinger Bands" in SI:
            df["BB_mid"] = close.rolling(bb_w).mean()
            bb_s = close.rolling(bb_w).std()
            df["BB_up"]  = df["BB_mid"] + bb_std*bb_s
            df["BB_dn"]  = df["BB_mid"] - bb_std*bb_s

        # ── subplot structure ─────────────────────────────────────────
        # Oscillator panels (each gets its own row below price)
        osc_panels = []
        if "RSI"            in SI: osc_panels.append("RSI")
        if "Stochastique"   in SI: osc_panels.append("Stoch")
        if "CCI"            in SI: osc_panels.append("CCI")
        if "Williams %R"    in SI: osc_panels.append("WPR")
        if "MFI"            in SI: osc_panels.append("MFI")
        if "MACD"           in SI: osc_panels.append("MACD")
        if "PPO"            in SI: osc_panels.append("PPO")
        if "DPO"            in SI: osc_panels.append("DPO")
        if "ADX"            in SI: osc_panels.append("ADX")
        if "Aroon"          in SI: osc_panels.append("Aroon")
        if "OBV"            in SI: osc_panels.append("OBV")
        if "Volatilité historique" in SI: osc_panels.append("HV")

        n_osc  = len(osc_panels)
        has_vol= "Volume" in SI
        rows   = 1 + (1 if has_vol else 0) + n_osc
        price_h= max(0.35, 0.75 - n_osc*0.06)
        vol_h  = 0.10 if has_vol else 0
        osc_h  = (1 - price_h - vol_h) / max(n_osc, 1) if n_osc else 0
        rh     = [price_h] + ([vol_h] if has_vol else []) + [osc_h]*n_osc

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, row_heights=rh)
        cr = 1

        # ── price candles ─────────────────────────────────────────────
        if chart_type=="Candlestick":
            fig.add_trace(go.Candlestick(x=df.index,open=open_,high=high,low=low,close=close,
                increasing_line_color="#00ff88",decreasing_line_color="#ff4466",
                increasing_fillcolor="#00ff88",decreasing_fillcolor="#ff4466",
                name=primary,line_width=1), row=cr,col=1)
        elif chart_type=="OHLC":
            fig.add_trace(go.Ohlc(x=df.index,open=open_,high=high,low=low,close=close,
                increasing_line_color="#00ff88",decreasing_line_color="#ff4466",name=primary), row=cr,col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index,y=close,mode="lines",
                line=dict(color="#00d4ff",width=1.5),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.04)",name=primary), row=cr,col=1)

        # ── overlays on price panel ───────────────────────────────────
        if "SMA" in SI:
            fig.add_trace(go.Scatter(x=df.index,y=df[f"SMA{sma1}"],mode="lines",
                line=dict(color="#00ff88",width=0.9),name=f"SMA{sma1}"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df[f"SMA{sma2}"],mode="lines",
                line=dict(color="#ffaa00",width=0.9,dash="dash"),name=f"SMA{sma2}"),row=cr,col=1)
        if "EMA" in SI:
            fig.add_trace(go.Scatter(x=df.index,y=df[f"EMA{ema_w}"],mode="lines",
                line=dict(color="#aa88ff",width=0.9),name=f"EMA{ema_w}"),row=cr,col=1)
        if "WMA" in SI and "WMA" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["WMA"],mode="lines",
                line=dict(color="#ff88aa",width=0.9),name=f"WMA{wma_w}"),row=cr,col=1)
        if "DEMA" in SI and "DEMA" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["DEMA"],mode="lines",
                line=dict(color="#44ffdd",width=0.9),name="DEMA"),row=cr,col=1)
        if "TEMA" in SI and "TEMA" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["TEMA"],mode="lines",
                line=dict(color="#ffdd44",width=0.9),name="TEMA"),row=cr,col=1)
        if "Bollinger Bands" in SI:
            fig.add_trace(go.Scatter(x=df.index,y=df["BB_up"],mode="lines",
                line=dict(color="#ffaa00",width=0.7,dash="dot"),name=f"BB+{bb_std}σ"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["BB_dn"],mode="lines",
                line=dict(color="#ffaa00",width=0.7,dash="dot"),fill="tonexty",
                fillcolor="rgba(255,170,0,0.04)",name=f"BB-{bb_std}σ"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["BB_mid"],mode="lines",
                line=dict(color="#ffaa00",width=0.5),name="BB Mid"),row=cr,col=1)
        if "Keltner Channel" in SI and "KC_mid" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["KC_up"],mode="lines",
                line=dict(color="#00ffdd",width=0.7,dash="dot"),name="KC+"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["KC_dn"],mode="lines",
                line=dict(color="#00ffdd",width=0.7,dash="dot"),fill="tonexty",
                fillcolor="rgba(0,255,221,0.04)",name="KC-"),row=cr,col=1)
        if "Donchian Channel" in SI and "DC_up" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["DC_up"],mode="lines",
                line=dict(color="#ff4466",width=0.7,dash="dot"),name="DC High"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["DC_dn"],mode="lines",
                line=dict(color="#00ff88",width=0.7,dash="dot"),name="DC Low"),row=cr,col=1)
        if "VWAP" in SI and "VWAP" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["VWAP"],mode="lines",
                line=dict(color="#ff88aa",width=1.2,dash="dash"),name="VWAP"),row=cr,col=1)
        if "Ichimoku" in SI and "ICH_conv" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["ICH_conv"],mode="lines",
                line=dict(color="#00d4ff",width=1),name="Tenkan"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["ICH_base"],mode="lines",
                line=dict(color="#ffaa00",width=1),name="Kijun"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["ICH_spanA"],mode="lines",
                line=dict(color="#00ff88",width=0.5),name="Span A"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["ICH_spanB"],mode="lines",
                line=dict(color="#ff4466",width=0.5),fill="tonexty",
                fillcolor="rgba(0,255,136,0.06)",name="Span B"),row=cr,col=1)
        if "Parabolic SAR" in SI and "PSAR" in df:
            psar_s = df["PSAR"]
            fig.add_trace(go.Scatter(x=df.index,y=psar_s,mode="markers",
                marker=dict(color="#ff88aa",size=3,symbol="circle"),name="PSAR"),row=cr,col=1)
        if "Pivot Points" in SI and "PP" in df:
            for level,color,lbl in [("PP","#ffaa00","PP"),("R1","#ff4466","R1"),
                                     ("S1","#00ff88","S1"),("R2","#ff6688","R2"),("S2","#44ff88","S2")]:
                fig.add_hline(y=float(df[level].iloc[-1]),line_dash="dot",
                    line_color=color,line_width=0.8,
                    annotation_text=lbl,annotation_font_color=color,row=cr,col=1)
        if "Support/Résistance" in SI and "SR_hi" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["SR_hi"],mode="markers",
                marker=dict(color="#ff4466",size=7,symbol="triangle-down"),name="Résistance"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["SR_lo"],mode="markers",
                marker=dict(color="#00ff88",size=7,symbol="triangle-up"),name="Support"),row=cr,col=1)

        # ── volume panel ──────────────────────────────────────────────
        if has_vol:
            cr+=1
            vc=["#00ff88" if c>=o else "#ff4466" for c,o in zip(close,open_)]
            fig.add_trace(go.Bar(x=df.index,y=vol,marker_color=vc,name="Vol",opacity=0.7),row=cr,col=1)
            if "OBV" not in SI:
                fig.update_yaxes(title_text="VOL",row=cr,col=1)

        # ── oscillator panels ─────────────────────────────────────────

        if "RSI" in osc_panels:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],mode="lines",
                line=dict(color="#aa88ff",width=1.2),name="RSI"),row=r,col=1)
            fig.add_hline(y=70,line_dash="dot",line_color="#ff4466",line_width=0.7,row=r,col=1)
            fig.add_hline(y=30,line_dash="dot",line_color="#00ff88",line_width=0.7,row=r,col=1)
            fig.update_yaxes(range=[0,100],title_text="RSI",row=r,col=1)

        if "Stoch" in osc_panels and "STOCH_K" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["STOCH_K"],mode="lines",
                line=dict(color="#00d4ff",width=1),name="%K"),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["STOCH_D"],mode="lines",
                line=dict(color="#ffaa00",width=1),name="%D"),row=r,col=1)
            fig.add_hline(y=80,line_dash="dot",line_color="#ff4466",line_width=0.7,row=r,col=1)
            fig.add_hline(y=20,line_dash="dot",line_color="#00ff88",line_width=0.7,row=r,col=1)
            fig.update_yaxes(range=[0,100],title_text="Stoch",row=r,col=1)

        if "CCI" in osc_panels and "CCI" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["CCI"],mode="lines",
                line=dict(color="#ffdd44",width=1),name="CCI"),row=r,col=1)
            fig.add_hline(y=100,line_dash="dot",line_color="#ff4466",line_width=0.7,row=r,col=1)
            fig.add_hline(y=-100,line_dash="dot",line_color="#00ff88",line_width=0.7,row=r,col=1)
            fig.update_yaxes(title_text="CCI",row=r,col=1)

        if "WPR" in osc_panels and "WPR" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["WPR"],mode="lines",
                line=dict(color="#ff88aa",width=1),name="Williams %R"),row=r,col=1)
            fig.add_hline(y=-20,line_dash="dot",line_color="#ff4466",line_width=0.7,row=r,col=1)
            fig.add_hline(y=-80,line_dash="dot",line_color="#00ff88",line_width=0.7,row=r,col=1)
            fig.update_yaxes(range=[-100,0],title_text="%R",row=r,col=1)

        if "MFI" in osc_panels and "MFI" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["MFI"],mode="lines",
                line=dict(color="#44ffdd",width=1),name="MFI"),row=r,col=1)
            fig.add_hline(y=80,line_dash="dot",line_color="#ff4466",line_width=0.7,row=r,col=1)
            fig.add_hline(y=20,line_dash="dot",line_color="#00ff88",line_width=0.7,row=r,col=1)
            fig.update_yaxes(range=[0,100],title_text="MFI",row=r,col=1)

        if "MACD" in osc_panels:
            cr+=1
            r=cr
            hc=["#00ff88" if v>=0 else "#ff4466" for v in df["MACD_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=df.index,y=df["MACD_hist"],marker_color=hc,
                name="MACD Hist",opacity=0.8),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],mode="lines",
                line=dict(color="#00d4ff",width=1),name="MACD"),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["MACD_sig"],mode="lines",
                line=dict(color="#ffaa00",width=1),name="Signal"),row=r,col=1)
            fig.update_yaxes(title_text="MACD",row=r,col=1)

        if "PPO" in osc_panels and "PPO" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["PPO"],mode="lines",
                line=dict(color="#00d4ff",width=1),name="PPO"),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["PPO_sig"],mode="lines",
                line=dict(color="#ffaa00",width=1),name="PPO Sig"),row=r,col=1)
            fig.update_yaxes(title_text="PPO",row=r,col=1)

        if "DPO" in osc_panels and "DPO" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["DPO"],mode="lines",
                line=dict(color="#ffdd44",width=1),fill="tozeroy",
                fillcolor="rgba(255,221,68,0.06)",name="DPO"),row=r,col=1)
            fig.add_hline(y=0,line_color="#4a7a99",line_width=0.8,row=r,col=1)
            fig.update_yaxes(title_text="DPO",row=r,col=1)

        if "ADX" in osc_panels and "ADX" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["ADX"],mode="lines",
                line=dict(color="#ffaa00",width=1.2),name="ADX"),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["ADX_plus"],mode="lines",
                line=dict(color="#00ff88",width=0.8),name="+DI"),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["ADX_minus"],mode="lines",
                line=dict(color="#ff4466",width=0.8),name="-DI"),row=r,col=1)
            fig.add_hline(y=25,line_dash="dot",line_color="#4a7a99",line_width=0.7,row=r,col=1)
            fig.update_yaxes(title_text="ADX",row=r,col=1)

        if "Aroon" in osc_panels and "AROON_up" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["AROON_up"],mode="lines",
                line=dict(color="#00ff88",width=1),name="Aroon Up"),row=r,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["AROON_dn"],mode="lines",
                line=dict(color="#ff4466",width=1),name="Aroon Dn"),row=r,col=1)
            fig.update_yaxes(range=[0,100],title_text="Aroon",row=r,col=1)

        if "OBV" in osc_panels and "OBV" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["OBV"],mode="lines",
                line=dict(color="#00d4ff",width=1),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.05)",name="OBV"),row=r,col=1)
            fig.update_yaxes(title_text="OBV",row=r,col=1)

        if "HV" in osc_panels and "HV" in df:
            cr+=1
            r=cr
            fig.add_trace(go.Scatter(x=df.index,y=df["HV"],mode="lines",
                line=dict(color="#ff88aa",width=1),fill="tozeroy",
                fillcolor="rgba(255,136,170,0.05)",name="Vol hist. (%)"),row=r,col=1)
            fig.update_yaxes(title_text="HV%",row=r,col=1)

        fig.update_layout(**PLOTLY_BASE, height=max(600, 180 + 180*rows),
            title=dict(text=f"◈ {primary} — {period_l} · {len(SI)} indicateur(s)",
                       font=dict(family="Orbitron",color="#00d4ff",size=13)),
            xaxis_rangeslider_visible=False,hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        last=float(close.iloc[-1]); first=float(close.iloc[0]); ret=(last-first)/first*100
        m1.metric("Open (période)", f"${first:,.2f}")
        m2.metric("Dernier",        f"${last:,.2f}")
        m3.metric("Rendement",      f"{ret:+.2f}%")
        m4.metric("High / Low",     f"${float(high.max()):,.2f} / ${float(low.min()):,.2f}")
        m5.metric("Volume moy.",    f"{float(vol.mean())/1e6:.1f}M")
        m6.metric("ATR(14)",        f"${float(df['ATR'].iloc[-1]):,.2f}" if "ATR" in df else "N/A")

        with st.expander("▸ Données brutes"):
            st.dataframe(df.sort_index(ascending=False).head(100), use_container_width=True, height=300)
        st.download_button("⬇ CSV", df.to_csv().encode(), f"{primary}_{period}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    blabel("Options Chain · IV Smile · Open Interest")
    st.markdown("<span class='tag-free'>GRATUIT</span><span class='tag-limit'>Actions US uniquement</span>",
                unsafe_allow_html=True)
    oc1,oc2 = st.columns([2,2])
    opt_t = oc1.text_input("Ticker", value=primary, key="opt_tk")
    opt_type = oc2.radio("Type", ["Calls","Puts","Les deux"], horizontal=True)

    if st.button("↺ Recharger options", key="reload_opt"):
        st.cache_data.clear()
        st.rerun()
    t_obj, _, all_exps = get_options(opt_t.upper().strip())
    if not all_exps:
        st.warning(f"⚠ Aucune option disponible pour **{opt_t}**. Yahoo Finance peut être temporairement limité.")
        st.info("💡 Cliquez sur **↺ Recharger** ci-dessus, ou vérifiez que le ticker est une action US.")
    else:
        sel_exp = st.selectbox("Expiration", all_exps)
        try:
            chain = t_obj.option_chain(sel_exp)
            calls, puts = chain.calls, chain.puts
        except Exception as e:
            st.error(f"Erreur: {e}"); calls=puts=pd.DataFrame()

        spot = (get_info(opt_t.upper().strip()).get("currentPrice") or
                get_info(opt_t.upper().strip()).get("regularMarketPrice"))
        if spot:
            st.markdown(f"<span style='font-family:Space Mono,monospace;font-size:11px;color:#4da8da;'>Spot: <b style='color:#00d4ff;'>${spot:,.2f}</b></span>", unsafe_allow_html=True)

        cols_show = [c for c in ["strike","lastPrice","bid","ask","impliedVolatility",
                                  "openInterest","volume","inTheMoney"] if c in calls.columns]
        def fmt_chain(df_c):
            d = df_c[cols_show].copy()
            if "impliedVolatility" in d:
                d["impliedVolatility"] = (d["impliedVolatility"]*100).round(2).astype(str)+"%"
            return d

        if opt_type in ["Calls","Les deux"] and not calls.empty:
            blabel("Calls"); st.dataframe(fmt_chain(calls), use_container_width=True, height=260)
        if opt_type in ["Puts","Les deux"] and not puts.empty:
            blabel("Puts");  st.dataframe(fmt_chain(puts),  use_container_width=True, height=260)

        if not calls.empty and not puts.empty and "openInterest" in calls.columns:
            c1,c2 = st.columns(2)
            with c1:
                blabel("Open Interest")
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=calls["strike"],y=calls["openInterest"],
                    name="Calls OI",marker_color="#00ff88",opacity=0.8))
                fig_oi.add_trace(go.Bar(x=puts["strike"],y=-puts["openInterest"],
                    name="Puts OI",marker_color="#ff4466",opacity=0.8))
                if spot: fig_oi.add_vline(x=spot,line_dash="dot",line_color="#00d4ff")
                fig_oi.update_layout(**PLOTLY_BASE,height=320,barmode="overlay",
                    title=dict(text="◈ OI Calls vs Puts",font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_oi, use_container_width=True)
            with c2:
                blabel("IV Smile")
                fig_iv = go.Figure()
                if "impliedVolatility" in calls.columns:
                    iv_c = calls[["strike","impliedVolatility"]].dropna()
                    iv_p = puts[["strike","impliedVolatility"]].dropna()
                    fig_iv.add_trace(go.Scatter(x=iv_c["strike"],y=iv_c["impliedVolatility"]*100,
                        mode="lines+markers",line=dict(color="#00ff88",width=1.5),name="IV Calls"))
                    fig_iv.add_trace(go.Scatter(x=iv_p["strike"],y=iv_p["impliedVolatility"]*100,
                        mode="lines+markers",line=dict(color="#ff4466",width=1.5),name="IV Puts"))
                    if spot: fig_iv.add_vline(x=spot,line_dash="dot",line_color="#00d4ff")
                fig_iv.update_layout(**PLOTLY_BASE,height=320,
                    title=dict(text="◈ IV Smile",font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    yaxis_title="IV (%)")
                st.plotly_chart(fig_iv, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — CRYPTO L2
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    blabel("ccxt · Orderbook Level 2 · Multi-Exchange")
    st.markdown("<span class='tag-free'>GRATUIT · Sans clé API</span>", unsafe_allow_html=True)
    if not CCXT_OK:
        st.error("pip install ccxt")
    else:
        cr1,cr2,cr3 = st.columns([2,2,1])
        exch_id = cr1.selectbox("Exchange", ["binance","kraken","okx","bybit","coinbase","kucoin"])
        sym_in  = cr2.text_input("Paire", value="BTC/USDT")
        ob_depth= cr3.slider("Profondeur", 5, 50, 20)
        tf_map  = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1j":"1d"}
        tf_l    = st.selectbox("Timeframe", list(tf_map.keys()), index=3)
        tf      = tf_map[tf_l]

        col_ob, col_chart = st.columns([1,2])
        with col_ob:
            blabel("Order Book L2")
            ob, err = get_ccxt_ob(exch_id, sym_in, ob_depth)
            if err: st.warning(err)
            elif ob:
                bids = pd.DataFrame(ob["bids"], columns=["Prix","Qté"])
                asks = pd.DataFrame(ob["asks"], columns=["Prix","Qté"])
                bids["Cumulé"] = bids["Qté"].cumsum()
                asks["Cumulé"] = asks["Qté"].cumsum()
                mid = (bids["Prix"].iloc[0]+asks["Prix"].iloc[0])/2
                spread = asks["Prix"].iloc[0]-bids["Prix"].iloc[0]
                st.markdown(f"""
                <div style='font-family:Space Mono,monospace;font-size:10px;line-height:2;color:#8aafc8;'>
                  Mid: <b style='color:#00d4ff;'>${mid:,.2f}</b><br>
                  Spread: <b style='color:#ffaa00;'>${spread:.4f} ({spread/mid*100:.4f}%)</b>
                </div>""", unsafe_allow_html=True)
                fig_ob = go.Figure()
                fig_ob.add_trace(go.Bar(x=bids["Cumulé"],y=bids["Prix"],orientation="h",
                    name="Bids",marker_color="rgba(0,255,136,0.6)"))
                fig_ob.add_trace(go.Bar(x=asks["Cumulé"],y=asks["Prix"],orientation="h",
                    name="Asks",marker_color="rgba(255,68,102,0.6)"))
                fig_ob.add_hline(y=mid,line_dash="dot",line_color="#00d4ff",line_width=0.8)
                fig_ob.update_layout(**PLOTLY_BASE,height=400,barmode="overlay",
                    title=dict(text="◈ Depth",font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    xaxis_title="Volume cumulé")
                st.plotly_chart(fig_ob, use_container_width=True)
        with col_chart:
            blabel("OHLCV Crypto")
            df_c, err_c = get_ccxt_ohlcv(exch_id, sym_in, tf, 300)
            if err_c: st.warning(err_c)
            elif not df_c.empty:
                fig_c = make_subplots(rows=2,cols=1,shared_xaxes=True,
                    vertical_spacing=0.03,row_heights=[0.72,0.28])
                fig_c.add_trace(go.Candlestick(x=df_c.index,open=df_c["Open"],high=df_c["High"],
                    low=df_c["Low"],close=df_c["Close"],
                    increasing_line_color="#00ff88",decreasing_line_color="#ff4466",
                    increasing_fillcolor="#00ff88",decreasing_fillcolor="#ff4466",
                    name=sym_in,line_width=1),row=1,col=1)
                vc=["#00ff88" if c>=o else "#ff4466" for c,o in zip(df_c["Close"],df_c["Open"])]
                fig_c.add_trace(go.Bar(x=df_c.index,y=df_c["Volume"],
                    marker_color=vc,name="Vol",opacity=0.7),row=2,col=1)
                fig_c.update_layout(**PLOTLY_BASE,height=440,
                    title=dict(text=f"◈ {sym_in} · {exch_id} · {tf_l}",
                               font=dict(family="Orbitron",color="#00d4ff",size=12)),
                    xaxis_rangeslider_visible=False,hovermode="x unified")
                st.plotly_chart(fig_c, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    blabel("Analyse Statistique & Quantitative")
    stat_tabs = st.tabs(["Régressions","PCA","Distributions","GARCH","Monte Carlo","Cointégration"])

    # fetch all returns
    all_closes = {}
    for t in selected:
        d = get_ohlcv(t, period, "1d")
        if not d.empty:
            s = d["Close"].squeeze()
            if isinstance(s, pd.Series) and len(s) > 1:
                all_closes[t] = s
    if len(all_closes) >= 1:
        price_df   = pd.DataFrame(all_closes).dropna()
        returns_df = price_df.pct_change().dropna()
    else:
        price_df   = pd.DataFrame()
        returns_df = pd.DataFrame()
    if price_df.empty:
        st.warning("Données insuffisantes pour l'analyse statistique. Vérifiez vos tickers et la période.")
        st.stop()

    # ── Régressions ──────────────────────────────────────────────────
    with stat_tabs[0]:
        blabel("Régression linéaire · Rolling Beta")
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            rc1,rc2 = st.columns(2)
            dep_t = rc1.selectbox("Variable dépendante (Y)", selected)
            ind_t = rc2.selectbox("Variable indépendante (X)", [t for t in selected if t!=dep_t])
            y = returns_df[dep_t].dropna(); x = returns_df[ind_t].dropna()
            common = y.index.intersection(x.index); y=y[common]; x=x[common]

            slope, intercept, r, p, se = stats.linregress(x, y)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Beta (pente)", f"{slope:.4f}")
            m2.metric("Alpha (ordonnée)", f"{intercept*252:.4f} ann.")
            m3.metric("R²", f"{r**2:.4f}")
            m4.metric("p-value", f"{p:.4f}")

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=x,y=y,mode="markers",
                marker=dict(color="#00d4ff",size=4,opacity=0.6),name="Observations"))
            x_line = np.linspace(x.min(),x.max(),100)
            fig_reg.add_trace(go.Scatter(x=x_line,y=slope*x_line+intercept,mode="lines",
                line=dict(color="#ffaa00",width=1.5),name=f"β={slope:.3f}"))
            fig_reg.update_layout(**PLOTLY_BASE,height=380,
                title=dict(text=f"◈ {dep_t} vs {ind_t}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                xaxis_title=f"{ind_t} returns",yaxis_title=f"{dep_t} returns")
            st.plotly_chart(fig_reg, use_container_width=True)

            # Rolling beta
            blabel("Rolling Beta (60j)")
            roll_beta = pd.Series(dtype=float, index=returns_df.index)
            for i in range(60, len(returns_df)):
                window = returns_df.iloc[i-60:i]
                sl,_,_,_,_ = stats.linregress(window[ind_t], window[dep_t])
                roll_beta.iloc[i] = sl
            roll_beta = roll_beta.dropna()
            fig_rb = go.Figure(go.Scatter(x=roll_beta.index,y=roll_beta,mode="lines",
                line=dict(color="#00d4ff",width=1.5),name="Rolling Beta"))
            fig_rb.add_hline(y=1,line_dash="dot",line_color="#ffaa00",line_width=0.8)
            fig_rb.add_hline(y=0,line_dash="dot",line_color="#4a7a99",line_width=0.8)
            fig_rb.update_layout(**PLOTLY_BASE,height=280,
                title=dict(text="◈ Rolling Beta (60j)",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)))
            st.plotly_chart(fig_rb, use_container_width=True)

    # ── PCA ──────────────────────────────────────────────────────────
    with stat_tabs[1]:
        blabel("Analyse en Composantes Principales (PCA)")
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(returns_df.dropna())
            pca = PCA()
            X_pca = pca.fit_transform(X_sc)
            expl = pca.explained_variance_ratio_

            pc1,pc2 = st.columns(2)
            with pc1:
                blabel("Variance expliquée")
                fig_var = go.Figure()
                fig_var.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(expl))],
                    y=expl*100,marker_color="#00d4ff",name="Variance"))
                fig_var.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(expl))],
                    y=np.cumsum(expl)*100,mode="lines+markers",
                    line=dict(color="#ffaa00",width=1.5),name="Cumulée"))
                fig_var.update_layout(**PLOTLY_BASE,height=320,
                    title=dict(text="◈ Variance expliquée",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    yaxis_title="%")
                st.plotly_chart(fig_var, use_container_width=True)
            with pc2:
                blabel("Loadings PC1 vs PC2")
                loadings = pd.DataFrame(pca.components_[:2].T,
                                         columns=["PC1","PC2"], index=returns_df.columns)
                fig_load = go.Figure()
                for j, ticker in enumerate(loadings.index):
                    fig_load.add_trace(go.Scatter(x=[loadings.loc[ticker,"PC1"]],
                        y=[loadings.loc[ticker,"PC2"]],mode="markers+text",
                        text=[ticker],textposition="top center",
                        marker=dict(color=COLORS[j%len(COLORS)],size=10),name=ticker))
                    fig_load.add_shape(type="line",x0=0,y0=0,
                        x1=loadings.loc[ticker,"PC1"],y1=loadings.loc[ticker,"PC2"],
                        line=dict(color=COLORS[j%len(COLORS)],width=1))
                fig_load.add_vline(x=0,line_color="#1a3a55",line_width=0.8)
                fig_load.add_hline(y=0,line_color="#1a3a55",line_width=0.8)
                fig_load.update_layout(**PLOTLY_BASE,height=320,
                    title=dict(text="◈ Loadings",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_load, use_container_width=True)

            blabel("Projection PC1 vs PC2 dans le temps")
            fig_pca = go.Figure(go.Scatter(x=X_pca[:,0],y=X_pca[:,1],mode="markers",
                marker=dict(color=np.arange(len(X_pca)),colorscale="Viridis",
                            size=4,opacity=0.7,showscale=True),name="Observations"))
            fig_pca.update_layout(**PLOTLY_BASE,height=320,
                title=dict(text="◈ Scores PCA",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)),
                xaxis_title=f"PC1 ({expl[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({expl[1]*100:.1f}%)" if len(expl)>1 else "PC2")
            st.plotly_chart(fig_pca, use_container_width=True)

    # ── Distributions ────────────────────────────────────────────────
    with stat_tabs[2]:
        blabel("Distribution des rendements · Tests de normalité")
        dist_t = st.selectbox("Ticker", selected, key="dist_tk")
        r = returns_df[dist_t].dropna() if dist_t in returns_df else pd.Series(dtype=float)
        if not r.empty:
            d1,d2,d3,d4 = st.columns(4)
            d1.metric("Moyenne ann.",  f"{r.mean()*252:.2%}")
            d2.metric("Volatilité ann.", f"{r.std()*np.sqrt(252):.2%}")
            d3.metric("Skewness",      f"{stats.skew(r):.4f}")
            d4.metric("Kurtosis (exc.)", f"{stats.kurtosis(r):.4f}")

            _, p_jb = stats.jarque_bera(r)
            _, p_sw = stats.shapiro(r[:5000])
            st.markdown(f"""
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#8aafc8;line-height:2;'>
              Jarque-Bera p-value: <b style='color:{"#ff4466" if p_jb<0.05 else "#00ff88"};'>{p_jb:.4f}</b>
              {'(non normal)' if p_jb<0.05 else '(normal)'}&nbsp;&nbsp;
              Shapiro-Wilk p-value: <b style='color:{"#ff4466" if p_sw<0.05 else "#00ff88"};'>{p_sw:.4f}</b>
            </div>""", unsafe_allow_html=True)

            fig_dist = make_subplots(rows=1,cols=2,subplot_titles=["Histogramme","QQ-Plot"])
            # Histogram
            fig_dist.add_trace(go.Histogram(x=r,nbinsx=80,
                marker_color="#00d4ff",opacity=0.7,name="Rendements",
                histnorm="probability density"),row=1,col=1)
            x_norm = np.linspace(r.min(),r.max(),200)
            fig_dist.add_trace(go.Scatter(x=x_norm,
                y=stats.norm.pdf(x_norm,r.mean(),r.std()),
                mode="lines",line=dict(color="#ffaa00",width=1.5),name="Normale"),row=1,col=1)
            # QQ
            qq = stats.probplot(r)
            fig_dist.add_trace(go.Scatter(x=qq[0][0],y=qq[0][1],mode="markers",
                marker=dict(color="#00d4ff",size=3),name="QQ"),row=1,col=2)
            fig_dist.add_trace(go.Scatter(x=qq[0][0],
                y=qq[1][0]*qq[0][0]+qq[1][1],mode="lines",
                line=dict(color="#ffaa00",width=1.5),name="Théorique"),row=1,col=2)
            fig_dist.update_layout(**PLOTLY_BASE,height=380,showlegend=False,
                title=dict(text=f"◈ {dist_t} — Distribution rendements",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)))
            st.plotly_chart(fig_dist, use_container_width=True)

    # ── GARCH ────────────────────────────────────────────────────────
    with stat_tabs[3]:
        blabel("GARCH(1,1) · Volatilité conditionnelle")
        if not ARCH_OK:
            st.error("pip install arch")
        else:
            garch_t = st.selectbox("Ticker", selected, key="garch_tk")
            r_g = returns_df[garch_t].dropna()*100 if garch_t in returns_df else pd.Series(dtype=float)
            if not r_g.empty:
                with st.spinner("Fitting GARCH(1,1)…"):
                    try:
                        am = arch_model(r_g, vol="Garch", p=1, q=1, dist="normal")
                        res = am.fit(disp="off")
                        cond_vol = res.conditional_volatility

                        g1,g2,g3,g4 = st.columns(4)
                        g1.metric("ω (omega)", f"{res.params['omega']:.6f}")
                        g2.metric("α (alpha)", f"{res.params['alpha[1]']:.4f}")
                        g3.metric("β (beta)",  f"{res.params['beta[1]']:.4f}")
                        g4.metric("Persist. α+β", f"{res.params['alpha[1]']+res.params['beta[1]']:.4f}")

                        fig_garch = make_subplots(rows=2,cols=1,shared_xaxes=True,
                            vertical_spacing=0.05,row_heights=[0.5,0.5])
                        fig_garch.add_trace(go.Scatter(x=r_g.index,y=r_g,mode="lines",
                            line=dict(color="#00d4ff",width=0.8),name="Returns (%)"),row=1,col=1)
                        fig_garch.add_trace(go.Scatter(x=cond_vol.index,y=cond_vol,mode="lines",
                            line=dict(color="#ff4466",width=1.5),
                            fill="tozeroy",fillcolor="rgba(255,68,102,0.08)",
                            name="Vol. cond. (%)"),row=2,col=1)
                        fig_garch.update_layout(**PLOTLY_BASE,height=460,
                            title=dict(text=f"◈ GARCH(1,1) — {garch_t}",
                                       font=dict(family="Orbitron",color="#00d4ff",size=12)),
                            hovermode="x unified")
                        st.plotly_chart(fig_garch, use_container_width=True)

                        # Forecast
                        blabel("Prévision volatilité (10 jours)")
                        forecast = res.forecast(horizon=10)
                        fvol = np.sqrt(forecast.variance.dropna().iloc[-1].values)
                        fig_fc = go.Figure(go.Bar(x=list(range(1,11)),y=fvol,
                            marker_color="#aa88ff",name="Vol prévue (%)"))
                        fig_fc.update_layout(**PLOTLY_BASE,height=260,
                            title=dict(text="◈ Prévision volatilité GARCH",
                                       font=dict(family="Orbitron",color="#00d4ff",size=11)),
                            xaxis_title="Jours",yaxis_title="Volatilité (%)")
                        st.plotly_chart(fig_fc, use_container_width=True)
                    except Exception as e:
                        st.error(f"GARCH error: {e}")

    # ── Monte Carlo ──────────────────────────────────────────────────
    with stat_tabs[4]:
        blabel("Simulation Monte Carlo · GBM")
        mc1,mc2,mc3 = st.columns(3)
        mc_t    = mc1.selectbox("Ticker", selected, key="mc_tk")
        n_sim   = mc2.slider("Simulations", 100, 2000, 500, step=100)
        n_days  = mc3.slider("Jours à projeter", 30, 365, 252)

        r_mc = returns_df[mc_t].dropna() if mc_t in returns_df else pd.Series(dtype=float)
        if not r_mc.empty:
            mu_d  = r_mc.mean(); sig_d = r_mc.std()
            df_mc = get_ohlcv(mc_t, period, "1d")
            S0    = float(df_mc["Close"].squeeze().iloc[-1]) if not df_mc.empty else 100.0

            with st.spinner(f"Monte Carlo {n_sim} simulations…"):
                np.random.seed(42)
                daily_ret = np.random.normal(mu_d, sig_d, (n_days, n_sim))
                price_paths = S0 * np.cumprod(1+daily_ret, axis=0)

            final = price_paths[-1]
            pct5  = np.percentile(final, 5)
            pct50 = np.percentile(final, 50)
            pct95 = np.percentile(final, 95)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Prix actuel",   f"${S0:,.2f}")
            m2.metric("P5 (baissier)", f"${pct5:,.2f}", f"{(pct5/S0-1)*100:+.1f}%")
            m3.metric("Médiane",       f"${pct50:,.2f}", f"{(pct50/S0-1)*100:+.1f}%")
            m4.metric("P95 (haussier)",f"${pct95:,.2f}", f"{(pct95/S0-1)*100:+.1f}%")

            fig_mc = go.Figure()
            for i in range(min(200, n_sim)):
                fig_mc.add_trace(go.Scatter(
                    y=price_paths[:,i], mode="lines",
                    line=dict(color="rgba(0,212,255,0.05)", width=0.5),
                    showlegend=False))
            fig_mc.add_trace(go.Scatter(y=np.percentile(price_paths,95,axis=1),mode="lines",
                line=dict(color="#00ff88",width=2),name="P95"))
            fig_mc.add_trace(go.Scatter(y=np.percentile(price_paths,50,axis=1),mode="lines",
                line=dict(color="#ffaa00",width=2),name="Médiane"))
            fig_mc.add_trace(go.Scatter(y=np.percentile(price_paths,5,axis=1),mode="lines",
                line=dict(color="#ff4466",width=2),name="P5"))
            fig_mc.add_hline(y=S0,line_dash="dot",line_color="#4a7a99",line_width=0.8)
            fig_mc.update_layout(**PLOTLY_BASE,height=420,
                title=dict(text=f"◈ Monte Carlo {n_sim} trajectoires — {mc_t} ({n_days}j)",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                yaxis_title="Prix ($)")
            st.plotly_chart(fig_mc, use_container_width=True)

            # Distribution finale
            fig_fdist = go.Figure()
            fig_fdist.add_trace(go.Histogram(x=final,nbinsx=80,
                marker_color="#00d4ff",opacity=0.7,histnorm="probability density",
                name="Distribution finale"))
            fig_fdist.add_vline(x=S0,line_dash="dot",line_color="#ffaa00",
                annotation_text=f"S₀ ${S0:.0f}")
            fig_fdist.add_vline(x=pct5,line_dash="dot",line_color="#ff4466",
                annotation_text=f"P5 ${pct5:.0f}")
            fig_fdist.add_vline(x=pct95,line_dash="dot",line_color="#00ff88",
                annotation_text=f"P95 ${pct95:.0f}")
            fig_fdist.update_layout(**PLOTLY_BASE,height=300,
                title=dict(text="◈ Distribution des prix finaux",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)))
            st.plotly_chart(fig_fdist, use_container_width=True)

    # ── Cointégration ────────────────────────────────────────────────
    with stat_tabs[5]:
        blabel("Cointégration · Pairs Trading")
        if not SM_OK:
            st.error("pip install statsmodels")
        elif len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            ci1,ci2 = st.columns(2)
            tA = ci1.selectbox("Actif A", selected, key="ci_a")
            tB = ci2.selectbox("Actif B", [t for t in selected if t!=tA], key="ci_b")
            pA = price_df[tA].dropna(); pB = price_df[tB].dropna()
            idx_c = pA.index.intersection(pB.index); pA=pA[idx_c]; pB=pB[idx_c]

            if len(pA) > 30:
                score, pval, crit = coint(pA, pB)
                adf_spread = adfuller(pA - pB * (pA.mean()/pB.mean()))

                col_ci1, col_ci2, col_ci3 = st.columns(3)
                col_ci1.metric("Score cointégration", f"{score:.4f}")
                col_ci2.metric("p-value",
                    f"{pval:.4f}",
                    "Cointégrés ✓" if pval<0.05 else "Non cointégrés",
                    delta_color="normal" if pval<0.05 else "inverse")
                col_ci3.metric("ADF spread p-value", f"{adf_spread[1]:.4f}")

                # Spread
                hedge = pA.mean()/pB.mean()
                spread = pA - pB*hedge
                z_score = (spread - spread.mean())/spread.std()

                fig_spread = make_subplots(rows=2,cols=1,shared_xaxes=True,
                    vertical_spacing=0.05,row_heights=[0.5,0.5])
                fig_spread.add_trace(go.Scatter(x=spread.index,y=spread,mode="lines",
                    line=dict(color="#00d4ff",width=1.2),name="Spread"),row=1,col=1)
                fig_spread.add_hline(y=spread.mean(),line_dash="dot",
                    line_color="#ffaa00",line_width=0.8,row=1,col=1)
                fig_spread.add_trace(go.Scatter(x=z_score.index,y=z_score,mode="lines",
                    line=dict(color="#aa88ff",width=1.2),name="Z-score"),row=2,col=1)
                for level in [2,-2,1,-1]:
                    col_z = "#ff4466" if abs(level)==2 else "#00ff88"
                    fig_spread.add_hline(y=level,line_dash="dot",
                        line_color=col_z,line_width=0.7,row=2,col=1)
                fig_spread.update_layout(**PLOTLY_BASE,height=400,
                    title=dict(text=f"◈ Spread & Z-score — {tA} vs {tB}",
                               font=dict(family="Orbitron",color="#00d4ff",size=12)),
                    hovermode="x unified")
                st.plotly_chart(fig_spread, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    blabel("Backtesting · Stratégies · Métriques de performance")
    bt_tabs = st.tabs(["SMA Crossover","RSI Mean-Reversion","Momentum","Résultats"])

    bt_ticker = st.selectbox("Ticker pour backtest", selected, key="bt_tk")
    df_bt_raw = get_ohlcv(bt_ticker, "2y", "1d")
    df_bt     = add_indicators(df_bt_raw) if not df_bt_raw.empty else pd.DataFrame()

    def run_backtest(signals, prices, cost=0.001):
        """Generic backtester with transaction costs."""
        pos = signals.shift(1).fillna(0)
        ret = prices.pct_change().fillna(0)
        trade = pos.diff().abs()
        strat_ret = pos*ret - trade*cost
        equity = (1+strat_ret).cumprod()*100
        bh_equity = (1+ret).cumprod()*100
        return strat_ret, equity, bh_equity, pos

    def perf_metrics(ret, equity, prices, rf=0.0):
        r = ret.dropna()
        return {
            "Rendement total":   f"{(equity.iloc[-1]/100-1)*100:+.2f}%",
            "Rendement annualisé": f"{r.mean()*252*100:+.2f}%",
            "Volatilité ann.":   f"{r.std()*np.sqrt(252)*100:.2f}%",
            "Sharpe":            f"{sharpe(r,rf):.3f}",
            "Sortino":           f"{sortino(r,rf):.3f}",
            "Max Drawdown":      f"{max_drawdown(equity)*100:.2f}%",
            "Calmar":            f"{calmar(r,equity):.3f}",
            "Win Rate":          f"{(r>0).mean()*100:.1f}%",
            "Nb trades":         f"{(ret.diff().abs()>0.001).sum()}",
        }

    with bt_tabs[0]:
        blabel("SMA Crossover Strategy")
        if not df_bt.empty:
            bc1,bc2,bc3 = st.columns(3)
            fast = bc1.slider("SMA Fast", 5,50,20,key="bt_f")
            slow = bc2.slider("SMA Slow",20,200,50,key="bt_s")
            cost = bc3.slider("Coût transaction (%)",0.0,0.5,0.1,step=0.05)/100
            close_bt = df_bt["Close"].squeeze()
            sma_f = close_bt.rolling(fast).mean(); sma_s = close_bt.rolling(slow).mean()
            signal = (sma_f > sma_s).astype(int)
            ret_bt, equity_bt, bh_bt, pos = run_backtest(signal, close_bt, cost)
            metrics = perf_metrics(ret_bt, equity_bt, close_bt)
            cols_m = st.columns(len(metrics))
            for i,(k,v) in enumerate(metrics.items()):
                cols_m[i].metric(k,v)
            fig_bt = make_subplots(rows=3,cols=1,shared_xaxes=True,
                vertical_spacing=0.04,row_heights=[0.45,0.3,0.25])
            fig_bt.add_trace(go.Scatter(x=close_bt.index,y=close_bt,mode="lines",
                line=dict(color="#4a7a99",width=1),name="Prix"),row=1,col=1)
            fig_bt.add_trace(go.Scatter(x=sma_f.index,y=sma_f,mode="lines",
                line=dict(color="#00ff88",width=1),name=f"SMA{fast}"),row=1,col=1)
            fig_bt.add_trace(go.Scatter(x=sma_s.index,y=sma_s,mode="lines",
                line=dict(color="#ffaa00",width=1,dash="dash"),name=f"SMA{slow}"),row=1,col=1)
            buys  = close_bt[signal.diff()==1]
            sells = close_bt[signal.diff()==-1]
            fig_bt.add_trace(go.Scatter(x=buys.index,y=buys,mode="markers",
                marker=dict(symbol="triangle-up",color="#00ff88",size=8),name="BUY"),row=1,col=1)
            fig_bt.add_trace(go.Scatter(x=sells.index,y=sells,mode="markers",
                marker=dict(symbol="triangle-down",color="#ff4466",size=8),name="SELL"),row=1,col=1)
            fig_bt.add_trace(go.Scatter(x=equity_bt.index,y=equity_bt,mode="lines",
                line=dict(color="#00d4ff",width=1.5),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.05)",name="Stratégie"),row=2,col=1)
            fig_bt.add_trace(go.Scatter(x=bh_bt.index,y=bh_bt,mode="lines",
                line=dict(color="#4a7a99",width=1,dash="dash"),name="Buy & Hold"),row=2,col=1)
            drawdown_series = (equity_bt/equity_bt.cummax()-1)*100
            fig_bt.add_trace(go.Scatter(x=drawdown_series.index,y=drawdown_series,
                mode="lines",fill="tozeroy",fillcolor="rgba(255,68,102,0.15)",
                line=dict(color="#ff4466",width=1),name="Drawdown"),row=3,col=1)
            fig_bt.update_layout(**PLOTLY_BASE,height=560,
                title=dict(text=f"◈ SMA{fast}/{slow} Crossover — {bt_ticker}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_bt, use_container_width=True)

    with bt_tabs[1]:
        blabel("RSI Mean-Reversion Strategy")
        if not df_bt.empty:
            rc1,rc2,rc3,rc4 = st.columns(4)
            rsi_period = rc1.slider("RSI Period",5,30,14,key="bt_rp")
            ob_level   = rc2.slider("Overbought",60,90,70,key="bt_ob")
            os_level   = rc3.slider("Oversold",10,40,30,key="bt_os")
            cost_r     = rc4.slider("Coût (%)",0.0,0.5,0.1,step=0.05,key="bt_cr")/100
            close_bt   = df_bt["Close"].squeeze()
            delta      = close_bt.diff()
            gain       = delta.clip(lower=0).rolling(rsi_period).mean()
            loss       = (-delta.clip(upper=0)).rolling(rsi_period).mean()
            rsi_bt     = 100-100/(1+gain/loss.replace(0,np.nan))
            signal_rsi = pd.Series(0, index=close_bt.index)
            signal_rsi[rsi_bt < os_level] =  1
            signal_rsi[rsi_bt > ob_level] = -1
            signal_rsi = signal_rsi.replace(0, np.nan).ffill().fillna(0)
            ret_rsi, eq_rsi, bh_rsi, _ = run_backtest(signal_rsi, close_bt, cost_r)
            metrics_rsi = perf_metrics(ret_rsi, eq_rsi, close_bt)
            cols_m = st.columns(len(metrics_rsi))
            for i,(k,v) in enumerate(metrics_rsi.items()): cols_m[i].metric(k,v)
            fig_rsi_bt = make_subplots(rows=3,cols=1,shared_xaxes=True,
                vertical_spacing=0.04,row_heights=[0.35,0.3,0.35])
            fig_rsi_bt.add_trace(go.Scatter(x=rsi_bt.index,y=rsi_bt,mode="lines",
                line=dict(color="#aa88ff",width=1.2),name="RSI"),row=1,col=1)
            fig_rsi_bt.add_hline(y=ob_level,line_dash="dot",line_color="#ff4466",
                line_width=0.8,row=1,col=1)
            fig_rsi_bt.add_hline(y=os_level,line_dash="dot",line_color="#00ff88",
                line_width=0.8,row=1,col=1)
            fig_rsi_bt.add_trace(go.Scatter(x=eq_rsi.index,y=eq_rsi,mode="lines",
                line=dict(color="#00d4ff",width=1.5),name="Stratégie"),row=2,col=1)
            fig_rsi_bt.add_trace(go.Scatter(x=bh_rsi.index,y=bh_rsi,mode="lines",
                line=dict(color="#4a7a99",width=1,dash="dash"),name="Buy & Hold"),row=2,col=1)
            dd_rsi=(eq_rsi/eq_rsi.cummax()-1)*100
            fig_rsi_bt.add_trace(go.Scatter(x=dd_rsi.index,y=dd_rsi,mode="lines",
                fill="tozeroy",fillcolor="rgba(255,68,102,0.15)",
                line=dict(color="#ff4466",width=1),name="Drawdown"),row=3,col=1)
            fig_rsi_bt.update_layout(**PLOTLY_BASE,height=520,
                title=dict(text=f"◈ RSI Mean-Reversion — {bt_ticker}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_rsi_bt, use_container_width=True)

    with bt_tabs[2]:
        blabel("Momentum Strategy")
        if not df_bt.empty:
            mc1,mc2 = st.columns(2)
            mom_period = mc1.slider("Fenêtre momentum (jours)",10,120,60,key="bt_mp")
            cost_m     = mc2.slider("Coût (%)",0.0,0.5,0.1,step=0.05,key="bt_mc")/100
            close_bt   = df_bt["Close"].squeeze()
            momentum   = close_bt.pct_change(mom_period)
            signal_mom = (momentum > 0).astype(int)
            ret_mom, eq_mom, bh_mom, _ = run_backtest(signal_mom, close_bt, cost_m)
            metrics_mom = perf_metrics(ret_mom, eq_mom, close_bt)
            cols_m = st.columns(len(metrics_mom))
            for i,(k,v) in enumerate(metrics_mom.items()): cols_m[i].metric(k,v)
            fig_mom = make_subplots(rows=3,cols=1,shared_xaxes=True,
                vertical_spacing=0.04,row_heights=[0.35,0.35,0.3])
            fig_mom.add_trace(go.Scatter(x=momentum.index,y=momentum*100,mode="lines",
                line=dict(color="#00d4ff",width=1),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.05)",name=f"Momentum {mom_period}j (%)"),row=1,col=1)
            fig_mom.add_hline(y=0,line_color="#4a7a99",line_width=0.8,row=1,col=1)
            fig_mom.add_trace(go.Scatter(x=eq_mom.index,y=eq_mom,mode="lines",
                line=dict(color="#00d4ff",width=1.5),name="Stratégie"),row=2,col=1)
            fig_mom.add_trace(go.Scatter(x=bh_mom.index,y=bh_mom,mode="lines",
                line=dict(color="#4a7a99",width=1,dash="dash"),name="Buy & Hold"),row=2,col=1)
            dd_mom=(eq_mom/eq_mom.cummax()-1)*100
            fig_mom.add_trace(go.Scatter(x=dd_mom.index,y=dd_mom,mode="lines",
                fill="tozeroy",fillcolor="rgba(255,68,102,0.15)",
                line=dict(color="#ff4466",width=1),name="Drawdown"),row=3,col=1)
            fig_mom.update_layout(**PLOTLY_BASE,height=520,
                title=dict(text=f"◈ Momentum ({mom_period}j) — {bt_ticker}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_mom, use_container_width=True)

    with bt_tabs[3]:
        blabel("Comparaison toutes stratégies")
        if not df_bt.empty:
            close_bt = df_bt["Close"].squeeze()
            sma_sig  = (close_bt.rolling(20).mean() > close_bt.rolling(50).mean()).astype(int)
            _,eq_sma,bh,_ = run_backtest(sma_sig, close_bt, 0.001)
            rsi_sig2 = pd.Series(0,index=close_bt.index)
            r_delta  = close_bt.diff(); r_gain=r_delta.clip(lower=0).rolling(14).mean()
            r_loss   = (-r_delta.clip(upper=0)).rolling(14).mean()
            rsi2     = 100-100/(1+r_gain/r_loss.replace(0,np.nan))
            rsi_sig2[rsi2<30]=1; rsi_sig2[rsi2>70]=-1
            rsi_sig2 = rsi_sig2.replace(0,np.nan).ffill().fillna(0)
            _,eq_rsi2,_,_ = run_backtest(rsi_sig2, close_bt, 0.001)
            mom_sig2 = (close_bt.pct_change(60)>0).astype(int)
            _,eq_mom2,_,_ = run_backtest(mom_sig2, close_bt, 0.001)
            fig_cmp = go.Figure()
            for label, eq, color in [("SMA 20/50",eq_sma,"#00d4ff"),
                                      ("RSI MR",   eq_rsi2,"#aa88ff"),
                                      ("Momentum", eq_mom2,"#ffaa00"),
                                      ("Buy & Hold",bh,    "#4a7a99")]:
                fig_cmp.add_trace(go.Scatter(x=eq.index,y=eq,mode="lines",
                    line=dict(color=color,width=1.5),name=label))
            fig_cmp.update_layout(**PLOTLY_BASE,height=400,
                title=dict(text=f"◈ Comparaison stratégies — {bt_ticker}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified",yaxis_title="Equity (base 100)")
            st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    blabel("Risk Management · VaR · CVaR · Stress Tests · Greeks")
    risk_tabs = st.tabs(["VaR / CVaR","Stress Tests","Greeks Options","Exposition"])

    with risk_tabs[0]:
        blabel("Value at Risk & Conditional VaR")
        risk_t  = st.selectbox("Ticker", selected, key="risk_tk")
        conf_lv = st.slider("Niveau de confiance (%)", 90, 99, 95) / 100
        portfolio_value = st.number_input("Valeur portefeuille ($)", value=100_000, step=10_000)
        r_risk  = returns_df[risk_t].dropna() if risk_t in returns_df else pd.Series(dtype=float)
        if not r_risk.empty:
            var_hist  = compute_var(r_risk, conf_lv, "historical")
            var_param = compute_var(r_risk, conf_lv, "parametric")
            var_cf    = compute_var(r_risk, conf_lv, "cornish_fisher")
            cvar_v    = compute_cvar(r_risk, conf_lv)

            rc1,rc2,rc3,rc4 = st.columns(4)
            rc1.metric(f"VaR Hist. ({conf_lv*100:.0f}%)",
                f"${abs(var_hist*portfolio_value):,.0f}", f"{var_hist*100:.2f}%",delta_color="inverse")
            rc2.metric(f"VaR Param. ({conf_lv*100:.0f}%)",
                f"${abs(var_param*portfolio_value):,.0f}", f"{var_param*100:.2f}%",delta_color="inverse")
            rc3.metric(f"VaR C-F ({conf_lv*100:.0f}%)",
                f"${abs(var_cf*portfolio_value):,.0f}", f"{var_cf*100:.2f}%",delta_color="inverse")
            rc4.metric(f"CVaR ({conf_lv*100:.0f}%)",
                f"${abs(cvar_v*portfolio_value):,.0f}", f"{cvar_v*100:.2f}%",delta_color="inverse")

            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(x=r_risk*100,nbinsx=80,
                marker_color="#00d4ff",opacity=0.6,name="Returns (%)",
                histnorm="probability density"))
            for v,label,color in [(var_hist,"VaR Hist","#ff4466"),
                                   (var_param,"VaR Param","#ffaa00"),
                                   (cvar_v,"CVaR","#ff88aa")]:
                fig_var.add_vline(x=v*100,line_dash="dot",line_color=color,
                    annotation_text=label,annotation_font_color=color)
            x_fill = np.linspace(r_risk.min()*100, var_hist*100, 200)
            fig_var.add_trace(go.Scatter(x=x_fill,
                y=stats.norm.pdf(x_fill/100,r_risk.mean(),r_risk.std())/100,
                mode="none",fill="tozeroy",fillcolor="rgba(255,68,102,0.2)",name="Tail"))
            fig_var.update_layout(**PLOTLY_BASE,height=380,
                title=dict(text=f"◈ Distribution VaR — {risk_t}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                xaxis_title="Rendement journalier (%)")
            st.plotly_chart(fig_var, use_container_width=True)

            # Rolling VaR
            blabel("Rolling VaR (60j)")
            roll_var = r_risk.rolling(60).quantile(1-conf_lv)
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(x=r_risk.index,y=r_risk*100,mode="lines",
                line=dict(color="#00d4ff",width=0.8),opacity=0.5,name="Returns"))
            fig_rv.add_trace(go.Scatter(x=roll_var.index,y=roll_var*100,mode="lines",
                line=dict(color="#ff4466",width=1.5),name=f"Rolling VaR {conf_lv*100:.0f}%"))
            fig_rv.update_layout(**PLOTLY_BASE,height=300,
                title=dict(text="◈ Rolling VaR (60j)",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)),
                yaxis_title="%",hovermode="x unified")
            st.plotly_chart(fig_rv, use_container_width=True)

    with risk_tabs[1]:
        blabel("Stress Tests · Scénarios historiques")
        stress_t = st.selectbox("Ticker", selected, key="stress_tk")
        df_stress = get_ohlcv(stress_t, "5y", "1d")
        if not df_stress.empty:
            r_stress = df_stress["Close"].squeeze().pct_change().dropna()
            scenarios = {
                "COVID Crash (Feb-Mar 2020)": ("2020-02-19","2020-03-23"),
                "Rate Hike 2022":             ("2022-01-03","2022-10-13"),
                "Q4 2018 Selloff":            ("2018-10-01","2018-12-24"),
                "Covid Recovery (2020)":      ("2020-03-23","2020-08-31"),
                "Dot-com (2000-2002)":        ("2000-03-10","2002-10-09"),
            }
            rows_sc = []
            for name, (start, end) in scenarios.items():
                try:
                    mask = (r_stress.index >= start) & (r_stress.index <= end)
                    r_sc = r_stress[mask]
                    if len(r_sc) < 5: continue
                    total = float((1+r_sc).prod()-1)
                    ann   = float(r_sc.mean()*252)
                    vol   = float(r_sc.std()*np.sqrt(252))
                    mdd   = float(max_drawdown((1+r_sc).cumprod()))
                    rows_sc.append({"Scénario":name,"Return":f"{total*100:+.1f}%",
                                    "Ann.":f"{ann*100:+.1f}%","Vol.":f"{vol*100:.1f}%",
                                    "Max DD":f"{mdd*100:.1f}%","Jours":len(r_sc)})
                except: pass

            st.dataframe(pd.DataFrame(rows_sc).set_index("Scénario"),
                         use_container_width=True)

            # Simulation stress -10%, -20%, -30%
            blabel("Simulation pertes sur portefeuille")
            pv = st.number_input("Valeur ($)", value=100_000, step=10_000, key="stress_pv")
            shocks = [-0.05,-0.10,-0.15,-0.20,-0.30,-0.40,-0.50]
            fig_shock = go.Figure(go.Bar(
                x=[f"{s*100:.0f}%" for s in shocks],
                y=[pv*s for s in shocks],
                marker_color=["#ff4466"]*len(shocks),
                text=[f"${pv*s:,.0f}" for s in shocks],
                textposition="outside",
                textfont=dict(family="Space Mono",color="#e8f4fd",size=10),
            ))
            fig_shock.update_layout(**PLOTLY_BASE,height=300,
                title=dict(text="◈ Impact chocs sur portefeuille",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)),
                yaxis_title="Perte ($)")
            st.plotly_chart(fig_shock, use_container_width=True)

    with risk_tabs[2]:
        blabel("Greeks Options (calcul analytique BS)")
        from scipy.stats import norm as norm_dist

        def bs_greeks(S, K, T, r, sigma, opt_type="call"):
            if T <= 0 or sigma <= 0: return {}
            d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
            d2 = d1-sigma*np.sqrt(T)
            delta = norm_dist.cdf(d1) if opt_type=="call" else norm_dist.cdf(d1)-1
            gamma = norm_dist.pdf(d1)/(S*sigma*np.sqrt(T))
            theta = (-(S*norm_dist.pdf(d1)*sigma)/(2*np.sqrt(T)) -
                     r*K*np.exp(-r*T)*norm_dist.cdf(d2 if opt_type=="call" else -d2))/365
            vega  = S*norm_dist.pdf(d1)*np.sqrt(T)/100
            rho   = K*T*np.exp(-r*T)*norm_dist.cdf(d2 if opt_type=="call" else -d2)/100
            price = (S*norm_dist.cdf(d1)-K*np.exp(-r*T)*norm_dist.cdf(d2)
                     if opt_type=="call" else
                     K*np.exp(-r*T)*norm_dist.cdf(-d2)-S*norm_dist.cdf(-d1))
            return dict(Price=price,Delta=delta,Gamma=gamma,Theta=theta,Vega=vega,Rho=rho)

        gc1,gc2,gc3,gc4,gc5 = st.columns(5)
        S_in  = gc1.number_input("Spot (S)", value=150.0, step=1.0)
        K_in  = gc2.number_input("Strike (K)", value=150.0, step=1.0)
        T_in  = gc3.number_input("Maturité (jours)", value=30, step=1)
        iv_in = gc4.number_input("IV (σ %)", value=25.0, step=0.5) / 100
        r_in  = gc5.number_input("Taux sans risque (%)", value=5.0, step=0.1) / 100
        opt_type_bs = st.radio("Type", ["call","put"], horizontal=True)

        greeks = bs_greeks(S_in, K_in, T_in/365, r_in, iv_in, opt_type_bs)
        if greeks:
            gm1,gm2,gm3,gm4,gm5,gm6 = st.columns(6)
            gm1.metric("Prix théorique", f"${greeks['Price']:.4f}")
            gm2.metric("Δ Delta", f"{greeks['Delta']:.4f}")
            gm3.metric("Γ Gamma", f"{greeks['Gamma']:.6f}")
            gm4.metric("Θ Theta/j", f"{greeks['Theta']:.4f}")
            gm5.metric("ν Vega/1%", f"{greeks['Vega']:.4f}")
            gm6.metric("ρ Rho/1%", f"{greeks['Rho']:.4f}")

            # Delta vs Spot
            spots = np.linspace(S_in*0.7, S_in*1.3, 100)
            deltas = [bs_greeks(s,K_in,T_in/365,r_in,iv_in,opt_type_bs).get("Delta",0) for s in spots]
            gammas = [bs_greeks(s,K_in,T_in/365,r_in,iv_in,opt_type_bs).get("Gamma",0) for s in spots]
            prices_bs = [bs_greeks(s,K_in,T_in/365,r_in,iv_in,opt_type_bs).get("Price",0) for s in spots]

            fig_greeks = make_subplots(rows=1,cols=3,subplot_titles=["Prix","Delta","Gamma"])
            fig_greeks.add_trace(go.Scatter(x=spots,y=prices_bs,mode="lines",
                line=dict(color="#00d4ff",width=1.5),name="Prix"),row=1,col=1)
            fig_greeks.add_trace(go.Scatter(x=spots,y=deltas,mode="lines",
                line=dict(color="#00ff88",width=1.5),name="Delta"),row=1,col=2)
            fig_greeks.add_trace(go.Scatter(x=spots,y=gammas,mode="lines",
                line=dict(color="#ffaa00",width=1.5),name="Gamma"),row=1,col=3)
            for col in [1,2,3]:
                fig_greeks.add_vline(x=S_in,line_dash="dot",line_color="#4a7a99",
                    line_width=0.8,row=1,col=col)
            fig_greeks.update_layout(**PLOTLY_BASE,height=320,showlegend=False,
                title=dict(text="◈ Greeks vs Spot",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)))
            st.plotly_chart(fig_greeks, use_container_width=True)

    with risk_tabs[3]:
        blabel("Exposition & Métriques de portefeuille")
        if len(selected) >= 2:
            exp_cols = st.columns(len(selected))
            for i, t in enumerate(selected):
                r_t = returns_df[t].dropna() if t in returns_df else pd.Series(dtype=float)
                if not r_t.empty:
                    vol_ann = r_t.std()*np.sqrt(252)*100
                    sharpe_v = sharpe(r_t)
                    mdd_v = max_drawdown(price_df[t]) if t in price_df else 0
                    with exp_cols[i]:
                        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:10px;color:#00d4ff;'>{t}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:10px;line-height:2;color:#8aafc8;'>Vol ann.: <b>{vol_ann:.1f}%</b><br>Sharpe: <b>{sharpe_v:.2f}</b><br>Max DD: <b>{mdd_v*100:.1f}%</b></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 6 — PORTFOLIO OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════
with tabs[6]:
    blabel("Optimisation de Portefeuille · Frontière Efficiente · Fama-French")
    port_tabs = st.tabs(["Optimisation","Frontière Efficiente","Factor Model"])

    with port_tabs[0]:
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            blabel("Allocation optimale")
            opt_method = st.radio("Méthode", ["Max Sharpe","Min Volatilité","Equal Weight"], horizontal=True)
            method_map = {"Max Sharpe":"sharpe","Min Volatilité":"min_vol","Equal Weight":"equal"}
            weights = optimize_portfolio(returns_df, method_map[opt_method])
            mu   = returns_df.mean()*252
            cov  = returns_df.cov()*252
            p_ret  = float(weights@mu)
            p_vol  = float(np.sqrt(weights@cov@weights))
            p_sh   = p_ret/(p_vol+1e-9)

            wm1,wm2,wm3 = st.columns(3)
            wm1.metric("Rendement attendu", f"{p_ret*100:.2f}%")
            wm2.metric("Volatilité",         f"{p_vol*100:.2f}%")
            wm3.metric("Sharpe ratio",        f"{p_sh:.3f}")

            wdf = pd.DataFrame({"Ticker":returns_df.columns,"Poids":weights,"Poids %":weights*100})
            wdf = wdf.sort_values("Poids",ascending=False)

            pw1,pw2 = st.columns([1,1])
            with pw1:
                fig_pie = go.Figure(go.Pie(labels=wdf["Ticker"],values=wdf["Poids"],
                    marker_colors=COLORS[:len(wdf)],
                    textinfo="label+percent",hole=0.4,
                    textfont=dict(family="Space Mono",size=11)))
                fig_pie.update_layout(**PLOTLY_BASE,height=340,
                    title=dict(text="◈ Allocation",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_pie, use_container_width=True)
            with pw2:
                fig_wbar = go.Figure(go.Bar(x=wdf["Ticker"],y=wdf["Poids %"],
                    marker_color=COLORS[:len(wdf)],
                    text=[f"{v:.1f}%" for v in wdf["Poids %"]],textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=10)))
                fig_wbar.update_layout(**PLOTLY_BASE,height=340,
                    title=dict(text="◈ Poids par actif",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    yaxis_title="%")
                st.plotly_chart(fig_wbar, use_container_width=True)

            # Performance historique du portefeuille
            blabel("Performance historique du portefeuille optimisé")
            port_returns = (returns_df*weights).sum(axis=1)
            port_equity  = (1+port_returns).cumprod()*100
            fig_port = go.Figure()
            fig_port.add_trace(go.Scatter(x=port_equity.index,y=port_equity,mode="lines",
                line=dict(color="#00d4ff",width=1.5),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.05)",name="Portefeuille"))
            for i,t in enumerate(returns_df.columns):
                eq_t = (1+returns_df[t]).cumprod()*100
                fig_port.add_trace(go.Scatter(x=eq_t.index,y=eq_t,mode="lines",
                    line=dict(color=COLORS[i%len(COLORS)],width=0.7,dash="dot"),name=t))
            fig_port.update_layout(**PLOTLY_BASE,height=360,
                title=dict(text=f"◈ Portefeuille {opt_method} vs actifs individuels",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_port, use_container_width=True)

    with port_tabs[1]:
        blabel("Frontière Efficiente (Monte Carlo)")
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            n_portfolios = st.slider("Nombre de portefeuilles simulés", 500, 5000, 2000, step=500)
            with st.spinner(f"Simulation {n_portfolios} portefeuilles…"):
                np.random.seed(42)
                mu_ef   = returns_df.mean()*252
                cov_ef  = returns_df.cov()*252
                n_assets = len(returns_df.columns)
                rets_sim = []; vols_sim = []; shs_sim = []; ws_sim = []
                for _ in range(n_portfolios):
                    w = np.random.dirichlet(np.ones(n_assets))
                    r_p = float(w@mu_ef)
                    v_p = float(np.sqrt(w@cov_ef@w))
                    rets_sim.append(r_p); vols_sim.append(v_p)
                    shs_sim.append(r_p/v_p); ws_sim.append(w)

            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=np.array(vols_sim)*100,
                y=np.array(rets_sim)*100,mode="markers",
                marker=dict(color=shs_sim,colorscale=[
                    [0,"#ff4466"],[0.5,"#ffaa00"],[1,"#00d4ff"]],
                    size=3,opacity=0.7,showscale=True,
                    colorbar=dict(title="Sharpe",
                                  tickfont=dict(family="Space Mono",color="#8aafc8"))),
                name="Portefeuilles"))
            # actifs individuels
            for i,t in enumerate(returns_df.columns):
                r_i=float(mu_ef[t]); v_i=float(np.sqrt(cov_ef.loc[t,t]))
                fig_ef.add_trace(go.Scatter(x=[v_i*100],y=[r_i*100],mode="markers+text",
                    text=[t],textposition="top center",
                    marker=dict(color=COLORS[i%len(COLORS)],size=10,symbol="diamond"),name=t))
            max_sh_idx = np.argmax(shs_sim)
            fig_ef.add_trace(go.Scatter(x=[vols_sim[max_sh_idx]*100],y=[rets_sim[max_sh_idx]*100],
                mode="markers",marker=dict(color="#00ff88",size=14,symbol="star"),
                name="Max Sharpe"))
            fig_ef.update_layout(**PLOTLY_BASE,height=480,
                title=dict(text="◈ Frontière Efficiente de Markowitz",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                xaxis_title="Volatilité (%)",yaxis_title="Rendement attendu (%)")
            st.plotly_chart(fig_ef, use_container_width=True)

    with port_tabs[2]:
        blabel("Factor Model · Alpha/Beta vs benchmark")
        if not SM_OK:
            st.error("pip install statsmodels")
        else:
            bench = st.selectbox("Benchmark", ["SPY","QQQ","^GSPC","^IXIC"], key="bench_tk")
            df_bench = get_ohlcv(bench, period, "1d")
            if df_bench.empty:
                st.warning(f"Impossible de charger {bench}.")
            else:
                r_bench = df_bench["Close"].squeeze().pct_change().dropna()
                rows_fm = []
                for t in selected:
                    r_t = returns_df[t].dropna() if t in returns_df else pd.Series(dtype=float)
                    if r_t.empty: continue
                    idx_c = r_t.index.intersection(r_bench.index)
                    y_t=r_t[idx_c]; x_t=r_bench[idx_c]
                    slope,intercept,r_val,p_val,_ = stats.linregress(x_t,y_t)
                    rows_fm.append({
                        "Ticker": t,
                        "Alpha (ann.)": f"{intercept*252*100:.2f}%",
                        "Beta": f"{slope:.3f}",
                        "R²": f"{r_val**2:.3f}",
                        "p-value Beta": f"{p_val:.4f}",
                        "Treynor": f"{r_t.mean()*252/slope:.3f}" if slope!=0 else "N/A",
                        "Info Ratio": f"{(r_t-r_bench[idx_c]).mean()/((r_t-r_bench[idx_c]).std()+1e-9)*np.sqrt(252):.3f}",
                    })
                st.dataframe(pd.DataFrame(rows_fm).set_index("Ticker"), use_container_width=True)

                # Beta bar chart
                betas = [float(r["Beta"]) for r in rows_fm]
                tickers_fm = [r["Ticker"] for r in rows_fm]
                fig_beta = go.Figure(go.Bar(x=tickers_fm,y=betas,
                    marker_color=["#00ff88" if b<1 else "#ff4466" for b in betas],
                    text=[f"{b:.3f}" for b in betas],textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=10)))
                fig_beta.add_hline(y=1,line_dash="dot",line_color="#ffaa00",line_width=0.8,
                    annotation_text="Beta = 1",annotation_font_color="#ffaa00")
                fig_beta.update_layout(**PLOTLY_BASE,height=300,
                    title=dict(text=f"◈ Beta vs {bench}",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    yaxis_title="Beta")
                st.plotly_chart(fig_beta, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 7 — MACRO / FRED
# ══════════════════════════════════════════════════════════════════════
with tabs[7]:
    blabel("FRED · Données Macroéconomiques · 800 000+ séries")
    st.markdown("<span class='tag-key'>Clé gratuite : fred.stlouisfed.org</span>", unsafe_allow_html=True)
    FRED_SERIES = {
        "Fed Funds Rate": "FEDFUNDS", "CPI Inflation": "CPIAUCSL",
        "Core PCE": "PCEPILFE", "PIB USA": "GDP", "Chômage": "UNRATE",
        "10Y Treasury": "DGS10", "2Y Treasury": "DGS2", "Spread 10-2Y": "T10Y2Y",
        "VIX": "VIXCLS", "M2 Money Supply": "M2SL", "Jobless Claims": "ICSA",
        "Housing Starts": "HOUST", "Retail Sales": "RSAFS",
    }
    fred_sel = st.multiselect("Séries FRED", list(FRED_SERIES.keys()),
                               default=["Fed Funds Rate","10Y Treasury","Spread 10-2Y","VIX"])
    custom_fred = st.text_input("Série custom (code FRED)", placeholder="ex: SP500")

    if not fred_key:
        st.info("Entrez votre clé FRED gratuite dans la barre latérale. fred.stlouisfed.org/docs/api/api_key.html")
    elif not FRED_OK:
        st.error("pip install fredapi")
    else:
        ids = [FRED_SERIES[l] for l in fred_sel]
        if custom_fred.strip(): ids.append(custom_fred.strip().upper())
        if ids:
            metric_cols = st.columns(min(len(ids),4))
            fig_fred = go.Figure()
            for i,sid in enumerate(ids):
                s = get_fred(sid, fred_key)
                if s.empty: continue
                fig_fred.add_trace(go.Scatter(x=s.index,y=s.values,mode="lines",
                    name=sid,line=dict(color=COLORS[i%len(COLORS)],width=1.5)))
                if i < len(metric_cols):
                    lv = float(s.iloc[-1]); pv = float(s.iloc[-2]) if len(s)>1 else lv
                    metric_cols[i].metric(sid, f"{lv:.2f}", f"{lv-pv:+.3f}")
            fig_fred.update_layout(**PLOTLY_BASE,height=420,
                title=dict(text="◈ Indicateurs Macro — FRED",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_fred, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 8 — FOREX
# ══════════════════════════════════════════════════════════════════════
with tabs[8]:
    blabel("Forex · Taux historiques et live")
    PAIRS = {"EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPY=X",
             "USD/CHF":"CHF=X","AUD/USD":"AUDUSD=X","USD/CAD":"CAD=X",
             "EUR/GBP":"EURGBP=X","EUR/JPY":"EURJPY=X","BTC/USD":"BTC-USD",
             "XAU/USD":"GC=F","WTI":"CL=F"}
    fx1,fx2,fx3 = st.columns([2,2,1])
    sel_pairs = fx1.multiselect("Paires", list(PAIRS.keys()),
                                default=["EUR/USD","GBP/USD","USD/JPY","XAU/USD"])
    fx_period = fx2.selectbox("Période", ["1mo","3mo","6mo","1y","2y"], index=2, key="fx_p")
    norm_fx   = fx3.checkbox("Normaliser", value=True)

    if forex_key and sel_pairs:
        live_cols = st.columns(min(len(sel_pairs),4))
        for i,pair in enumerate(sel_pairs[:4]):
            parts = pair.split("/")
            if len(parts)==2:
                rate, _ = get_forex_rate(parts[0],parts[1],forex_key)
                if rate and i < len(live_cols):
                    live_cols[i].metric(pair, f"{rate:.4f}")

    closes_fx = {}
    for pair in sel_pairs:
        sym = PAIRS[pair]; d = get_ohlcv(sym, fx_period, "1d")
        if not d.empty: closes_fx[pair] = d["Close"].squeeze()

    if closes_fx:
        pdf = pd.DataFrame(closes_fx).dropna()
        plot_df = (pdf/pdf.iloc[0]*100) if norm_fx else pdf
        fig_fx = go.Figure()
        for i,col in enumerate(plot_df.columns):
            fig_fx.add_trace(go.Scatter(x=plot_df.index,y=plot_df[col],mode="lines",
                name=col,line=dict(color=COLORS[i%len(COLORS)],width=1.5)))
        fig_fx.update_layout(**PLOTLY_BASE,height=400,
            title=dict(text=f"◈ Forex {'normalisé' if norm_fx else 'prix'} — {fx_period}",
                       font=dict(family="Orbitron",color="#00d4ff",size=12)),
            hovermode="x unified",yaxis_title="Base 100" if norm_fx else "Prix")
        st.plotly_chart(fig_fx, use_container_width=True)

        # Recap table
        recap = []
        for pair in sel_pairs:
            sym = PAIRS[pair]; d = get_ohlcv(sym, fx_period, "1d")
            if d.empty: continue
            c = d["Close"].squeeze()
            recap.append({"Paire":pair,"Dernier":f"{float(c.iloc[-1]):.4f}",
                "Variation":f"{(float(c.iloc[-1])/float(c.iloc[0])-1)*100:+.2f}%",
                "High":f"{float(d['High'].squeeze().max()):.4f}",
                "Low":f"{float(d['Low'].squeeze().min()):.4f}",
                "Vol 20j σ":f"{float(c.pct_change().rolling(20).std().iloc[-1]*100):.2f}%"})
        st.dataframe(pd.DataFrame(recap).set_index("Paire"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 9 — NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════
with tabs[9]:
    blabel("News & Analyse de Sentiment")
    nc1,nc2 = st.columns([3,1])
    news_query = nc1.text_input("Recherche", value=f"{primary} earnings", key="nq")
    news_days  = nc2.slider("Jours", 1, 30, 7, key="nd")

    if not news_key:
        st.info("Clé NewsAPI gratuite : newsapi.org  |  En attendant, actualités via yfinance :")
        try:
            t_news = yf.Ticker(primary)
            for article in (t_news.news or [])[:8]:
                title = article.get("title",""); link = article.get("link","")
                pub   = article.get("publisher","")
                ts    = article.get("providerPublishTime",0)
                dt    = datetime.fromtimestamp(ts).strftime("%d %b %Y") if ts else ""
                sentiment_html = ""
                if VADER_OK and title:
                    s = sia.polarity_scores(title)["compound"]
                    sentiment_html = ("<span style='color:#00ff88;'>▲ POS</span>" if s>=0.05
                                     else "<span style='color:#ff4466;'>▼ NEG</span>" if s<=-0.05
                                     else "<span style='color:#ffaa00;'>◆ NEU</span>")
                st.markdown(f"""<div class='news-card'>
                  <div class='news-title'><a href='{link}' style='color:#e8f4fd;text-decoration:none;' target='_blank'>{title}</a></div>
                  <div class='news-meta'>{pub} · {dt} · {sentiment_html}</div>
                </div>""", unsafe_allow_html=True)
        except: pass
    else:
        articles, err = get_news_api(news_query, news_key, news_days)
        if err: st.error(f"NewsAPI: {err}")
        elif not articles: st.info("Aucun article.")
        else:
            if VADER_OK:
                scores = [sia.polarity_scores(a.get("title","")+(" "+a.get("description","") or ""))["compound"] for a in articles]
                avg = np.mean(scores)
                sm1,sm2,sm3 = st.columns(3)
                sm1.metric("Sentiment moyen", f"{avg:+.3f}")
                sm2.metric("Positifs", sum(1 for s in scores if s>=0.05))
                sm3.metric("Négatifs", sum(1 for s in scores if s<=-0.05))
                fig_sent = go.Figure(go.Bar(x=list(range(len(scores))),y=scores,
                    marker_color=["#00ff88" if s>=0.05 else "#ff4466" if s<=-0.05
                                  else "#ffaa00" for s in scores],name="Score VADER"))
                fig_sent.add_hline(y=0,line_color="#4a7a99",line_width=0.8)
                fig_sent.update_layout(**PLOTLY_BASE,height=220,
                    title=dict(text="◈ Scores VADER",font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_sent, use_container_width=True)
            for article in articles[:15]:
                title  = article.get("title",""); url = article.get("url","")
                source = article.get("source",{}).get("name",""); pubdt = article.get("publishedAt","")[:10]
                desc   = article.get("description","") or ""
                sh = ""
                if VADER_OK:
                    sc = sia.polarity_scores(title+" "+desc)["compound"]
                    sh = ("▲ POS" if sc>=0.05 else "▼ NEG" if sc<=-0.05 else "◆ NEU")
                    sh = f"<span style='color:{'#00ff88' if sc>=0.05 else '#ff4466' if sc<=-0.05 else '#ffaa00'};'>{sh}</span>"
                st.markdown(f"""<div class='news-card'>
                  <div class='news-title'><a href='{url}' style='color:#e8f4fd;text-decoration:none;' target='_blank'>{title}</a></div>
                  <div class='news-meta'>{source} · {pubdt} · {sh}</div>
                  <div style='font-family:IBM Plex Sans,sans-serif;font-size:12px;color:#7aafc8;margin-top:6px;'>{desc[:200]}{'…' if len(desc)>200 else ''}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 10 — FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════
with tabs[10]:
    blabel("Fondamentaux · Valorisation · Financiers")
    fund_t = st.selectbox("Ticker", selected, key="fund_tk")
    if st.button("↺ Recharger les données", key="reload_fund"):
        st.cache_data.clear()
        st.rerun()
    info = get_info(fund_t)
    if not info or len(info) <= 5:
        st.warning(f"⚠ Données indisponibles pour **{fund_t}**. Yahoo Finance peut être temporairement indisponible.")
        st.info("💡 Cliquez sur **↺ Recharger** ci-dessus, ou attendez 30 secondes et réessayez.")
        st.stop()
    else:
        fc1,fc2,fc3 = st.columns(3)
        with fc1:
            blabel("Entreprise")
            for label, key in [("Nom","longName"),("Secteur","sector"),
                                ("Industrie","industry"),("Pays","country"),("Change","exchange")]:
                st.markdown(f"<span style='color:#4a7a99;font-family:Space Mono,monospace;font-size:10px;'>{label}:</span> <span style='font-family:Space Mono,monospace;font-size:10px;'>{info.get(key,'N/A')}</span>", unsafe_allow_html=True)
            emp = info.get("fullTimeEmployees")
            if isinstance(emp,(int,float)):
                st.markdown(f"<span style='color:#4a7a99;font-family:Space Mono,monospace;font-size:10px;'>Employés:</span> <span style='font-family:Space Mono,monospace;font-size:10px;'>{emp:,.0f}</span>", unsafe_allow_html=True)
        with fc2:
            blabel("Valorisation")
            for k,v in [("Market Cap",fmt_large(info.get("marketCap"))),
                        ("Enterprise Value",fmt_large(info.get("enterpriseValue"))),
                        ("P/E TTM", f"{info.get('trailingPE',0):.2f}" if isinstance(info.get("trailingPE"),float) else "N/A"),
                        ("Fwd P/E", f"{info.get('forwardPE',0):.2f}" if isinstance(info.get("forwardPE"),float) else "N/A"),
                        ("P/B", f"{info.get('priceToBook',0):.2f}" if isinstance(info.get("priceToBook"),float) else "N/A"),
                        ("EV/EBITDA",f"{info.get('enterpriseToEbitda',0):.2f}" if isinstance(info.get("enterpriseToEbitda"),float) else "N/A")]:
                st.markdown(f"<span style='color:#4a7a99;font-family:Space Mono,monospace;font-size:10px;'>{k}:</span> <span style='font-family:Space Mono,monospace;font-size:10px;'>{v}</span>", unsafe_allow_html=True)
        with fc3:
            blabel("Financiers")
            for k,v in [("Revenue",fmt_large(info.get("totalRevenue"))),
                        ("EBITDA",fmt_large(info.get("ebitda"))),
                        ("Net Income",fmt_large(info.get("netIncomeToCommon"))),
                        ("Free Cash Flow",fmt_large(info.get("freeCashflow"))),
                        ("Total Cash",fmt_large(info.get("totalCash"))),
                        ("Total Debt",fmt_large(info.get("totalDebt")))]:
                st.markdown(f"<span style='color:#4a7a99;font-family:Space Mono,monospace;font-size:10px;'>{k}:</span> <span style='font-family:Space Mono,monospace;font-size:10px;'>{v}</span>", unsafe_allow_html=True)

        st.divider()
        blabel("Marges & Croissance")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Gross Margin",  pct_fmt(info.get("grossMargins")))
        m2.metric("Op Margin",     pct_fmt(info.get("operatingMargins")))
        m3.metric("Profit Margin", pct_fmt(info.get("profitMargins")))
        m4.metric("Rev. Growth",   pct_fmt(info.get("revenueGrowth")))
        m5.metric("Earn. Growth",  pct_fmt(info.get("earningsGrowth")))
        m6.metric("ROE",           pct_fmt(info.get("returnOnEquity")))

        summary = info.get("longBusinessSummary","")
        if summary:
            with st.expander("▸ Description"):
                st.markdown(f"<p style='font-family:IBM Plex Sans,sans-serif;font-size:13px;color:#8aafc8;line-height:1.7;'>{summary}</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 11 — CORRELATION
# ══════════════════════════════════════════════════════════════════════
with tabs[11]:
    blabel("Matrice de Corrélation · Performance · Distribution")
    if len(selected) < 2:
        st.info("Sélectionnez ≥ 2 tickers.")
    else:
        corr = returns_df.corr()

        cc1,cc2 = st.columns(2)
        with cc1:
            blabel("Heatmap de corrélation")
            fig_hm = go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale=[[0,"#ff4466"],[0.25,"#aa2244"],[0.5,"#0d1f30"],
                             [0.75,"#004488"],[1,"#00d4ff"]],
                zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",textfont=dict(family="Space Mono",size=11,color="#e8f4fd"),
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ=%{z:.4f}<extra></extra>",
                colorbar=dict(tickfont=dict(family="Space Mono",color="#8aafc8"))))
            fig_hm.update_layout(**PLOTLY_BASE,height=420,
                title=dict(text="◈ Corrélation rendements journaliers",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)))
            st.plotly_chart(fig_hm, use_container_width=True)

        with cc2:
            blabel("RSI comparatif")
            rsi_vals=[]; rsi_tks=[]
            for t in selected:
                if t in returns_df:
                    r_t = returns_df[t].dropna()
                    c_t = price_df[t].dropna() if t in price_df else pd.Series(dtype=float)
                    if len(c_t)>=14:
                        delta_t=c_t.diff(); gain_t=delta_t.clip(lower=0).rolling(14).mean()
                        loss_t=(-delta_t.clip(upper=0)).rolling(14).mean()
                        rsi_t=100-100/(1+gain_t/loss_t.replace(0,np.nan))
                        v=float(rsi_t.iloc[-1])
                        if not np.isnan(v): rsi_vals.append(v); rsi_tks.append(t)
            if rsi_vals:
                fig_rsi_bar = go.Figure(go.Bar(x=rsi_tks,y=rsi_vals,
                    marker_color=["#ff4466" if v>70 else "#00ff88" if v<30 else "#00d4ff" for v in rsi_vals],
                    text=[f"{v:.1f}" for v in rsi_vals],textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=10)))
                fig_rsi_bar.add_hline(y=70,line_dash="dot",line_color="#ff4466",line_width=0.8)
                fig_rsi_bar.add_hline(y=30,line_dash="dot",line_color="#00ff88",line_width=0.8)
                fig_rsi_bar.update_layout(**PLOTLY_BASE,height=420,
                    title=dict(text="◈ RSI(14) comparatif",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    showlegend=False)
                fig_rsi_bar.update_yaxes(range=[0,100],title_text="RSI")
                st.plotly_chart(fig_rsi_bar, use_container_width=True)

        blabel("Performance normalisée (Base 100)")
        norm_df = (price_df/price_df.iloc[0]*100)
        fig_norm = go.Figure()
        for i,col in enumerate(norm_df.columns):
            fig_norm.add_trace(go.Scatter(x=norm_df.index,y=norm_df[col],mode="lines",
                name=col,line=dict(color=COLORS[i%len(COLORS)],width=1.5)))
        fig_norm.update_layout(**PLOTLY_BASE,height=360,
            title=dict(text="◈ Performance normalisée",
                       font=dict(family="Orbitron",color="#00d4ff",size=12)),
            hovermode="x unified",yaxis_title="Base 100")
        st.plotly_chart(fig_norm, use_container_width=True)

        blabel("Distribution des rendements")
        fig_violin = go.Figure()
        for i,col in enumerate(returns_df.columns):
            fig_violin.add_trace(go.Violin(y=returns_df[col]*100,name=col,
                box_visible=True,meanline_visible=True,
                line_color=COLORS[i%len(COLORS)],opacity=0.6))
        fig_violin.update_layout(**PLOTLY_BASE,height=360,
            title=dict(text="◈ Distribution rendements journaliers (%)",
                       font=dict(family="Orbitron",color="#00d4ff",size=12)),
            yaxis_title="%",violinmode="overlay")
        st.plotly_chart(fig_violin, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 12 — HEATMAPS
# ══════════════════════════════════════════════════════════════════════
with tabs[12]:
    blabel("Heatmaps · Corrélation · Calendaire · Volatilité · Secteurs")
    hm_tabs = st.tabs([
        "🔗  Corrélation",
        "📅  Calendaire",
        "📉  Volatilité",
        "🏭  Secteurs",
    ])

    # ── shared data ───────────────────────────────────────────────────
    hm_closes = {}
    for t in selected:
        d = get_ohlcv(t, "1y", "1d")
        if not d.empty:
            s = d["Close"].squeeze()
            if isinstance(s, pd.Series) and len(s) > 1:
                hm_closes[t] = s
    if len(hm_closes) >= 1:
        hm_price_df   = pd.DataFrame(hm_closes).dropna()
        hm_returns_df = hm_price_df.pct_change().dropna()
    else:
        hm_price_df   = pd.DataFrame()
        hm_returns_df = pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════
    #  HEATMAP 1 — CORRELATION
    # ══════════════════════════════════════════════════════════════════
    with hm_tabs[0]:
        blabel("Matrice de corrélation des rendements journaliers")
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            hc1, hc2 = st.columns([2,1])
            with hc1:
                corr_method = st.radio("Méthode", ["Pearson","Spearman","Kendall"],
                                       horizontal=True, key="corr_m")
            with hc2:
                corr_period = st.selectbox("Fenêtre glissante (jours)",
                                           [None,30,60,90,120,180],
                                           format_func=lambda x: "Complète" if x is None else f"{x}j",
                                           key="corr_p")

            if corr_period:
                corr_data = hm_returns_df.tail(corr_period)
            else:
                corr_data = hm_returns_df

            if corr_method == "Pearson":
                corr_mx = corr_data.corr(method="pearson")
            elif corr_method == "Spearman":
                corr_mx = corr_data.corr(method="spearman")
            else:
                corr_mx = corr_data.corr(method="kendall")

            # Annotations avec valeur + significance stars
            annot = []
            for i, row in enumerate(corr_mx.index):
                row_ann = []
                for j, col in enumerate(corr_mx.columns):
                    v = corr_mx.loc[row, col]
                    if i == j:
                        row_ann.append("1.00")
                    else:
                        # t-test significance
                        n = len(corr_data)
                        t_stat = v * np.sqrt(n-2) / np.sqrt(max(1-v**2, 1e-9))
                        p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
                        stars  = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else ""
                        row_ann.append(f"{v:.2f}{stars}")
                annot.append(row_ann)

            fig_corr = go.Figure(go.Heatmap(
                z=corr_mx.values,
                x=corr_mx.columns.tolist(),
                y=corr_mx.index.tolist(),
                colorscale=[[0,"#ff4466"],[0.25,"#aa1133"],[0.5,"#0d1f30"],
                            [0.75,"#003388"],[1,"#00d4ff"]],
                zmid=0, zmin=-1, zmax=1,
                text=annot,
                texttemplate="%{text}",
                textfont=dict(family="Space Mono", size=11, color="#e8f4fd"),
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ = %{z:.4f}<extra></extra>",
                colorbar=dict(
                    title=dict(text="ρ", font=dict(color="#4da8da")),
                    tickfont=dict(family="Space Mono", color="#8aafc8"),
                    tickvals=[-1,-0.5,0,0.5,1],
                ),
            ))
            fig_corr.update_layout(**PLOTLY_BASE, height=500,
                title=dict(text=f"◈ Corrélation {corr_method} {'('+str(corr_period)+'j)' if corr_period else '(complète)'}  · * p<0.05  ** p<0.01  *** p<0.001",
                           font=dict(family="Orbitron", color="#00d4ff", size=12)))
            st.plotly_chart(fig_corr, use_container_width=True)

            # Rolling correlation between 2 assets
            if len(selected) >= 2:
                st.markdown(""); blabel("Corrélation glissante entre deux actifs")
                rc1, rc2, rc3 = st.columns(3)
                ra = rc1.selectbox("Actif A", selected, key="hm_ra")
                rb = rc2.selectbox("Actif B", [t for t in selected if t != ra], key="hm_rb")
                rw = rc3.slider("Fenêtre (jours)", 10, 120, 30, key="hm_rw")
                if ra in hm_returns_df and rb in hm_returns_df:
                    roll_corr = hm_returns_df[ra].rolling(rw).corr(hm_returns_df[rb])
                    fig_rc = go.Figure()
                    fig_rc.add_trace(go.Scatter(
                        x=roll_corr.index, y=roll_corr,
                        mode="lines", line=dict(color="#00d4ff", width=1.5),
                        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
                        name=f"Corr {ra}/{rb}"))
                    fig_rc.add_hline(y=0,  line_dash="dot", line_color="#4a7a99", line_width=0.8)
                    fig_rc.add_hline(y=0.7, line_dash="dot", line_color="#ff4466", line_width=0.7,
                                     annotation_text="Forte +", annotation_font_color="#ff4466")
                    fig_rc.add_hline(y=-0.7, line_dash="dot", line_color="#00ff88", line_width=0.7,
                                     annotation_text="Forte -", annotation_font_color="#00ff88")
                    fig_rc.update_layout(**PLOTLY_BASE, height=300,
                        title=dict(text=f"◈ Corrélation glissante {rw}j — {ra} vs {rb}",
                                   font=dict(family="Orbitron", color="#00d4ff", size=11)),
                        hovermode="x unified")
                    fig_rc.update_yaxes(range=[-1, 1], title_text="ρ")
                    st.plotly_chart(fig_rc, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    #  HEATMAP 2 — CALENDAIRE
    # ══════════════════════════════════════════════════════════════════
    with hm_tabs[1]:
        blabel("Performance calendaire · Rendements par mois et par année")
        cal_t = st.selectbox("Ticker", selected, key="cal_tk")

        df_cal = get_ohlcv(cal_t, "5y", "1d")
        if df_cal.empty:
            st.warning("Données insuffisantes.")
        else:
            r_cal = df_cal["Close"].squeeze().pct_change().dropna()
            r_cal.index = pd.to_datetime(r_cal.index)

            # ── Monthly returns heatmap ───────────────────────────────
            monthly = r_cal.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
            monthly_df = pd.DataFrame({
                "Year":  monthly.index.year,
                "Month": monthly.index.month,
                "Return": monthly.values,
            })
            pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Return")
            pivot.columns = ["Jan","Fév","Mar","Avr","Mai","Jun",
                             "Jul","Aoû","Sep","Oct","Nov","Déc"]

            # text annotations
            text_ann = [[f"{v:+.1f}%" if not np.isnan(v) else ""
                         for v in row] for row in pivot.values]

            max_abs = max(abs(pivot.values[~np.isnan(pivot.values)].max()),
                          abs(pivot.values[~np.isnan(pivot.values)].min()), 5)

            fig_cal = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index.tolist()],
                colorscale=[[0,"#ff4466"],[0.35,"#661122"],[0.5,"#0d1f30"],
                            [0.65,"#113366"],[1,"#00d4ff"]],
                zmid=0, zmin=-max_abs, zmax=max_abs,
                text=text_ann,
                texttemplate="%{text}",
                textfont=dict(family="Space Mono", size=10, color="#e8f4fd"),
                hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
                colorbar=dict(
                    title=dict(text="%", font=dict(color="#4da8da")),
                    tickfont=dict(family="Space Mono", color="#8aafc8"),
                ),
            ))
            fig_cal.update_layout(**PLOTLY_BASE, height=max(350, len(pivot)*40+100),
                title=dict(text=f"◈ Rendements mensuels — {cal_t} (%)",
                           font=dict(family="Orbitron", color="#00d4ff", size=12)))
            st.plotly_chart(fig_cal, use_container_width=True)

            # ── Annual summary bar ────────────────────────────────────
            blabel("Rendements annuels")
            annual = r_cal.resample("YE").apply(lambda x: (1+x).prod()-1) * 100
            fig_ann = go.Figure(go.Bar(
                x=[str(y.year) for y in annual.index],
                y=annual.values,
                marker_color=["#00ff88" if v >= 0 else "#ff4466" for v in annual.values],
                text=[f"{v:+.1f}%" for v in annual.values],
                textposition="outside",
                textfont=dict(family="Space Mono", color="#e8f4fd", size=10),
            ))
            fig_ann.add_hline(y=0, line_color="#4a7a99", line_width=0.8)
            fig_ann.update_layout(**PLOTLY_BASE, height=300,
                title=dict(text=f"◈ Rendements annuels — {cal_t}",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                yaxis_title="%")
            st.plotly_chart(fig_ann, use_container_width=True)

            # ── Day-of-week heatmap ───────────────────────────────────
            blabel("Rendement moyen par jour de la semaine × heure (si données intraday)")
            dow_returns = r_cal.copy()
            dow_returns.index = pd.to_datetime(dow_returns.index)
            dow_avg = dow_returns.groupby(dow_returns.index.day_of_week).mean() * 100
            days_map = {0:"Lun",1:"Mar",2:"Mer",3:"Jeu",4:"Ven",5:"Sam",6:"Dim"}
            dow_avg.index = [days_map.get(i, str(i)) for i in dow_avg.index]

            fig_dow = go.Figure(go.Bar(
                x=dow_avg.index.tolist(),
                y=dow_avg.values,
                marker_color=["#00ff88" if v >= 0 else "#ff4466" for v in dow_avg.values],
                text=[f"{v:+.3f}%" for v in dow_avg.values],
                textposition="outside",
                textfont=dict(family="Space Mono", color="#e8f4fd", size=10),
            ))
            fig_dow.add_hline(y=0, line_color="#4a7a99", line_width=0.8)
            fig_dow.update_layout(**PLOTLY_BASE, height=280,
                title=dict(text=f"◈ Rendement moyen par jour — {cal_t}",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                yaxis_title="Rend. moy. (%)")
            st.plotly_chart(fig_dow, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    #  HEATMAP 3 — VOLATILITY
    # ══════════════════════════════════════════════════════════════════
    with hm_tabs[2]:
        blabel("Heatmap de volatilité · Réalisée vs Implicite")
        if hm_returns_df.empty:
            st.warning("Données insuffisantes.")
        else:
            vc1, vc2 = st.columns(2)
            vol_window = vc1.slider("Fenêtre vol. réalisée (jours)", 5, 60, 21, key="vol_w")
            vol_ann    = vc2.checkbox("Annualiser (×√252)", value=True, key="vol_ann")

            mult = np.sqrt(252) if vol_ann else 1.0

            # Rolling vol for all tickers
            vol_df = hm_returns_df.rolling(vol_window).std() * mult * 100
            vol_df = vol_df.dropna()

            if not vol_df.empty:
                # ── Rolling vol heatmap over time ─────────────────────
                # Resample to weekly for readability
                vol_weekly = vol_df.resample("W").mean()

                fig_vhm = go.Figure(go.Heatmap(
                    z=vol_weekly.T.values,
                    x=[str(d.date()) for d in vol_weekly.index],
                    y=vol_weekly.columns.tolist(),
                    colorscale=[[0,"#070b10"],[0.3,"#003388"],[0.6,"#ffaa00"],[1,"#ff4466"]],
                    hovertemplate="<b>%{y}</b><br>%{x}<br>Vol: %{z:.1f}%<extra></extra>",
                    colorbar=dict(
                        title=dict(text="Vol%", font=dict(color="#4da8da")),
                        tickfont=dict(family="Space Mono", color="#8aafc8"),
                    ),
                ))
                fig_vhm.update_layout(**PLOTLY_BASE, height=max(300, len(selected)*60+100),
                    title=dict(text=f"◈ Volatilité réalisée {vol_window}j {'annualisée' if vol_ann else ''} — hebdomadaire",
                               font=dict(family="Orbitron", color="#00d4ff", size=12)))
                st.plotly_chart(fig_vhm, use_container_width=True)

                # ── Current vol snapshot ──────────────────────────────
                blabel("Snapshot volatilité courante")
                last_vols = vol_df.iloc[-1].sort_values(ascending=False)
                avg_vols  = vol_df.mean().reindex(last_vols.index)
                fig_vsnap = go.Figure()
                fig_vsnap.add_trace(go.Bar(
                    x=last_vols.index.tolist(), y=last_vols.values,
                    name="Vol actuelle",
                    marker_color=["#ff4466" if v > avg_vols[t]*1.2
                                  else "#ffaa00" if v > avg_vols[t]
                                  else "#00ff88"
                                  for t, v in zip(last_vols.index, last_vols.values)],
                    text=[f"{v:.1f}%" for v in last_vols.values],
                    textposition="outside",
                    textfont=dict(family="Space Mono", color="#e8f4fd", size=10),
                ))
                fig_vsnap.add_trace(go.Scatter(
                    x=avg_vols.reindex(last_vols.index).index.tolist(),
                    y=avg_vols.reindex(last_vols.index).values,
                    mode="markers",
                    marker=dict(color="#00d4ff", size=10, symbol="diamond"),
                    name="Vol moyenne (période)",
                ))
                fig_vsnap.update_layout(**PLOTLY_BASE, height=320,
                    title=dict(text="◈ Vol actuelle vs moyenne  (rouge=au-dessus moy, vert=en-dessous)",
                               font=dict(family="Orbitron", color="#00d4ff", size=11)),
                    yaxis_title="Volatilité (%)")
                st.plotly_chart(fig_vsnap, use_container_width=True)

                # ── Vol correlation heatmap ───────────────────────────
                blabel("Corrélation des volatilités")
                vol_corr = vol_df.corr()
                fig_vcorr = go.Figure(go.Heatmap(
                    z=vol_corr.values,
                    x=vol_corr.columns.tolist(),
                    y=vol_corr.index.tolist(),
                    colorscale=[[0,"#ff4466"],[0.5,"#0d1f30"],[1,"#00d4ff"]],
                    zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in vol_corr.values],
                    texttemplate="%{text}",
                    textfont=dict(family="Space Mono", size=11, color="#e8f4fd"),
                    colorbar=dict(tickfont=dict(family="Space Mono", color="#8aafc8")),
                ))
                fig_vcorr.update_layout(**PLOTLY_BASE, height=400,
                    title=dict(text="◈ Corrélation des volatilités réalisées",
                               font=dict(family="Orbitron", color="#00d4ff", size=11)))
                st.plotly_chart(fig_vcorr, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    #  HEATMAP 4 — SECTOR
    # ══════════════════════════════════════════════════════════════════
    with hm_tabs[3]:
        blabel("Heatmap sectorielle · Performance S&P 500 par secteur")

        SECTOR_ETFS = {
            "Tech (XLK)":         "XLK",
            "Finance (XLF)":      "XLF",
            "Santé (XLV)":        "XLV",
            "Conso. Disc. (XLY)": "XLY",
            "Conso. Base (XLP)":  "XLP",
            "Énergie (XLE)":      "XLE",
            "Industrie (XLI)":    "XLI",
            "Matériaux (XLB)":    "XLB",
            "Immobilier (XLRE)":  "XLRE",
            "Services pub. (XLU)":"XLU",
            "Comm. (XLC)":        "XLC",
        }

        SECTOR_STOCKS = {
            "Tech":        ["AAPL","MSFT","NVDA","GOOGL","META","AVGO","ORCL","CRM","AMD","INTC"],
            "Finance":     ["JPM","BAC","WFC","GS","MS","BLK","AXP","SPGI","MCO","ICE"],
            "Santé":       ["JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","ABT","MDT","AMGN"],
            "Énergie":     ["XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO","PSX","HAL"],
            "Industrie":   ["CAT","DE","HON","UPS","RTX","LMT","BA","GE","MMM","FDX"],
            "Conso. Disc.":["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TGT","BKNG","TJX"],
            "Conso. Base": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KHC"],
        }

        sc1, sc2 = st.columns(2)
        sector_period = sc1.selectbox("Période", ["1sem","1mois","3mois","6mois","1an"],
                                      index=2, key="sec_p")
        sector_view   = sc2.radio("Vue", ["ETF sectoriels","Titres par secteur"],
                                  horizontal=True, key="sec_v")

        period_yf_map = {"1sem":"7d","1mois":"1mo","3mois":"3mo","6mois":"6mo","1an":"1y"}
        yf_p = period_yf_map[sector_period]

        if sector_view == "ETF sectoriels":
            blabel(f"Performance ETF sectoriels — {sector_period}")
            with st.spinner("Chargement ETF sectoriels…"):
                sector_rets = {}
                for name, sym in SECTOR_ETFS.items():
                    d = get_ohlcv(sym, yf_p, "1d")
                    if not d.empty:
                        c = d["Close"].squeeze()
                        sector_rets[name] = float((c.iloc[-1]/c.iloc[0]-1)*100)

            if sector_rets:
                sr_df = pd.Series(sector_rets).sort_values(ascending=False)

                # Treemap-style bar
                fig_sec = go.Figure(go.Bar(
                    x=sr_df.values,
                    y=sr_df.index.tolist(),
                    orientation="h",
                    marker_color=["#00ff88" if v >= 0 else "#ff4466" for v in sr_df.values],
                    text=[f"{v:+.2f}%" for v in sr_df.values],
                    textposition="auto",
                    textfont=dict(family="Space Mono", color="#e8f4fd", size=10),
                ))
                fig_sec.add_vline(x=0, line_color="#4a7a99", line_width=0.8)
                fig_sec.update_layout(**PLOTLY_BASE, height=420,
                    title=dict(text=f"◈ Performance sectorielle — {sector_period}",
                               font=dict(family="Orbitron", color="#00d4ff", size=12)),
                    xaxis_title="Performance (%)")
                st.plotly_chart(fig_sec, use_container_width=True)

                # Heatmap: secteurs × sous-périodes
                blabel("Heatmap sectorielle multi-périodes")
                with st.spinner("Calcul multi-périodes…"):
                    multi_periods = {"1sem":"7d","1mois":"1mo","3mois":"3mo","6mois":"6mo","1an":"1y"}
                    heat_data = {}
                    for name, sym in SECTOR_ETFS.items():
                        row = {}
                        for plabel_m, pyf_m in multi_periods.items():
                            d = get_ohlcv(sym, pyf_m, "1d")
                            if not d.empty:
                                c = d["Close"].squeeze()
                                row[plabel_m] = float((c.iloc[-1]/c.iloc[0]-1)*100)
                            else:
                                row[plabel_m] = np.nan
                        heat_data[name] = row
                    heat_df = pd.DataFrame(heat_data).T

                text_hd = [[f"{v:+.1f}%" if not np.isnan(v) else "N/A"
                            for v in row] for row in heat_df.values]
                fig_mhm = go.Figure(go.Heatmap(
                    z=heat_df.values,
                    x=heat_df.columns.tolist(),
                    y=heat_df.index.tolist(),
                    colorscale=[[0,"#ff4466"],[0.4,"#661122"],[0.5,"#0d1f30"],
                                [0.6,"#113366"],[1,"#00d4ff"]],
                    zmid=0,
                    text=text_hd,
                    texttemplate="%{text}",
                    textfont=dict(family="Space Mono", size=10, color="#e8f4fd"),
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>",
                    colorbar=dict(
                        title=dict(text="%", font=dict(color="#4da8da")),
                        tickfont=dict(family="Space Mono", color="#8aafc8"),
                    ),
                ))
                fig_mhm.update_layout(**PLOTLY_BASE, height=480,
                    title=dict(text="◈ Heatmap sectorielle multi-périodes",
                               font=dict(family="Orbitron", color="#00d4ff", size=12)))
                st.plotly_chart(fig_mhm, use_container_width=True)

        else:
            blabel(f"Performance titres par secteur — {sector_period}")
            sel_sector = st.selectbox("Secteur", list(SECTOR_STOCKS.keys()), key="sec_sel")
            stocks = SECTOR_STOCKS[sel_sector]

            with st.spinner(f"Chargement {sel_sector}…"):
                stock_rets = {}
                for sym in stocks:
                    d = get_ohlcv(sym, yf_p, "1d")
                    if not d.empty:
                        c = d["Close"].squeeze()
                        stock_rets[sym] = float((c.iloc[-1]/c.iloc[0]-1)*100)

            if stock_rets:
                sr2 = pd.Series(stock_rets).sort_values(ascending=False)
                max_abs2 = max(abs(sr2.max()), abs(sr2.min()), 1)

                # Grid heatmap (1 row × N cols)
                fig_stk = go.Figure(go.Heatmap(
                    z=[sr2.values],
                    x=sr2.index.tolist(),
                    y=[sel_sector],
                    colorscale=[[0,"#ff4466"],[0.5,"#0d1f30"],[1,"#00d4ff"]],
                    zmid=0, zmin=-max_abs2, zmax=max_abs2,
                    text=[[f"{v:+.1f}%" for v in sr2.values]],
                    texttemplate="%{text}",
                    textfont=dict(family="Space Mono", size=12, color="#e8f4fd"),
                    colorbar=dict(tickfont=dict(family="Space Mono", color="#8aafc8")),
                ))
                fig_stk.update_layout(**PLOTLY_BASE, height=200,
                    title=dict(text=f"◈ {sel_sector} — Performance {sector_period}",
                               font=dict(family="Orbitron", color="#00d4ff", size=12)))
                st.plotly_chart(fig_stk, use_container_width=True)

                # Bar chart détaillé
                fig_stk2 = go.Figure(go.Bar(
                    x=sr2.index.tolist(), y=sr2.values,
                    marker_color=["#00ff88" if v >= 0 else "#ff4466" for v in sr2.values],
                    text=[f"{v:+.2f}%" for v in sr2.values],
                    textposition="outside",
                    textfont=dict(family="Space Mono", color="#e8f4fd", size=10),
                ))
                fig_stk2.add_hline(y=0, line_color="#4a7a99", line_width=0.8)
                fig_stk2.update_layout(**PLOTLY_BASE, height=340,
                    title=dict(text=f"◈ {sel_sector} — Titres individuels",
                               font=dict(family="Orbitron", color="#00d4ff", size=11)),
                    yaxis_title="Performance (%)")
                st.plotly_chart(fig_stk2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 13 — EARNINGS CALENDAR
# ══════════════════════════════════════════════════════════════════════
with tabs[13]:
    blabel("Earnings Calendar · Résultats & Prévisions · Surprise EPS")

    earn_tabs = st.tabs(["📋  Calendrier & Historique", "📊  Analyse Surprise EPS", "📈  Réaction du marché"])

    with earn_tabs[0]:
        blabel("Historique des résultats trimestriels")
        earn_t = st.selectbox("Ticker", selected, key="earn_tk")
        info_e = get_info(earn_t)
        tk_obj = yf.Ticker(earn_t)

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("EPS TTM",       f"${info_e.get('trailingEps', 0):.2f}"   if isinstance(info_e.get('trailingEps'), float) else "N/A")
        col_e2.metric("EPS Fwd",       f"${info_e.get('forwardEps', 0):.2f}"    if isinstance(info_e.get('forwardEps'), float) else "N/A")
        col_e3.metric("Croiss. EPS",   f"{info_e.get('earningsGrowth',0)*100:.1f}%" if isinstance(info_e.get('earningsGrowth'), float) else "N/A")
        col_e4.metric("Prochaine date",str(info_e.get('earningsTimestamp', 'N/A'))[:10] if info_e.get('earningsTimestamp') else "N/A")

        try:
            earnings_hist = tk_obj.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                blabel("Historique EPS — Estimé vs Réel")
                eh = earnings_hist.copy()
                # Normalize columns
                cols_map = {}
                for c in eh.columns:
                    cl = c.lower()
                    if 'actual' in cl: cols_map[c] = 'EPS Réel'
                    elif 'estimate' in cl: cols_map[c] = 'EPS Estimé'
                    elif 'surprise' in cl and '%' in cl: cols_map[c] = 'Surprise %'
                    elif 'surprise' in cl: cols_map[c] = 'Surprise $'
                eh = eh.rename(columns=cols_map)

                if 'EPS Réel' in eh.columns and 'EPS Estimé' in eh.columns:
                    fig_eps = go.Figure()
                    fig_eps.add_trace(go.Bar(
                        x=[str(i)[:10] for i in eh.index],
                        y=eh['EPS Estimé'],
                        name='EPS Estimé',
                        marker_color='#4a7a99',
                        opacity=0.8,
                    ))
                    fig_eps.add_trace(go.Bar(
                        x=[str(i)[:10] for i in eh.index],
                        y=eh['EPS Réel'],
                        name='EPS Réel',
                        marker_color=['#00ff88' if r >= e else '#ff4466'
                                      for r, e in zip(eh['EPS Réel'].fillna(0),
                                                       eh['EPS Estimé'].fillna(0))],
                        opacity=0.9,
                    ))
                    fig_eps.update_layout(**PLOTLY_BASE, height=340, barmode='group',
                        title=dict(text=f"◈ EPS Trimestriel — {earn_t}",
                                   font=dict(family="Orbitron", color="#00d4ff", size=12)),
                        hovermode="x unified")
                    st.plotly_chart(fig_eps, use_container_width=True)
                st.dataframe(eh, use_container_width=True)
            else:
                st.info("Historique EPS non disponible pour ce ticker.")
        except Exception as ex:
            st.info(f"Données earnings non disponibles : {ex}")

        # Next earnings + consensus
        blabel("Recommandations analystes")
        try:
            reco = tk_obj.recommendations
            if reco is not None and not reco.empty:
                reco_recent = reco.tail(20).copy()
                st.dataframe(reco_recent, use_container_width=True, height=250)

                # Consensus pie
                if 'To Grade' in reco_recent.columns:
                    grade_counts = reco_recent['To Grade'].value_counts()
                    fig_pie_r = go.Figure(go.Pie(
                        labels=grade_counts.index.tolist(),
                        values=grade_counts.values,
                        marker_colors=COLORS[:len(grade_counts)],
                        hole=0.4,
                        textfont=dict(family="Space Mono", size=10),
                    ))
                    fig_pie_r.update_layout(**PLOTLY_BASE, height=300,
                        title=dict(text="◈ Consensus analystes",
                                   font=dict(family="Orbitron", color="#00d4ff", size=11)))
                    st.plotly_chart(fig_pie_r, use_container_width=True)
        except:
            st.info("Recommandations non disponibles.")

    with earn_tabs[1]:
        blabel("Analyse de la surprise EPS · Impact sur le cours")
        earn_t2 = st.selectbox("Ticker", selected, key="earn_tk2")
        try:
            tk2 = yf.Ticker(earn_t2)
            eh2 = tk2.earnings_history
            if eh2 is not None and not eh2.empty:
                eh2 = eh2.copy()
                cols_map2 = {}
                for c in eh2.columns:
                    cl = c.lower()
                    if 'actual' in cl: cols_map2[c] = 'actual'
                    elif 'estimate' in cl: cols_map2[c] = 'estimate'
                    elif 'surprise' in cl and '%' in cl: cols_map2[c] = 'surprise_pct'
                eh2 = eh2.rename(columns=cols_map2)

                if 'actual' in eh2.columns and 'estimate' in eh2.columns:
                    eh2['surprise_pct_calc'] = ((eh2['actual'] - eh2['estimate']) /
                                                 eh2['estimate'].abs().replace(0, np.nan) * 100)
                    beat_count = (eh2['surprise_pct_calc'] > 0).sum()
                    miss_count = (eh2['surprise_pct_calc'] < 0).sum()
                    avg_surp   = eh2['surprise_pct_calc'].mean()

                    s1,s2,s3,s4 = st.columns(4)
                    s1.metric("Beats",        str(int(beat_count)))
                    s2.metric("Misses",       str(int(miss_count)))
                    s3.metric("Taux de beat", f"{beat_count/(beat_count+miss_count)*100:.0f}%")
                    s4.metric("Surprise moy.", f"{avg_surp:+.1f}%")

                    fig_surp = go.Figure(go.Bar(
                        x=[str(i)[:10] for i in eh2.index],
                        y=eh2['surprise_pct_calc'],
                        marker_color=['#00ff88' if v >= 0 else '#ff4466'
                                      for v in eh2['surprise_pct_calc'].fillna(0)],
                        text=[f"{v:+.1f}%" for v in eh2['surprise_pct_calc'].fillna(0)],
                        textposition="outside",
                        textfont=dict(family="Space Mono", color="#e8f4fd", size=9),
                        name="Surprise EPS %",
                    ))
                    fig_surp.add_hline(y=0, line_color="#4a7a99", line_width=0.8)
                    fig_surp.update_layout(**PLOTLY_BASE, height=320,
                        title=dict(text=f"◈ Surprise EPS (%) — {earn_t2}",
                                   font=dict(family="Orbitron", color="#00d4ff", size=11)),
                        yaxis_title="Surprise (%)")
                    st.plotly_chart(fig_surp, use_container_width=True)
        except Exception as ex:
            st.info(f"Données non disponibles : {ex}")

    with earn_tabs[2]:
        blabel("Réaction du marché autour des earnings (Event Study)")
        earn_t3 = st.selectbox("Ticker", selected, key="earn_tk3")
        window  = st.slider("Fenêtre ±jours", 1, 10, 5)
        try:
            tk3 = yf.Ticker(earn_t3)
            eh3 = tk3.earnings_history
            df_ev = get_ohlcv(earn_t3, "2y", "1d")
            if (eh3 is not None and not eh3.empty and not df_ev.empty):
                close_ev = df_ev["Close"].squeeze()
                ret_ev   = close_ev.pct_change()
                dates    = [pd.Timestamp(str(d)[:10]) for d in eh3.index if pd.Timestamp(str(d)[:10]) in ret_ev.index]
                event_rets = []
                for d in dates[-8:]:
                    try:
                        idx = ret_ev.index.get_loc(d)
                        start = max(0, idx-window); end = min(len(ret_ev), idx+window+1)
                        window_r = ret_ev.iloc[start:end].values
                        if len(window_r) == 2*window+1:
                            event_rets.append(window_r)
                    except: pass
                if event_rets:
                    avg_path = np.mean(event_rets, axis=0) * 100
                    x_axis   = list(range(-window, window+1))
                    fig_ev   = go.Figure()
                    for i, path in enumerate(event_rets):
                        fig_ev.add_trace(go.Scatter(
                            x=x_axis, y=np.array(path)*100, mode="lines",
                            line=dict(color="rgba(0,212,255,0.2)", width=1),
                            showlegend=False))
                    fig_ev.add_trace(go.Scatter(
                        x=x_axis, y=avg_path, mode="lines+markers",
                        line=dict(color="#ffaa00", width=2.5),
                        marker=dict(size=6), name="Moy. événements"))
                    fig_ev.add_vline(x=0, line_dash="dot", line_color="#ff4466",
                                     annotation_text="Earnings Day",
                                     annotation_font_color="#ff4466")
                    fig_ev.add_hline(y=0, line_color="#4a7a99", line_width=0.8)
                    fig_ev.update_layout(**PLOTLY_BASE, height=380,
                        title=dict(text=f"◈ Event Study Earnings — {earn_t3} (±{window}j)",
                                   font=dict(family="Orbitron", color="#00d4ff", size=12)),
                        xaxis_title="Jours relatifs à l'annonce",
                        yaxis_title="Rendement (%)", hovermode="x unified")
                    st.plotly_chart(fig_ev, use_container_width=True)
                else:
                    st.info("Pas assez d'événements pour l'event study.")
        except Exception as ex:
            st.info(f"Event study indisponible : {ex}")

# ══════════════════════════════════════════════════════════════════════
#  TAB 14 — LIQUIDITÉ
# ══════════════════════════════════════════════════════════════════════
with tabs[14]:
    blabel("Analyse de Liquidité · Bid-Ask · Volume · Microstructure")

    liq_tabs = st.tabs(["📊  Métriques Liquidité", "📉  Spread & Coût", "🔄  Flux de volume"])

    with liq_tabs[0]:
        blabel("Métriques de liquidité de marché")
        liq_t = st.selectbox("Ticker", selected, key="liq_tk")
        info_l = get_info(liq_t)
        df_l   = get_ohlcv(liq_t, "1y", "1d")

        if not df_l.empty:
            close_l = df_l["Close"].squeeze()
            high_l  = df_l["High"].squeeze()
            low_l   = df_l["Low"].squeeze()
            vol_l   = df_l["Volume"].squeeze()
            price_l = float(close_l.iloc[-1])

            # Liquidité metrics
            avg_vol_20  = float(vol_l.rolling(20).mean().iloc[-1])
            avg_vol_50  = float(vol_l.rolling(50).mean().iloc[-1])
            avg_vol_200 = float(vol_l.rolling(200).mean().iloc[-1])
            rel_vol     = float(vol_l.iloc[-1]) / avg_vol_20 if avg_vol_20 > 0 else 1

            # Amihud illiquidity ratio
            daily_ret_l = close_l.pct_change().abs()
            dollar_vol  = close_l * vol_l
            amihud      = (daily_ret_l / dollar_vol.replace(0, np.nan)).rolling(21).mean() * 1e6
            amihud_last = float(amihud.iloc[-1]) if not np.isnan(amihud.iloc[-1]) else 0

            # High-Low spread proxy
            hl_spread   = (high_l - low_l) / ((high_l + low_l) / 2) * 100
            avg_hl      = float(hl_spread.rolling(20).mean().iloc[-1])

            # Turnover ratio
            shares_out  = info_l.get("sharesOutstanding", 0)
            turnover    = float(vol_l.iloc[-1]) / shares_out * 100 if shares_out else 0

            lm1,lm2,lm3,lm4 = st.columns(4)
            lm1.metric("Vol aujourd'hui",   f"{float(vol_l.iloc[-1])/1e6:.1f}M")
            lm2.metric("Vol moy. 20j",      f"{avg_vol_20/1e6:.1f}M")
            lm3.metric("Volume relatif",    f"{rel_vol:.2f}x",
                        "Élevé" if rel_vol > 1.5 else "Normal" if rel_vol > 0.5 else "Faible")
            lm4.metric("Turnover journalier", f"{turnover:.3f}%")

            lm5,lm6,lm7,lm8 = st.columns(4)
            lm5.metric("Spread H-L moy. 20j", f"{avg_hl:.2f}%")
            lm6.metric("Amihud illiq. (×10⁶)", f"{amihud_last:.4f}")
            lm7.metric("Market Cap",           fmt_large(info_l.get("marketCap")))
            lm8.metric("Float",                fmt_large(info_l.get("floatShares")))

            # Rolling volume profile
            blabel("Profil de volume — Rolling 20j vs 50j vs 200j")
            fig_vol_p = go.Figure()
            fig_vol_p.add_trace(go.Scatter(x=vol_l.index, y=vol_l/1e6,
                mode="lines", line=dict(color="#4a7a99", width=0.7), name="Vol journalier", opacity=0.5))
            fig_vol_p.add_trace(go.Scatter(x=vol_l.index, y=vol_l.rolling(20).mean()/1e6,
                mode="lines", line=dict(color="#00d4ff", width=1.5), name="Moy 20j"))
            fig_vol_p.add_trace(go.Scatter(x=vol_l.index, y=vol_l.rolling(50).mean()/1e6,
                mode="lines", line=dict(color="#ffaa00", width=1.2, dash="dash"), name="Moy 50j"))
            fig_vol_p.add_trace(go.Scatter(x=vol_l.index, y=vol_l.rolling(200).mean()/1e6,
                mode="lines", line=dict(color="#ff4466", width=1, dash="dot"), name="Moy 200j"))
            fig_vol_p.update_layout(**PLOTLY_BASE, height=320,
                title=dict(text=f"◈ Volume (M) — {liq_t}",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                yaxis_title="Volume (M)", hovermode="x unified")
            st.plotly_chart(fig_vol_p, use_container_width=True)

            # Amihud rolling
            blabel("Ratio d'illiquidité Amihud (plus haut = moins liquide)")
            fig_amihud = go.Figure(go.Scatter(x=amihud.index, y=amihud,
                mode="lines", line=dict(color="#ff88aa", width=1.2),
                fill="tozeroy", fillcolor="rgba(255,136,170,0.06)", name="Amihud"))
            fig_amihud.update_layout(**PLOTLY_BASE, height=260,
                title=dict(text="◈ Amihud Illiquidity Ratio",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                hovermode="x unified")
            st.plotly_chart(fig_amihud, use_container_width=True)

    with liq_tabs[1]:
        blabel("Coût de transaction estimé · Spread implicite")
        liq_t2 = st.selectbox("Ticker", selected, key="liq_tk2")
        pv_liq  = st.number_input("Taille de position ($)", value=100_000, step=10_000, key="liq_pv")
        df_l2   = get_ohlcv(liq_t2, "3mo", "1d")

        if not df_l2.empty:
            close_l2 = df_l2["Close"].squeeze()
            high_l2  = df_l2["High"].squeeze()
            low_l2   = df_l2["Low"].squeeze()
            vol_l2   = df_l2["Volume"].squeeze()
            price_l2 = float(close_l2.iloc[-1])

            # Roll's spread estimator
            ret_l2   = close_l2.pct_change().dropna()
            cov_ret  = ret_l2.autocorr(lag=1)
            roll_spread = 2 * np.sqrt(max(-cov_ret, 0)) * price_l2

            # Kyle's lambda (price impact)
            dollar_ret = ret_l2 * price_l2
            signed_vol = vol_l2.pct_change().dropna() * price_l2
            if len(dollar_ret) > 10 and len(signed_vol) > 10:
                idx_k = dollar_ret.index.intersection(signed_vol.index)
                if len(idx_k) > 5:
                    slope_k, _, _, _, _ = stats.linregress(signed_vol[idx_k].fillna(0),
                                                            dollar_ret[idx_k].fillna(0))
                    kyle_lambda = abs(slope_k)
                else:
                    kyle_lambda = 0
            else:
                kyle_lambda = 0

            # Market impact estimate
            shares_to_buy = pv_liq / price_l2
            avg_vol_d     = float(vol_l2.mean())
            pct_adv       = shares_to_buy / avg_vol_d * 100 if avg_vol_d > 0 else 0
            impact_est    = 0.1 * np.sqrt(pct_adv / 100) * price_l2  # simplified square-root model

            cl1,cl2,cl3,cl4 = st.columns(4)
            cl1.metric("Roll Spread estimé",  f"${roll_spread:.4f}")
            cl2.metric("Spread en bps",       f"{roll_spread/price_l2*10000:.1f} bps")
            cl3.metric("% ADV (position)",    f"{pct_adv:.2f}%")
            cl4.metric("Impact marché estimé",f"${impact_est:.4f}")

            cl5,cl6,cl7 = st.columns(3)
            cl5.metric("Coût spread (aller)", f"${roll_spread/2*shares_to_buy:.2f}")
            cl6.metric("Coût aller-retour",   f"${roll_spread*shares_to_buy:.2f}")
            cl7.metric("Coût total estimé",   f"${(roll_spread+impact_est)*shares_to_buy:.2f}")

            # H-L spread rolling
            hl_s2 = (high_l2-low_l2)/((high_l2+low_l2)/2)*100
            fig_hl = go.Figure()
            fig_hl.add_trace(go.Scatter(x=hl_s2.index, y=hl_s2,
                mode="lines", line=dict(color="#00d4ff", width=0.8), name="H-L Spread %"))
            fig_hl.add_trace(go.Scatter(x=hl_s2.index, y=hl_s2.rolling(20).mean(),
                mode="lines", line=dict(color="#ffaa00", width=1.5), name="Moy 20j"))
            fig_hl.update_layout(**PLOTLY_BASE, height=300,
                title=dict(text=f"◈ Spread High-Low (%) — {liq_t2}",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                yaxis_title="Spread (%)", hovermode="x unified")
            st.plotly_chart(fig_hl, use_container_width=True)

    with liq_tabs[2]:
        blabel("Flux de volume · Distribution intraday")
        liq_t3  = st.selectbox("Ticker", selected, key="liq_tk3")
        df_intra = get_ohlcv(liq_t3, "5d", "15m")

        if not df_intra.empty:
            vol_intra = df_intra["Volume"].squeeze()
            close_i   = df_intra["Close"].squeeze()
            open_i    = df_intra["Open"].squeeze()

            # Buy/Sell volume proxy (close > open = buy pressure)
            buy_vol  = vol_intra.where(close_i >= open_i, 0)
            sell_vol = vol_intra.where(close_i < open_i, 0)

            fig_flow = go.Figure()
            fig_flow.add_trace(go.Bar(x=df_intra.index, y=buy_vol/1e3,
                name="Buy Vol (k)", marker_color="#00ff88", opacity=0.8))
            fig_flow.add_trace(go.Bar(x=df_intra.index, y=-sell_vol/1e3,
                name="Sell Vol (k)", marker_color="#ff4466", opacity=0.8))
            fig_flow.add_hline(y=0, line_color="#4a7a99", line_width=0.8)
            fig_flow.update_layout(**PLOTLY_BASE, height=340, barmode="overlay",
                title=dict(text=f"◈ Flux de volume intraday 15m — {liq_t3} (5 derniers jours)",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                yaxis_title="Volume (k)", hovermode="x unified")
            st.plotly_chart(fig_flow, use_container_width=True)

            # VWAP intraday
            tp_i = (df_intra["High"].squeeze() + df_intra["Low"].squeeze() + close_i) / 3
            vwap_i = (tp_i * vol_intra).cumsum() / vol_intra.cumsum()
            fig_vwap_i = make_subplots(rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.04, row_heights=[0.65, 0.35])
            fig_vwap_i.add_trace(go.Candlestick(
                x=df_intra.index, open=open_i,
                high=df_intra["High"].squeeze(), low=df_intra["Low"].squeeze(), close=close_i,
                increasing_line_color="#00ff88", decreasing_line_color="#ff4466",
                name="Prix", line_width=1), row=1, col=1)
            fig_vwap_i.add_trace(go.Scatter(x=vwap_i.index, y=vwap_i,
                mode="lines", line=dict(color="#ffaa00", width=1.5, dash="dash"), name="VWAP"),
                row=1, col=1)
            fig_vwap_i.add_trace(go.Bar(x=df_intra.index, y=vol_intra/1e3,
                marker_color="#00d4ff", name="Vol (k)", opacity=0.6), row=2, col=1)
            fig_vwap_i.update_layout(**PLOTLY_BASE, height=440,
                title=dict(text=f"◈ Prix + VWAP intraday — {liq_t3}",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                xaxis_rangeslider_visible=False, hovermode="x unified")
            st.plotly_chart(fig_vwap_i, use_container_width=True)
        else:
            st.info("Données intraday non disponibles pour ce ticker.")

# ══════════════════════════════════════════════════════════════════════
#  TAB 15 — SANTÉ FINANCIÈRE
# ══════════════════════════════════════════════════════════════════════
with tabs[15]:
    blabel("Score de Santé Financière · Altman Z-Score · Piotroski · Beneish M-Score")

    sh_t   = st.selectbox("Ticker", selected, key="sh_tk")
    info_sh = get_info(sh_t)

    sh1, sh2, sh3 = st.columns(3)

    # ── Altman Z-Score ─────────────────────────────────────────────
    with sh1:
        blabel("Altman Z-Score (risque de faillite)")
        try:
            mktcap   = info_sh.get("marketCap", 0) or 0
            tot_debt = info_sh.get("totalDebt", 1) or 1
            tot_rev  = info_sh.get("totalRevenue", 0) or 0
            tot_ass  = info_sh.get("totalAssets", 1) or 1
            cur_ass  = info_sh.get("totalCurrentAssets", 0) or 0
            cur_lib  = info_sh.get("totalCurrentLiabilities", 0) or 0
            ret_earn = info_sh.get("retainedEarnings", 0) or 0
            ebit     = info_sh.get("ebit", 0) or 0

            wc = cur_ass - cur_lib
            X1 = wc / tot_ass
            X2 = ret_earn / tot_ass
            X3 = ebit / tot_ass
            X4 = mktcap / tot_debt
            X5 = tot_rev / tot_ass
            Z  = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

            z_color  = "#00ff88" if Z > 2.99 else "#ffaa00" if Z > 1.81 else "#ff4466"
            z_label  = "Zone sûre ✓" if Z > 2.99 else "Zone grise ⚠" if Z > 1.81 else "Zone danger ✗"
            st.markdown(f"""
            <div style='text-align:center; padding:16px;
                        background:#0d1f30; border:1px solid {z_color};
                        border-radius:4px; margin-bottom:12px;'>
              <div style='font-family:Orbitron,sans-serif; font-size:32px;
                          font-weight:900; color:{z_color};'>{Z:.2f}</div>
              <div style='font-family:Space Mono,monospace; font-size:10px;
                          color:{z_color}; margin-top:4px;'>{z_label}</div>
            </div>""", unsafe_allow_html=True)

            for label, val, ref in [
                ("X1 — Fonds de roulement/Actifs", X1, ">0.1"),
                ("X2 — Résultats cumulés/Actifs",  X2, ">0.1"),
                ("X3 — EBIT/Actifs",               X3, ">0.05"),
                ("X4 — Cap. boursière/Dettes",      X4, ">1.0"),
                ("X5 — CA/Actifs",                  X5, ">0.5"),
            ]:
                st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:9px;color:#8aafc8;line-height:1.8;'>"
                            f"<span style='color:#4a7a99;'>{label}</span><br>"
                            f"<b style='color:#e8f4fd;'>{val:.3f}</b> (réf: {ref})</div>",
                            unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Z-Score indisponible: {e}")

    # ── Piotroski F-Score ─────────────────────────────────────────
    with sh2:
        blabel("Piotroski F-Score (qualité fondamentale)")
        try:
            roa      = info_sh.get("returnOnAssets", 0) or 0
            fcf      = info_sh.get("freeCashflow", 0) or 0
            roe      = info_sh.get("returnOnEquity", 0) or 0
            op_margin= info_sh.get("operatingMargins", 0) or 0
            curr_r   = info_sh.get("currentRatio", 0) or 0
            debt_eq  = info_sh.get("debtToEquity", 999) or 999
            rev_growth=info_sh.get("revenueGrowth", 0) or 0
            earn_growth=info_sh.get("earningsGrowth", 0) or 0
            gross_m  = info_sh.get("grossMargins", 0) or 0

            scores = {
                "ROA > 0":           int(roa > 0),
                "FCF > 0":           int(fcf > 0),
                "ROE > 0":           int(roe > 0),
                "Marge op. > 0":     int(op_margin > 0),
                "Current ratio > 1": int(curr_r > 1),
                "Leverage < 50%":    int(debt_eq < 50),
                "Croiss. revenu > 0":int(rev_growth > 0),
                "Croiss. BPA > 0":   int(earn_growth > 0),
                "Marge brute > 20%": int(gross_m > 0.20),
            }
            F = sum(scores.values())
            f_color = "#00ff88" if F >= 7 else "#ffaa00" if F >= 4 else "#ff4466"
            f_label = "Fort ✓" if F >= 7 else "Moyen ◆" if F >= 4 else "Faible ✗"

            st.markdown(f"""
            <div style='text-align:center; padding:16px;
                        background:#0d1f30; border:1px solid {f_color};
                        border-radius:4px; margin-bottom:12px;'>
              <div style='font-family:Orbitron,sans-serif; font-size:32px;
                          font-weight:900; color:{f_color};'>{F}/9</div>
              <div style='font-family:Space Mono,monospace; font-size:10px;
                          color:{f_color}; margin-top:4px;'>{f_label}</div>
            </div>""", unsafe_allow_html=True)

            for criterion, score in scores.items():
                icon = "✓" if score else "✗"
                color = "#00ff88" if score else "#ff4466"
                st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:9px;"
                            f"line-height:2;'><span style='color:{color};'>{icon}</span> "
                            f"<span style='color:#8aafc8;'>{criterion}</span></div>",
                            unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"F-Score indisponible: {e}")

    # ── Composite Score ───────────────────────────────────────────
    with sh3:
        blabel("Score composite personnalisé")
        try:
            # Composantes pondérées
            components = {
                "Valorisation (P/E)": max(0, min(10, 10 - (info_sh.get("trailingPE", 30) or 30) / 5)),
                "Croissance revenu":  min(10, max(0, (info_sh.get("revenueGrowth", 0) or 0) * 100)),
                "Marge nette":        min(10, max(0, (info_sh.get("profitMargins", 0) or 0) * 50)),
                "ROE":                min(10, max(0, (info_sh.get("returnOnEquity", 0) or 0) * 50)),
                "Solidité bilan":     min(10, max(0, 10 - (info_sh.get("debtToEquity", 100) or 100) / 20)),
                "FCF yield":          min(10, max(0, (info_sh.get("freeCashflow", 0) or 0) /
                                          max(info_sh.get("marketCap", 1) or 1, 1) * 100)),
            }
            total_score = sum(components.values()) / len(components)
            sc_color = "#00ff88" if total_score >= 7 else "#ffaa00" if total_score >= 4 else "#ff4466"

            st.markdown(f"""
            <div style='text-align:center; padding:16px;
                        background:#0d1f30; border:1px solid {sc_color};
                        border-radius:4px; margin-bottom:12px;'>
              <div style='font-family:Orbitron,sans-serif; font-size:32px;
                          font-weight:900; color:{sc_color};'>{total_score:.1f}/10</div>
              <div style='font-family:Space Mono,monospace; font-size:10px;
                          color:{sc_color}; margin-top:4px;'>Score composite</div>
            </div>""", unsafe_allow_html=True)

            fig_radar = go.Figure(go.Scatterpolar(
                r=list(components.values()) + [list(components.values())[0]],
                theta=list(components.keys()) + [list(components.keys())[0]],
                fill="toself",
                fillcolor="rgba(0,212,255,0.1)",
                line=dict(color="#00d4ff", width=1.5),
                name=sh_t,
            ))
            fig_radar.update_layout(
                paper_bgcolor="#070b10", plot_bgcolor="#070b10",
                font=dict(family="Space Mono", color="#8aafc8", size=9),
                polar=dict(
                    bgcolor="#0a1520",
                    angularaxis=dict(linecolor="#1a3a55", gridcolor="#1a3a55", color="#8aafc8"),
                    radialaxis=dict(range=[0,10], linecolor="#1a3a55",
                                   gridcolor="#1a3a55", color="#8aafc8"),
                ),
                height=300, margin=dict(l=40,r=40,t=30,b=30),
                title=dict(text="◈ Radar fondamental",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.warning(f"Score composite indisponible: {e}")

    # Comparaison multi-tickers
    if len(selected) > 1:
        blabel("Comparaison scores — Multi-tickers")
        comp_rows = []
        for t in selected:
            inf = get_info(t)
            roa_c = inf.get("returnOnAssets", 0) or 0
            roe_c = inf.get("returnOnEquity", 0) or 0
            pm_c  = inf.get("profitMargins", 0) or 0
            pe_c  = inf.get("trailingPE", 30) or 30
            dg_c  = inf.get("debtToEquity", 100) or 100
            rg_c  = inf.get("revenueGrowth", 0) or 0
            comp_rows.append({
                "Ticker": t,
                "ROA %":  f"{roa_c*100:.1f}%",
                "ROE %":  f"{roe_c*100:.1f}%",
                "Marge nette": f"{pm_c*100:.1f}%",
                "P/E":    f"{pe_c:.1f}",
                "D/E":    f"{dg_c:.1f}",
                "Rev Growth": f"{rg_c*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(comp_rows).set_index("Ticker"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 16 — RÉGIMES & IA
# ══════════════════════════════════════════════════════════════════════
with tabs[16]:
    blabel("Détection de Régimes · Signaux IA · Prévisions ML")

    ai_tabs = st.tabs(["🔍  Régimes de marché", "🤖  Signaux ML", "📡  Prévision ARIMA"])

    with ai_tabs[0]:
        blabel("Détection de régimes de marché — HMM simplifié & Clustering")
        reg_t  = st.selectbox("Ticker", selected, key="reg_tk")
        df_reg = get_ohlcv(reg_t, "2y", "1d")

        if not df_reg.empty:
            close_r = df_reg["Close"].squeeze()
            ret_r   = close_r.pct_change().dropna()
            vol_r   = ret_r.rolling(21).std() * np.sqrt(252) * 100
            mom_r   = close_r.pct_change(60) * 100

            features = pd.DataFrame({
                "Return 1j (%)":   ret_r * 100,
                "Vol 21j (%)":     vol_r,
                "Momentum 60j (%)": mom_r,
            }).dropna()

            from sklearn.cluster import KMeans
            sc_reg = StandardScaler()
            X_reg  = sc_reg.fit_transform(features)
            n_reg  = st.slider("Nombre de régimes", 2, 5, 3, key="n_reg")
            km     = KMeans(n_clusters=n_reg, random_state=42, n_init=10)
            labels = km.fit_predict(X_reg)
            features["Régime"] = labels

            regime_colors = ["#00d4ff","#00ff88","#ffaa00","#ff4466","#aa88ff"]
            fig_reg_ts = go.Figure()
            fig_reg_ts.add_trace(go.Scatter(x=close_r.index, y=close_r,
                mode="lines", line=dict(color="#4a7a99", width=1), name="Prix"))
            for r_id in range(n_reg):
                mask = features["Régime"] == r_id
                idx_r = features[mask].index
                p_r   = close_r[close_r.index.isin(idx_r)]
                fig_reg_ts.add_trace(go.Scatter(x=p_r.index, y=p_r,
                    mode="markers",
                    marker=dict(color=regime_colors[r_id], size=4, opacity=0.7),
                    name=f"Régime {r_id}"))
            fig_reg_ts.update_layout(**PLOTLY_BASE, height=380,
                title=dict(text=f"◈ Régimes de marché ({n_reg} clusters) — {reg_t}",
                           font=dict(family="Orbitron", color="#00d4ff", size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_reg_ts, use_container_width=True)

            # Caractéristiques par régime
            blabel("Caractéristiques moyennes par régime")
            regime_stats = features.groupby("Régime").agg({
                "Return 1j (%)":    ["mean","std"],
                "Vol 21j (%)":      "mean",
                "Momentum 60j (%)": "mean",
            }).round(3)
            regime_stats.columns = ["Rend. moy.","Rend. σ","Vol moy.","Momentum moy."]
            st.dataframe(regime_stats, use_container_width=True)

            # Scatter 3D features
            blabel("Espace des features par régime")
            fig_sc3 = go.Figure()
            for r_id in range(n_reg):
                mask = features["Régime"] == r_id
                fig_sc3.add_trace(go.Scatter(
                    x=features[mask]["Vol 21j (%)"],
                    y=features[mask]["Momentum 60j (%)"],
                    mode="markers",
                    marker=dict(color=regime_colors[r_id], size=4, opacity=0.6),
                    name=f"Régime {r_id}"))
            fig_sc3.update_layout(**PLOTLY_BASE, height=340,
                title=dict(text="◈ Vol vs Momentum par régime",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                xaxis_title="Volatilité 21j (%)", yaxis_title="Momentum 60j (%)")
            st.plotly_chart(fig_sc3, use_container_width=True)

    with ai_tabs[1]:
        blabel("Signaux ML — XGBoost Feature Importance & Prédiction direction")
        if not XGB_OK:
            st.error("pip install xgboost")
        else:
            ml_t   = st.selectbox("Ticker", selected, key="ml_tk")
            df_ml  = get_ohlcv(ml_t, "3y", "1d")

            if not df_ml.empty:
                df_ml2 = add_indicators(df_ml)
                close_ml = df_ml2["Close"].squeeze()
                ret_ml   = close_ml.pct_change()

                # Build features
                feat_df = pd.DataFrame({
                    "ret_1":   ret_ml,
                    "ret_5":   close_ml.pct_change(5),
                    "ret_20":  close_ml.pct_change(20),
                    "vol_10":  ret_ml.rolling(10).std(),
                    "vol_21":  ret_ml.rolling(21).std(),
                    "rsi":     df_ml2["RSI"] if "RSI" in df_ml2 else ret_ml.rolling(14).mean(),
                    "sma_rat": close_ml / close_ml.rolling(20).mean() - 1,
                    "bb_pos":  (close_ml - df_ml2["BB_dn"]) / (df_ml2["BB_up"] - df_ml2["BB_dn"] + 1e-9)
                               if "BB_up" in df_ml2 else pd.Series(0, index=close_ml.index),
                    "mom_5":   close_ml.pct_change(5),
                    "mom_10":  close_ml.pct_change(10),
                }).dropna()

                # Target: direction t+1
                target = (ret_ml.shift(-1) > 0).astype(int).reindex(feat_df.index).dropna()
                feat_df = feat_df.reindex(target.index).dropna()

                split = int(len(feat_df) * 0.8)
                X_train, X_test = feat_df.iloc[:split], feat_df.iloc[split:]
                y_train, y_test = target.iloc[:split], target.iloc[split:]

                with st.spinner("Training XGBoost…"):
                    model = xgb.XGBClassifier(
                        n_estimators=100, max_depth=3, learning_rate=0.05,
                        subsample=0.8, random_state=42, eval_metric="logloss",
                        verbosity=0)
                    model.fit(X_train, y_train)
                    acc = model.score(X_test, y_test)
                    preds = model.predict(X_test)
                    proba = model.predict_proba(X_test)[:, 1]

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Accuracy test",     f"{acc*100:.1f}%")
                m2.metric("Train size",        str(len(X_train)))
                m3.metric("Test size",         str(len(X_test)))
                m4.metric("Signal actuel",
                    "▲ HAUSSIER" if proba[-1] > 0.55 else "▼ BAISSIER" if proba[-1] < 0.45 else "◆ NEUTRE",
                    f"P(hausse)={proba[-1]:.2f}")

                # Feature importance
                blabel("Feature Importance")
                fi = pd.Series(model.feature_importances_, index=feat_df.columns).sort_values(ascending=True)
                fig_fi = go.Figure(go.Bar(
                    x=fi.values, y=fi.index.tolist(),
                    orientation="h",
                    marker_color="#00d4ff",
                    text=[f"{v:.3f}" for v in fi.values],
                    textposition="outside",
                    textfont=dict(family="Space Mono", color="#e8f4fd", size=9),
                ))
                fig_fi.update_layout(**PLOTLY_BASE, height=360,
                    title=dict(text="◈ XGBoost Feature Importance",
                               font=dict(family="Orbitron", color="#00d4ff", size=11)),
                    xaxis_title="Importance")
                st.plotly_chart(fig_fi, use_container_width=True)

                # Proba rolling on test set
                blabel("Probabilité haussière (test set)")
                fig_proba = go.Figure()
                fig_proba.add_trace(go.Scatter(
                    x=X_test.index, y=proba,
                    mode="lines", line=dict(color="#00d4ff", width=1.5),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.05)", name="P(hausse)"))
                fig_proba.add_hline(y=0.55, line_dash="dot", line_color="#00ff88",
                                    line_width=0.8, annotation_text="Signal BUY",
                                    annotation_font_color="#00ff88")
                fig_proba.add_hline(y=0.45, line_dash="dot", line_color="#ff4466",
                                    line_width=0.8, annotation_text="Signal SELL",
                                    annotation_font_color="#ff4466")
                fig_proba.add_hline(y=0.5, line_color="#4a7a99", line_width=0.5)
                fig_proba.update_layout(**PLOTLY_BASE, height=280,
                    title=dict(text=f"◈ Proba direction haussière — {ml_t}",
                               font=dict(family="Orbitron", color="#00d4ff", size=11)),
                    hovermode="x unified")
                fig_proba.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_proba, use_container_width=True)

    with ai_tabs[2]:
        blabel("Prévision ARIMA · Séries temporelles")
        if not SM_OK:
            st.error("pip install statsmodels")
        else:
            ar_t   = st.selectbox("Ticker", selected, key="ar_tk")
            ac1,ac2,ac3,ac4 = st.columns(4)
            p_ar   = ac1.slider("p (AR)",  0, 5, 1)
            d_ar   = ac2.slider("d (I)",   0, 2, 1)
            q_ar   = ac3.slider("q (MA)",  0, 5, 1)
            h_ar   = ac4.slider("Horizon (jours)", 5, 60, 30)

            df_ar  = get_ohlcv(ar_t, "1y", "1d")
            if not df_ar.empty:
                close_ar = df_ar["Close"].squeeze()
                with st.spinner(f"Fitting ARIMA({p_ar},{d_ar},{q_ar})…"):
                    try:
                        model_ar = ARIMA(close_ar, order=(p_ar, d_ar, q_ar))
                        res_ar   = model_ar.fit()
                        forecast = res_ar.forecast(steps=h_ar)
                        conf_int = res_ar.get_forecast(h_ar).conf_int()
                        fut_idx  = pd.date_range(start=close_ar.index[-1] + timedelta(days=1),
                                                 periods=h_ar, freq="B")

                        aic = res_ar.aic; bic = res_ar.bic
                        am1,am2 = st.columns(2)
                        am1.metric("AIC", f"{aic:.2f}")
                        am2.metric("BIC", f"{bic:.2f}")

                        fig_arima = go.Figure()
                        fig_arima.add_trace(go.Scatter(
                            x=close_ar.index[-120:], y=close_ar.iloc[-120:],
                            mode="lines", line=dict(color="#00d4ff", width=1.5),
                            name="Historique"))
                        fig_arima.add_trace(go.Scatter(
                            x=fut_idx, y=forecast,
                            mode="lines", line=dict(color="#ffaa00", width=2),
                            name=f"Prévision {h_ar}j"))
                        fig_arima.add_trace(go.Scatter(
                            x=list(fut_idx) + list(fut_idx[::-1]),
                            y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0][::-1]),
                            fill="toself", fillcolor="rgba(255,170,0,0.1)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="IC 95%"))
                        fig_arima.add_vline(x=str(close_ar.index[-1].date()),
                            line_dash="dot", line_color="#4a7a99", line_width=1)
                        fig_arima.update_layout(**PLOTLY_BASE, height=420,
                            title=dict(text=f"◈ ARIMA({p_ar},{d_ar},{q_ar}) — {ar_t} +{h_ar}j",
                                       font=dict(family="Orbitron", color="#00d4ff", size=12)),
                            hovermode="x unified")
                        st.plotly_chart(fig_arima, use_container_width=True)

                        # Résidus
                        blabel("Diagnostic des résidus")
                        resid = res_ar.resid
                        rf1,rf2 = st.columns(2)
                        with rf1:
                            fig_resid = go.Figure(go.Scatter(
                                x=close_ar.index, y=resid,
                                mode="lines", line=dict(color="#aa88ff", width=0.8),
                                name="Résidus"))
                            fig_resid.add_hline(y=0, line_color="#4a7a99", line_width=0.8)
                            fig_resid.update_layout(**PLOTLY_BASE, height=240,
                                title=dict(text="◈ Résidus",
                                           font=dict(family="Orbitron", color="#00d4ff", size=10)))
                            st.plotly_chart(fig_resid, use_container_width=True)
                        with rf2:
                            fig_resid_hist = go.Figure(go.Histogram(
                                x=resid, nbinsx=50, marker_color="#aa88ff",
                                opacity=0.7, histnorm="probability density"))
                            fig_resid_hist.update_layout(**PLOTLY_BASE, height=240,
                                title=dict(text="◈ Distribution résidus",
                                           font=dict(family="Orbitron", color="#00d4ff", size=10)))
                            st.plotly_chart(fig_resid_hist, use_container_width=True)
                    except Exception as e:
                        st.error(f"ARIMA error: {e}")

# ══════════════════════════════════════════════════════════════════════
#  TAB 17 — BINANCE API
# ══════════════════════════════════════════════════════════════════════
with tabs[17]:
    blabel("Binance API · Marché · Portfolio · Trades · Funding")

    st.markdown("""
    <span class='tag-free'>Sans clé : marché public</span>
    <span class='tag-key'>Avec clé : compte & trades</span>
    <span class='tag-free'>binance.com/en/my/settings/api-management</span>
    """, unsafe_allow_html=True)

    # ── Binance REST helpers ──────────────────────────────────────────
    BINANCE_BASE = "https://api.binance.com"
    BINANCE_FUTURES = "https://fapi.binance.com"

    def binance_public(endpoint, params=None, base=BINANCE_BASE):
        try:
            r = requests.get(f"{base}{endpoint}", params=params or {}, timeout=10)
            r.raise_for_status()
            return r.json(), None
        except Exception as e:
            return None, str(e)

    def binance_signed(endpoint, params=None, base=BINANCE_BASE,
                       api_key="", api_secret=""):
        if not api_key or not api_secret:
            return None, "Clé API requise"
        try:
            import hmac, hashlib
            params = params or {}
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 10000
            query = "&".join([f"{k}={v}" for k, v in params.items()])
            sig = hmac.new(api_secret.encode(), query.encode(),
                            hashlib.sha256).hexdigest()
            params["signature"] = sig
            headers = {"X-MBX-APIKEY": api_key}
            r = requests.get(f"{base}{endpoint}", params=params,
                             headers=headers, timeout=10)
            r.raise_for_status()
            return r.json(), None
        except Exception as e:
            return None, str(e)

    @st.cache_data(ttl=10)
    def get_binance_ticker(symbol):
        data, err = binance_public("/api/v3/ticker/24hr", {"symbol": symbol})
        return data, err

    @st.cache_data(ttl=60)
    def get_binance_orderbook(symbol, limit=20):
        data, err = binance_public("/api/v3/depth", {"symbol": symbol, "limit": limit})
        return data, err

    @st.cache_data(ttl=30)
    def get_binance_klines(symbol, interval="1h", limit=200):
        data, err = binance_public("/api/v3/klines",
                                    {"symbol": symbol, "interval": interval, "limit": limit})
        if err or not data: return pd.DataFrame(), err
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base",
            "taker_buy_quote","ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        return df, None

    @st.cache_data(ttl=30)
    def get_binance_trades(symbol, limit=50):
        data, err = binance_public("/api/v3/trades",
                                    {"symbol": symbol, "limit": limit})
        return data, err

    @st.cache_data(ttl=60)
    def get_binance_exchange_info():
        data, err = binance_public("/api/v3/exchangeInfo")
        return data, err

    @st.cache_data(ttl=30)
    def get_binance_funding(symbol):
        data, err = binance_public("/fapi/v1/fundingRate",
                                    {"symbol": symbol, "limit": 100},
                                    base=BINANCE_FUTURES)
        return data, err

    @st.cache_data(ttl=30)
    def get_binance_open_interest(symbol):
        data, err = binance_public("/fapi/v1/openInterest",
                                    {"symbol": symbol},
                                    base=BINANCE_FUTURES)
        return data, err

    @st.cache_data(ttl=30)
    def get_binance_long_short(symbol, period="5m", limit=50):
        data, err = binance_public("/futures/data/globalLongShortAccountRatio",
                                    {"symbol": symbol, "period": period, "limit": limit},
                                    base=BINANCE_FUTURES)
        return data, err

    # ── Sub-tabs ─────────────────────────────────────────────────────
    bn_tabs = st.tabs([
        "📊  Marché Live",
        "📖  Orderbook",
        "📈  OHLCV",
        "⚡  Trades Récents",
        "🌊  Futures & Funding",
        "💼  Mon Portfolio",
    ])

    # Common symbol selector
    POPULAR_SYMBOLS = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
        "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
        "LINKUSDT","LTCUSDT","UNIUSDT","ATOMUSDT","NEARUSDT",
    ]
    with st.sidebar:
        st.markdown("<div class='blabel'>▸ Binance Symbol</div>", unsafe_allow_html=True)
        bn_sym_input = st.text_input("Symbol Binance", value="BTCUSDT",
                                      key="bn_sym",
                                      help="Ex: BTCUSDT, ETHUSDT, SOLUSDT")
        bn_symbol = bn_sym_input.upper().strip().replace("/","").replace("-","")

    # ── TAB: Marché Live ─────────────────────────────────────────────
    with bn_tabs[0]:
        blabel(f"Ticker 24h — {bn_symbol}")
        if st.button("↺ Rafraîchir", key="bn_refresh_ticker"):
            st.cache_data.clear()
            st.rerun()

        ticker_data, ticker_err = get_binance_ticker(bn_symbol)
        if ticker_err:
            st.error(f"Erreur Binance: {ticker_err}")
        elif ticker_data:
            price     = float(ticker_data.get("lastPrice", 0))
            change    = float(ticker_data.get("priceChangePercent", 0))
            high_24   = float(ticker_data.get("highPrice", 0))
            low_24    = float(ticker_data.get("lowPrice", 0))
            vol_24    = float(ticker_data.get("volume", 0))
            vol_quote = float(ticker_data.get("quoteVolume", 0))
            trades_24 = int(ticker_data.get("count", 0))
            bid       = float(ticker_data.get("bidPrice", 0))
            ask       = float(ticker_data.get("askPrice", 0))
            spread    = ask - bid
            spread_bps= spread/price*10000 if price else 0

            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Prix",        f"${price:,.4f}", f"{change:+.2f}%",
                       delta_color="normal" if change >= 0 else "inverse")
            t2.metric("High 24h",    f"${high_24:,.4f}")
            t3.metric("Low 24h",     f"${low_24:,.4f}")
            t4.metric("Var 24h",     f"{change:+.2f}%",
                       delta_color="normal" if change >= 0 else "inverse")

            t5,t6,t7,t8 = st.columns(4)
            t5.metric("Volume 24h",  f"{vol_24:,.0f}")
            t6.metric("Vol. Quote",  f"${vol_quote/1e6:,.1f}M")
            t7.metric("Trades 24h",  f"{trades_24:,}")
            t8.metric("Spread",      f"${spread:.4f} ({spread_bps:.1f} bps)")

            t9,t10,t11,t12 = st.columns(4)
            t9.metric("Bid",         f"${bid:,.4f}")
            t10.metric("Ask",        f"${ask:,.4f}")
            open_p = float(ticker_data.get("openPrice",0))
            t11.metric("Open 24h",   f"${open_p:,.4f}")
            prev_c = float(ticker_data.get("prevClosePrice",0))
            t12.metric("Close préc.",f"${prev_c:,.4f}")

        # Top movers
        blabel("Top 20 — Variation 24h (USDT)")
        with st.spinner("Chargement top movers…"):
            all_tickers, err_all = binance_public("/api/v3/ticker/24hr")
        if all_tickers and not err_all:
            df_movers = pd.DataFrame(all_tickers)
            df_movers = df_movers[df_movers["symbol"].str.endswith("USDT")].copy()
            df_movers["priceChangePercent"] = df_movers["priceChangePercent"].astype(float)
            df_movers["lastPrice"]          = df_movers["lastPrice"].astype(float)
            df_movers["volume"]             = df_movers["volume"].astype(float)
            df_movers["quoteVolume"]        = df_movers["quoteVolume"].astype(float)

            mv1, mv2 = st.columns(2)
            with mv1:
                blabel("Top gainers")
                top_gain = df_movers.nlargest(10, "priceChangePercent")[
                    ["symbol","lastPrice","priceChangePercent","quoteVolume"]]
                top_gain.columns = ["Symbol","Prix","Var%","Vol USDT"]
                top_gain["Var%"] = top_gain["Var%"].apply(lambda x: f"{x:+.2f}%")
                top_gain["Vol USDT"] = top_gain["Vol USDT"].apply(lambda x: f"${x/1e6:.1f}M")
                st.dataframe(top_gain.set_index("Symbol"), use_container_width=True)

            with mv2:
                blabel("Top losers")
                top_lose = df_movers.nsmallest(10, "priceChangePercent")[
                    ["symbol","lastPrice","priceChangePercent","quoteVolume"]]
                top_lose.columns = ["Symbol","Prix","Var%","Vol USDT"]
                top_lose["Var%"] = top_lose["Var%"].apply(lambda x: f"{x:+.2f}%")
                top_lose["Vol USDT"] = top_lose["Vol USDT"].apply(lambda x: f"${x/1e6:.1f}M")
                st.dataframe(top_lose.set_index("Symbol"), use_container_width=True)

            # Volume leaders
            blabel("Top 10 — Volume USDT 24h")
            top_vol = df_movers.nlargest(10, "quoteVolume")[
                ["symbol","lastPrice","priceChangePercent","quoteVolume"]]
            top_vol.columns = ["Symbol","Prix","Var%","Vol USDT"]
            colors_v = ["#00ff88" if v >= 0 else "#ff4466"
                        for v in top_vol["Var%"].str.replace("%","").astype(float)]
            fig_vol_bar = go.Figure(go.Bar(
                x=top_vol["Symbol"],
                y=top_vol["Vol USDT"]/1e6,
                marker_color=colors_v,
                text=[f"${v/1e6:.0f}M" for v in top_vol["Vol USDT"]],
                textposition="outside",
                textfont=dict(family="Space Mono", color="#e8f4fd", size=9),
            ))
            fig_vol_bar.update_layout(**PLOTLY_BASE, height=300,
                title=dict(text="◈ Volume USDT 24h (M$)",
                           font=dict(family="Orbitron", color="#00d4ff", size=11)),
                yaxis_title="Volume (M$)")
            st.plotly_chart(fig_vol_bar, use_container_width=True)

    # ── TAB: Orderbook ───────────────────────────────────────────────
    with bn_tabs[1]:
        blabel(f"Orderbook Level 2 — {bn_symbol}")
        ob_limit = st.select_slider("Profondeur", options=[5,10,20,50,100,500,1000], value=20)

        ob_data, ob_err = get_binance_orderbook(bn_symbol, ob_limit)
        if ob_err:
            st.error(f"Erreur: {ob_err}")
        elif ob_data:
            bids = pd.DataFrame(ob_data["bids"][:ob_limit],
                                columns=["Prix","Quantité"]).astype(float)
            asks = pd.DataFrame(ob_data["asks"][:ob_limit],
                                columns=["Prix","Quantité"]).astype(float)
            bids["Cumulé"] = bids["Quantité"].cumsum()
            asks["Cumulé"] = asks["Quantité"].cumsum()

            mid   = (float(bids["Prix"].iloc[0]) + float(asks["Prix"].iloc[0])) / 2
            spread_ob = float(asks["Prix"].iloc[0]) - float(bids["Prix"].iloc[0])

            om1,om2,om3,om4 = st.columns(4)
            om1.metric("Mid Price",   f"${mid:,.4f}")
            om2.metric("Spread",      f"${spread_ob:.4f}")
            om3.metric("Spread bps",  f"{spread_ob/mid*10000:.2f}")
            total_bid_vol = float(bids["Quantité"].sum())
            total_ask_vol = float(asks["Quantité"].sum())
            imbalance = (total_bid_vol - total_ask_vol)/(total_bid_vol + total_ask_vol)*100
            om4.metric("Imbalance Bid/Ask",
                        f"{imbalance:+.1f}%",
                        "Pression achat" if imbalance > 0 else "Pression vente",
                        delta_color="normal" if imbalance > 0 else "inverse")

            ob_col1, ob_col2 = st.columns(2)
            with ob_col1:
                blabel("Bids (achat)")
                bids_display = bids.copy()
                bids_display["Prix"] = bids_display["Prix"].apply(lambda x: f"${x:,.4f}")
                st.dataframe(bids_display, use_container_width=True, height=300)
            with ob_col2:
                blabel("Asks (vente)")
                asks_display = asks.copy()
                asks_display["Prix"] = asks_display["Prix"].apply(lambda x: f"${x:,.4f}")
                st.dataframe(asks_display, use_container_width=True, height=300)

            # Depth chart
            fig_depth = go.Figure()
            fig_depth.add_trace(go.Scatter(
                x=bids["Cumulé"], y=bids["Prix"],
                mode="lines", line=dict(color="#00ff88", width=2),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.1)",
                name="Bids"))
            fig_depth.add_trace(go.Scatter(
                x=asks["Cumulé"], y=asks["Prix"],
                mode="lines", line=dict(color="#ff4466", width=2),
                fill="tozeroy", fillcolor="rgba(255,68,102,0.1)",
                name="Asks"))
            fig_depth.add_hline(y=mid, line_dash="dot",
                                 line_color="#00d4ff", line_width=0.8)
            fig_depth.update_layout(**PLOTLY_BASE, height=380,
                title=dict(text=f"◈ Market Depth — {bn_symbol}",
                           font=dict(family="Orbitron", color="#00d4ff", size=12)),
                xaxis_title="Volume cumulé", yaxis_title="Prix")
            st.plotly_chart(fig_depth, use_container_width=True)

    # ── TAB: OHLCV ───────────────────────────────────────────────────
    with bn_tabs[2]:
        blabel(f"OHLCV Binance — {bn_symbol}")
        kc1, kc2 = st.columns(2)
        kline_intervals = {
            "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
            "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
            "1j":"1d","3j":"3d","1sem":"1w","1mois":"1M",
        }
        kl_interval = kc1.selectbox("Intervalle", list(kline_intervals.keys()),
                                     index=5, key="bn_kl_iv")
        kl_limit    = kc2.slider("Bougies", 50, 1000, 200, key="bn_kl_lim")
        iv_str = kline_intervals[kl_interval]

        df_kl, kl_err = get_binance_klines(bn_symbol, iv_str, kl_limit)
        if kl_err:
            st.error(f"Erreur: {kl_err}")
        elif not df_kl.empty:
            fig_kl = make_subplots(rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.03, row_heights=[0.72, 0.28])
            fig_kl.add_trace(go.Candlestick(
                x=df_kl.index, open=df_kl["open"],
                high=df_kl["high"], low=df_kl["low"], close=df_kl["close"],
                increasing_line_color="#00ff88", decreasing_line_color="#ff4466",
                increasing_fillcolor="#00ff88", decreasing_fillcolor="#ff4466",
                name=bn_symbol, line_width=1), row=1, col=1)

            # SMA overlays
            df_kl["sma20"] = df_kl["close"].rolling(20).mean()
            df_kl["sma50"] = df_kl["close"].rolling(50).mean()
            fig_kl.add_trace(go.Scatter(x=df_kl.index, y=df_kl["sma20"],
                mode="lines", line=dict(color="#00ff88", width=1),
                name="SMA20"), row=1, col=1)
            fig_kl.add_trace(go.Scatter(x=df_kl.index, y=df_kl["sma50"],
                mode="lines", line=dict(color="#ffaa00", width=1, dash="dash"),
                name="SMA50"), row=1, col=1)

            # Volume with taker buy coloring
            taker_buy = df_kl["taker_buy_base"].astype(float)
            total_vol = df_kl["volume"]
            buy_pct   = taker_buy / total_vol.replace(0, np.nan)
            vol_colors = ["#00ff88" if bp >= 0.5 else "#ff4466"
                          for bp in buy_pct.fillna(0.5)]
            fig_kl.add_trace(go.Bar(x=df_kl.index, y=df_kl["volume"],
                marker_color=vol_colors, name="Volume", opacity=0.7),
                row=2, col=1)

            fig_kl.update_layout(**PLOTLY_BASE, height=500,
                title=dict(text=f"◈ {bn_symbol} — {kl_interval} ({kl_limit} bougies) · Binance",
                           font=dict(family="Orbitron", color="#00d4ff", size=12)),
                xaxis_rangeslider_visible=False, hovermode="x unified")
            st.plotly_chart(fig_kl, use_container_width=True)

            # Stats
            last_p = float(df_kl["close"].iloc[-1])
            first_p= float(df_kl["close"].iloc[0])
            ret_pct= (last_p/first_p-1)*100
            ks1,ks2,ks3,ks4,ks5 = st.columns(5)
            ks1.metric("Dernier",      f"${last_p:,.4f}")
            ks2.metric("Variation",    f"{ret_pct:+.2f}%")
            ks3.metric("High période", f"${float(df_kl['high'].max()):,.4f}")
            ks4.metric("Low période",  f"${float(df_kl['low'].min()):,.4f}")
            ks5.metric("Nb trades moy.",f"{int(df_kl['trades'].astype(float).mean()):,}")

    # ── TAB: Trades Récents ──────────────────────────────────────────
    with bn_tabs[3]:
        blabel(f"Trades récents — {bn_symbol}")
        tr_limit = st.slider("Nombre de trades", 10, 1000, 50, key="bn_tr_lim")
        trades_data, tr_err = get_binance_trades(bn_symbol, tr_limit)

        if tr_err:
            st.error(f"Erreur: {tr_err}")
        elif trades_data:
            df_trades = pd.DataFrame(trades_data)
            df_trades["time"] = pd.to_datetime(df_trades["time"], unit="ms")
            df_trades["price"] = df_trades["price"].astype(float)
            df_trades["qty"]   = df_trades["qty"].astype(float)
            df_trades["quoteQty"] = df_trades["quoteQty"].astype(float)

            # Stats
            buy_trades  = df_trades[df_trades["isBuyerMaker"] == False]
            sell_trades = df_trades[df_trades["isBuyerMaker"] == True]
            buy_vol     = float(buy_trades["quoteQty"].sum())
            sell_vol    = float(sell_trades["quoteQty"].sum())
            total_vol_t = buy_vol + sell_vol

            tr1,tr2,tr3,tr4 = st.columns(4)
            tr1.metric("Trades achat",  f"{len(buy_trades)}")
            tr2.metric("Trades vente",  f"{len(sell_trades)}")
            tr3.metric("Vol achat $",   f"${buy_vol:,.0f}")
            tr4.metric("Vol vente $",   f"${sell_vol:,.0f}")

            # Delta volume bar
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Bar(
                x=["Buy Volume","Sell Volume"],
                y=[buy_vol, sell_vol],
                marker_color=["#00ff88","#ff4466"],
                text=[f"${buy_vol:,.0f}",f"${sell_vol:,.0f}"],
                textposition="outside",
                textfont=dict(family="Space Mono",color="#e8f4fd",size=10),
            ))
            fig_delta.update_layout(**PLOTLY_BASE, height=260,
                title=dict(text="◈ Buy vs Sell Volume ($)",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)))
            st.plotly_chart(fig_delta, use_container_width=True)

            # Trade scatter
            fig_tr = go.Figure()
            fig_tr.add_trace(go.Scatter(
                x=df_trades["time"],
                y=df_trades["price"],
                mode="markers",
                marker=dict(
                    color=["#00ff88" if not m else "#ff4466"
                           for m in df_trades["isBuyerMaker"]],
                    size=np.clip(df_trades["qty"]/df_trades["qty"].max()*15+3, 3, 18),
                    opacity=0.8,
                ),
                text=[f"{'BUY' if not m else 'SELL'} {q:.4f} @ ${p:.4f}"
                      for m,q,p in zip(df_trades["isBuyerMaker"],
                                        df_trades["qty"],df_trades["price"])],
                hoverinfo="text+x",
                name="Trades",
            ))
            fig_tr.update_layout(**PLOTLY_BASE, height=340,
                title=dict(text=f"◈ Trades récents — {bn_symbol} (vert=achat, rouge=vente, taille=quantité)",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)),
                hovermode="closest")
            st.plotly_chart(fig_tr, use_container_width=True)

            # Raw table
            with st.expander("▸ Données brutes"):
                df_display = df_trades[["time","price","qty","quoteQty","isBuyerMaker"]].copy()
                df_display.columns = ["Heure","Prix","Quantité","Valeur USDT","Vendeur?"]
                st.dataframe(df_display, use_container_width=True, height=300)

    # ── TAB: Futures & Funding ───────────────────────────────────────
    with bn_tabs[4]:
        blabel(f"Futures Perpétuels — {bn_symbol}")
        st.markdown("<span class='tag-free'>Données Binance Futures (FAPI) · Sans clé</span>",
                    unsafe_allow_html=True)

        # Funding rate
        fut_symbol = bn_symbol  # e.g. BTCUSDT works for both spot and perp
        fc1, fc2 = st.columns(2)

        with fc1:
            blabel("Funding Rate historique")
            fr_data, fr_err = get_binance_funding(fut_symbol)
            if fr_err:
                st.warning(f"Futures: {fr_err} (ticker peut ne pas avoir de perp)")
            elif fr_data:
                df_fr = pd.DataFrame(fr_data)
                df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], unit="ms")
                df_fr["fundingRate"] = df_fr["fundingRate"].astype(float) * 100

                last_fr = float(df_fr["fundingRate"].iloc[-1])
                avg_fr  = float(df_fr["fundingRate"].mean())
                ann_fr  = avg_fr * 3 * 365  # 3x/day × 365

                fm1,fm2,fm3 = st.columns(3)
                fm1.metric("Funding actuel", f"{last_fr:.4f}%",
                            "Longs paient" if last_fr > 0 else "Shorts paient",
                            delta_color="inverse" if last_fr > 0 else "normal")
                fm2.metric("Funding moy.",   f"{avg_fr:.4f}%")
                fm3.metric("Funding ann.",   f"{ann_fr:.2f}%")

                fig_fr = go.Figure()
                fig_fr.add_trace(go.Bar(
                    x=df_fr["fundingTime"],
                    y=df_fr["fundingRate"],
                    marker_color=["#ff4466" if v > 0 else "#00ff88"
                                  for v in df_fr["fundingRate"]],
                    name="Funding Rate %"))
                fig_fr.add_hline(y=0, line_color="#4a7a99", line_width=0.8)
                fig_fr.add_hline(y=0.01, line_dash="dot", line_color="#ff4466",
                                  line_width=0.7, annotation_text="Neutre +",
                                  annotation_font_color="#ff4466")
                fig_fr.add_hline(y=-0.01, line_dash="dot", line_color="#00ff88",
                                  line_width=0.7, annotation_text="Neutre -",
                                  annotation_font_color="#00ff88")
                fig_fr.update_layout(**PLOTLY_BASE, height=320,
                    title=dict(text=f"◈ Funding Rate (%) — {fut_symbol}",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    hovermode="x unified")
                st.plotly_chart(fig_fr, use_container_width=True)

        with fc2:
            blabel("Open Interest & Long/Short Ratio")
            oi_data, oi_err = get_binance_open_interest(fut_symbol)
            if oi_err:
                st.warning(f"OI: {oi_err}")
            elif oi_data:
                oi_val = float(oi_data.get("openInterest", 0))
                oi_price = float(oi_data.get("time", 0))
                st.metric("Open Interest", f"{oi_val:,.0f} {fut_symbol[:3]}")

            ls_data, ls_err = get_binance_long_short(fut_symbol)
            if ls_err:
                st.warning(f"L/S: {ls_err}")
            elif ls_data:
                df_ls = pd.DataFrame(ls_data)
                df_ls["timestamp"] = pd.to_datetime(df_ls["timestamp"], unit="ms")
                df_ls["longShortRatio"] = df_ls["longShortRatio"].astype(float)
                df_ls["longAccount"]    = df_ls["longAccount"].astype(float) * 100
                df_ls["shortAccount"]   = df_ls["shortAccount"].astype(float) * 100

                last_ls  = float(df_ls["longShortRatio"].iloc[-1])
                last_lng = float(df_ls["longAccount"].iloc[-1])
                last_sht = float(df_ls["shortAccount"].iloc[-1])

                lm1,lm2,lm3 = st.columns(3)
                lm1.metric("L/S Ratio",   f"{last_ls:.3f}")
                lm2.metric("Longs %",     f"{last_lng:.1f}%")
                lm3.metric("Shorts %",    f"{last_sht:.1f}%")

                fig_ls = go.Figure()
                fig_ls.add_trace(go.Scatter(
                    x=df_ls["timestamp"], y=df_ls["longShortRatio"],
                    mode="lines", line=dict(color="#00d4ff", width=1.5),
                    name="L/S Ratio"))
                fig_ls.add_hline(y=1, line_dash="dot", line_color="#ffaa00",
                                  line_width=0.8, annotation_text="Neutre",
                                  annotation_font_color="#ffaa00")
                fig_ls.update_layout(**PLOTLY_BASE, height=280,
                    title=dict(text="◈ Long/Short Ratio",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    hovermode="x unified")
                st.plotly_chart(fig_ls, use_container_width=True)

    # ── TAB: Mon Portfolio ───────────────────────────────────────────
    with bn_tabs[5]:
        blabel("Mon Portfolio Binance · Balances · P&L")
        if not binance_key or not binance_secret:
            st.info("🔑 Entrez votre clé API Binance dans la barre latérale pour accéder à votre compte.")
            st.markdown("""
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#4a7a99;
                        line-height:2;background:#0d1f30;padding:12px;border-radius:4px;
                        border:1px solid #1a3a55;'>
            <b style='color:#00d4ff;'>Comment créer une clé API Binance (lecture seule) :</b><br>
            1. Connectez-vous sur binance.com<br>
            2. Mon Compte → Gestion des API<br>
            3. Créer une API → <b>Activer uniquement "Lire les infos"</b><br>
            4. Désactivez tout le reste (trading, retrait, etc.)<br>
            5. Whitelist votre IP pour plus de sécurité<br>
            <br>
            <b style='color:#ffaa00;'>⚠ Ne partagez jamais votre clé secrète</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("↺ Recharger le portfolio", key="bn_reload_pf"):
                st.cache_data.clear()
                st.rerun()

            # Account info
            account_data, acc_err = binance_signed(
                "/api/v3/account", {}, api_key=binance_key, api_secret=binance_secret)

            if acc_err:
                st.error(f"Erreur API: {acc_err}")
                st.info("Vérifiez que vos clés sont correctes et que la permission 'Lire' est activée.")
            elif account_data:
                balances = account_data.get("balances", [])
                df_bal = pd.DataFrame(balances)
                df_bal["free"]   = df_bal["free"].astype(float)
                df_bal["locked"] = df_bal["locked"].astype(float)
                df_bal["total"]  = df_bal["free"] + df_bal["locked"]
                df_bal = df_bal[df_bal["total"] > 0].sort_values("total", ascending=False)

                st.metric("Nombre d'actifs", str(len(df_bal)))

                # Get USDT values for each asset
                usdt_values = []
                for _, row in df_bal.iterrows():
                    asset = row["asset"]
                    qty   = row["total"]
                    if asset == "USDT":
                        usdt_values.append(qty)
                    else:
                        sym_try = f"{asset}USDT"
                        td, _ = binance_public("/api/v3/ticker/price",
                                               {"symbol": sym_try})
                        if td and "price" in td:
                            usdt_values.append(qty * float(td["price"]))
                        else:
                            usdt_values.append(0.0)

                df_bal["USDT Value"] = usdt_values
                total_usdt = sum(usdt_values)
                df_bal["Poids %"] = df_bal["USDT Value"] / total_usdt * 100 if total_usdt > 0 else 0

                st.metric("Valeur totale estimée", f"${total_usdt:,.2f} USDT")

                # Portfolio table
                df_display_bal = df_bal[["asset","free","locked","total","USDT Value","Poids %"]].copy()
                df_display_bal.columns = ["Actif","Libre","Bloqué","Total","Valeur USDT","Poids %"]
                df_display_bal["Valeur USDT"] = df_display_bal["Valeur USDT"].apply(lambda x: f"${x:,.2f}")
                df_display_bal["Poids %"] = df_display_bal["Poids %"].apply(lambda x: f"{x:.2f}%")
                st.dataframe(df_display_bal.set_index("Actif"), use_container_width=True)

                # Portfolio pie
                df_pie = df_bal[df_bal["USDT Value"] > 0]
                if not df_pie.empty:
                    fig_pf_pie = go.Figure(go.Pie(
                        labels=df_pie["asset"].tolist(),
                        values=df_pie["USDT Value"].tolist(),
                        marker_colors=COLORS * (len(df_pie)//len(COLORS)+1),
                        textinfo="label+percent",
                        hole=0.4,
                        textfont=dict(family="Space Mono", size=10),
                    ))
                    fig_pf_pie.update_layout(**PLOTLY_BASE, height=400,
                        title=dict(text="◈ Allocation portfolio (USDT)",
                                   font=dict(family="Orbitron",color="#00d4ff",size=12)))
                    st.plotly_chart(fig_pf_pie, use_container_width=True)

                # Account type info
                blabel("Informations compte")
                acc_info = {
                    "Maker Commission":  f"{account_data.get('makerCommission',0)/100:.2f}%",
                    "Taker Commission":  f"{account_data.get('takerCommission',0)/100:.2f}%",
                    "Buyer Commission":  f"{account_data.get('buyerCommission',0)/100:.2f}%",
                    "Seller Commission": f"{account_data.get('sellerCommission',0)/100:.2f}%",
                    "Can Trade":         str(account_data.get("canTrade", False)),
                    "Can Withdraw":      str(account_data.get("canWithdraw", False)),
                    "Can Deposit":       str(account_data.get("canDeposit", False)),
                }
                for k, v in acc_info.items():
                    st.markdown(
                        f"<span style='font-family:Space Mono,monospace;font-size:10px;"
                        f"color:#4a7a99;'>{k}:</span> "
                        f"<span style='font-family:Space Mono,monospace;font-size:10px;"
                        f"color:#e8f4fd;'>{v}</span>",
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(f"""
<div style='text-align:center;font-family:Space Mono,monospace;font-size:8px;color:#1e3a55;padding:8px 0;'>
  QUANTUM TERMINAL v3.0 · Market Data · Statistics · Backtesting · Risk · Portfolio · Macro · Forex · News · ML ·
  yfinance · ccxt · FRED · Alpha Vantage · NewsAPI · scipy · arch · statsmodels ·
  Usage informatif uniquement · Pas de conseil financier ·
  {datetime.now().strftime('%d %b %Y %H:%M')}
</div>""", unsafe_allow_html=True)
