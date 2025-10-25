import os
import time
import hmac
import json
import math
import signal
import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque

import requests


# =======================
# Config
# =======================
API_KEY = os.getenv("COINDCX_API_KEY", "")
API_SECRET = os.getenv("COINDCX_API_SECRET", "")
BASE_URL = "https://api.coindcx.com"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM = TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

NO_TRADE_NOTIFICATION_INTERVAL = 30 * 60

PAPER_MODE = True
PORT = int(os.getenv("PORT", 8080))

STARTING_CAPITAL_INR = 1000.0
MAX_CONCURRENT_POSITIONS = 1

# 🎯 Better coin selection - major coins only
TARGET_COINS = ['BTC', 'ETH', 'ADA', 'MATIC', 'LINK']

# =======================
# 🎯 MEAN REVERSION PARAMETERS
# =======================
SL_PCT = 0.008           # 0.8% stop loss (wider for mean reversion)
TP_PCT = 0.012           # 1.2% take profit (wider targets)
PARTIAL_TP_PCT = 0.008   # 0.8% partial profit
PARTIAL_TAKE = 0.60
TSL_ACTIVATE = 0.010     # Activate trailing at 1.0%
TSL_STEP = 0.003         # Trail by 0.3%

MIN_HOLD_TIME_SEC = 300  # 5 minutes minimum (was 2 min)
EMERGENCY_STOP_PCT = 0.02
MAX_HOLD_SEC = 60 * 60

# Mean Reversion specific
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Indicators
RSI_LEN = 14
EMA_FAST = 20
EMA_SLOW = 50

LEVERAGE = 3
RISK_PER_TRADE_PCT = 0.5  # 🔥 Reduced from 1% due to losses
MIN_MARGIN_INR = 800.0    # 🔥 Increased from 100
MAX_MARGIN_INR = 1200.0   # 🔥 Increased from 500

MIN_ATR_PERCENT = 0.5
ATR_PERIOD = 14

MIN_24H_VOLUME_INR = 50000000  # Higher - major coins only
MAX_SPREAD_PERCENT = 0.5
MIN_PRICE_INR = 10.0           # Higher minimum price

AVOID_HOURS_UTC = [3, 4, 11, 12]

TAKER_FEE = 0.001
MAKER_FEE = 0.001

MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# Risk Management
MAX_DAILY_LOSS_PCT = 5.0
MAX_CONSECUTIVE_LOSSES = 3
CONSECUTIVE_LOSS_COOLDOWN = 1800
MAX_DAILY_TRADES = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)


# =======================
# Performance Tracker, Market Regime, Risk Manager
# =======================
class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.hourly_performance = {}
        self.symbol_performance = {}
    
    def record_trade(self, trade_data):
        self.trades.append(trade_data)
        
        if 'pnl' in trade_data:
            is_win = trade_data['pnl'] > 0
            
            hour = trade_data['ts'].hour
            if hour not in self.hourly_performance:
                self.hourly_performance[hour] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
            
            if is_win:
                self.hourly_performance[hour]['wins'] += 1
            else:
                self.hourly_performance[hour]['losses'] += 1
            self.hourly_performance[hour]['pnl'] += trade_data['pnl']
            
            symbol = trade_data['symbol']
            if symbol not in self.symbol_performance:
                self.symbol_performance[symbol] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
            
            if is_win:
                self.symbol_performance[symbol]['wins'] += 1
            else:
                self.symbol_performance[symbol]['losses'] += 1
            self.symbol_performance[symbol]['pnl'] += trade_data['pnl']
    
    def get_best_hours(self, top_n=3):
        hours = [(h, data['pnl']) for h, data in self.hourly_performance.items() if data['wins'] + data['losses'] >= 2]
        hours.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in hours[:top_n]]
    
    def get_best_symbols(self, top_n=3):
        symbols = [(s, data['pnl']) for s, data in self.symbol_performance.items() if data['wins'] + data['losses'] >= 2]
        symbols.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in symbols[:top_n]]
    
    def get_win_rate(self):
        closed = [t for t in self.trades if 'pnl' in t]
        if not closed:
            return 0.0
        wins = len([t for t in closed if t['pnl'] > 0])
        return (wins / len(closed)) * 100


class MarketRegime:
    def __init__(self):
        self.current_regime = 'normal'
        self.volatility_history = deque(maxlen=20)
    
    def update(self, atr_percent):
        self.volatility_history.append(atr_percent)
        
        if len(self.volatility_history) < 10:
            return 'normal'
        
        avg_volatility = sum(self.volatility_history) / len(self.volatility_history)
        
        if avg_volatility > 1.5:
            self.current_regime = 'high'
        elif avg_volatility < 0.3:
            self.current_regime = 'low'
        else:
            self.current_regime = 'normal'
        
        return self.current_regime
    
    def get_adapted_params(self):
        if self.current_regime == 'high':
            return {
                'tp_pct': 0.015,
                'sl_pct': 0.010,
                'partial_pct': 0.012,
                'risk_multiplier': 0.5  # 🔥 Cut risk in half
            }
        elif self.current_regime == 'low':
            return {
                'tp_pct': 0.010,
                'sl_pct': 0.006,
                'partial_pct': 0.007,
                'risk_multiplier': 1.2
            }
        else:
            return {
                'tp_pct': TP_PCT,
                'sl_pct': SL_PCT,
                'partial_pct': PARTIAL_TP_PCT,
                'risk_multiplier': 1.0
            }


class RiskManager:
    def __init__(self, starting_equity):
        self.starting_equity = starting_equity
        self.daily_starting_equity = starting_equity
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.cooldown_until = None
        self.is_trading_paused = False
    
    def reset_daily_stats(self):
        current_date = datetime.now(timezone.utc).date()
        if current_date != self.last_reset_date:
            self.daily_starting_equity = paper_broker.equity
            self.daily_trade_count = 0
            self.last_reset_date = current_date
            self.is_trading_paused = False
            logging.info(f"📅 Daily reset | Equity: ₹{self.daily_starting_equity:.2f}")
    
    def check_daily_loss_limit(self, current_equity):
        daily_pnl_pct = ((current_equity - self.daily_starting_equity) / self.daily_starting_equity) * 100
        
        if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
            self.is_trading_paused = True
            logging.warning(f"🛑 DAILY LOSS LIMIT: {daily_pnl_pct:.2f}%")
            
            if ENABLE_TELEGRAM:
                send_telegram(f"""
⚠️ <b>DAILY LOSS LIMIT HIT</b>

📉 Daily P&L: {daily_pnl_pct:.2f}%
🛑 Trading paused until next day
💰 Equity: ₹{current_equity:.2f}
""", silent=False)
            return False
        
        return True
    
    def check_trade_limit(self):
        return self.daily_trade_count < MAX_DAILY_TRADES
    
    def record_trade_result(self, pnl):
        self.daily_trade_count += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
            
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=CONSECUTIVE_LOSS_COOLDOWN)
                logging.warning(f"⚠️ {MAX_CONSECUTIVE_LOSSES} LOSSES | Cooldown {CONSECUTIVE_LOSS_COOLDOWN//60}min")
                
                if ENABLE_TELEGRAM:
                    send_telegram(f"""
⚠️ <b>LOSS BREAKER ACTIVATED</b>

📉 {self.consecutive_losses} consecutive losses
⏸️ Cooling down for {CONSECUTIVE_LOSS_COOLDOWN//60} minutes
🧘 Switching to defensive mode
""", silent=False)
        else:
            self.consecutive_losses = 0
    
    def can_trade(self):
        self.reset_daily_stats()
        
        if self.is_trading_paused:
            return False
        
        if self.cooldown_until and datetime.now(timezone.utc) < self.cooldown_until:
            return False
        else:
            self.cooldown_until = None
        
        if not self.check_trade_limit():
            return False
        
        return True


# =======================
# API & Telegram
# =======================
def api_call_with_retry(func, *args, max_retries=MAX_API_RETRIES, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                logging.error(f"API failed after {max_retries} attempts: {e}")
                return None
            
            wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
            logging.warning(f"API retry {attempt + 1}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
    
    return None


last_no_trade_notification_time = 0


def send_telegram(message, silent=False):
    if not ENABLE_TELEGRAM:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": silent
        }
        response = requests.post(url, data=data, timeout=5)
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.text}")
    except Exception as e:
        logging.error(f"Telegram failed: {e}")


def send_minute_update():
    global last_no_trade_notification_time
    
    marks = {s: float(states[s].candles[-1]["c"]) for s in states if states[s].candles}
    
    has_positions = len(paper_broker.positions) > 0
    
    if not has_positions:
        current_time = time.time()
        if current_time - last_no_trade_notification_time < NO_TRADE_NOTIFICATION_INTERVAL:
            return
        last_no_trade_notification_time = current_time
    
    total_pnl = paper_broker.equity - paper_broker.start_cash
    pnl_pct = (total_pnl / paper_broker.start_cash) * 100
    
    daily_pnl = paper_broker.equity - risk_manager.daily_starting_equity
    daily_pnl_pct = (daily_pnl / risk_manager.daily_starting_equity) * 100 if risk_manager.daily_starting_equity > 0 else 0
    
    pos_text = ""
    if has_positions:
        for sym, pos in paper_broker.positions.items():
            if sym in marks:
                upnl_pct = pos.uPnL_pct(marks[sym]) * 100
                upnl_inr = pos.uPnL_inr(marks[sym])
                hold_minutes = (datetime.now(timezone.utc) - pos.open_time).seconds // 60
                pos_text += f"\n📊 {sym}: {pos.side.upper()} | ₹{upnl_inr:+.2f} ({upnl_pct:+.2f}%) | {hold_minutes}m"
    else:
        pos_text = "\n💤 No positions"
    
    regime_emoji = "🔥" if market_regime.current_regime == 'high' else "❄️" if market_regime.current_regime == 'low' else "🌊"
    
    message = f"""
⏱️ <b>{'Active' if has_positions else 'Status'}</b>

💰 <b>Total:</b> ₹{total_pnl:+.2f} ({pnl_pct:+.2f}%)
📅 <b>Daily:</b> ₹{daily_pnl:+.2f} ({daily_pnl_pct:+.2f}%)
💼 <b>Equity:</b> ₹{paper_broker.equity:.2f}
{regime_emoji} <b>Regime:</b> {market_regime.current_regime.upper()}
{pos_text}

📊 {risk_manager.daily_trade_count}/{MAX_DAILY_TRADES} | {risk_manager.consecutive_losses} losses
"""
    
    send_telegram(message, silent=not has_positions)


def send_trade_notification(action, symbol, qty, price, pnl=None, pnl_pct=None, reason=None, margin_used=None, strategy=None):
    if action in ["buy", "sell"]:
        notional = qty * price
        adapted = market_regime.get_adapted_params()
        
        message = f"""
🚀 <b>TRADE OPENED - MEAN REVERSION</b>

📊 <b>Symbol:</b> {symbol}
{'🟢 LONG' if action == 'buy' else '🔴 SHORT'}
💰 <b>Entry:</b> ₹{price:.4f}
📦 <b>Qty:</b> {qty:.4f}
💵 <b>Notional:</b> ₹{notional:.2f}
💼 <b>Margin:</b> ₹{margin_used:.2f}

🎯 <b>Targets:</b>
├ TP: +{adapted['tp_pct']*100:.1f}%
├ Partial: +{adapted['partial_pct']*100:.1f}%
└ SL: -{adapted['sl_pct']*100:.1f}%

📈 <b>Strategy:</b> {strategy or 'Mean Reversion'}
{' 🔥' if market_regime.current_regime == 'high' else '❄️' if market_regime.current_regime == 'low' else '🌊'} <b>Regime:</b> {market_regime.current_regime}

💼 Equity: ₹{paper_broker.equity:.2f}
"""
    else:
        emoji = "✅" if pnl > 0 else "🛑"
        
        reason_map = {
            "tp": "🎯 Take Profit",
            "sl": "🛑 Stop Loss",
            "partial": "📊 Partial",
            "time": "⏰ Time",
            "trailing": "📉 Trailing",
            "emergency": "🚨 Emergency"
        }
        reason_text = reason_map.get(reason, "")
        
        message = f"""
{emoji} <b>TRADE CLOSED</b>

📊 {symbol}
💰 Exit: ₹{price:.4f}
{'🎯' if pnl > 0 else '🛑'} P&L: ₹{pnl:+.2f} ({pnl_pct:+.2f}%)
{reason_text}

💼 Equity: ₹{paper_broker.equity:.2f}
📅 Daily: {risk_manager.daily_trade_count} | {risk_manager.consecutive_losses} losses
📈 Win Rate: {performance_tracker.get_win_rate():.1f}%
"""
    
    send_telegram(message, silent=False)


def send_final_report():
    closed_trades = [t for t in paper_broker.trades if 'pnl' in t]
    winning_trades = [t for t in closed_trades if t['pnl'] > 0]
    losing_trades = [t for t in closed_trades if t['pnl'] < 0]
    
    win_rate = performance_tracker.get_win_rate()
    
    total_pnl = paper_broker.equity - paper_broker.start_cash
    pnl_pct = (total_pnl / paper_broker.start_cash) * 100
    
    avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    total_fees = sum(t.get('fee', 0) for t in paper_broker.trades)
    
    avg_hold = sum(t.get('hold_time_sec', 0) for t in closed_trades) / len(closed_trades) if closed_trades else 0
    avg_hold_min = int(avg_hold // 60)
    
    runtime = datetime.now(timezone.utc) - start_time if start_time else timedelta(0)
    hours = runtime.seconds // 3600
    minutes = (runtime.seconds % 3600) // 60
    
    best_hours = performance_tracker.get_best_hours()
    best_symbols = performance_tracker.get_best_symbols()
    
    message = f"""
📊 <b>═══ FINAL REPORT ═══</b>

⏱️ <b>Runtime:</b> {hours}h {minutes}m

💰 <b>ACCOUNT</b>
├ Starting: ₹{paper_broker.start_cash:.2f}
├ Final: ₹{paper_broker.equity:.2f}
└ P&L: ₹{total_pnl:+.2f} ({pnl_pct:+.2f}%)

📈 <b>TRADES</b>
├ Total: {len(closed_trades)}
├ Wins: {len(winning_trades)} ✅
├ Losses: {len(losing_trades)} ❌
├ Win Rate: {win_rate:.1f}%
├ Avg Win: ₹{avg_win:.2f}
├ Avg Loss: ₹{avg_loss:.2f}
├ Avg Hold: {avg_hold_min}m
└ Fees: ₹{total_fees:.2f}

🎯 <b>ANALYTICS</b>
├ Best Hours: {', '.join([f'{h}:00' for h in best_hours]) if best_hours else 'N/A'}
├ Best Symbols: {', '.join([s.replace('I-','').replace('_INR','') for s in best_symbols]) if best_symbols else 'N/A'}

🔧 <b>STRATEGY</b>
• Mean Reversion (Bollinger Bands)
• ₹800-1200 margin per trade
• Skip high volatility periods

{'🎉 PROFITABLE!' if total_pnl > 0 else '⚠️ IN LOSS' if total_pnl < 0 else '⚪ BREAKEVEN'}
"""
    
    if closed_trades:
        message += "\n\n<b>Last 5:</b>\n"
        for t in closed_trades[-5:]:
            emoji = "✅" if t['pnl'] > 0 else "❌"
            hold = t.get('hold_time_sec', 0)
            message += f"{emoji} {t['symbol']}: ₹{t['pnl']:+.2f} ({hold//60}m)\n"
    
    send_telegram(message, silent=False)


# =======================
# Health Check
# =======================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        uptime = (datetime.now(timezone.utc) - start_time).seconds if start_time else 0
        
        status = f"""
        <html>
        <head><title>Mean Reversion Bot</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>🚀 Mean Reversion Bot: RUNNING ✅</h1>
            <p><strong>Strategy:</strong> Bollinger Bands Mean Reversion</p>
            <p><strong>Regime:</strong> {market_regime.current_regime.upper()}</p>
            <p><strong>Margin:</strong> ₹800-1200 per trade</p>
            <p><strong>Daily Trades:</strong> {risk_manager.daily_trade_count}/{MAX_DAILY_TRADES}</p>
            <p><strong>Losses:</strong> {risk_manager.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}</p>
            <p><strong>Equity:</strong> ₹{paper_broker.equity:.2f}</p>
            <p><strong>Win Rate:</strong> {performance_tracker.get_win_rate():.1f}%</p>
            <p><strong>Status:</strong> {'🟢 ACTIVE' if risk_manager.can_trade() else '🔴 PAUSED'}</p>
        </body>
        </html>
        """
        self.wfile.write(status.encode())
    
    def log_message(self, format, *args):
        pass

def start_health_server():
    try:
        server = HTTPServer(('0.0.0.0', PORT), HealthCheckHandler)
        logging.info(f"✓ Health server on port {PORT}")
        server.serve_forever()
    except Exception as e:
        logging.error(f"Health server error: {e}")


# =======================
# Market Functions
# =======================
def calculate_atr(candles, period=ATR_PERIOD):
    if len(candles) < period + 1:
        return None
    
    true_ranges = []
    for i in range(-period, 0):
        high = float(candles[i]["h"])
        low = float(candles[i]["l"])
        prev_close = float(candles[i-1]["c"])
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    
    return sum(true_ranges) / len(true_ranges)


def sufficient_volatility(state):
    if len(state.candles) < ATR_PERIOD + 1:
        return True
    
    atr = calculate_atr(state.candles, ATR_PERIOD)
    if atr is None:
        return True
    
    current_price = float(state.candles[-1]["c"])
    atr_percent = (atr / current_price) * 100
    
    market_regime.update(atr_percent)
    
    return atr_percent >= MIN_ATR_PERCENT


def is_good_trading_hour():
    current_hour = datetime.now(timezone.utc).hour
    
    best_hours = performance_tracker.get_best_hours()
    if best_hours and len(performance_tracker.trades) > 10:
        if current_hour in best_hours:
            return True
    
    if current_hour in AVOID_HOURS_UTC:
        return False
    
    return True


def get_available_markets():
    def fetch():
        url = f"{BASE_URL}/exchange/v1/markets_details"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    
    markets = api_call_with_retry(fetch)
    if markets:
        logging.info(f"✓ Retrieved {len(markets)} markets")
    return markets or []


def select_tradable_symbols():
    markets = get_available_markets()
    
    if not markets:
        return [f"I-{coin}_INR" for coin in TARGET_COINS[:3]]
    
    tickers = fetch_all_tickers()
    
    quality_markets = []
    for market in markets:
        pair = market.get('pair', '')
        
        if not (pair.startswith('I-') and '_INR' in pair and market.get('status') == 'active'):
            continue
        
        coin = pair.replace('I-', '').replace('_INR', '')
        if coin not in TARGET_COINS:
            continue
        
        ticker_format = pair.replace('I-', '').replace('_', '')
        if ticker_format not in tickers:
            continue
        
        ticker = tickers[ticker_format]
        price = ticker['price']
        volume_24h = ticker['volume'] * price
        
        if price < MIN_PRICE_INR or volume_24h < MIN_24H_VOLUME_INR:
            continue
        
        quality_markets.append({'pair': pair, 'price': price, 'volume_24h': volume_24h})
    
    quality_markets.sort(key=lambda x: x['volume_24h'], reverse=True)
    
    selected = [m['pair'] for m in quality_markets[:3]]  # Top 3 only
    
    if selected:
        logging.info(f"✅ Selected {len(selected)} markets")
    else:
        selected = [f"I-{coin}_INR" for coin in TARGET_COINS[:3]]
    
    return selected


def hmac_signature(secret: str, body: dict) -> str:
    payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def ema(prev_ema, price, length):
    k = 2 / (length + 1)
    return price * k + prev_ema * (1 - k)


def rsi(prices, length=14):
    if len(prices) < length + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, length + 1):
        ch = prices[-i] - prices[-i-1]
        if ch > 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / length) / (losses / length)
    return 100.0 - (100.0 / (1.0 + rs))


def fetch_all_tickers():
    def fetch():
        url = f"{BASE_URL}/exchange/ticker"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            tickers = response.json()
            ticker_dict = {}
            for ticker in tickers:
                market = ticker.get('market')
                if market:
                    ticker_dict[market] = {
                        'price': float(ticker.get('last_price', 0)),
                        'volume': float(ticker.get('volume', 0)),
                        'high': float(ticker.get('high', 0)),
                        'low': float(ticker.get('low', 0)),
                        'bid': float(ticker.get('bid', 0)),
                        'ask': float(ticker.get('ask', 0)),
                        'change_24h': float(ticker.get('change_24_hour', 0)),
                    }
            return ticker_dict
        return {}
    
    return api_call_with_retry(fetch) or {}


def fetch_ticker_data(symbol):
    all_tickers = fetch_all_tickers()
    ticker_format = symbol.replace('I-', '').replace('_', '')
    return all_tickers.get(ticker_format)


def create_candle_from_ticker(symbol, ticker_data):
    if not ticker_data or ticker_data['price'] == 0:
        return None
    
    price = ticker_data['price']
    return {
        "t": int(time.time() * 1000),
        "o": price,
        "h": ticker_data.get('high', price),
        "l": ticker_data.get('low', price),
        "c": price,
        "v": ticker_data.get('volume', 0),
    }


# =======================
# Paper Broker
# =======================
class Position:
    def __init__(self, symbol, side, qty, entry_price):
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.entry_price = entry_price
        self.open_time = datetime.now(timezone.utc)
        self.trailing_stop_price = None
        self.max_favorable_price = entry_price
        self.partial_taken = False

    def uPnL_pct(self, last_price):
        if self.side == "buy":
            return (last_price / self.entry_price) - 1.0
        else:
            return (self.entry_price / last_price) - 1.0
    
    def uPnL_inr(self, last_price):
        if self.side == "buy":
            return (last_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - last_price) * self.qty


class PaperBroker:
    def __init__(self, start_cash_inr=350.0):
        self.cash = start_cash_inr
        self.equity = start_cash_inr
        self.positions = {}
        self.trades = []
        self.start_cash = start_cash_inr

    def _fee_cost(self, notional, taker=True):
        fee = TAKER_FEE if taker else MAKER_FEE
        return notional * fee

    def place_market(self, symbol, side, qty, price, margin_used, strategy=None):
        if qty <= 0:
            return False, "qty<=0"
        if symbol in self.positions:
            return False, "Position exists"
        
        notional = qty * price
        fee = self._fee_cost(notional, taker=True)
        
        if self.cash < fee:
            return False, "Insufficient cash"
        
        self.cash -= fee
        self.positions[symbol] = Position(symbol, "buy" if side == "buy" else "sell", qty, price)
        self.trades.append({
            "ts": datetime.now(timezone.utc),
            "symbol": symbol,
            "action": side,
            "qty": qty,
            "price": price,
            "fee": fee,
            "notional": notional,
            "margin_used": margin_used,
            "strategy": strategy
        })
        
        logging.info(f"✅ OPENED {side.upper()} {symbol} @ ₹{price:.4f} | {strategy}")
        
        send_trade_notification(side, symbol, qty, price, margin_used=margin_used, strategy=strategy)
        
        return True, {"status": "filled", "avg_price": price}

    def close_market(self, symbol, price, qty=None, reason=None):
        if symbol not in self.positions:
            return False, "No position"
        pos = self.positions[symbol]
        q = pos.qty if qty is None else min(qty, pos.qty)
        if q <= 0:
            return False, "qty<=0"
        
        notional = q * price
        fee = self._fee_cost(notional, taker=True)
        
        if pos.side == "buy":
            pnl = (price - pos.entry_price) * q
        else:
            pnl = (pos.entry_price - price) * q
        
        self.cash += pnl
        self.cash -= fee
        
        hold_time = (datetime.now(timezone.utc) - pos.open_time).seconds
        pnl_pct = (pnl / (pos.entry_price * q)) * 100
        
        self.trades.append({
            "ts": datetime.now(timezone.utc),
            "symbol": symbol,
            "action": "close",
            "qty": q,
            "price": price,
            "fee": fee,
            "pnl": pnl,
            "notional": notional,
            "hold_time_sec": hold_time,
            "reason": reason
        })
        
        logging.info(f"📉 CLOSED {symbol} | ₹{pnl:+.2f} ({pnl_pct:+.2f}%) | {reason}")
        
        send_trade_notification("close", symbol, q, price, pnl, pnl_pct, reason)
        
        risk_manager.record_trade_result(pnl)
        risk_manager.check_daily_loss_limit(self.equity)
        
        performance_tracker.record_trade({
            "ts": datetime.now(timezone.utc),
            "symbol": symbol,
            "pnl": pnl,
            "reason": reason,
            "hold_time_sec": hold_time
        })
        
        pos.qty -= q
        if pos.qty <= 1e-12:
            del self.positions[symbol]
        
        return True, {"status": "closed", "pnl": pnl}

    def update_equity(self, marks):
        upnl = 0.0
        for sym, pos in list(self.positions.items()):
            lp = marks.get(sym)
            if lp is None:
                continue
            upnl += pos.uPnL_inr(lp)
        self.equity = self.cash + upnl


# =======================
# 🎯 MEAN REVERSION STRATEGY
# =======================
class SymbolState:
    def __init__(self, symbol, broker):
        self.symbol = symbol
        self.candles = []
        self.closes = []
        self.volumes = []
        self.ema_fast = None
        self.ema_slow = None
        self.position = None
        self.broker = broker
        self.last_update = None


def update_indicators(state):
    if not state.candles:
        return
    closes = [float(c["c"]) for c in state.candles]
    vols = [float(c.get("v", 0)) for c in state.candles]
    state.closes = closes
    state.volumes = vols
    
    if state.ema_fast is None:
        if len(closes) >= EMA_SLOW:
            state.ema_fast = sum(closes[-EMA_FAST:]) / EMA_FAST
            state.ema_slow = sum(closes[-EMA_SLOW:]) / EMA_SLOW
    else:
        state.ema_fast = ema(state.ema_fast, closes[-1], EMA_FAST)
        state.ema_slow = ema(state.ema_slow, closes[-1], EMA_SLOW)


def mean_reversion_signal(state):
    """🎯 BOLLINGER BANDS MEAN REVERSION"""
    if len(state.closes) < BOLLINGER_PERIOD:
        return None, None
    
    current_price = state.closes[-1]
    
    # Calculate Bollinger Bands
    sma = sum(state.closes[-BOLLINGER_PERIOD:]) / BOLLINGER_PERIOD
    variance = sum((x - sma) ** 2 for x in state.closes[-BOLLINGER_PERIOD:]) / BOLLINGER_PERIOD
    std = math.sqrt(variance)
    
    upper_band = sma + (std * BOLLINGER_STD)
    lower_band = sma - (std * BOLLINGER_STD)
    
    # RSI for confirmation
    r = rsi(state.closes, RSI_LEN)
    if r is None:
        return None, None
    
    # Distance from bands
    distance_from_lower = ((current_price - lower_band) / lower_band) * 100
    distance_from_upper = ((upper_band - current_price) / upper_band) * 100
    
    # 🎯 LONG: Price at/below lower band + oversold
    if current_price <= lower_band * 1.005 and r < RSI_OVERSOLD:
        logging.info(f"🎯 {state.symbol} OVERSOLD | Price: ₹{current_price:.4f} | Lower BB: ₹{lower_band:.4f} | RSI: {r:.1f}")
        return 'long', f"Oversold (RSI:{r:.1f})"
    
    # 🎯 SHORT: Price at/above upper band + overbought
    elif current_price >= upper_band * 0.995 and r > RSI_OVERBOUGHT:
        logging.info(f"🎯 {state.symbol} OVERBOUGHT | Price: ₹{current_price:.4f} | Upper BB: ₹{upper_band:.4f} | RSI: {r:.1f}")
        return 'short', f"Overbought (RSI:{r:.1f})"
    
    return None, None


def calculate_dynamic_margin(broker_equity):
    margin = (broker_equity * RISK_PER_TRADE_PCT) / 100.0
    return max(MIN_MARGIN_INR, min(margin, MAX_MARGIN_INR))


def calc_qty_from_inr(price_in_inr, per_trade_margin_inr, leverage):
    notional = per_trade_margin_inr * leverage
    qty = max(0.0, notional / max(1e-9, price_in_inr))
    if price_in_inr < 1:
        return round(qty, 2)
    elif price_in_inr < 10:
        return round(qty, 4)
    elif price_in_inr < 100:
        return round(qty, 6)
    else:
        return round(qty, 8)


def maybe_enter(state):
    """🎯 MEAN REVERSION ENTRY"""
    if len(state.candles) < 50 or state.position is not None:
        return
    
    if len(state.broker.positions) >= MAX_CONCURRENT_POSITIONS:
        return
    
    if not risk_manager.can_trade():
        return
    
    # 🚨 SKIP HIGH VOLATILITY
    if market_regime.current_regime == 'high':
        logging.debug(f"⏸️ Skipping {state.symbol} - HIGH volatility")
        return
    
    if not is_good_trading_hour():
        return
    
    if not sufficient_volatility(state):
        return
    
    # 🎯 Get mean reversion signal
    direction, reason = mean_reversion_signal(state)
    
    if direction is None:
        return
    
    adapted_params = market_regime.get_adapted_params()
    dynamic_margin = calculate_dynamic_margin(state.broker.equity) * adapted_params['risk_multiplier']
    
    last = state.candles[-1]
    close = float(last["c"])
    qty = calc_qty_from_inr(close, dynamic_margin, LEVERAGE)
    
    strategy_name = f"Mean Rev {reason}"
    
    if direction == 'long':
        ok, resp = state.broker.place_market(state.symbol, "buy", qty, close, dynamic_margin, strategy=strategy_name)
        if ok:
            state.position = Position(state.symbol, "buy", qty, close)
            
    elif direction == 'short':
        ok, resp = state.broker.place_market(state.symbol, "sell", qty, close, dynamic_margin, strategy=strategy_name)
        if ok:
            state.position = Position(state.symbol, "sell", qty, close)


def manage_exit(state):
    if state.position is None or len(state.candles) == 0:
        return
    
    last = float(state.candles[-1]["c"])
    pos = state.position
    pnl = pos.uPnL_pct(last)
    hold_time = (datetime.now(timezone.utc) - pos.open_time).seconds
    
    adapted = market_regime.get_adapted_params()

    # Emergency stop
    if hold_time < MIN_HOLD_TIME_SEC:
        if pnl <= -EMERGENCY_STOP_PCT:
            ok, resp = state.broker.close_market(pos.symbol, last, reason="emergency")
            if ok:
                state.position = None
            return
        
        if pnl <= -adapted['sl_pct']:
            return
    
    # Partial profit
    if pnl >= adapted['partial_pct'] and not pos.partial_taken:
        reduce_qty = round(pos.qty * PARTIAL_TAKE, 6)
        ok, resp = state.broker.close_market(pos.symbol, last, qty=reduce_qty, reason="partial")
        if ok:
            pos.qty -= reduce_qty
            pos.partial_taken = True
            if pos.qty <= 1e-12:
                state.position = None
            return

    # Take profit
    if pnl >= adapted['tp_pct']:
        ok, resp = state.broker.close_market(pos.symbol, last, reason="tp")
        if ok:
            state.position = None
        return
    
    # Stop loss
    if hold_time >= MIN_HOLD_TIME_SEC and pnl <= -adapted['sl_pct']:
        ok, resp = state.broker.close_market(pos.symbol, last, reason="sl")
        if ok:
            state.position = None
        return

    # Time exit
    if hold_time >= MAX_HOLD_SEC:
        ok, resp = state.broker.close_market(pos.symbol, last, reason="time")
        if ok:
            state.position = None
        return

    # Trailing stop
    if pnl >= TSL_ACTIVATE:
        if pos.side == "buy":
            pos.max_favorable_price = max(pos.max_favorable_price, last)
            new_trailing = pos.max_favorable_price * (1.0 - TSL_STEP)
            if pos.trailing_stop_price is None or new_trailing > pos.trailing_stop_price:
                pos.trailing_stop_price = new_trailing
            if last <= pos.trailing_stop_price:
                ok, resp = state.broker.close_market(pos.symbol, last, reason="trailing")
                if ok:
                    state.position = None
        else:
            pos.max_favorable_price = min(pos.max_favorable_price, last)
            new_trailing = pos.max_favorable_price * (1.0 + TSL_STEP)
            if pos.trailing_stop_price is None or new_trailing < pos.trailing_stop_price:
                pos.trailing_stop_price = new_trailing
            if last >= pos.trailing_stop_price:
                ok, resp = state.broker.close_market(pos.symbol, last, reason="trailing")
                if ok:
                    state.position = None


# =======================
# Globals & Main
# =======================
performance_tracker = PerformanceTracker()
market_regime = MarketRegime()
risk_manager = None
paper_broker = PaperBroker(start_cash_inr=STARTING_CAPITAL_INR)
states = {}
stop_flag = False
SYMBOLS = []
start_time = None


def ticker_polling_thread():
    while not stop_flag:
        try:
            time.sleep(30)
            
            for sym in SYMBOLS:
                if sym not in states:
                    continue
                
                st = states[sym]
                ticker = fetch_ticker_data(sym)
                
                if ticker and ticker['price'] > 0:
                    candle = create_candle_from_ticker(sym, ticker)
                    if candle:
                        current_minute = candle['t'] // 60000
                        last_minute = st.candles[-1]['t'] // 60000 if st.candles else 0
                        
                        if current_minute != last_minute:
                            st.candles.append(candle)
                            st.last_update = datetime.now(timezone.utc)
                            
                            if len(st.candles) > 500:
                                st.candles = st.candles[-500:]
                            
                            logging.info(f"🕯️ {sym:12} ₹{candle['c']:8.4f}")
                            
                            update_indicators(st)
                            maybe_enter(st)
                            manage_exit(st)
            
            marks = {s: float(states[s].candles[-1]["c"]) for s in states if states[s].candles}
            if marks:
                paper_broker.update_equity(marks)
                send_minute_update()
                
        except Exception as e:
            logging.exception(f"✗ Error: {e}")


def initialize_with_ticker():
    logging.info("\n📊 Initializing mean reversion bot...")
    
    for sym in SYMBOLS:
        ticker = fetch_ticker_data(sym)
        if ticker and ticker['price'] > 0:
            states[sym].candles = []
            base_price = ticker['price']
            
            for i in range(50):
                fake_candle = {
                    "t": int((time.time() - (50-i)*60) * 1000),
                    "o": base_price * (1 + (i-25)*0.0005),
                    "h": base_price * (1 + (i-25)*0.0005 + 0.001),
                    "l": base_price * (1 + (i-25)*0.0005 - 0.001),
                    "c": base_price * (1 + (i-24)*0.0005),
                    "v": 100000,
                }
                states[sym].candles.append(fake_candle)
            
            states[sym].last_update = datetime.now(timezone.utc)
            update_indicators(states[sym])
            
            logging.info(f"✓ {sym:12} @ ₹{base_price:8.4f}")


def main():
    global stop_flag, SYMBOLS, states, start_time, last_no_trade_notification_time, risk_manager
    
    start_time = datetime.now(timezone.utc)
    last_no_trade_notification_time = time.time()
    risk_manager = RiskManager(STARTING_CAPITAL_INR)
    
    logging.info("=" * 70)
    logging.info("🚀 Mean Reversion Bot")
    logging.info(f"Capital: ₹{STARTING_CAPITAL_INR}")
    logging.info(f"Strategy: Bollinger Bands Mean Reversion")
    logging.info(f"Margin: ₹{MIN_MARGIN_INR}-{MAX_MARGIN_INR} | SL: {SL_PCT*100:.1f}% | TP: {TP_PCT*100:.1f}%")
    logging.info(f"SKIPS: High volatility periods")
    logging.info("=" * 70)
    
    if ENABLE_TELEGRAM:
        send_telegram(f"""
🚀 <b>MEAN REVERSION BOT STARTED</b>

💰 Capital: ₹{STARTING_CAPITAL_INR}

🎯 <b>Strategy:</b>
• Bollinger Bands (20-period, 2σ)
• Buy oversold (RSI below 35)
• Sell overbought (RSI above 65)
• 🚨 SKIPS high volatility

💼 <b>Risk:</b>
• Margin: ₹{MIN_MARGIN_INR}-{MAX_MARGIN_INR}/trade
• SL: {SL_PCT*100:.1f}% | TP: {TP_PCT*100:.1f}%
• Min hold: {MIN_HOLD_TIME_SEC//60} min

🎯 <b>Target:</b>
• Win rate: 60-70%
• Better fit for ranging markets

{'💰 PAPER MODE' if PAPER_MODE else '🔴 LIVE'}
""")
    
    SYMBOLS = select_tradable_symbols()
    
    if not SYMBOLS:
        logging.error("✗ No symbols")
        return
    
    states = {sym: SymbolState(sym, paper_broker) for sym in SYMBOLS}
    
    initialize_with_ticker()
    
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    polling_thread = threading.Thread(target=ticker_polling_thread, daemon=True)
    polling_thread.start()
    
    logging.info("\n✓ Mean reversion bot running | Press Ctrl+C to stop\n")
    
    try:
        while not stop_flag:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\n⚠️  Shutting down...")
        stop_flag = True
        time.sleep(1)
        send_final_report()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        global stop_flag
        stop_flag = True
        time.sleep(1)
        send_final_report()
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
