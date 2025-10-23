import os
import time
import hmac
import json
import math
import signal
import hashlib
import logging
import threading
from datetime import datetime, timedelta

import requests


# =======================
# Config
# =======================
API_KEY = os.getenv("COINDCX_API_KEY", "")
API_SECRET = os.getenv("COINDCX_API_SECRET", "")
BASE_URL = "https://api.coindcx.com"
FUTURES_ORDER_CREATE = "/exchange/v1/derivatives/futures/orders/create"

# Paper mode toggle
PAPER_MODE = True

# ⚡ YOUR ACTUAL ACCOUNT SIZE
STARTING_CAPITAL_INR = 350.0  # Your futures wallet balance
MAX_CONCURRENT_POSITIONS = 1   # With ₹350, only 1 trade at a time

# TARGET COINS - Cheaper altcoins
TARGET_COINS = ['MATIC', 'TRX', 'ADA', 'ALGO', 'XRP']

# Strategy parameters
SL_PCT = 0.0075        # 0.75% stop loss
TP_PCT = 0.015         # 1.5% take profit
PARTIAL_TP_PCT = 0.01  # 1.0% partial profit
PARTIAL_TAKE = 0.50    # reduce 50% at partial TP
TSL_ACTIVATE = 0.008   # activate trailing after +0.8%
TSL_STEP = 0.002       # trail by 0.2% steps
MAX_HOLD_SEC = 60 * 60 # 60 minutes

# Indicators
RSI_LEN = 14
EMA_FAST = 20
EMA_SLOW = 50
VOL_MA = 20
BREAKOUT_LOOKBACK = 15
RSI_BUY_MAX = 65.0
RSI_SELL_MIN = 35.0

# Risk & leverage
LEVERAGE = 3
PER_TRADE_MARGIN_INR = 300.0  # ₹300 per trade with 3x leverage = ₹900 position

# Fees
TAKER_FEE = 0.001
MAKER_FEE = 0.001

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)


# =======================
# Discover Available Markets
# =======================
def get_available_markets():
    """Query CoinDCX to get available trading pairs"""
    try:
        url = f"{BASE_URL}/exchange/v1/markets_details"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            markets = response.json()
            logging.info(f"✓ Retrieved {len(markets)} markets from CoinDCX")
            return markets
        else:
            logging.error(f"✗ Failed to fetch markets: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"✗ Error fetching markets: {e}")
        return []


def select_tradable_symbols():
    """Select specific target coins for trading"""
    markets = get_available_markets()
    
    if not markets:
        logging.warning("⚠️ Could not fetch markets, using defaults")
        return [f"I-{coin}_INR" for coin in TARGET_COINS]
    
    # Find active INR spot markets for target coins
    active_inr = [
        m['pair'] for m in markets 
        if 'INR' in m.get('pair', '') 
        and m.get('status') == 'active'
        and m['pair'].startswith('I-')
    ]
    
    # Match our target coins
    selected = []
    for coin in TARGET_COINS:
        pair = f"I-{coin}_INR"
        if pair in active_inr:
            selected.append(pair)
            logging.info(f"✓ Found {pair}")
        else:
            logging.warning(f"⚠️ {pair} not found or inactive")
    
    if not selected:
        logging.error("✗ None of the target coins are available!")
        return []
    
    logging.info(f"\n✅ Selected {len(selected)} symbols: {', '.join(selected)}")
    return selected


# =======================
# Utils & Indicators
# =======================
def hmac_signature(secret: str, body: dict) -> str:
    payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return sig


def ema(prev_ema, price, length):
    k = 2 / (length + 1)
    return price * k + prev_ema * (1 - k)


def rsi(prices, length=14):
    if len(prices) < length + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-length, 0):
        ch = prices[i] - prices[i-1]
        if ch > 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / length) / (losses / length)
    return 100.0 - (100.0 / (1.0 + rs))


# =======================
# Ticker Data
# =======================
def fetch_all_tickers():
    """Fetch all tickers from CoinDCX"""
    try:
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
    except Exception as e:
        logging.error(f"✗ Error fetching tickers: {e}")
        return {}


def convert_symbol_to_ticker_format(symbol):
    """Convert I-BTC_INR to BTCINR format"""
    return symbol.replace('I-', '').replace('_', '')


def fetch_ticker_data(symbol):
    """Fetch ticker data for a specific symbol"""
    try:
        all_tickers = fetch_all_tickers()
        ticker_format = convert_symbol_to_ticker_format(symbol)
        
        if ticker_format in all_tickers:
            return all_tickers[ticker_format]
        
        logging.warning(f"⚠️ No ticker data for {symbol} (looking for {ticker_format})")
        return None
    except Exception as e:
        logging.error(f"✗ Error fetching ticker for {symbol}: {e}")
        return None


def create_candle_from_ticker(symbol, ticker_data):
    """Create a synthetic 1-minute candle from ticker data"""
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
        self.open_time = datetime.utcnow()
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

    def place_market(self, symbol, side, qty, price):
        if qty <= 0:
            return False, "qty<=0"
        if symbol in self.positions:
            return False, "Position already open"
        
        notional = qty * price
        fee = self._fee_cost(notional, taker=True)
        
        # Check if we have enough cash
        if self.cash < fee:
            logging.warning(f"⚠️ Insufficient cash for fees: Need ₹{fee:.2f}, Have ₹{self.cash:.2f}")
            return False, "Insufficient cash"
        
        self.cash -= fee
        self.positions[symbol] = Position(symbol, "buy" if side == "buy" else "sell", qty, price)
        self.trades.append({
            "ts": datetime.utcnow(),
            "symbol": symbol,
            "action": side,
            "qty": qty,
            "price": price,
            "fee": fee,
            "notional": notional
        })
        logging.info(f"📈 OPENED {side.upper()} {symbol} qty={qty:.4f} @ ₹{price:.4f} (Notional: ₹{notional:.2f}, Fee: ₹{fee:.2f})")
        return True, {"status": "filled", "avg_price": price}

    def close_market(self, symbol, price, qty=None):
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
        
        hold_time = (datetime.utcnow() - pos.open_time).seconds
        
        self.trades.append({
            "ts": datetime.utcnow(),
            "symbol": symbol,
            "action": "close",
            "qty": q,
            "price": price,
            "fee": fee,
            "pnl": pnl,
            "notional": notional,
            "hold_time_sec": hold_time
        })
        
        pnl_pct = (pnl / (pos.entry_price * q)) * 100
        logging.info(f"📉 CLOSED {symbol} qty={q:.4f} @ ₹{price:.4f} | PnL: ₹{pnl:+.2f} ({pnl_pct:+.2f}%) | Hold: {hold_time//60}m | Fee: ₹{fee:.2f}")
        
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
            if pos.side == "buy":
                upnl += (lp - pos.entry_price) * pos.qty
            else:
                upnl += (pos.entry_price - lp) * pos.qty
        self.equity = self.cash + upnl


# =======================
# Symbol State & Strategy
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

    def last_close(self):
        return float(self.candles[-1]["c"]) if self.candles else None


def update_indicators(state: SymbolState):
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


def compute_breakout_levels(state: SymbolState):
    if len(state.candles) < BREAKOUT_LOOKBACK + 1:
        return None, None
    highs = [float(c["h"]) for c in state.candles[-(BREAKOUT_LOOKBACK+1):-1]]
    lows = [float(c["l"]) for c in state.candles[-(BREAKOUT_LOOKBACK+1):-1]]
    return max(highs), min(lows)


def volume_ok(state: SymbolState):
    if len(state.candles) < VOL_MA + 1:
        return True
    vols = [float(c.get("v", 0)) for c in state.candles]
    if all(v == 0 for v in vols):
        return True
    last_v = vols[-1]
    ma = sum(vols[-VOL_MA:]) / VOL_MA
    return last_v > ma if ma > 0 else True


def eligible_momentum(state: SymbolState, long=True):
    if state.ema_fast is None or state.ema_slow is None or len(state.closes) < RSI_LEN + 1:
        return False
    r = rsi(state.closes, RSI_LEN)
    if r is None:
        return False
    if long:
        return (r < RSI_BUY_MAX) and (state.ema_fast > state.ema_slow) and volume_ok(state)
    else:
        return (r > RSI_SELL_MIN) and (state.ema_fast < state.ema_slow) and volume_ok(state)


def calc_qty_from_inr(price_in_inr, per_trade_margin_inr, leverage):
    """Calculate quantity based on margin and leverage"""
    notional = per_trade_margin_inr * leverage
    qty = max(0.0, notional / max(1e-9, price_in_inr))
    # Round to appropriate decimals based on price
    if price_in_inr < 1:
        return round(qty, 2)
    elif price_in_inr < 10:
        return round(qty, 4)
    elif price_in_inr < 100:
        return round(qty, 6)
    else:
        return round(qty, 8)


def maybe_enter(state: SymbolState):
    if len(state.candles) < BREAKOUT_LOOKBACK + 2 or state.position is not None:
        return
    
    # ⚡ CHECK: Don't enter if we already have max positions open
    if len(state.broker.positions) >= MAX_CONCURRENT_POSITIONS:
        return
    
    # ⚡ CHECK: Do we have enough cash for this trade?
    required_margin = PER_TRADE_MARGIN_INR * 0.002  # Just need fees (0.1% entry + 0.1% exit = 0.2%)
    if state.broker.cash < required_margin:
        logging.warning(f"⚠️ Insufficient cash for {state.symbol}: Need ₹{required_margin:.2f}, Have ₹{state.broker.cash:.2f}")
        return
    
    last = state.candles[-1]
    close = float(last["c"])
    hi, lo = compute_breakout_levels(state)
    if hi is None:
        return
    
    qty = calc_qty_from_inr(close, PER_TRADE_MARGIN_INR, LEVERAGE)
    
    if close > hi and eligible_momentum(state, long=True):
        ok, resp = state.broker.place_market(state.symbol, "buy", qty, close)
        if ok:
            state.position = Position(state.symbol, "buy", qty, close)
    elif close < lo and eligible_momentum(state, long=False):
        ok, resp = state.broker.place_market(state.symbol, "sell", qty, close)
        if ok:
            state.position = Position(state.symbol, "sell", qty, close)


def manage_exit(state: SymbolState):
    if state.position is None or len(state.candles) == 0:
        return
    last = float(state.candles[-1]["c"])
    pos = state.position
    pnl = pos.uPnL_pct(last)

    # Partial profit
    if pnl >= PARTIAL_TP_PCT and not pos.partial_taken:
        reduce_qty = round(pos.qty * PARTIAL_TAKE, 6)
        ok, resp = state.broker.close_market(pos.symbol, last, qty=reduce_qty)
        if ok:
            pos.qty -= reduce_qty
            pos.partial_taken = True
            if pos.qty <= 1e-12:
                state.position = None
            return

    # Hard TP/SL
    if pnl >= TP_PCT or pnl <= -SL_PCT:
        ok, resp = state.broker.close_market(pos.symbol, last)
        if ok:
            state.position = None
        return

    # Time exit
    if datetime.utcnow() - pos.open_time >= timedelta(seconds=MAX_HOLD_SEC):
        ok, resp = state.broker.close_market(pos.symbol, last)
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
                ok, resp = state.broker.close_market(pos.symbol, last)
                if ok:
                    state.position = None


# =======================
# Main Bot Logic
# =======================
paper_broker = PaperBroker(start_cash_inr=STARTING_CAPITAL_INR)
states = {}
stop_flag = False
SYMBOLS = []
start_time = None


def ticker_polling_thread():
    """Poll ticker API for price updates"""
    while not stop_flag:
        try:
            time.sleep(60)  # Every minute
            
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
                            st.last_update = datetime.utcnow()
                            
                            if len(st.candles) > 500:
                                st.candles = st.candles[-500:]
                            
                            # Show 24h change if available
                            change = ticker.get('change_24h', 0)
                            change_str = f"({change:+.2f}%)" if change != 0 else ""
                            
                            logging.info(f"🕯️ {sym:12} ₹{candle['c']:8.4f} {change_str} | H: ₹{candle['h']:8.4f} | L: ₹{candle['l']:8.4f}")
                            
                            update_indicators(st)
                            maybe_enter(st)
                            manage_exit(st)
                else:
                    logging.warning(f"⚠️ No ticker data for {sym}")
                
            marks = {s: float(states[s].candles[-1]["c"]) for s in states if states[s].candles}
            if marks:
                paper_broker.update_equity(marks)
                
                pos_info = []
                for sym, pos in paper_broker.positions.items():
                    if sym in marks:
                        upnl_pct = pos.uPnL_pct(marks[sym]) * 100
                        upnl_inr = pos.uPnL_inr(marks[sym])
                        pos_info.append(f"{sym} {pos.side} ₹{upnl_inr:+.2f} ({upnl_pct:+.2f}%)")
                
                pos_str = " | ".join(pos_info) if pos_info else "No open positions"
                logging.info(f"💰 Equity: ₹{paper_broker.equity:.2f} | Cash: ₹{paper_broker.cash:.2f} | {pos_str}\n")
                
        except Exception as e:
            logging.exception(f"✗ Ticker polling error: {e}")


def initialize_with_ticker():
    """Initialize candles using ticker data"""
    logging.info("\n📊 Initializing with ticker data...")
    
    for sym in SYMBOLS:
        ticker = fetch_ticker_data(sym)
        if ticker and ticker['price'] > 0:
            states[sym].candles = []
            base_price = ticker['price']
            
            # Generate 50 synthetic candles
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
            
            states[sym].last_update = datetime.utcnow()
            update_indicators(states[sym])
            
            # Calculate how many units we can buy
            qty = calc_qty_from_inr(base_price, PER_TRADE_MARGIN_INR, LEVERAGE)
            notional = qty * base_price
            
            logging.info(f"✓ {sym:12} @ ₹{base_price:8.4f} | Qty: {qty:10.4f} units | Notional: ₹{notional:.2f}")
        else:
            logging.error(f"✗ Failed to initialize {sym}")


def generate_report():
    """Generate final trading report"""
    logging.info("\n" + "=" * 70)
    logging.info("📊 FINAL TRADING REPORT")
    logging.info("=" * 70)
    
    if start_time:
        runtime = datetime.utcnow() - start_time
        hours = runtime.seconds // 3600
        minutes = (runtime.seconds % 3600) // 60
        logging.info(f"⏱️  Runtime: {hours}h {minutes}m")
    
    logging.info(f"\n💰 Account Summary:")
    logging.info(f"   Starting Capital: ₹{paper_broker.start_cash:.2f}")
    logging.info(f"   Final Equity:     ₹{paper_broker.equity:.2f}")
    logging.info(f"   Final Cash:       ₹{paper_broker.cash:.2f}")
    
    total_pnl = paper_broker.equity - paper_broker.start_cash
    pnl_pct = (total_pnl / paper_broker.start_cash) * 100
    logging.info(f"   Total P&L:        ₹{total_pnl:+.2f} ({pnl_pct:+.2f}%)")
    
    logging.info(f"\n📈 Trading Statistics:")
    logging.info(f"   Total Trades:     {len(paper_broker.trades)}")
    
    if paper_broker.trades:
        closed_trades = [t for t in paper_broker.trades if 'pnl' in t]
        if closed_trades:
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] < 0]
            
            logging.info(f"   Winning Trades:   {len(winning_trades)}")
            logging.info(f"   Losing Trades:    {len(losing_trades)}")
            
            if closed_trades:
                win_rate = (len(winning_trades) / len(closed_trades)) * 100
                logging.info(f"   Win Rate:         {win_rate:.1f}%")
            
            if winning_trades:
                avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
                max_win = max(t['pnl'] for t in winning_trades)
                logging.info(f"   Avg Win:          ₹{avg_win:.2f}")
                logging.info(f"   Max Win:          ₹{max_win:.2f}")
            
            if losing_trades:
                avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
                max_loss = min(t['pnl'] for t in losing_trades)
                logging.info(f"   Avg Loss:         ₹{avg_loss:.2f}")
                logging.info(f"   Max Loss:         ₹{max_loss:.2f}")
            
            total_fees = sum(t.get('fee', 0) for t in paper_broker.trades)
            logging.info(f"   Total Fees Paid:  ₹{total_fees:.2f}")
            
            if closed_trades:
                avg_hold = sum(t.get('hold_time_sec', 0) for t in closed_trades) / len(closed_trades)
                logging.info(f"   Avg Hold Time:    {int(avg_hold//60)}m {int(avg_hold%60)}s")
    
    if paper_broker.positions:
        logging.info(f"\n⚠️  Open Positions:")
        marks = {s: float(states[s].candles[-1]["c"]) for s in states if states[s].candles}
        for sym, pos in paper_broker.positions.items():
            if sym in marks:
                upnl_pct = pos.uPnL_pct(marks[sym]) * 100
                upnl_inr = pos.uPnL_inr(marks[sym])
                logging.info(f"   {sym:12} {pos.side.upper():4} {pos.qty:10.4f} @ ₹{pos.entry_price:8.4f} | Current: ₹{marks[sym]:8.4f} | uPnL: ₹{upnl_inr:+.2f} ({upnl_pct:+.2f}%)")
    
    if paper_broker.trades:
        logging.info(f"\n📋 Recent Trades (Last 10):")
        for t in paper_broker.trades[-10:]:
            pnl_str = f"| P&L: ₹{t.get('pnl', 0):+7.2f}" if 'pnl' in t else " " * 18
            logging.info(f"   {t['ts'].strftime('%Y-%m-%d %H:%M:%S')} | {t['symbol']:12} {t['action']:5} @ ₹{t.get('price', 0):8.4f} {pnl_str}")
    
    logging.info("=" * 70)
    logging.info("✅ Report saved to trading_bot.log")
    logging.info("=" * 70 + "\n")


def main():
    global stop_flag, SYMBOLS, states, start_time
    
    start_time = datetime.utcnow()
    
    logging.info("=" * 70)
    logging.info("🚀 CoinDCX Intraday Trading Bot")
    logging.info(f"Mode: {'💰 PAPER TRADING' if PAPER_MODE else '🔴 LIVE TRADING'}")
    logging.info(f"Starting Capital: ₹{STARTING_CAPITAL_INR}")
    logging.info(f"Target Coins: {', '.join(TARGET_COINS)}")
    logging.info(f"Margin/Trade: ₹{PER_TRADE_MARGIN_INR} | Leverage: {LEVERAGE}x | Max Position: ₹{PER_TRADE_MARGIN_INR * LEVERAGE}")
    logging.info(f"Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS}")
    logging.info("=" * 70)
    
    SYMBOLS = select_tradable_symbols()
    
    if not SYMBOLS:
        logging.error("✗ No trading symbols available. Exiting.")
        return
    
    states = {sym: SymbolState(sym, paper_broker) for sym in SYMBOLS}
    
    initialize_with_ticker()
    
    polling_thread = threading.Thread(target=ticker_polling_thread, daemon=True)
    polling_thread.start()
    logging.info("\n✓ Ticker polling thread started")
    logging.info("✓ Press Ctrl+C to stop and generate report\n")
    
    try:
        while not stop_flag:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\n⚠️  Shutting down gracefully...")
        stop_flag = True
        time.sleep(1)
        generate_report()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        global stop_flag
        logging.info("\n⚠️  Interrupt signal received...")
        stop_flag = True
        time.sleep(1)
        generate_report()
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
