"""
simulate.py — Realistic multi-trade offline simulation using synthetic BTC price data.
Uses Geometric Brownian Motion with regime switching, slippage, and funding costs.
No API keys required.

Usage:
    python simulate.py                # 100 trades, 10x leverage
    python simulate.py 50             # 50 trades, 10x leverage
    python simulate.py 100 25         # 100 trades, 25x leverage
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
import time

import numpy as np

# ── Parse args before importing bot modules ────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("trades",   nargs="?", type=int, default=100)
parser.add_argument("leverage", nargs="?", type=int, default=10)
_args = parser.parse_args()

if _args.leverage < 1 or _args.leverage > 100:
    sys.exit(f"Error: leverage must be between 1 and 100 (got {_args.leverage})")

os.environ.update({
    "PHEMEX_API_KEY":           "SIM_KEY",
    "PHEMEX_API_SECRET":        "SIM_SECRET",
    "PHEMEX_TESTNET":           "true",
    "TRADING_SYMBOL":           "BTC/USDT:USDT",
    "TRADING_TIMEFRAME":        "1m",
    "LEVERAGE":                 str(_args.leverage),
    "TRADE_SIZE_USDT":          "10",    # $10 fixed per trade on a $100 account
    "RISK_PER_TRADE_PCT":       "0",     # no compounding — fixed size only
    "EMA_FAST":                 "9",
    "EMA_SLOW":                 "21",
    "RSI_PERIOD":               "14",
    "RSI_OVERBOUGHT":           "70",
    "RSI_OVERSOLD":             "30",
    "RSI_LONG_MAX":             "60",
    "RSI_SHORT_MIN":            "40",
    "TAKE_PROFIT_PCT":          "0.6",
    "STOP_LOSS_PCT":            "0.35",
    "MAX_SESSION_LOSS_PCT":     "30",
    "MAX_CONCURRENT_TRADES":    "1",
    "TRADE_COOLDOWN_SECONDS":   "0",
    "LOG_LEVEL":                "WARNING",
    "LOG_FILE":                 "logs/sim.log",
    # L2 / OB settings — thresholds match defaults
    "OB_DEPTH":                 "20",
    "OB_IMBALANCE_THRESHOLD":   "0.10",
    "MAX_SPREAD_PCT":           "0.05",
    "USE_VWAP_FILTER":          "false",  # OB can't be accurately synthesised for back-testing
    "USE_MAKER_ENTRY":          "false",  # no live orders in simulation
    "MAKER_ENTRY_TIMEOUT_S":    "0",
})

import logging
logging.disable(logging.WARNING)

from strategy import ScalpStrategy, Signal   # noqa: E402
from risk_manager import RiskManager, OpenTrade  # noqa: E402
from config import Config                    # noqa: E402

# ── ANSI colours ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def g(t):  return f"{GREEN}{t}{RESET}"
def r(t):  return f"{RED}{t}{RESET}"
def y(t):  return f"{YELLOW}{t}{RESET}"
def cy(t): return f"{CYAN}{t}{RESET}"
def b(t):  return f"{BOLD}{t}{RESET}"
def d(t):  return f"{DIM}{t}{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# Candle generator — Geometric Brownian Motion
# ══════════════════════════════════════════════════════════════════════════════

def generate_candles(n_candles: int = 8000, seed: int = 42) -> list[list]:
    """
    Geometric Brownian Motion with BTC-realistic parameters.
    No engineered crossovers or RSI resets — genuine market noise.

    Parameters calibrated to real BTC/USDT 1m data:
      σ_annual ≈ 75%  →  σ_1min ≈ 0.001 (0.1% per candle)
      Occasional volatility spikes (news / liquidation events)
      Ranging periods mixed with trending periods
    """
    rng = np.random.default_rng(seed)
    sigma_base = 0.00100   # 0.10% per 1-minute candle — BTC realistic
    drift_base = 0.000003  # tiny positive drift (long-run BTC tendency)

    price = 43_000.0
    closes: list[float] = []

    while len(closes) < n_candles:
        # Regime: trend (40%), range (40%), high-vol (20%)
        regime_len = int(rng.integers(20, 80))
        regime     = rng.choice(["trend", "range", "highvol"],
                                p=[0.40, 0.40, 0.20])

        if regime == "trend":
            drift = rng.choice([-1, 1]) * rng.uniform(0.00010, 0.00025)
            sigma = sigma_base
        elif regime == "range":
            drift = 0.0
            sigma = sigma_base * 0.6          # quieter during ranging
        else:  # highvol
            drift = rng.choice([-1, 1]) * rng.uniform(0.00005, 0.00015)
            sigma = sigma_base * rng.uniform(2.0, 4.0)   # spiky

        for _ in range(regime_len):
            if len(closes) >= n_candles:
                break
            # Occasional fat-tail spike (flash crash / pump: 0.5% chance)
            spike = rng.uniform(2.0, 5.0) if rng.random() < 0.005 else 1.0
            ret   = rng.normal(drift_base + drift, sigma * spike)
            price = max(1_000.0, price * (1.0 + ret))
            closes.append(round(price, 2))

    ts_base = int(time.time()) * 1000 - len(closes) * 60_000
    candles = []
    for i, close in enumerate(closes):
        prev   = closes[i - 1] if i > 0 else close
        # Realistic wicks: bigger during high-vol candles
        move   = abs(close - prev) / prev if prev else 0
        wick   = max(0.0002, move * 0.5)
        hi_ext = rng.uniform(wick * 0.5, wick * 2.0) * close
        lo_ext = rng.uniform(wick * 0.5, wick * 2.0) * close
        candles.append([
            ts_base + i * 60_000,
            round(prev, 2),
            round(max(prev, close) + hi_ext, 2),
            round(min(prev, close) - lo_ext, 2),
            close,
            round(rng.uniform(0.1, 15.0), 3),
        ])
    return candles


# ══════════════════════════════════════════════════════════════════════════════
# Slippage model
# ══════════════════════════════════════════════════════════════════════════════

def apply_slippage(price: float, side: str, rng) -> float:
    """Market orders on 1m BTC futures incur 0.02–0.08% slippage."""
    slip = rng.uniform(0.0002, 0.0008)
    return price * (1 + slip) if side == "long" else price * (1 - slip)


# ══════════════════════════════════════════════════════════════════════════════
# Funding rate
# ══════════════════════════════════════════════════════════════════════════════

# Phemex charges funding every 8 hours = 480 minutes.
# Typical rate: 0.01% per 8h. Longs pay shorts when positive.
_FUNDING_PER_CANDLE = 0.0001 / 480   # fraction of notional per 1-minute candle


def funding_cost(notional: float, candles_held: int) -> float:
    return notional * _FUNDING_PER_CANDLE * candles_held


# ══════════════════════════════════════════════════════════════════════════════
# Intra-candle TP / SL check
# ══════════════════════════════════════════════════════════════════════════════

def check_candle_exit(trade: OpenTrade, candle: list) -> tuple[str | None, float]:
    _ts, _o, high, low, _c, _v = candle
    if trade.side == "long":
        if low  <= trade.stop_loss:   return "stop_loss",   trade.stop_loss
        if high >= trade.take_profit: return "take_profit", trade.take_profit
    else:
        if high >= trade.stop_loss:   return "stop_loss",   trade.stop_loss
        if low  <= trade.take_profit: return "take_profit", trade.take_profit
    return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ASCII equity curve
# ══════════════════════════════════════════════════════════════════════════════

def equity_curve(balances: list[float], target_width: int = 60, height: int = 8) -> str:
    if len(balances) < 2:
        return ""
    n = len(balances)
    if n > target_width:
        idx     = [int(round(i * (n - 1) / (target_width - 1))) for i in range(target_width)]
        sampled = [balances[i] for i in idx]
    else:
        sampled = list(balances)
    lo   = min(sampled) * 0.9995
    hi   = max(sampled) * 1.0005
    span = hi - lo or 1.0
    w    = len(sampled)
    lines = []
    for row in range(height, 0, -1):
        thresh      = lo + span * (row / height)
        prev_thresh = lo + span * ((row - 1) / height)
        label = f"{thresh:>9,.0f}" if row % 2 == 0 else " " * 9
        bar   = "".join(
            "▓" if v >= thresh else "░" if v >= prev_thresh else " "
            for v in sampled
        )
        lines.append(f"  {label} │{bar}│")
    lines.append(f"           └{'─' * w}┘")
    mid = w // 2
    lines.append(f"           1{' ' * (mid - 2)}{mid}{' ' * (w - mid - len(str(w)))}{w}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main simulation
# ══════════════════════════════════════════════════════════════════════════════

W = 76  # display width


def run_simulation(target_trades: int = 100) -> None:
    starting_balance = 100.0
    rng = np.random.default_rng(42)

    all_candles = generate_candles(
        n_candles=max(20_000, target_trades * 200), seed=42
    )

    strategy = ScalpStrategy()
    rm        = RiskManager(starting_balance)

    # ── Header ────────────────────────────────────────────────────────────────
    print(cy("═" * W))
    print(cy(b(f"  PHEMEX SCALP BOT  ─  {target_trades}-TRADE SIMULATION")))
    print(cy(f"  GBM price data  │  slippage 0.02–0.08%  │  funding 0.01%/8h"))
    print(cy(f"  {Config.SYMBOL}  │  {Config.TIMEFRAME}  │  {Config.LEVERAGE}x leverage  │  "
             f"Fixed ${Config.TRADE_SIZE_USDT:.0f} USDT per trade"))
    print(cy(f"  Balance: {starting_balance:,.2f} USDT  │  "
             f"Max session loss: {Config.MAX_SESSION_LOSS_PCT}%  │  "
             f"TP: +{Config.TAKE_PROFIT_PCT}%  │  SL: -{Config.STOP_LOSS_PCT}%"))
    print(cy("═" * W))
    print()

    # ── Table header ──────────────────────────────────────────────────────────
    print(d(f"  {'#':>4}  {'Side':<6}  {'Entry':>10}  {'Exit':>10}  "
            f"{'Result':<10}  {'PnL':>8}  {'Slip+Fund':>9}  {'Balance':>10}"))
    print(d("  " + "─" * (W - 2)))

    # ── State ─────────────────────────────────────────────────────────────────
    balance:        float            = starting_balance
    balances:       list[float]      = [starting_balance]
    completed:      int              = 0
    current_trade:  OpenTrade | None = None
    trade_log:      list[dict]       = []
    candles_held:   int              = 0
    trade_notional: float            = 0.0

    for i in range(Config.MIN_CANDLES, len(all_candles)):
        candle = all_candles[i]
        close  = float(candle[4])

        # ── Manage open position ───────────────────────────────────────────────
        if current_trade is not None:
            candles_held += 1
            reason, exit_price = check_candle_exit(current_trade, candle)

            if reason:
                record = rm.register_close(exit_price, reason)
                completed += 1
                pnl = record.pnl_usdt

                # Apply funding cost on top of the fee already in register_close
                fund        = funding_cost(trade_notional, candles_held)
                pnl        -= fund
                balance     = starting_balance + rm._realised_pnl - fund
                rm._realised_pnl -= fund

                balances.append(balance)
                tp_hit     = reason == "take_profit"
                result_str = g("TP ✓") if tp_hit else r("SL ✗")
                pnl_str    = (g if pnl >= 0 else r)(f"{pnl:>+7.2f}")
                side_str   = g("LONG  ") if current_trade.side == "long" else r("SHORT ")

                print(
                    f"  {completed:>4}  {side_str}"
                    f"  {current_trade.entry_price:>10,.2f}"
                    f"  {exit_price:>10,.2f}"
                    f"  {result_str}        "
                    f"{pnl_str}"
                    f"  {DIM}{fund:>+8.4f}{RESET}"
                    f"  {balance:>10,.4f}"
                )

                trade_log.append({
                    "num":   completed,
                    "side":  current_trade.side,
                    "entry": current_trade.entry_price,
                    "exit":  exit_price,
                    "pnl":   pnl,
                    "tp":    tp_hit,
                })
                current_trade  = None
                candles_held   = 0
                trade_notional = 0.0

                if completed % 10 == 0 and completed < target_trades:
                    wins_n  = sum(1 for t in trade_log if t["pnl"] > 0)
                    pct_chg = (balance / starting_balance - 1) * 100
                    chk_col = g if pct_chg >= 0 else r
                    print(d(
                        f"  ── {completed} trades  Balance: {balance:,.2f} USDT "
                        f"({chk_col(f'{pct_chg:+.2f}%')})  "
                        f"W/L: {wins_n}/{completed - wins_n} "
                        + "─" * 10
                    ))

                if rm.session_locked:
                    print()
                    print(r(b(
                        f"  !! SESSION LOCKED — loss {rm.session_loss_pct:.1f}% "
                        f"hit {Config.MAX_SESSION_LOSS_PCT}% limit !!"
                    )))
                    break

                if completed >= target_trades:
                    break

            continue

        if rm.session_locked or completed >= target_trades:
            break

        # ── Evaluate strategy ──────────────────────────────────────────────────
        # orderbook=None: real L2 OB can't be synthesised accurately enough to
        # model its predictive benefit. The live bot fetches a real order book on
        # every tick and applies the imbalance + spread filters.
        window_start = max(0, i - 199)
        result = strategy.analyse(all_candles[window_start: i + 1], orderbook=None)
        if result is None or result.signal == Signal.NONE:
            continue

        allowed, _ = rm.can_open_trade()
        if not allowed:
            continue

        # ── Open position ──────────────────────────────────────────────────────
        signal_side = result.signal.value  # 'long' or 'short'
        fill_price  = apply_slippage(close, signal_side, rng)

        notional       = Config.TRADE_SIZE_USDT * Config.LEVERAGE
        contracts      = round(notional / fill_price, 6)
        trade_notional = notional

        tp_pct = Config.TAKE_PROFIT_PCT / 100
        sl_pct = Config.STOP_LOSS_PCT   / 100
        if signal_side == "long":
            take_profit = fill_price * (1 + tp_pct)
            stop_loss   = fill_price * (1 - sl_pct)
        else:
            take_profit = fill_price * (1 - tp_pct)
            stop_loss   = fill_price * (1 + sl_pct)

        current_trade = OpenTrade(
            trade_id    = str(uuid.uuid4())[:8],
            side        = signal_side,
            entry_price = fill_price,
            contracts   = contracts,
            take_profit = take_profit,
            stop_loss   = stop_loss,
        )
        rm.register_open(current_trade)

    # ── Force-close any remaining open trade ──────────────────────────────────
    if current_trade is not None:
        last_close = float(all_candles[-1][4])
        record = rm.register_close(last_close, "session_end")
        if record:
            completed += 1
            balance    = starting_balance + rm._realised_pnl
            balances.append(balance)
            pnl = record.pnl_usdt
            print(
                f"  {completed:>4}  "
                f"{'LONG  ' if current_trade.side == 'long' else 'SHORT '}"
                f"  {current_trade.entry_price:>10,.2f}"
                f"  {last_close:>10,.2f}"
                f"  {d('END       ')}"
                f"  {(g if pnl >= 0 else r)(f'{pnl:>+7.2f}')}"
                f"  {balance:>11,.4f}"
            )
            trade_log.append({
                "num": completed, "side": current_trade.side,
                "entry": current_trade.entry_price, "exit": last_close,
                "pnl": pnl, "tp": False,
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    wins    = [t for t in trade_log if t["pnl"] > 0]
    losses  = [t for t in trade_log if t["pnl"] <= 0]
    net     = balance - starting_balance
    net_pct = (net / starting_balance) * 100

    best  = max(trade_log, key=lambda t: t["pnl"]) if trade_log else None
    worst = min(trade_log, key=lambda t: t["pnl"]) if trade_log else None

    peak   = starting_balance
    max_dd = 0.0
    for bal in balances:
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd: max_dd = dd

    def row(label, value, col=RESET):
        print(f"  {label:<30}{col}{value}{RESET}")

    print()
    print(cy("═" * W))
    print(cy(b(f"  SESSION SUMMARY — {completed} TRADES")))
    print(cy("═" * W))

    bal_col = GREEN if net >= 0 else RED
    row("Starting balance",   f"{starting_balance:,.2f} USDT")
    row("Final balance",      f"{balance:,.2f} USDT", bal_col)
    print()
    row("Total trades",       str(completed))
    print(f"  {'Wins / Losses':<30}"
          f"{g(str(len(wins)) + ' wins')}  /  {r(str(len(losses)) + ' losses')}")
    wr = len(wins) / completed * 100 if completed else 0
    row("Win rate",           f"{wr:.1f}%", GREEN if wr >= 50 else RED)
    print()
    net_col = GREEN if net >= 0 else RED
    row("Net PnL",            f"{net:>+.4f} USDT  ({net_pct:>+.2f}%)", net_col)
    if wins:
        row("  Avg win",      f"{sum(t['pnl'] for t in wins)/len(wins):>+.4f} USDT", GREEN)
    if losses:
        row("  Avg loss",     f"{sum(t['pnl'] for t in losses)/len(losses):>+.4f} USDT", RED)
    if best:
        row("  Best trade",   f"{best['pnl']:>+.4f} USDT  (trade #{best['num']})", GREEN)
    if worst:
        row("  Worst trade",  f"{worst['pnl']:>+.4f} USDT  (trade #{worst['num']})", RED)
    print()
    dd_col = RED if max_dd > 15 else YELLOW if max_dd > 8 else GREEN
    row("Max drawdown",       f"{max_dd:.2f}%", dd_col)
    row("Session loss",       f"{rm.session_loss_pct:.2f}%  (limit {Config.MAX_SESSION_LOSS_PCT}%)")
    row("Session locked",     str(rm.session_locked))

    print()
    print(d("  Includes: taker fees (0.075%/side) + slippage (0.02–0.08%) + funding (0.01%/8h)"))
    print(d("  Price data: GBM with regime switching (trend/range/highvol) + fat-tail spikes"))
    print(d("  Live bot additionally applies real-time OB imbalance + spread filters + maker entry"))

    print()
    print(cy("  Equity Curve  (balance over completed trades)"))
    print(equity_curve(balances, target_width=60))
    print(cy("═" * W))
    print()


if __name__ == "__main__":
    run_simulation(target_trades=_args.trades)
