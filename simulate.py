"""
simulate.py — Multi-trade offline simulation using synthetic BTC price data.
Exercises ScalpStrategy + RiskManager with intra-candle TP/SL detection.
No API keys required.

Usage:
    python simulate.py           # 100 trades (default)
    python simulate.py 50        # custom trade count
"""

from __future__ import annotations

import os
import sys
import uuid
import time

import numpy as np

# ── Patch env before importing bot modules ─────────────────────────────────
os.environ.update({
    "PHEMEX_API_KEY":          "SIM_KEY",
    "PHEMEX_API_SECRET":       "SIM_SECRET",
    "PHEMEX_TESTNET":          "true",
    "TRADING_SYMBOL":          "BTC/USDT:USDT",
    "TRADING_TIMEFRAME":       "1m",
    "TRADE_SIZE_USDT":         "200",    # base fallback size (USDT)
    "LEVERAGE":                "25",     # risk-on: 25× leverage
    "EMA_FAST":                "9",
    "EMA_SLOW":                "21",
    "RSI_PERIOD":              "14",
    "RSI_OVERBOUGHT":          "70",
    "RSI_OVERSOLD":            "30",
    "MAX_SESSION_LOSS_PCT":    "30",
    "TAKE_PROFIT_PCT":         "0.6",
    "STOP_LOSS_PCT":           "0.35",
    "MAX_CONCURRENT_TRADES":   "1",
    "TRADE_COOLDOWN_SECONDS":  "0",
    "RISK_PER_TRADE_PCT":      "20",     # compound sizing: 20% of balance per trade
    "LOG_LEVEL":               "WARNING",
    "LOG_FILE":                "logs/sim.log",
})

import logging
logging.disable(logging.WARNING)

from strategy import ScalpStrategy, Signal, StrategyResult  # noqa: E402
from risk_manager import RiskManager, OpenTrade             # noqa: E402
from config import Config                                   # noqa: E402

# ── ANSI colours ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def g(t: str) -> str: return f"{GREEN}{t}{RESET}"
def r(t: str) -> str: return f"{RED}{t}{RESET}"
def y(t: str) -> str: return f"{YELLOW}{t}{RESET}"
def cy(t: str) -> str: return f"{CYAN}{t}{RESET}"
def b(t: str) -> str: return f"{BOLD}{t}{RESET}"
def d(t: str) -> str: return f"{DIM}{t}{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic price generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_candles(n_candles: int = 8000, seed: int = 42) -> list[list]:
    """
    Produces alternating trend cycles with moderate drift and deliberate
    RSI cool-down phases so that EMA crossovers fire when RSI is in the
    strategy's acceptance window (30–60 for LONG, 40–70 for SHORT).

    Cycle ≈ 76 candles → ~2 trades:
        8  flat      → RSI resets to ~50, EMAs converge
        30 up        → gentle slope; EMA9 crosses EMA21 ~candle 18 at RSI ≈ 54
        8  flat      → RSI cools before downtrend
        30 down      → gentle slope; EMA9 crosses below EMA21 ~candle 18 at RSI ≈ 46
    """
    rng = np.random.default_rng(seed)
    closes: list[float] = [43_000.0]

    def add(n: int, drift: float, vol: float) -> None:
        p = closes[-1]
        for _ in range(n):
            p = max(100.0, p * (1.0 + rng.normal(drift, vol)))
            closes.append(round(p, 2))

    add(40, 0.0, 0.0003)           # warm-up

    while len(closes) < n_candles:
        add(8,   0.0,       0.0002)  # flat: RSI → 50, EMAs converge
        add(30,  0.00030,   0.00040) # gentle uptrend: cross at RSI ~54
        add(8,   0.0,       0.0002)  # flat: RSI cools
        add(30, -0.00030,   0.00040) # gentle downtrend: cross at RSI ~46

    closes = closes[:n_candles]
    ts_base = int(time.time()) * 1000 - len(closes) * 60_000

    candles: list[list] = []
    for i, close in enumerate(closes):
        prev   = closes[i - 1] if i > 0 else close
        hi_ext = rng.uniform(0.0001, 0.0005) * close
        lo_ext = rng.uniform(0.0001, 0.0005) * close
        candles.append([
            ts_base + i * 60_000,
            round(prev, 2),
            round(max(prev, close) + hi_ext, 2),
            round(min(prev, close) - lo_ext, 2),
            close,
            round(rng.uniform(0.5, 5.0), 3),
        ])

    return candles


# ══════════════════════════════════════════════════════════════════════════════
# Intra-candle TP / SL check  (uses candle high/low, not just close)
# ══════════════════════════════════════════════════════════════════════════════

def check_candle_exit(trade: OpenTrade, candle: list) -> tuple[str | None, float]:
    """
    Returns (reason, exact_exit_price) if TP or SL is breached inside the candle.
    SL takes priority when both levels are touched (gap candle).
    """
    _ts, _o, high, low, _c, _v = candle

    if trade.side == "long":
        sl_hit = low  <= trade.stop_loss
        tp_hit = high >= trade.take_profit
        if sl_hit:
            return "stop_loss",   trade.stop_loss
        if tp_hit:
            return "take_profit", trade.take_profit
    else:
        sl_hit = high >= trade.stop_loss
        tp_hit = low  <= trade.take_profit
        if sl_hit:
            return "stop_loss",   trade.stop_loss
        if tp_hit:
            return "take_profit", trade.take_profit

    return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ASCII equity curve
# ══════════════════════════════════════════════════════════════════════════════

def equity_curve(balances: list[float], target_width: int = 60, height: int = 8) -> str:
    """Area chart of cumulative balance across all completed trades."""
    if len(balances) < 2:
        return ""

    # Downsample to target width
    n = len(balances)
    if n > target_width:
        idx = [int(round(i * (n - 1) / (target_width - 1))) for i in range(target_width)]
        sampled = [balances[i] for i in idx]
    else:
        sampled = list(balances)

    lo = min(sampled) * 0.9995
    hi = max(sampled) * 1.0005
    span = hi - lo or 1.0
    w = len(sampled)

    lines: list[str] = []
    for row in range(height, 0, -1):
        thresh      = lo + span * (row / height)
        prev_thresh = lo + span * ((row - 1) / height)
        label = f"{thresh:>9,.0f}" if row % 2 == 0 else " " * 9
        bar = ""
        for val in sampled:
            if val >= thresh:
                bar += "▓"
            elif val >= prev_thresh:
                bar += "░"
            else:
                bar += " "
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
    starting_balance = 1_000.0
    all_candles = generate_candles(n_candles=max(12000, target_trades * 120))

    strategy = ScalpStrategy()
    rm       = RiskManager(starting_balance)

    # ── Header ────────────────────────────────────────────────────────────────
    effective_x = Config.RISK_PER_TRADE_PCT / 100 * Config.LEVERAGE
    print(cy("═" * W))
    print(cy(b(f"  PHEMEX SCALP BOT  ─  {target_trades}-TRADE SIMULATION  [RISK-ON]")))
    print(cy(f"  {Config.SYMBOL}  │  {Config.TIMEFRAME}  │  {Config.LEVERAGE}x leverage"))
    print(cy(f"  Compound sizing: {Config.RISK_PER_TRADE_PCT:.0f}% of balance × "
             f"{Config.LEVERAGE}x = {effective_x:.1f}× account exposure per trade"))
    print(cy(f"  Balance: {starting_balance:,.2f} USDT  │  "
             f"Max session loss: {Config.MAX_SESSION_LOSS_PCT}%  │  "
             f"TP: +{Config.TAKE_PROFIT_PCT}%  │  SL: -{Config.STOP_LOSS_PCT}%"))
    print(cy("═" * W))
    print()

    # ── Table header ──────────────────────────────────────────────────────────
    print(d(f"  {'#':>4}  {'Side':<6}  {'Entry':>10}  {'Exit':>10}  "
            f"{'Result':<10}  {'PnL':>8}  {'Balance':>11}  {'Loss%':>6}"))
    print(d("  " + "─" * (W - 2)))

    # ── State ─────────────────────────────────────────────────────────────────
    balance:      float             = starting_balance
    balances:     list[float]       = [starting_balance]
    completed:    int               = 0
    current_trade: OpenTrade | None = None
    trade_log:    list[dict]        = []

    for i in range(Config.MIN_CANDLES, len(all_candles)):
        candle = all_candles[i]
        close  = float(candle[4])

        # ── Manage open position ───────────────────────────────────────────────
        if current_trade is not None:
            reason, exit_price = check_candle_exit(current_trade, candle)

            if reason:
                record = rm.register_close(exit_price, reason)
                completed += 1
                pnl     = record.pnl_usdt
                balance = starting_balance + rm._realised_pnl
                balances.append(balance)

                tp_hit     = reason == "take_profit"
                result_str = (g("TP ✓") if tp_hit else r("SL ✗"))
                pnl_str    = (g if pnl >= 0 else r)(f"{pnl:>+7.2f}")
                side_str   = g("LONG  ") if current_trade.side == "long" else r("SHORT ")
                loss_col   = RED if rm.session_loss_pct > 15 else YELLOW if rm.session_loss_pct > 5 else DIM

                print(
                    f"  {completed:>4}  {side_str}"
                    f"  {current_trade.entry_price:>10,.2f}"
                    f"  {exit_price:>10,.2f}"
                    f"  {result_str}        "
                    f"{pnl_str}"
                    f"  {balance:>11,.4f}"
                    f"  {loss_col}{rm.session_loss_pct:>5.1f}%{RESET}"
                )

                trade_log.append({
                    "num":   completed,
                    "side":  current_trade.side,
                    "entry": current_trade.entry_price,
                    "exit":  exit_price,
                    "pnl":   pnl,
                    "tp":    tp_hit,
                })
                current_trade = None

                # ── Checkpoint every 10 trades ─────────────────────────────────
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

            continue  # hold — don't look for new signal this tick

        # ── Stop if done ───────────────────────────────────────────────────────
        if rm.session_locked or completed >= target_trades:
            break

        # ── Evaluate strategy (rolling 200-candle window for speed) ───────────
        window_start = max(0, i - 199)
        result = strategy.analyse(all_candles[window_start: i + 1])
        if result is None or result.signal == Signal.NONE:
            continue

        allowed, _ = rm.can_open_trade()
        if not allowed:
            continue

        # ── Open position with compound sizing ────────────────────────────────
        # Risk RISK_PER_TRADE_PCT% of the current (growing) balance each trade.
        # This compounds profits: bigger balance → bigger notional → bigger wins.
        trade_usdt = balance * (Config.RISK_PER_TRADE_PCT / 100)
        notional   = trade_usdt * Config.LEVERAGE
        contracts  = round(notional / close, 6)
        current_trade = OpenTrade(
            trade_id    = str(uuid.uuid4())[:8],
            side        = result.signal.value,
            entry_price = close,
            contracts   = contracts,
            take_profit = result.take_profit,
            stop_loss   = result.stop_loss,
        )
        rm.register_open(current_trade)

    # ── Force-close any remaining open trade ──────────────────────────────────
    if current_trade is not None:
        last_close = float(all_candles[-1][4])
        record = rm.register_close(last_close, "session_end")
        if record:
            completed += 1
            balance = starting_balance + rm._realised_pnl
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
                f"  {d(f'{rm.session_loss_pct:>5.1f}%')}"
            )
            trade_log.append({
                "num": completed, "side": current_trade.side,
                "entry": current_trade.entry_price, "exit": last_close,
                "pnl": pnl, "tp": False,
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    wins   = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    net    = rm._realised_pnl
    net_pct = (net / starting_balance) * 100

    best  = max(trade_log, key=lambda t: t["pnl"]) if trade_log else None
    worst = min(trade_log, key=lambda t: t["pnl"]) if trade_log else None

    # Max drawdown
    peak   = starting_balance
    max_dd = 0.0
    for bal in balances:
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd:
            max_dd = dd

    def row(label: str, value: str, col: str = RESET) -> None:
        print(f"  {label:<28}{col}{value}{RESET}")

    print()
    print(cy("═" * W))
    print(cy(b(f"  SESSION SUMMARY — {completed} TRADES")))
    print(cy("═" * W))

    bal_col = GREEN if net >= 0 else RED
    row("Starting balance",  f"{starting_balance:,.2f} USDT")
    row("Final balance",     f"{balance:,.2f} USDT", bal_col)
    print()
    row("Total trades",      str(completed))
    print(f"  {'Wins / Losses':<28}"
          f"{g(str(len(wins)) + ' wins')}  /  {r(str(len(losses)) + ' losses')}")
    wr = len(wins) / completed * 100 if completed else 0
    row("Win rate",          f"{wr:.1f}%", GREEN if wr >= 50 else YELLOW)
    print()
    net_col = GREEN if net >= 0 else RED
    row("Net PnL",           f"{net:>+.4f} USDT  ({net_pct:>+.2f}%)", net_col)
    if wins:
        avg_win = sum(t["pnl"] for t in wins) / len(wins)
        row("  Avg win",     f"{avg_win:>+.4f} USDT", GREEN)
    if losses:
        avg_loss = sum(t["pnl"] for t in losses) / len(losses)
        row("  Avg loss",    f"{avg_loss:>+.4f} USDT", RED)
    if best:
        row("  Best trade",  f"{best['pnl']:>+.4f} USDT  (trade #{best['num']})", GREEN)
    if worst:
        row("  Worst trade", f"{worst['pnl']:>+.4f} USDT  (trade #{worst['num']})", RED)
    print()
    dd_col = RED if max_dd > 15 else YELLOW if max_dd > 8 else GREEN
    row("Max drawdown",      f"{max_dd:.2f}%", dd_col)
    row("Session loss",      f"{rm.session_loss_pct:.2f}%  (limit {Config.MAX_SESSION_LOSS_PCT}%)")
    row("Session locked",    str(rm.session_locked))

    print()
    print(cy("  Equity Curve  (balance over completed trades)"))
    print(equity_curve(balances, target_width=60))
    print(cy("═" * W))
    print()


if __name__ == "__main__":
    trades = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_simulation(target_trades=trades)
