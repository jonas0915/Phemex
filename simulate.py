"""
simulate.py — Offline trade simulation using synthetic BTC price data.
No API keys or live exchange connection required.

Usage:
    python simulate.py
"""

from __future__ import annotations

import os
import sys
import uuid
import time

import numpy as np

# ── Inject dummy env vars before any bot module is imported ────────────────
os.environ.update({
    "PHEMEX_API_KEY":            "SIM_KEY",
    "PHEMEX_API_SECRET":         "SIM_SECRET",
    "PHEMEX_TESTNET":            "true",
    "TRADING_SYMBOL":            "BTC/USDT:USDT",
    "TRADING_TIMEFRAME":         "1m",
    "TRADE_SIZE_USDT":           "100",
    "LEVERAGE":                  "5",
    "EMA_FAST":                  "9",
    "EMA_SLOW":                  "21",
    "RSI_PERIOD":                "14",
    "RSI_OVERBOUGHT":            "70",
    "RSI_OVERSOLD":              "30",
    "MAX_SESSION_LOSS_PCT":      "30",
    "TAKE_PROFIT_PCT":           "0.6",
    "STOP_LOSS_PCT":             "0.35",
    "MAX_CONCURRENT_TRADES":     "1",
    "TRADE_COOLDOWN_SECONDS":    "0",   # no cooldown delay in simulation
    "LOG_LEVEL":                 "WARNING",
    "LOG_FILE":                  "logs/sim.log",
})

# Suppress logging noise during simulation
import logging
logging.disable(logging.WARNING)

from strategy import ScalpStrategy, Signal, StrategyResult  # noqa: E402
from risk_manager import RiskManager, OpenTrade             # noqa: E402
from config import Config                                   # noqa: E402

# ── Colour helpers ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic price generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_candles(n: int = 200, seed: int = 42) -> list[list]:
    """
    Generate synthetic 1-minute OHLCV candles designed to produce at least
    two EMA crossover signals (one bullish, one bearish).

    Segments:
        0-35   : flat noise (warmup — lets EMAs stabilise)
        35-75  : uptrend   → bullish EMA9/EMA21 crossover ~candle 55
        75-100 : sharp drop → stop-loss test
        100-140: slow recovery with new uptrend
        140-170: downtrend  → bearish EMA9/EMA21 crossover ~candle 155
        170-200: flat tail
    """
    rng = np.random.default_rng(seed)
    base = 43_000.0
    closes: list[float] = [base]

    def _segment(n_bars: int, drift: float, vol: float) -> list[float]:
        prices = []
        p = closes[-1]
        for _ in range(n_bars):
            p = p * (1 + rng.normal(drift, vol))
            prices.append(round(p, 2))
        return prices

    closes += _segment(35,  0.00000, 0.0008)   # flat
    closes += _segment(40,  0.00060, 0.0006)   # uptrend
    closes += _segment(25, -0.00120, 0.0010)   # sharp drop
    closes += _segment(40,  0.00045, 0.0007)   # recovery
    closes += _segment(30, -0.00070, 0.0007)   # downtrend
    closes += _segment(30,  0.00010, 0.0006)   # flat tail

    closes = closes[:n]
    ts_start = int(time.time()) * 1000 - len(closes) * 60_000

    candles = []
    for i, close in enumerate(closes):
        noise_hi = rng.uniform(0.0001, 0.0006) * close
        noise_lo = rng.uniform(0.0001, 0.0006) * close
        open_  = closes[i - 1] if i > 0 else close
        high   = max(open_, close) + noise_hi
        low    = min(open_, close) - noise_lo
        vol    = rng.uniform(0.5, 5.0)
        candles.append([ts_start + i * 60_000, open_, high, low, close, vol])

    return candles


# ══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════════════════

W = 66  # box width

def header(text: str) -> None:
    print(c("═" * W, CYAN))
    print(c(f"  {text}", BOLD + CYAN))
    print(c("═" * W, CYAN))


def box(lines: list[tuple[str, str, str]]) -> None:
    """Print a box. Each line is (label, value, colour)."""
    print(c("  ┌" + "─" * (W - 4) + "┐", DIM))
    for label, value, col in lines:
        content = f"  {label:<22}{c(value, col)}"
        print(f"  │  {label:<20}{c(value, col)}")
    print(c("  └" + "─" * (W - 4) + "┘", DIM))


def _box(title: str, rows: list[tuple[str, str, str]]) -> None:
    bar = "─" * (W - 4)
    print(f"  {DIM}┌─ {RESET}{BOLD}{title}{RESET}{DIM} {'─' * (W - 7 - len(title))}┐{RESET}")
    for label, value, col in rows:
        pad = W - 8 - len(label) - len(value)
        print(f"  {DIM}│{RESET}  {label:<24}{col}{value}{RESET}{' ' * max(0, pad)}{DIM}│{RESET}")
    print(f"  {DIM}└{bar}┘{RESET}")


def signal_badge(signal: Signal) -> str:
    if signal == Signal.LONG:
        return c("▲ LONG", GREEN + BOLD)
    if signal == Signal.SHORT:
        return c("▼ SHORT", RED + BOLD)
    return c("· hold", DIM)


def pnl_colour(pnl: float) -> str:
    return GREEN if pnl >= 0 else RED


# ══════════════════════════════════════════════════════════════════════════════
# Simulation engine
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation() -> None:
    starting_balance = 1_000.0
    candles = generate_candles(n=200)

    strategy = ScalpStrategy()
    rm = RiskManager(starting_balance)

    header(
        f"PHEMEX SCALP BOT — TRADE SIMULATION  "
        f"({len(candles)} × 1m candles)"
    )
    print(f"  Symbol   : {Config.SYMBOL}")
    print(f"  Leverage : {Config.LEVERAGE}x")
    print(f"  Trade sz : {Config.TRADE_SIZE_USDT} USDT  "
          f"(notional {Config.TRADE_SIZE_USDT * Config.LEVERAGE:.0f} USDT)")
    print(f"  Balance  : {starting_balance:,.2f} USDT  |  "
          f"Max session loss: {Config.MAX_SESSION_LOSS_PCT}%  "
          f"({starting_balance * Config.MAX_SESSION_LOSS_PCT / 100:.2f} USDT)")
    print(f"  TP: +{Config.TAKE_PROFIT_PCT}%   SL: -{Config.STOP_LOSS_PCT}%")
    print()

    print(c(f"  {'Candle':>6}  {'Close':>10}  {'EMA9':>10}  {'EMA21':>10}  "
            f"{'RSI':>6}  Signal", DIM))
    print(c("  " + "─" * 62, DIM))

    trade_num = 0

    for i in range(Config.MIN_CANDLES, len(candles)):
        window = candles[: i + 1]
        current_price = candles[i][4]  # close

        # ── Manage open trade ────────────────────────────────────────────────
        if rm.open_trade is not None:
            exit_reason = rm.check_exit_conditions(current_price)

            if exit_reason:
                record = rm.register_close(current_price, exit_reason)
                col = GREEN if record.pnl_usdt >= 0 else RED
                label = "TAKE PROFIT ✓" if exit_reason == "take_profit" else "STOP LOSS ✗"

                print()
                _box(f"TRADE #{trade_num} CLOSED — {label}", [
                    ("Reason",        label,                        col),
                    ("Exit price",    f"{current_price:>12,.2f} USDT", col),
                    ("PnL",           f"{record.pnl_usdt:>+.4f} USDT",  col),
                    ("Session PnL",   f"{rm._realised_pnl:>+.4f} USDT",
                                      pnl_colour(rm._realised_pnl)),
                    ("Session loss",  f"{rm.session_loss_pct:.2f}%",
                                      RED if rm.session_loss_pct > 10 else RESET),
                    ("Remaining risk",f"{rm.remaining_risk_usdt:.2f} USDT", YELLOW),
                ])
                print()

                if rm.session_locked:
                    print(c(f"\n  !! SESSION LOCKED — cumulative loss "
                            f"{rm.session_loss_pct:.1f}% reached {Config.MAX_SESSION_LOSS_PCT}% limit !!\n",
                            RED + BOLD))
                    break

            else:
                t = rm.open_trade
                side_str = c("LONG  ▲", GREEN) if t.side == "long" else c("SHORT ▼", RED)
                dist_tp = abs(t.take_profit - current_price)
                dist_sl = abs(t.stop_loss - current_price)
                print(f"  {i:>6}  {current_price:>10,.2f}  "
                      f"{'─':>10}  {'─':>10}  {'─':>6}  "
                      f"Holding {side_str}  "
                      f"{DIM}TP-{dist_tp:.1f}  SL-{dist_sl:.1f}{RESET}")
            continue

        # ── Analyse strategy ─────────────────────────────────────────────────
        result: StrategyResult | None = strategy.analyse(window)
        if result is None:
            continue

        sig_str = signal_badge(result.signal)
        print(f"  {i:>6}  {result.current_price:>10,.2f}  "
              f"{result.ema_fast:>10,.2f}  {result.ema_slow:>10,.2f}  "
              f"{result.rsi:>6.1f}  {sig_str}")

        # ── Open trade on signal ─────────────────────────────────────────────
        if result.signal == Signal.NONE:
            continue

        allowed, reason = rm.can_open_trade()
        if not allowed:
            print(c(f"         Trade blocked: {reason}", DIM))
            continue

        trade_num += 1
        notional = Config.TRADE_SIZE_USDT * Config.LEVERAGE
        contracts = round(notional / current_price, 6)

        open_trade = OpenTrade(
            trade_id=str(uuid.uuid4())[:8],
            side=result.signal.value,
            entry_price=current_price,
            contracts=contracts,
            take_profit=result.take_profit,
            stop_loss=result.stop_loss,
        )
        rm.register_open(open_trade)

        side_col = GREEN if result.signal == Signal.LONG else RED
        tp_pct = (result.take_profit / current_price - 1) * 100
        sl_pct = (result.stop_loss  / current_price - 1) * 100

        print()
        _box(f"TRADE #{trade_num} OPENED", [
            ("Side",         result.signal.value.upper(),      side_col),
            ("Entry price",  f"{current_price:>12,.2f} USDT",  side_col),
            ("Contracts",    f"{contracts:.6f}",                RESET),
            ("Notional",     f"{notional:,.2f} USDT ({Config.LEVERAGE}x)", RESET),
            ("Take profit",  f"{result.take_profit:,.2f} USDT  ({tp_pct:+.2f}%)", GREEN),
            ("Stop loss",    f"{result.stop_loss:,.2f} USDT  ({sl_pct:+.2f}%)",   RED),
            ("EMA9 / EMA21", f"{result.ema_fast:,.2f} / {result.ema_slow:,.2f}",  RESET),
            ("RSI",          f"{result.rsi:.1f}",              YELLOW),
        ])
        print()

    # ── Final summary ────────────────────────────────────────────────────────
    # Close any remaining open position at last price
    if rm.open_trade is not None:
        last_price = candles[-1][4]
        record = rm.register_close(last_price, "session_end")
        if record:
            col = pnl_colour(record.pnl_usdt)
            print()
            _box("TRADE CLOSED — SESSION END", [
                ("Exit price", f"{last_price:,.2f} USDT", col),
                ("PnL",        f"{record.pnl_usdt:+.4f} USDT", col),
            ])

    s = rm.session_summary()
    net_col = pnl_colour(s["realised_pnl"])

    print()
    print(c("═" * W, CYAN))
    print(c(f"  SESSION SUMMARY", BOLD + CYAN))
    print(c("═" * W, CYAN))
    print(f"  Starting balance : {s['start_balance']:>10,.2f} USDT")
    print(f"  Total trades     : {s['total_trades']:>10}  "
          f"({c(str(s['wins']) + ' wins', GREEN)}  /  "
          f"{c(str(s['losses']) + ' losses', RED)})")
    print(f"  Win rate         : {s['win_rate_pct']:>9.1f}%")
    pnl_str = f"{s['realised_pnl']:+.4f} USDT"
    print(f"  Realised PnL     : {c(pnl_str, net_col):>10}")
    print(f"  Session loss     : {s['session_loss_pct']:>9.2f}%  "
          f"(limit {Config.MAX_SESSION_LOSS_PCT}%)")
    print(f"  Session locked   : {s['session_locked']}")
    print(c("═" * W, CYAN))
    print()


if __name__ == "__main__":
    run_simulation()
