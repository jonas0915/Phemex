"""
Phemex Scalp Trading Bot — main entry point.

Usage:
    python bot.py

The bot runs an infinite loop, ticking every ~TIMEFRAME seconds.
Press Ctrl+C to stop gracefully.
"""

from __future__ import annotations

import signal
import sys
import time

from config import Config
from exchange import PhemexExchange
from logger import get_logger
from risk_manager import RiskManager
from strategy import ScalpStrategy
from trader import Trader

log = get_logger(__name__)

# Approximate seconds per candle for each supported timeframe
_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}

# How often to run a tick (fraction of one candle period)
_TICK_FRACTION = 0.5   # tick twice per candle


def _candle_sleep() -> float:
    secs = _TIMEFRAME_SECONDS.get(Config.TIMEFRAME, 60)
    return secs * _TICK_FRACTION


def main() -> None:
    log.info("=" * 60)
    log.info("Phemex Scalp Bot starting")
    log.info("Symbol      : %s", Config.SYMBOL)
    log.info("Timeframe   : %s", Config.TIMEFRAME)
    log.info("Leverage    : %dx", Config.LEVERAGE)
    if Config.RISK_PER_TRADE_PCT > 0:
        log.info("Sizing      : compound %.1f%% of balance × %dx leverage",
                 Config.RISK_PER_TRADE_PCT, Config.LEVERAGE)
    else:
        log.info("Sizing      : fixed %.2f USDT × %dx = %.2f USDT notional",
                 Config.TRADE_SIZE_USDT, Config.LEVERAGE,
                 Config.TRADE_SIZE_USDT * Config.LEVERAGE)
    log.info("Max loss    : %.1f%%", Config.MAX_SESSION_LOSS_PCT)
    log.info("Testnet     : %s", Config.TESTNET)
    log.info("=" * 60)

    # ── Bootstrap ────────────────────────────────────────────────────────────
    try:
        exchange = PhemexExchange()
    except Exception as exc:
        log.error("Failed to initialise exchange connection: %s", exc)
        sys.exit(1)

    try:
        starting_balance = exchange.fetch_usdt_balance()
    except Exception as exc:
        log.error("Failed to fetch starting balance: %s", exc)
        sys.exit(1)

    log.info("Session starting balance: %.4f USDT", starting_balance)
    if starting_balance <= 0:
        log.error("Balance is zero — check API credentials and testnet setting.")
        sys.exit(1)

    # ── Startup position reconciliation ──────────────────────────────────────
    # If the bot crashed while a position was open, close it cleanly before
    # starting a new session so we never hold a ghost position.
    try:
        orphan_positions = exchange.fetch_positions()
        if orphan_positions:
            log.warning(
                "Found %d open position(s) from a previous session — closing before start",
                len(orphan_positions),
            )
            for pos in orphan_positions:
                exchange.close_position(pos)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not check for orphan positions: %s", exc)

    risk_manager = RiskManager(starting_balance)
    strategy = ScalpStrategy()
    trader = Trader(exchange, strategy, risk_manager)

    tick_interval = _candle_sleep()
    log.info("Tick interval: %.0fs", tick_interval)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    shutdown_requested = False

    def _shutdown(signum, frame):  # noqa: ANN001
        nonlocal shutdown_requested
        log.info("Shutdown signal received (%s)", signal.Signals(signum).name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Main loop ────────────────────────────────────────────────────────────
    try:
        while not shutdown_requested:
            loop_start = time.time()

            try:
                trader.tick()
            except Exception as exc:  # noqa: BLE001
                log.exception("Unhandled error in tick: %s", exc)

            # Check session locked (might have been set inside tick)
            if risk_manager.session_locked:
                log.warning("Session locked. Printing summary and exiting.")
                _print_summary(risk_manager)
                break

            elapsed = time.time() - loop_start
            sleep_for = max(0, tick_interval - elapsed)
            log.debug("Tick done in %.2fs — sleeping %.2fs", elapsed, sleep_for)
            # Sleep in 1-second increments so SIGTERM is handled promptly
            deadline = time.time() + sleep_for
            while time.time() < deadline and not shutdown_requested:
                time.sleep(min(1.0, deadline - time.time()))

    finally:
        log.info("Shutting down — closing all open positions...")
        try:
            trader.emergency_close_all()
        except Exception as exc:  # noqa: BLE001
            log.error("Error during emergency close: %s", exc)

        _print_summary(risk_manager)
        log.info("Bot stopped.")


def _print_summary(rm: RiskManager) -> None:
    s = rm.session_summary()
    log.info("=" * 60)
    log.info("SESSION SUMMARY")
    log.info("  Start balance : %.4f USDT", s["start_balance"])
    log.info("  Total trades  : %d (W:%d / L:%d)", s["total_trades"], s["wins"], s["losses"])
    log.info("  Win rate      : %.1f%%", s["win_rate_pct"])
    log.info("  Realised PnL  : %.4f USDT", s["realised_pnl"])
    log.info("  Session loss  : %.2f%%", s["session_loss_pct"])
    log.info("  Session locked: %s", s["session_locked"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
