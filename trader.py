"""
Trade executor.
Bridges the strategy signals with the exchange API and risk manager.
"""

from __future__ import annotations

import uuid
from typing import Optional

from config import Config
from exchange import PhemexExchange
from logger import get_logger
from risk_manager import OpenTrade, RiskManager
from strategy import ScalpStrategy, Signal, StrategyResult

log = get_logger(__name__)


class Trader:
    """
    Combines exchange, strategy, and risk manager into one execution unit.
    Called on every tick by the main bot loop.
    """

    def __init__(
        self,
        exchange: PhemexExchange,
        strategy: ScalpStrategy,
        risk_manager: RiskManager,
    ) -> None:
        self.exchange = exchange
        self.strategy = strategy
        self.rm = risk_manager

    # ── Main tick ─────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Execute one iteration of the bot loop."""

        # 1. Fetch fresh candles
        ohlcv = self.exchange.fetch_ohlcv(limit=max(100, Config.MIN_CANDLES + 5))
        if not ohlcv:
            log.warning("No OHLCV data received")
            return

        # 2. Get current price
        ticker = self.exchange.fetch_ticker()
        current_price = float(ticker.get("last", ohlcv[-1][4]))

        # 3. Fetch live balance for session limit check and compound sizing
        balance: Optional[float] = None
        try:
            balance = self.exchange.fetch_usdt_balance()
            self.rm.update_current_balance(balance)
        except Exception as exc:  # noqa: BLE001
            log.warning("Balance check failed: %s", exc)

        # 4. Manage open position (exit checks take priority)
        if self.rm.open_trade is not None:
            self._manage_open_trade(current_price)
            return      # one action per tick

        # 5. Session locked? No new entries.
        if self.rm.session_locked:
            log.info("Session locked — skipping signal evaluation")
            return

        # 6. Evaluate strategy
        result: Optional[StrategyResult] = self.strategy.analyse(ohlcv)
        if result is None or result.signal == Signal.NONE:
            return

        # 7. Check risk manager gate
        allowed, reason = self.rm.can_open_trade()
        if not allowed:
            log.debug("Trade blocked: %s", reason)
            return

        # 8. Open trade (pass live balance for compound sizing)
        self._open_trade(result, current_price, balance)

    # ── Trade management ──────────────────────────────────────────────────────

    def _open_trade(
        self,
        result: StrategyResult,
        current_price: float,
        balance: Optional[float] = None,
    ) -> None:
        side = "buy" if result.signal == Signal.LONG else "sell"

        # Compound sizing: risk a % of current balance when configured
        if Config.RISK_PER_TRADE_PCT > 0 and balance is not None and balance > 0:
            trade_usdt = balance * (Config.RISK_PER_TRADE_PCT / 100)
            notional = trade_usdt * Config.LEVERAGE
            try:
                market = self.exchange._exchange.market(Config.SYMBOL)
                precision = market.get("precision", {}).get("amount", 6)
            except Exception:  # noqa: BLE001
                precision = 6
            qty = round(notional / current_price, precision)
            log.info(
                "Compound sizing | balance=%.2f risk=%.1f%% notional=%.2f qty=%.6f",
                balance, Config.RISK_PER_TRADE_PCT, notional, qty,
            )
        else:
            qty = self.exchange.calculate_order_qty(current_price)

        order = self.exchange.place_market_order(side, qty)
        if order is None:
            log.error("Order placement failed — trade aborted")
            return

        fill_price = float(order.get("average") or order.get("price") or current_price)

        open_trade = OpenTrade(
            trade_id=str(order.get("id") or uuid.uuid4()),
            side=result.signal.value,
            entry_price=fill_price,
            contracts=qty,
            take_profit=result.take_profit,
            stop_loss=result.stop_loss,
        )
        self.rm.register_open(open_trade)

    def _manage_open_trade(self, current_price: float) -> None:
        """Check TP/SL and close if triggered."""
        exit_reason = self.rm.check_exit_conditions(current_price)
        if exit_reason is None:
            # Log position status periodically
            t = self.rm.open_trade
            if t:
                log.debug(
                    "Holding %s | entry=%.4f current=%.4f TP=%.4f SL=%.4f",
                    t.side, t.entry_price, current_price, t.take_profit, t.stop_loss,
                )
            return

        # Close position
        positions = self.exchange.fetch_positions()
        if positions:
            self.exchange.close_position(positions[0])
        else:
            # Fallback: send opposing market order
            t = self.rm.open_trade
            if t:
                close_side = "sell" if t.side == "long" else "buy"
                self.exchange.place_market_order(close_side, t.contracts)

        self.rm.register_close(current_price, exit_reason)

    def emergency_close_all(self) -> None:
        """Force-close everything — called on bot shutdown or max-loss breach."""
        log.warning("Emergency close all triggered")
        self.exchange.cancel_all_orders()

        if self.rm.open_trade is not None:
            positions = self.exchange.fetch_positions()
            if positions:
                for pos in positions:
                    self.exchange.close_position(pos)

            ticker = self.exchange.fetch_ticker()
            price = float(ticker.get("last", 0))
            self.rm.register_close(price, "manual")
