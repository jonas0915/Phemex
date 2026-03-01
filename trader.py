"""
Trade executor.
Bridges the strategy signals with the exchange API and risk manager.
"""

from __future__ import annotations

import time
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
        self.rm       = risk_manager

    # ── Main tick ─────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Execute one iteration of the bot loop."""

        # 1. Fetch fresh candles
        ohlcv = self.exchange.fetch_ohlcv(limit=max(100, Config.MIN_CANDLES + 5))
        if not ohlcv:
            log.warning("No OHLCV data received")
            return

        # 2. Get current price
        ticker        = self.exchange.fetch_ticker()
        current_price = float(ticker.get("last", ohlcv[-1][4]))

        # 3. Fetch live balance for session limit check and compound sizing
        balance: Optional[float] = None
        try:
            balance = self.exchange.fetch_usdt_balance()
            self.rm.update_current_balance(balance)
        except Exception as exc:  # noqa: BLE001
            log.warning("Balance check failed: %s", exc)

        # 4. Sync in-memory position against actual exchange positions.
        #    Detects closes triggered by exchange-native SL/TP orders or
        #    manual intervention while the bot was running.
        if self.rm.open_trade is not None:
            self._sync_position(current_price)

        # 5. Manage open position — software TP/SL as safety net
        if self.rm.open_trade is not None:
            self._manage_open_trade(current_price)
            return   # one action per tick

        # 6. Session locked? No new entries.
        if self.rm.session_locked:
            log.info("Session locked — skipping signal evaluation")
            return

        # 7. Evaluate strategy
        result: Optional[StrategyResult] = self.strategy.analyse(ohlcv)
        if result is None or result.signal == Signal.NONE:
            return

        # 8. Check risk manager gate
        allowed, reason = self.rm.can_open_trade()
        if not allowed:
            log.debug("Trade blocked: %s", reason)
            return

        # 9. Open trade (pass live balance for compound sizing)
        self._open_trade(result, current_price, balance)

    # ── Position sync ─────────────────────────────────────────────────────────

    def _sync_position(self, current_price: float) -> None:
        """
        Reconcile the in-memory open trade with actual exchange positions.

        If the exchange reports no open position while we believe one exists,
        a native SL/TP order (or manual action) closed it since the last tick.
        We record the close and cancel any stale orders.

        A 5-second grace period after entry avoids false positives from
        Phemex API propagation lag right after order placement.
        """
        t = self.rm.open_trade
        if t is None:
            return

        # Skip sync in the first few seconds — API lag after entry
        if time.time() - t.opened_at < 5:
            return

        try:
            positions = self.exchange.fetch_positions()
        except Exception as exc:  # noqa: BLE001
            log.warning("Position sync skipped (fetch failed): %s", exc)
            return

        if positions:
            return  # Position still open on exchange — nothing to reconcile

        # Exchange shows no position; it was closed externally.
        log.warning(
            "Position sync: exchange shows no open position for trade %s "
            "— recording close at current price %.4f (native SL/TP or manual).",
            t.trade_id, current_price,
        )

        # Cancel any remaining native orders to avoid ghost fills
        for order_id, label in ((t.sl_order_id, "SL"), (t.tp_order_id, "TP")):
            if order_id:
                try:
                    self.exchange.cancel_order(order_id)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Could not cancel stale %s order %s: %s", label, order_id, exc)

        self.rm.register_close(current_price, "exchange_closed")

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
            notional   = trade_usdt * Config.LEVERAGE
            try:
                market    = self.exchange._exchange.market(Config.SYMBOL)
                precision = market.get("precision", {}).get("amount", 6)
                if not isinstance(precision, int):
                    precision = 6
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

        # Recompute TP/SL from actual fill price to account for slippage.
        tp_pct = Config.TAKE_PROFIT_PCT / 100
        sl_pct = Config.STOP_LOSS_PCT   / 100
        if result.signal == Signal.LONG:
            take_profit = fill_price * (1 + tp_pct)
            stop_loss   = fill_price * (1 - sl_pct)
        else:
            take_profit = fill_price * (1 - tp_pct)
            stop_loss   = fill_price * (1 + sl_pct)

        open_trade = OpenTrade(
            trade_id    = str(order.get("id") or uuid.uuid4()),
            side        = result.signal.value,
            entry_price = fill_price,
            contracts   = qty,
            take_profit = take_profit,
            stop_loss   = stop_loss,
        )
        self.rm.register_open(open_trade)

        # Place exchange-native SL and TP orders for crash-safe protection
        if Config.USE_EXCHANGE_SL_TP:
            close_side = "sell" if result.signal == Signal.LONG else "buy"

            sl_order = self.exchange.place_stop_loss_order(close_side, qty, stop_loss)
            tp_order = self.exchange.place_take_profit_order(close_side, qty, take_profit)

            sl_id = str(sl_order.get("id", "")) if sl_order else ""
            tp_id = str(tp_order.get("id", "")) if tp_order else ""

            if sl_id or tp_id:
                self.rm.set_order_ids(sl_id, tp_id)

            if not sl_order:
                log.warning(
                    "Exchange-native SL not placed — position protected by software SL only. "
                    "A bot crash would leave this position unprotected."
                )
            if not tp_order:
                log.warning(
                    "Exchange-native TP not placed — relying on software TP only."
                )

    def _manage_open_trade(self, current_price: float) -> None:
        """Software TP/SL safety net — fires if native orders haven't triggered."""
        exit_reason = self.rm.check_exit_conditions(current_price)
        if exit_reason is None:
            t = self.rm.open_trade
            if t:
                log.debug(
                    "Holding %s | entry=%.4f current=%.4f TP=%.4f SL=%.4f",
                    t.side, t.entry_price, current_price, t.take_profit, t.stop_loss,
                )
            return

        log.info("Software %s triggered at %.4f", exit_reason, current_price)
        self._close_position(current_price, exit_reason)

    def _close_position(self, current_price: float, reason: str) -> None:
        """
        Cancel native SL/TP orders, close the position on exchange,
        and record the close with the risk manager.
        """
        t = self.rm.open_trade
        if t is None:
            return

        # Cancel outstanding native orders first to avoid double-fill
        for order_id, label in ((t.sl_order_id, "SL"), (t.tp_order_id, "TP")):
            if order_id:
                try:
                    self.exchange.cancel_order(order_id)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Could not cancel %s order %s: %s", label, order_id, exc)

        # Close the position on exchange
        positions = self.exchange.fetch_positions()
        if positions:
            self.exchange.close_position(positions[0])
        else:
            # Fallback: opposing market order
            close_side = "sell" if t.side == "long" else "buy"
            self.exchange.place_market_order(close_side, t.contracts)

        self.rm.register_close(current_price, reason)

    def emergency_close_all(self) -> None:
        """Force-close everything — called on bot shutdown or max-loss breach."""
        log.warning("Emergency close all triggered")
        self.exchange.cancel_all_orders()

        if self.rm.open_trade is not None:
            t = self.rm.open_trade

            # Cancel any native SL/TP orders
            for order_id, label in ((t.sl_order_id, "SL"), (t.tp_order_id, "TP")):
                if order_id:
                    try:
                        self.exchange.cancel_order(order_id)
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "Could not cancel %s order %s on emergency: %s",
                            label, order_id, exc,
                        )

            positions = self.exchange.fetch_positions()
            if positions:
                for pos in positions:
                    self.exchange.close_position(pos)

            # Fetch last price for PnL accounting; fall back to entry price
            # if the ticker call fails so we never record a close at price 0.
            try:
                ticker = self.exchange.fetch_ticker()
                price  = float(ticker.get("last") or 0)
                if price <= 0:
                    raise ValueError("ticker returned zero/null price")
            except Exception as exc:  # noqa: BLE001
                price = self.rm.open_trade.entry_price
                log.warning(
                    "Ticker failed during emergency close (%s) — using entry price %.4f",
                    exc, price,
                )
            self.rm.register_close(price, "manual")
