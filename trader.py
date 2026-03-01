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

        # 0. Advance per-candle risk state (cooldown counters, etc.)
        self.rm.on_new_candle()

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

        # 4. Fetch L2 order book for signal filtering and smart execution
        orderbook: Optional[dict] = self.exchange.fetch_order_book(Config.OB_DEPTH)
        if not orderbook.get("bids"):
            log.debug("Order book unavailable — L2 filters will be skipped this tick")
            orderbook = None

        # 5. Sync in-memory position against actual exchange positions.
        if self.rm.open_trade is not None:
            self._sync_position(current_price)

        # 6. Manage open position — software TP/SL as safety net
        if self.rm.open_trade is not None:
            self._manage_open_trade(current_price)
            return   # one action per tick

        # 7. Session locked? No new entries.
        if self.rm.session_locked:
            log.info("Session locked — skipping signal evaluation")
            return

        # 8. Evaluate strategy (with L2 context for filtering)
        result: Optional[StrategyResult] = self.strategy.analyse(ohlcv, orderbook=orderbook)
        if result is None or result.signal == Signal.NONE:
            return

        # 9. Check risk manager gate
        allowed, reason = self.rm.can_open_trade()
        if not allowed:
            log.debug("Trade blocked: %s", reason)
            return

        # 10. Open trade (pass live balance for compound sizing + OB for execution)
        self._open_trade(result, current_price, balance, orderbook)

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

        if time.time() - t.opened_at < 5:
            return

        try:
            positions = self.exchange.fetch_positions()
        except Exception as exc:  # noqa: BLE001
            log.warning("Position sync skipped (fetch failed): %s", exc)
            return

        if positions:
            return  # Position still open on exchange — nothing to reconcile

        log.warning(
            "Position sync: exchange shows no open position for trade %s "
            "— recording close at current price %.4f (native SL/TP or manual).",
            t.trade_id, current_price,
        )

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
        orderbook: Optional[dict] = None,
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

        # ── Smart execution: try maker limit first, fall back to market ─────────
        if Config.USE_MAKER_ENTRY and orderbook:
            order = self._try_maker_entry(side, qty, orderbook, result.signal)
        else:
            order = self.exchange.place_market_order(side, qty)

        if order is None:
            log.error("Order placement failed — trade aborted")
            return

        fill_price = float(order.get("average") or order.get("price") or current_price)

        # Use the strategy's computed TP/SL percentages (ATR-based or fixed).
        # Re-apply them to the actual fill price so slippage is accounted for.
        raw_price = result.current_price
        tp_pct = abs(result.take_profit - raw_price) / raw_price if raw_price > 0 else Config.TAKE_PROFIT_PCT / 100
        sl_pct = abs(result.stop_loss   - raw_price) / raw_price if raw_price > 0 else Config.STOP_LOSS_PCT   / 100
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
                log.warning("Exchange-native TP not placed — relying on software TP only.")

    def _try_maker_entry(
        self,
        side: str,
        qty: float,
        orderbook: dict,
        signal: Signal,
    ) -> Optional[dict]:
        """
        Attempt entry with a limit order posted inside the spread.

        Phemex pays a maker rebate of −0.025% vs a taker fee of +0.075%.
        Saving 0.10% per fill has a real impact against a 0.6% TP target.

        Strategy:
          • LONG  → limit at (best_bid + mid) / 2  (slightly above best bid)
          • SHORT → limit at (best_ask + mid) / 2  (slightly below best ask)

        If not filled within MAKER_ENTRY_TIMEOUT_S seconds, the limit order
        is cancelled and we fall back to a market order.
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return self.exchange.place_market_order(side, qty)

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid      = (best_bid + best_ask) / 2

        if signal == Signal.LONG:
            # Place between best bid and mid — post as maker, slight pull toward mid
            limit_price = round((best_bid + mid) / 2, 1)
        else:
            limit_price = round((best_ask + mid) / 2, 1)

        order = self.exchange.place_limit_order(side, qty, limit_price)
        if order is None:
            log.debug("Limit order placement failed — falling back to market")
            return self.exchange.place_market_order(side, qty)

        order_id = str(order.get("id", ""))
        log.info(
            "Maker limit posted | side=%s qty=%.6f price=%.4f id=%s — "
            "waiting up to %ds for fill",
            side, qty, limit_price, order_id, Config.MAKER_ENTRY_TIMEOUT_S,
        )

        deadline = time.time() + Config.MAKER_ENTRY_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(1)
            status = self.exchange.fetch_order_status(order_id)
            if status is None:
                break
            order_status = status.get("status", "")
            if order_status == "closed":
                log.info(
                    "Maker limit filled | id=%s avg=%.4f",
                    order_id, status.get("average", limit_price),
                )
                return status
            if order_status in ("canceled", "rejected", "expired"):
                log.warning(
                    "Maker limit %s — falling back to market", order_status
                )
                return self.exchange.place_market_order(side, qty)

        # Timed out — cancel the pending limit and use market
        log.info(
            "Maker limit not filled in %ds — cancelling, using market",
            Config.MAKER_ENTRY_TIMEOUT_S,
        )
        self.exchange.cancel_order(order_id)
        return self.exchange.place_market_order(side, qty)

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

        for order_id, label in ((t.sl_order_id, "SL"), (t.tp_order_id, "TP")):
            if order_id:
                try:
                    self.exchange.cancel_order(order_id)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Could not cancel %s order %s: %s", label, order_id, exc)

        positions = self.exchange.fetch_positions()
        if positions:
            self.exchange.close_position(positions[0])
        else:
            close_side = "sell" if t.side == "long" else "buy"
            self.exchange.place_market_order(close_side, t.contracts)

        self.rm.register_close(current_price, reason)

    def emergency_close_all(self) -> None:
        """Force-close everything — called on bot shutdown or max-loss breach."""
        log.warning("Emergency close all triggered")
        self.exchange.cancel_all_orders()

        if self.rm.open_trade is not None:
            t = self.rm.open_trade

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
