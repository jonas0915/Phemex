"""
Phemex exchange wrapper built on top of CCXT.
Handles connection, market data, and order execution.
"""

from __future__ import annotations

import time
from typing import Optional

import ccxt

from config import Config
from logger import get_logger

log = get_logger(__name__)


class PhemexExchange:
    """Thin wrapper around ccxt.phemex with retry logic."""

    _MAX_RETRIES = 3
    _RETRY_DELAY = 2  # seconds

    def __init__(self) -> None:
        options = {
            "apiKey": Config.API_KEY,
            "secret": Config.API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},   # perpetual futures
        }
        if Config.TESTNET:
            log.info("Connecting to Phemex TESTNET")
            options["urls"] = {
                "api": {
                    "public": "https://testnet-api.phemex.com",
                    "private": "https://testnet-api.phemex.com",
                }
            }
        else:
            log.info("Connecting to Phemex MAINNET")

        self._exchange = ccxt.phemex(options)

        # Set leverage immediately
        self._set_leverage()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call(self, fn, *args, **kwargs):
        """Call a ccxt method with automatic retry on network errors."""
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                if attempt == self._MAX_RETRIES:
                    raise
                log.warning("Network error (attempt %d/%d): %s", attempt, self._MAX_RETRIES, exc)
                time.sleep(self._RETRY_DELAY * attempt)
            except ccxt.ExchangeError as exc:
                log.error("Exchange error: %s", exc)
                raise

    def _set_leverage(self) -> None:
        try:
            self._call(
                self._exchange.set_leverage,
                Config.LEVERAGE,
                Config.SYMBOL,
            )
            log.info("Leverage set to %dx on %s", Config.LEVERAGE, Config.SYMBOL)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not set leverage: %s", exc)

    # ── Market data ───────────────────────────────────────────────────────────

    def fetch_ohlcv(self, limit: int = 100) -> list[list]:
        """Return OHLCV candles [[ts, o, h, l, c, v], ...]."""
        return self._call(
            self._exchange.fetch_ohlcv,
            Config.SYMBOL,
            timeframe=Config.TIMEFRAME,
            limit=limit,
        )

    def fetch_ticker(self) -> dict:
        return self._call(self._exchange.fetch_ticker, Config.SYMBOL)

    def fetch_balance(self) -> dict:
        """Return balance dict with 'total', 'free', 'used' sub-dicts."""
        return self._call(self._exchange.fetch_balance)

    def fetch_usdt_balance(self) -> float:
        """Return total USDT equity (including unrealised PnL)."""
        bal = self.fetch_balance()
        # Phemex futures accounts use 'USDT' as the settle currency
        usdt = bal.get("USDT") or bal.get("total", {})
        if isinstance(usdt, dict):
            return float(usdt.get("total", 0))
        return float(usdt)

    def fetch_positions(self) -> list[dict]:
        """Return open positions for the configured symbol."""
        try:
            positions = self._call(
                self._exchange.fetch_positions,
                symbols=[Config.SYMBOL],
            )
            return [p for p in positions if float(p.get("contracts", 0) or 0) != 0]
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch positions: %s", exc)
            return []

    def fetch_open_orders(self) -> list[dict]:
        return self._call(self._exchange.fetch_open_orders, Config.SYMBOL)

    # ── Order management ──────────────────────────────────────────────────────

    def place_market_order(self, side: str, amount: float) -> Optional[dict]:
        """
        Place a market order.

        Args:
            side:   'buy' or 'sell'
            amount: contract size (USDT notional / price → qty)
        """
        try:
            order = self._call(
                self._exchange.create_market_order,
                Config.SYMBOL,
                side,
                amount,
            )
            log.info("Market order placed | side=%s amount=%.6f id=%s", side, amount, order.get("id"))
            return order
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to place market %s order: %s", side, exc)
            return None

    def place_limit_order(self, side: str, amount: float, price: float) -> Optional[dict]:
        try:
            order = self._call(
                self._exchange.create_limit_order,
                Config.SYMBOL,
                side,
                amount,
                price,
            )
            log.info("Limit order placed | side=%s amount=%.6f price=%.4f id=%s",
                     side, amount, price, order.get("id"))
            return order
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to place limit %s order: %s", side, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._call(self._exchange.cancel_order, order_id, Config.SYMBOL)
            log.info("Order %s cancelled", order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not cancel order %s: %s", order_id, exc)
            return False

    def cancel_all_orders(self) -> None:
        orders = self.fetch_open_orders()
        for o in orders:
            self.cancel_order(o["id"])

    def close_position(self, position: dict) -> Optional[dict]:
        """
        Immediately close an open position with a market order.
        """
        side = position.get("side", "")          # 'long' or 'short'
        contracts = float(position.get("contracts", 0))
        if contracts == 0:
            return None

        close_side = "sell" if side == "long" else "buy"
        log.info("Closing %s position | contracts=%.6f", side, contracts)
        return self.place_market_order(close_side, contracts)

    # ── Utility ───────────────────────────────────────────────────────────────

    def calculate_order_qty(self, price: float) -> float:
        """
        Convert a USDT notional trade size to contract qty, accounting for leverage.
        Most Phemex USDT perpetuals are linear (1 contract = 1 base coin).
        """
        notional = Config.TRADE_SIZE_USDT * Config.LEVERAGE
        qty = notional / price
        # Round to the market's precision
        try:
            market = self._exchange.market(Config.SYMBOL)
            precision = market.get("precision", {}).get("amount", 6)
            qty = round(qty, precision)
        except Exception:  # noqa: BLE001
            qty = round(qty, 6)
        return qty
