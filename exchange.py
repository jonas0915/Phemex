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
        """Call a ccxt method with automatic retry on network/rate-limit errors."""
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except ccxt.RateLimitExceeded as exc:
                if attempt == self._MAX_RETRIES:
                    raise
                wait = self._RETRY_DELAY * attempt * 2   # longer back-off for rate limits
                log.warning("Rate limit (attempt %d/%d) — sleeping %ds: %s",
                            attempt, self._MAX_RETRIES, wait, exc)
                time.sleep(wait)
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
        result = self._call(
            self._exchange.fetch_ohlcv,
            Config.SYMBOL,
            timeframe=Config.TIMEFRAME,
            limit=limit,
        )
        if not result:
            log.warning("fetch_ohlcv returned empty or None for %s", Config.SYMBOL)
            return []
        return result

    def fetch_ticker(self) -> dict:
        return self._call(self._exchange.fetch_ticker, Config.SYMBOL)

    def fetch_balance(self) -> dict:
        """Return balance dict with 'total', 'free', 'used' sub-dicts."""
        return self._call(self._exchange.fetch_balance)

    def fetch_usdt_balance(self) -> float:
        """Return total USDT equity (including unrealised PnL)."""
        bal = self.fetch_balance()
        # Walk through common CCXT balance structures for Phemex futures
        for currency in ("USDT", "USD"):
            entry = bal.get(currency)
            if isinstance(entry, dict):
                for field in ("total", "free", "used"):
                    val = entry.get(field)
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            pass
            elif entry is not None:
                try:
                    return float(entry)
                except (ValueError, TypeError):
                    pass
        # Last resort: walk bal["total"] sub-dict
        total_dict = bal.get("total") or {}
        if isinstance(total_dict, dict):
            for currency in ("USDT", "USD"):
                val = total_dict.get(currency)
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
        log.warning("Could not parse USDT balance from exchange response")
        return 0.0

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

    def fetch_order_book(self, depth: int = 20) -> dict:
        """
        Return L2 order book snapshot.
        {"bids": [[price, size], ...], "asks": [[price, size], ...]}
        Bids are sorted descending (highest first); asks ascending (lowest first).
        Returns empty book dict on failure so callers can treat missing OB gracefully.
        """
        try:
            return self._call(self._exchange.fetch_order_book, Config.SYMBOL, depth)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_order_book failed: %s", exc)
            return {"bids": [], "asks": []}

    def fetch_order_status(self, order_id: str) -> Optional[dict]:
        """Fetch a single order by ID. Returns None on failure."""
        try:
            return self._call(self._exchange.fetch_order, order_id, Config.SYMBOL)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_order_status %s failed: %s", order_id, exc)
            return None

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
        if amount <= 0:
            log.error("Refused market order — qty must be > 0, got %.6f", amount)
            return None
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

    def place_stop_loss_order(
        self, side: str, amount: float, stop_price: float
    ) -> Optional[dict]:
        """
        Place a reduce-only stop-market order on the exchange (native SL).
        Triggered when last price crosses stop_price in the adverse direction.
        side: 'sell' to protect a long, 'buy' to protect a short.
        Returns order dict or None on failure (caller falls back to software SL).
        """
        if amount <= 0:
            log.error("Refused SL order — qty must be > 0, got %.6f", amount)
            return None
        try:
            order = self._call(
                self._exchange.create_order,
                Config.SYMBOL,
                "Stop",   # Phemex stop-market type for G-contracts
                side,
                amount,
                None,     # no limit price — triggers a market fill
                {
                    "stopPrice":   stop_price,
                    "reduceOnly":  True,
                    "triggerType": "ByLastPrice",
                },
            )
            log.info(
                "Exchange SL order placed | side=%s qty=%.6f trigger=%.4f id=%s",
                side, amount, stop_price, order.get("id"),
            )
            return order
        except Exception as exc:  # noqa: BLE001
            log.warning("Exchange-native SL placement failed: %s", exc)
            return None

    def place_take_profit_order(
        self, side: str, amount: float, tp_price: float
    ) -> Optional[dict]:
        """
        Place a reduce-only limit order at the take-profit price (native TP).
        side: 'sell' to close a long, 'buy' to close a short.
        Returns order dict or None on failure (caller falls back to software TP).
        """
        if amount <= 0:
            log.error("Refused TP order — qty must be > 0, got %.6f", amount)
            return None
        try:
            order = self._call(
                self._exchange.create_limit_order,
                Config.SYMBOL,
                side,
                amount,
                tp_price,
                {"reduceOnly": True},
            )
            log.info(
                "Exchange TP order placed | side=%s qty=%.6f price=%.4f id=%s",
                side, amount, tp_price, order.get("id"),
            )
            return order
        except Exception as exc:  # noqa: BLE001
            log.warning("Exchange-native TP placement failed: %s", exc)
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
        Handles the various side-field names CCXT may use for Phemex.
        """
        contracts = float(position.get("contracts", 0) or 0)
        if contracts == 0:
            return None

        # CCXT normalises side to 'long'/'short'; fallback to info dict
        side = (position.get("side") or "").lower()
        if side not in ("long", "short"):
            info = position.get("info", {})
            side = (info.get("side") or info.get("posSide") or "").lower()

        if side == "long":
            close_side = "sell"
        elif side == "short":
            close_side = "buy"
        else:
            log.error("Cannot determine position side from: %s — skipping close", position)
            return None

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
