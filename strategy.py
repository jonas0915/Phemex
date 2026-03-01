"""
Scalp trading strategy using EMA crossover + RSI confirmation,
filtered by VWAP trend bias and order-book imbalance.

Signal logic:
  LONG  → EMA_fast crosses ABOVE EMA_slow
          AND RSI_OVERSOLD < RSI < RSI_LONG_MAX
          AND price ≥ VWAP  (volume-confirmed uptrend)
          AND OB imbalance ≥ +OB_IMBALANCE_THRESHOLD  (bid-side dominance)
          AND bid-ask spread ≤ MAX_SPREAD_PCT

  SHORT → EMA_fast crosses BELOW EMA_slow
          AND RSI_SHORT_MIN < RSI < RSI_OVERBOUGHT
          AND price ≤ VWAP  (volume-confirmed downtrend)
          AND OB imbalance ≤ −OB_IMBALANCE_THRESHOLD  (ask-side dominance)
          AND bid-ask spread ≤ MAX_SPREAD_PCT

Order book context is optional (pass None to skip L2 filters, e.g. during
back-testing without synthetic books).  VWAP filter is always applied when
USE_VWAP_FILTER=true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from logger import get_logger

log = get_logger(__name__)


class Signal(Enum):
    LONG  = "long"
    SHORT = "short"
    NONE  = "none"


@dataclass
class OrderBookContext:
    """Microstructure snapshot derived from the L2 order book."""
    imbalance:  float   # -1.0 (ask-heavy) to +1.0 (bid-heavy)
    spread_pct: float   # bid-ask spread as % of mid-price
    mid_price:  float   # (best_bid + best_ask) / 2


_EMPTY_OB = OrderBookContext(imbalance=0.0, spread_pct=0.0, mid_price=0.0)


@dataclass
class StrategyResult:
    signal:        Signal
    current_price: float
    ema_fast:      float
    ema_slow:      float
    rsi:           float
    vwap:          float
    take_profit:   float
    stop_loss:     float
    ob_imbalance:  float = 0.0
    spread_pct:    float = 0.0


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price over the rolling candle window.
    typical_price = (high + low + close) / 3
    Returns a Series aligned with df.index.
    """
    typical   = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol   = df["volume"].cumsum().replace(0, np.nan)
    return (typical * df["volume"]).cumsum() / cum_vol


# ── Main strategy class ────────────────────────────────────────────────────────

class ScalpStrategy:
    """
    Generates BUY / SELL / HOLD signals from OHLCV data, optionally
    filtered by L2 order-book context and VWAP.
    """

    def __init__(self) -> None:
        self.ema_fast_period = Config.EMA_FAST
        self.ema_slow_period = Config.EMA_SLOW
        self.rsi_period      = Config.RSI_PERIOD
        self.rsi_overbought  = Config.RSI_OVERBOUGHT
        self.rsi_oversold    = Config.RSI_OVERSOLD
        self.rsi_long_max    = Config.RSI_LONG_MAX
        self.rsi_short_min   = Config.RSI_SHORT_MIN
        self.take_profit_pct = Config.TAKE_PROFIT_PCT / 100
        self.stop_loss_pct   = Config.STOP_LOSS_PCT   / 100

    # ── Order-book analysis ────────────────────────────────────────────────────

    @staticmethod
    def analyse_orderbook(
        orderbook: dict,
        depth_pct: float = 0.005,
    ) -> OrderBookContext:
        """
        Compute bid/ask imbalance and spread from a CCXT order book dict.

        depth_pct: radius around mid-price to count as 'near market' (default 0.5%).
        Levels further than this are large wall orders, not immediate pressure.

        Returns OrderBookContext with:
          imbalance  = (bid_vol − ask_vol) / (bid_vol + ask_vol)  ∈ [−1, +1]
          spread_pct = (best_ask − best_bid) / mid × 100
          mid_price  = (best_bid + best_ask) / 2
        """
        bids = orderbook.get("bids", [])   # [[price, size], ...] descending
        asks = orderbook.get("asks", [])   # [[price, size], ...] ascending

        if not bids or not asks:
            return _EMPTY_OB

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])

        if best_ask <= best_bid or best_bid <= 0:
            return _EMPTY_OB

        mid      = (best_bid + best_ask) / 2
        spread   = (best_ask - best_bid) / mid * 100

        lower = mid * (1 - depth_pct)
        upper = mid * (1 + depth_pct)

        bid_vol = sum(float(s) for p, s in bids if float(p) >= lower)
        ask_vol = sum(float(s) for p, s in asks if float(p) <= upper)
        total   = bid_vol + ask_vol

        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

        return OrderBookContext(
            imbalance  = float(np.clip(imbalance, -1.0, 1.0)),
            spread_pct = spread,
            mid_price  = mid,
        )

    # ── Main entry point ───────────────────────────────────────────────────────

    def analyse(
        self,
        ohlcv: list[list],
        orderbook: Optional[dict] = None,
    ) -> Optional[StrategyResult]:
        """
        Analyse OHLCV candles and return a StrategyResult (or None if not
        enough candles).

        ohlcv:     list of [timestamp, open, high, low, close, volume]
        orderbook: CCXT order book dict (optional).  When supplied, the
                   bid/ask imbalance and spread filters are applied.
                   When None, those filters are skipped (optimistic mode).
        """
        if len(ohlcv) < Config.MIN_CANDLES:
            log.debug("Not enough candles (%d / %d)", len(ohlcv), Config.MIN_CANDLES)
            return None

        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["close"]  = df["close"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float).clip(lower=0)

        df["ema_fast"] = _ema(df["close"], self.ema_fast_period)
        df["ema_slow"] = _ema(df["close"], self.ema_slow_period)
        df["rsi"]      = _rsi(df["close"], self.rsi_period)
        df["vwap"]     = _vwap(df)

        # Use last two completed candles to detect crossovers.
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        ema_fast_curr = float(curr["ema_fast"])
        ema_slow_curr = float(curr["ema_slow"])
        ema_fast_prev = float(prev["ema_fast"])
        ema_slow_prev = float(prev["ema_slow"])
        rsi           = float(curr["rsi"])
        price         = float(curr["close"])
        vwap          = float(curr["vwap"])

        # Guard against NaN (flat price, zero volume, early candles)
        if any(math.isnan(v) for v in (
            ema_fast_curr, ema_slow_curr, ema_fast_prev, ema_slow_prev, rsi, vwap
        )):
            log.warning("NaN indicator on candle %d — skipping signal", len(ohlcv))
            return StrategyResult(
                signal=Signal.NONE, current_price=price,
                ema_fast=0.0, ema_slow=0.0, rsi=0.0, vwap=0.0,
                take_profit=0.0, stop_loss=0.0,
            )

        # ── Order-book context ─────────────────────────────────────────────────
        ob_ctx = self.analyse_orderbook(orderbook) if orderbook else _EMPTY_OB

        # Spread guard — skip entry when bid-ask spread is too wide
        if orderbook and ob_ctx.spread_pct > Config.MAX_SPREAD_PCT:
            log.debug(
                "Spread %.4f%% > limit %.4f%% — no entry",
                ob_ctx.spread_pct, Config.MAX_SPREAD_PCT,
            )
            return StrategyResult(
                signal=Signal.NONE, current_price=price,
                ema_fast=ema_fast_curr, ema_slow=ema_slow_curr,
                rsi=rsi, vwap=vwap,
                take_profit=0.0, stop_loss=0.0,
                ob_imbalance=ob_ctx.imbalance, spread_pct=ob_ctx.spread_pct,
            )

        # ── Crossover detection ────────────────────────────────────────────────
        bullish_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast_curr > ema_slow_curr)
        bearish_cross = (ema_fast_prev >= ema_slow_prev) and (ema_fast_curr < ema_slow_curr)

        log.debug(
            "EMA fast=%.4f slow=%.4f | RSI=%.2f | VWAP=%.4f | "
            "OB_imb=%.3f spread=%.4f%% | bull_x=%s bear_x=%s | price=%.4f",
            ema_fast_curr, ema_slow_curr, rsi, vwap,
            ob_ctx.imbalance, ob_ctx.spread_pct,
            bullish_cross, bearish_cross, price,
        )

        signal = Signal.NONE
        tp = sl = 0.0

        if bullish_cross and self.rsi_oversold < rsi < self.rsi_long_max:
            # VWAP filter: price should be at or above the volume-weighted mean
            vwap_ok = (not Config.USE_VWAP_FILTER) or (price >= vwap * 0.999)
            # OB filter: bid-side pressure must confirm bullish momentum
            ob_ok   = (not orderbook) or (ob_ctx.imbalance >= Config.OB_IMBALANCE_THRESHOLD)

            if vwap_ok and ob_ok:
                signal = Signal.LONG
                tp     = price * (1 + self.take_profit_pct)
                sl     = price * (1 - self.stop_loss_pct)
                log.info(
                    "LONG signal | price=%.4f TP=%.4f SL=%.4f "
                    "RSI=%.2f VWAP=%.4f OB_imb=%.3f",
                    price, tp, sl, rsi, vwap, ob_ctx.imbalance,
                )
            else:
                log.debug(
                    "LONG cross BLOCKED | vwap_ok=%s ob_ok=%s "
                    "(price=%.4f vwap=%.4f imb=%.3f)",
                    vwap_ok, ob_ok, price, vwap, ob_ctx.imbalance,
                )

        elif bearish_cross and self.rsi_short_min < rsi < self.rsi_overbought:
            # VWAP filter: price should be at or below the volume-weighted mean
            vwap_ok = (not Config.USE_VWAP_FILTER) or (price <= vwap * 1.001)
            # OB filter: ask-side pressure must confirm bearish momentum
            ob_ok   = (not orderbook) or (ob_ctx.imbalance <= -Config.OB_IMBALANCE_THRESHOLD)

            if vwap_ok and ob_ok:
                signal = Signal.SHORT
                tp     = price * (1 - self.take_profit_pct)
                sl     = price * (1 + self.stop_loss_pct)
                log.info(
                    "SHORT signal | price=%.4f TP=%.4f SL=%.4f "
                    "RSI=%.2f VWAP=%.4f OB_imb=%.3f",
                    price, tp, sl, rsi, vwap, ob_ctx.imbalance,
                )
            else:
                log.debug(
                    "SHORT cross BLOCKED | vwap_ok=%s ob_ok=%s "
                    "(price=%.4f vwap=%.4f imb=%.3f)",
                    vwap_ok, ob_ok, price, vwap, ob_ctx.imbalance,
                )

        return StrategyResult(
            signal        = signal,
            current_price = price,
            ema_fast      = ema_fast_curr,
            ema_slow      = ema_slow_curr,
            rsi           = rsi,
            vwap          = vwap,
            take_profit   = tp,
            stop_loss     = sl,
            ob_imbalance  = ob_ctx.imbalance,
            spread_pct    = ob_ctx.spread_pct,
        )
