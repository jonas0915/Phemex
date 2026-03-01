"""
Scalp trading strategy using EMA crossover + RSI confirmation.

Signal logic:
  LONG  → EMA_fast crosses ABOVE EMA_slow AND RSI_OVERSOLD < RSI < RSI_LONG_MAX
  SHORT → EMA_fast crosses BELOW EMA_slow AND RSI_SHORT_MIN < RSI < RSI_OVERBOUGHT

Both signals require RSI to be in a 'neutral zone' to avoid fading
an already exhausted move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
class StrategyResult:
    signal:        Signal
    current_price: float
    ema_fast:      float
    ema_slow:      float
    rsi:           float
    take_profit:   float
    stop_loss:     float


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


class ScalpStrategy:
    """
    Generates BUY / SELL / HOLD signals from OHLCV data.
    """

    def __init__(self) -> None:
        self.ema_fast_period = Config.EMA_FAST
        self.ema_slow_period = Config.EMA_SLOW
        self.rsi_period      = Config.RSI_PERIOD
        self.rsi_overbought  = Config.RSI_OVERBOUGHT
        self.rsi_oversold    = Config.RSI_OVERSOLD
        self.rsi_long_max    = Config.RSI_LONG_MAX    # upper RSI bound for LONG entries
        self.rsi_short_min   = Config.RSI_SHORT_MIN   # lower RSI bound for SHORT entries
        self.take_profit_pct = Config.TAKE_PROFIT_PCT / 100
        self.stop_loss_pct   = Config.STOP_LOSS_PCT   / 100

    # ── Main entry point ──────────────────────────────────────────────────────

    def analyse(self, ohlcv: list[list]) -> Optional[StrategyResult]:
        """
        Analyse OHLCV candles and return a StrategyResult or None if
        there are not enough candles to compute indicators.

        ohlcv: list of [timestamp, open, high, low, close, volume]
        """
        if len(ohlcv) < Config.MIN_CANDLES:
            log.debug("Not enough candles (%d / %d)", len(ohlcv), Config.MIN_CANDLES)
            return None

        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["close"] = df["close"].astype(float)

        df["ema_fast"] = _ema(df["close"], self.ema_fast_period)
        df["ema_slow"] = _ema(df["close"], self.ema_slow_period)
        df["rsi"]      = _rsi(df["close"], self.rsi_period)

        # Use last two completed candles (index -2) to detect crossovers.
        # Current (index -1) is the forming candle — we act on its close.
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        ema_fast_curr = curr["ema_fast"]
        ema_slow_curr = curr["ema_slow"]
        ema_fast_prev = prev["ema_fast"]
        ema_slow_prev = prev["ema_slow"]
        rsi           = curr["rsi"]
        price         = curr["close"]

        # Guard against NaN indicators (e.g. flat price series, zero volume)
        if any(math.isnan(v) for v in (
            ema_fast_curr, ema_slow_curr, ema_fast_prev, ema_slow_prev, float(rsi)
        )):
            log.warning("NaN indicator on candle %d — skipping signal", len(ohlcv))
            return StrategyResult(
                signal=Signal.NONE, current_price=float(price),
                ema_fast=0.0, ema_slow=0.0, rsi=0.0,
                take_profit=0.0, stop_loss=0.0,
            )

        # Detect crossover
        bullish_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast_curr > ema_slow_curr)
        bearish_cross = (ema_fast_prev >= ema_slow_prev) and (ema_fast_curr < ema_slow_curr)

        log.debug(
            "EMA fast=%.4f slow=%.4f | RSI=%.2f | bull_x=%s bear_x=%s | price=%.4f",
            ema_fast_curr, ema_slow_curr, rsi, bullish_cross, bearish_cross, price,
        )

        signal = Signal.NONE
        tp = sl = 0.0

        if bullish_cross and self.rsi_oversold < rsi < self.rsi_long_max:
            signal = Signal.LONG
            tp     = float(price) * (1 + self.take_profit_pct)
            sl     = float(price) * (1 - self.stop_loss_pct)
            log.info(
                "LONG signal | price=%.4f TP=%.4f SL=%.4f RSI=%.2f",
                price, tp, sl, rsi,
            )

        elif bearish_cross and self.rsi_short_min < rsi < self.rsi_overbought:
            signal = Signal.SHORT
            tp     = float(price) * (1 - self.take_profit_pct)
            sl     = float(price) * (1 + self.stop_loss_pct)
            log.info(
                "SHORT signal | price=%.4f TP=%.4f SL=%.4f RSI=%.2f",
                price, tp, sl, rsi,
            )

        return StrategyResult(
            signal=signal,
            current_price=float(price),
            ema_fast=float(ema_fast_curr),
            ema_slow=float(ema_slow_curr),
            rsi=float(rsi),
            take_profit=tp,
            stop_loss=sl,
        )
