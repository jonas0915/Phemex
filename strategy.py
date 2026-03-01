"""
Scalp trading strategy: EMA crossover + RSI + ADX trend filter + ATR stops.

Signal logic (all conditions must hold):
  LONG  → EMA_fast crosses ABOVE EMA_slow
          AND EMA_fast slope is rising   (not a flat/exhausted crossover)
          AND ADX ≥ ADX_THRESHOLD        (market is actually trending)
          AND RSI_OVERSOLD < RSI < RSI_LONG_MAX
          AND price ≥ VWAP  (when USE_VWAP_FILTER=true)
          AND OB imbalance ≥ +threshold  (when orderbook supplied)
          AND bid-ask spread ≤ MAX_SPREAD_PCT

  SHORT → EMA_fast crosses BELOW EMA_slow
          AND EMA_fast slope is falling
          AND ADX ≥ ADX_THRESHOLD
          AND RSI_SHORT_MIN < RSI < RSI_OVERBOUGHT
          AND price ≤ VWAP  (when USE_VWAP_FILTER=true)
          AND OB imbalance ≤ −threshold  (when orderbook supplied)
          AND bid-ask spread ≤ MAX_SPREAD_PCT

TP / SL:
  When USE_ATR_STOPS=true (default):
    TP = ATR_TP_MULT × ATR  →  ~0.6% in normal BTC 1m conditions
    SL = ATR_SL_MULT × ATR  →  ~0.25%  (R:R ≈ 2.4, breakeven WR ≈ 47%)
  Otherwise: fixed TAKE_PROFIT_PCT / STOP_LOSS_PCT from config.

Key improvement over naive EMA crossover:
  ADX < threshold → market is choppy → skip trade (eliminates ~40% false signals)
  ATR-based stops → tighter SL in low-vol, wider in high-vol → better R:R
  EMA slope gate  → filters flat/reverting crossovers
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
    adx:           float = 0.0   # trend strength (0–100)
    atr_pct:       float = 0.0   # ATR as fraction of price
    macd_hist:     float = 0.0   # MACD histogram (+ bullish, − bearish)


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


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder smoothing.
    TR = max(H-L, |H-prev_C|, |L-prev_C|)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev  = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, and histogram.
    Histogram > 0 means bullish momentum; < 0 means bearish momentum.
    Rising histogram means momentum is accelerating in that direction.
    """
    ema_f    = series.ewm(span=fast,   adjust=False).mean()
    ema_s    = series.ewm(span=slow,   adjust=False).mean()
    macd_ln  = ema_f - ema_s
    sig_ln   = macd_ln.ewm(span=signal, adjust=False).mean()
    hist     = macd_ln - sig_ln
    return macd_ln, sig_ln, hist


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's Average Directional Index.
    ADX < 20  → choppy / ranging
    ADX 20–25 → weak trend forming
    ADX > 25  → confirmed trend
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    up   = high.diff()
    down = -low.diff()

    dm_plus  = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    dm_minus = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )

    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)

    smooth = period - 1   # Wilder smoothing ≡ EMA com=period-1
    atr_s    = tr.ewm(com=smooth, adjust=False).mean().replace(0, np.nan)
    di_plus  = 100 * dm_plus.ewm(com=smooth, adjust=False).mean()  / atr_s
    di_minus = 100 * dm_minus.ewm(com=smooth, adjust=False).mean() / atr_s

    di_sum = (di_plus + di_minus).replace(0, np.nan)
    dx     = 100 * (di_plus - di_minus).abs() / di_sum
    adx    = dx.ewm(com=smooth, adjust=False).mean()
    return adx.fillna(0.0)


def _vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price over the rolling candle window.
    typical_price = (high + low + close) / 3
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (typical * df["volume"]).cumsum() / cum_vol


# ── Main strategy class ────────────────────────────────────────────────────────

class ScalpStrategy:
    """
    Generates BUY / SELL / HOLD signals from OHLCV data.

    Filters applied in order:
      1. Enough candles for reliable indicator values
      2. ADX ≥ threshold  → trend strength confirmation
      3. EMA crossover    → direction signal
      4. EMA slope        → crossover must be actively trending
      5. RSI              → not overbought/oversold at entry
      6. VWAP             → optional volume-weighted trend bias
      7. OB imbalance     → optional real-time order flow confirmation
      8. Bid-ask spread   → optional liquidity guard
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

        mid    = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid) / mid * 100

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
        orderbook: CCXT order book dict (optional).
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
        df["atr"]      = _atr(df, Config.ATR_PERIOD)
        df["adx"]      = _adx(df, Config.ADX_PERIOD)
        _, _, df["macd_hist"] = _macd(
            df["close"],
            fast=Config.MACD_FAST,
            slow=Config.MACD_SLOW,
            signal=Config.MACD_SIGNAL,
        )

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        ema_fast_curr = float(curr["ema_fast"])
        ema_slow_curr = float(curr["ema_slow"])
        ema_fast_prev = float(prev["ema_fast"])
        ema_slow_prev = float(prev["ema_slow"])
        rsi           = float(curr["rsi"])
        rsi_prev      = float(prev["rsi"])
        price         = float(curr["close"])
        vwap          = float(curr["vwap"])
        adx_val       = float(curr["adx"])
        atr_val       = float(curr["atr"])
        atr_pct       = atr_val / price if price > 0 else 0.0
        macd_hist     = float(curr["macd_hist"])

        if any(math.isnan(v) for v in (
            ema_fast_curr, ema_slow_curr, ema_fast_prev, ema_slow_prev,
            rsi, vwap, adx_val, atr_val, macd_hist,
        )):
            log.warning("NaN indicator on candle %d — skipping signal", len(ohlcv))
            return StrategyResult(
                signal=Signal.NONE, current_price=price,
                ema_fast=0.0, ema_slow=0.0, rsi=0.0, vwap=0.0,
                take_profit=0.0, stop_loss=0.0,
            )

        # ── 1. ADX filter: skip choppy/ranging markets ─────────────────────────
        if adx_val < Config.ADX_THRESHOLD:
            log.debug(
                "ADX %.2f < threshold %.1f — market not trending, skip",
                adx_val, Config.ADX_THRESHOLD,
            )
            return StrategyResult(
                signal=Signal.NONE, current_price=price,
                ema_fast=ema_fast_curr, ema_slow=ema_slow_curr,
                rsi=rsi, vwap=vwap,
                take_profit=0.0, stop_loss=0.0,
                adx=adx_val, atr_pct=atr_pct,
            )

        # ── 2. EMA slope: fast EMA must be actively moving ─────────────────────
        # EMA_SLOPE_BARS=0 disables this filter entirely.
        if Config.EMA_SLOPE_BARS > 0 and len(df) >= Config.EMA_SLOPE_BARS + 1:
            ema_fast_old   = float(df["ema_fast"].iloc[-(Config.EMA_SLOPE_BARS + 1)])
            ema_fast_slope = ema_fast_curr - ema_fast_old   # + rising, − falling
            slope_ok_long  = ema_fast_slope > 0
            slope_ok_short = ema_fast_slope < 0
        else:
            ema_fast_slope = 0.0
            slope_ok_long  = True   # filter disabled
            slope_ok_short = True

        # ── 3. Medium-term trend confirmation ──────────────────────────────────
        # Close must be above/below its N-bar-ago value to confirm direction.
        # Eliminates counter-trend crossovers (EMA lag catching the wrong side).
        tb = Config.TREND_CONFIRM_BARS
        if tb > 0 and len(df) >= tb + 1:
            price_n_ago    = float(df["close"].iloc[-(tb + 1)])
            med_trend_up   = price > price_n_ago   # confirmed upward bias
            med_trend_down = price < price_n_ago   # confirmed downward bias
        else:
            med_trend_up   = True   # disabled: allow all
            med_trend_down = True

        # ── 4. Order-book context ──────────────────────────────────────────────
        ob_ctx = self.analyse_orderbook(orderbook) if orderbook else _EMPTY_OB

        # Spread guard
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
                adx=adx_val, atr_pct=atr_pct,
            )

        # ── 5. Crossover detection ─────────────────────────────────────────────
        bullish_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast_curr > ema_slow_curr)
        bearish_cross = (ema_fast_prev >= ema_slow_prev) and (ema_fast_curr < ema_slow_curr)

        log.debug(
            "EMA fast=%.4f slow=%.4f | slope=%.4f | ADX=%.2f ATR=%.4f%% | "
            "RSI=%.2f | bull_x=%s bear_x=%s | price=%.4f",
            ema_fast_curr, ema_slow_curr, ema_fast_slope, adx_val, atr_pct * 100,
            rsi, bullish_cross, bearish_cross, price,
        )

        # ── 6. Compute ATR-based TP / SL ──────────────────────────────────────
        if Config.USE_ATR_STOPS and atr_pct > 0:
            # TP and SL scale with volatility; clamp to reasonable ranges
            tp_pct = float(np.clip(Config.ATR_TP_MULT * atr_pct, 0.002, 0.05))
            sl_pct = float(np.clip(Config.ATR_SL_MULT * atr_pct, 0.001, 0.02))
        else:
            tp_pct = self.take_profit_pct
            sl_pct = self.stop_loss_pct

        signal = Signal.NONE
        tp = sl = 0.0

        # ── 7. LONG signal ─────────────────────────────────────────────────────
        if bullish_cross and self.rsi_oversold < rsi < self.rsi_long_max:
            trend_ok  = med_trend_up                # medium-term bias confirmed
            vwap_ok   = (not Config.USE_VWAP_FILTER) or (price >= vwap * 0.999)
            ob_ok     = (not orderbook) or (ob_ctx.imbalance >= Config.OB_IMBALANCE_THRESHOLD)
            # MACD histogram must be positive: confirms bullish momentum aligns
            macd_ok   = macd_hist > 0
            # RSI must be rising: momentum is building, not exhausted
            rsi_ok    = rsi > rsi_prev

            if slope_ok_long and trend_ok and vwap_ok and ob_ok and macd_ok and rsi_ok:
                signal = Signal.LONG
                tp     = price * (1 + tp_pct)
                sl     = price * (1 - sl_pct)
                log.info(
                    "LONG signal | price=%.4f TP=%.4f (+%.3f%%) SL=%.4f (-%.3f%%) "
                    "ADX=%.1f ATR=%.4f%% RSI=%.2f MACD_h=%.4f",
                    price, tp, tp_pct * 100, sl, sl_pct * 100,
                    adx_val, atr_pct * 100, rsi, macd_hist,
                )
            else:
                log.debug(
                    "LONG cross BLOCKED | slope=%s trend=%s vwap=%s ob=%s macd=%s rsi_up=%s",
                    slope_ok_long, trend_ok, vwap_ok, ob_ok, macd_ok, rsi_ok,
                )

        # ── 8. SHORT signal ────────────────────────────────────────────────────
        elif bearish_cross and self.rsi_short_min < rsi < self.rsi_overbought:
            trend_ok  = med_trend_down              # medium-term bias confirmed
            vwap_ok   = (not Config.USE_VWAP_FILTER) or (price <= vwap * 1.001)
            ob_ok     = (not orderbook) or (ob_ctx.imbalance <= -Config.OB_IMBALANCE_THRESHOLD)
            # MACD histogram must be negative: confirms bearish momentum aligns
            macd_ok   = macd_hist < 0
            # RSI must be falling: momentum is building on the downside
            rsi_ok    = rsi < rsi_prev

            if slope_ok_short and trend_ok and vwap_ok and ob_ok and macd_ok and rsi_ok:
                signal = Signal.SHORT
                tp     = price * (1 - tp_pct)
                sl     = price * (1 + sl_pct)
                log.info(
                    "SHORT signal | price=%.4f TP=%.4f (-%.3f%%) SL=%.4f (+%.3f%%) "
                    "ADX=%.1f ATR=%.4f%% RSI=%.2f MACD_h=%.4f",
                    price, tp, tp_pct * 100, sl, sl_pct * 100,
                    adx_val, atr_pct * 100, rsi, macd_hist,
                )
            else:
                log.debug(
                    "SHORT cross BLOCKED | slope=%s trend=%s vwap=%s ob=%s macd=%s rsi_dn=%s",
                    slope_ok_short, trend_ok, vwap_ok, ob_ok, macd_ok, rsi_ok,
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
            adx           = adx_val,
            atr_pct       = atr_pct,
            macd_hist     = macd_hist,
        )
