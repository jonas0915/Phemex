"""
Configuration loader for the Phemex scalp trading bot.
Reads all settings from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default=None, cast=str, required: bool = False):
    val = os.environ.get(key, default)
    if required and val is None:
        raise EnvironmentError(f"Required env var '{key}' is not set. Check your .env file.")
    if val is None:
        return None
    try:
        return cast(val)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid value for '{key}': {val}") from exc


class Config:
    # ── Phemex API ──────────────────────────────────────────────────────────
    API_KEY: str = _get("PHEMEX_API_KEY", required=True)
    API_SECRET: str = _get("PHEMEX_API_SECRET", required=True)
    TESTNET: bool = _get("PHEMEX_TESTNET", default="true").lower() == "true"

    # ── Market ───────────────────────────────────────────────────────────────
    SYMBOL: str = _get("TRADING_SYMBOL", default="BTC/USDT:USDT")
    TIMEFRAME: str = _get("TRADING_TIMEFRAME", default="1m")
    TRADE_SIZE_USDT: float = _get("TRADE_SIZE_USDT", default="100", cast=float)
    LEVERAGE: int = _get("LEVERAGE", default="5", cast=int)

    # ── Strategy ─────────────────────────────────────────────────────────────
    EMA_FAST: int = _get("EMA_FAST", default="9", cast=int)
    EMA_SLOW: int = _get("EMA_SLOW", default="21", cast=int)
    RSI_PERIOD: int = _get("RSI_PERIOD", default="14", cast=int)
    RSI_OVERBOUGHT: float = _get("RSI_OVERBOUGHT", default="70", cast=float)
    RSI_OVERSOLD: float = _get("RSI_OVERSOLD", default="30", cast=float)

    # ── Risk management ──────────────────────────────────────────────────────
    MAX_SESSION_LOSS_PCT: float = _get("MAX_SESSION_LOSS_PCT", default="30", cast=float)
    TAKE_PROFIT_PCT: float = _get("TAKE_PROFIT_PCT", default="0.6", cast=float)
    STOP_LOSS_PCT: float = _get("STOP_LOSS_PCT", default="0.35", cast=float)
    MAX_CONCURRENT_TRADES: int = _get("MAX_CONCURRENT_TRADES", default="1", cast=int)
    TRADE_COOLDOWN_SECONDS: int = _get("TRADE_COOLDOWN_SECONDS", default="30", cast=int)

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = _get("LOG_LEVEL", default="INFO")
    LOG_FILE: str = _get("LOG_FILE", default="logs/trading_bot.log")

    # Minimum candles needed before generating signals
    MIN_CANDLES: int = max(EMA_SLOW, RSI_PERIOD) + 5
