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
    API_KEY:    str  = _get("PHEMEX_API_KEY",    required=True)
    API_SECRET: str  = _get("PHEMEX_API_SECRET", required=True)
    TESTNET:    bool = _get("PHEMEX_TESTNET", default="true").lower() == "true"

    # ── Market ───────────────────────────────────────────────────────────────
    SYMBOL:          str   = _get("TRADING_SYMBOL",    default="BTC/USDT:USDT")
    TIMEFRAME:       str   = _get("TRADING_TIMEFRAME", default="1m")
    TRADE_SIZE_USDT: float = _get("TRADE_SIZE_USDT",   default="100",  cast=float)
    LEVERAGE:        int   = _get("LEVERAGE",           default="5",    cast=int)

    # ── Strategy ─────────────────────────────────────────────────────────────
    EMA_FAST:       int   = _get("EMA_FAST",       default="9",  cast=int)
    EMA_SLOW:       int   = _get("EMA_SLOW",       default="21", cast=int)
    RSI_PERIOD:     int   = _get("RSI_PERIOD",     default="14", cast=int)
    RSI_OVERBOUGHT: float = _get("RSI_OVERBOUGHT", default="70", cast=float)
    RSI_OVERSOLD:   float = _get("RSI_OVERSOLD",   default="30", cast=float)

    # Neutral zone bounds: only enter LONG when RSI < RSI_LONG_MAX,
    # only enter SHORT when RSI > RSI_SHORT_MIN.
    # Prevents entries when momentum is already over-extended.
    RSI_LONG_MAX:  float = _get("RSI_LONG_MAX",  default="60", cast=float)
    RSI_SHORT_MIN: float = _get("RSI_SHORT_MIN", default="40", cast=float)

    # ── Risk management ──────────────────────────────────────────────────────
    MAX_SESSION_LOSS_PCT:   float = _get("MAX_SESSION_LOSS_PCT",   default="30",   cast=float)
    TAKE_PROFIT_PCT:        float = _get("TAKE_PROFIT_PCT",        default="0.6",  cast=float)
    STOP_LOSS_PCT:          float = _get("STOP_LOSS_PCT",          default="0.35", cast=float)
    MAX_CONCURRENT_TRADES:  int   = _get("MAX_CONCURRENT_TRADES",  default="1",    cast=int)
    TRADE_COOLDOWN_SECONDS: int   = _get("TRADE_COOLDOWN_SECONDS", default="30",   cast=int)

    # Compound sizing: when > 0, risk this % of current account balance per trade
    # (multiplied by LEVERAGE). Set to 0 to use the fixed TRADE_SIZE_USDT instead.
    RISK_PER_TRADE_PCT: float = _get("RISK_PER_TRADE_PCT", default="0", cast=float)

    # ── Exchange-native SL / TP orders ────────────────────────────────────────
    # When True, place exchange-side stop-loss and take-profit orders immediately
    # after entry so the position is protected even if the bot crashes or loses
    # connectivity.  Set False only for paper-trading / testnet debugging.
    USE_EXCHANGE_SL_TP: bool = _get("USE_EXCHANGE_SL_TP", default="true").lower() == "true"

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = _get("LOG_LEVEL", default="INFO")
    LOG_FILE:  str = _get("LOG_FILE",  default="logs/trading_bot.log")

    # Minimum candles needed before generating signals
    MIN_CANDLES: int = max(EMA_SLOW, RSI_PERIOD) + 5

    # ── Startup validation ────────────────────────────────────────────────────

    @classmethod
    def validate(cls) -> None:
        """
        Raise ValueError listing every configuration problem found.
        Call this at bot startup before connecting to the exchange.
        """
        errors: list[str] = []

        if cls.LEVERAGE < 1 or cls.LEVERAGE > 100:
            errors.append(f"LEVERAGE={cls.LEVERAGE} must be between 1 and 100")

        if cls.TAKE_PROFIT_PCT <= 0:
            errors.append(f"TAKE_PROFIT_PCT={cls.TAKE_PROFIT_PCT} must be > 0")

        if cls.STOP_LOSS_PCT <= 0:
            errors.append(f"STOP_LOSS_PCT={cls.STOP_LOSS_PCT} must be > 0")

        if cls.TAKE_PROFIT_PCT <= cls.STOP_LOSS_PCT:
            errors.append(
                f"TAKE_PROFIT_PCT ({cls.TAKE_PROFIT_PCT}%) must be greater than "
                f"STOP_LOSS_PCT ({cls.STOP_LOSS_PCT}%) — negative risk/reward will "
                "guarantee long-term losses even with a high win rate"
            )

        if cls.MAX_SESSION_LOSS_PCT <= 0 or cls.MAX_SESSION_LOSS_PCT > 100:
            errors.append(
                f"MAX_SESSION_LOSS_PCT={cls.MAX_SESSION_LOSS_PCT} must be between 0 and 100"
            )

        if cls.TRADE_SIZE_USDT <= 0:
            errors.append(f"TRADE_SIZE_USDT={cls.TRADE_SIZE_USDT} must be > 0")

        if not (0 <= cls.RISK_PER_TRADE_PCT <= 100):
            errors.append(
                f"RISK_PER_TRADE_PCT={cls.RISK_PER_TRADE_PCT} must be between 0 and 100"
            )

        if cls.RSI_OVERSOLD >= cls.RSI_OVERBOUGHT:
            errors.append(
                f"RSI_OVERSOLD ({cls.RSI_OVERSOLD}) must be < "
                f"RSI_OVERBOUGHT ({cls.RSI_OVERBOUGHT})"
            )

        if not (cls.RSI_OVERSOLD < cls.RSI_LONG_MAX <= cls.RSI_OVERBOUGHT):
            errors.append(
                f"RSI_LONG_MAX ({cls.RSI_LONG_MAX}) must be between "
                f"RSI_OVERSOLD ({cls.RSI_OVERSOLD}) and RSI_OVERBOUGHT ({cls.RSI_OVERBOUGHT})"
            )

        if not (cls.RSI_OVERSOLD <= cls.RSI_SHORT_MIN < cls.RSI_OVERBOUGHT):
            errors.append(
                f"RSI_SHORT_MIN ({cls.RSI_SHORT_MIN}) must be between "
                f"RSI_OVERSOLD ({cls.RSI_OVERSOLD}) and RSI_OVERBOUGHT ({cls.RSI_OVERBOUGHT})"
            )

        if cls.EMA_FAST >= cls.EMA_SLOW:
            errors.append(
                f"EMA_FAST ({cls.EMA_FAST}) must be less than EMA_SLOW ({cls.EMA_SLOW})"
            )

        supported_tf = ("1m", "3m", "5m", "15m", "30m", "1h")
        if cls.TIMEFRAME not in supported_tf:
            errors.append(
                f"TRADING_TIMEFRAME={cls.TIMEFRAME!r} is not supported. "
                f"Choose from: {', '.join(supported_tf)}"
            )

        if errors:
            raise ValueError(
                "Configuration errors found — fix your .env file before starting:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )
