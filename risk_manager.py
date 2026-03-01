"""
Session-level risk manager.

Responsibilities:
  1. Record starting balance at the beginning of each trading session.
  2. Check after every trade / on every tick whether the 30 % session loss
     threshold has been breached.
  3. Gate each new trade: refuse entry if the session is already locked out.
  4. Track open position status vs. its individual stop-loss / take-profit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from logger import get_logger

log = get_logger(__name__)


@dataclass
class TradeRecord:
    """Immutable snapshot of a completed trade."""
    trade_id:    str
    side:        str    # 'long' | 'short'
    entry_price: float
    exit_price:  float
    contracts:   float
    pnl_usdt:    float
    reason:      str    # 'take_profit' | 'stop_loss' | 'manual' | 'session_close' | 'exchange_closed'
    opened_at:   float = field(default_factory=time.time)
    closed_at:   float = field(default_factory=time.time)


@dataclass
class OpenTrade:
    """Tracks a currently open position."""
    trade_id:    str
    side:        str
    entry_price: float
    contracts:   float
    take_profit: float
    stop_loss:   float
    opened_at:   float = field(default_factory=time.time)
    sl_order_id: str   = ""   # exchange-native SL order ID (empty = software-only)
    tp_order_id: str   = ""   # exchange-native TP order ID (empty = software-only)


class RiskManager:
    """
    Enforces risk rules at the session level.

    A *session* starts when the bot is launched and ends either when:
      - The max-loss threshold is hit, OR
      - The bot is stopped manually.

    Max session loss = 30 % (configurable) of the balance at session start.
    """

    def __init__(self, starting_balance: float) -> None:
        self.session_start_balance: float = starting_balance
        self.max_loss_pct: float = Config.MAX_SESSION_LOSS_PCT
        self.max_loss_usdt: float = starting_balance * (self.max_loss_pct / 100)

        self._session_locked: bool = False
        self._realised_pnl: float = 0.0          # sum of closed-trade PnL
        self._trade_history: list[TradeRecord] = []
        self._open_trade: Optional[OpenTrade] = None
        self._last_trade_at: float = 0.0

        log.info(
            "RiskManager initialised | start_balance=%.2f USDT | "
            "max_loss=%.1f%% (%.2f USDT)",
            starting_balance, self.max_loss_pct, self.max_loss_usdt,
        )

    # ── Session state ─────────────────────────────────────────────────────────

    @property
    def session_locked(self) -> bool:
        return self._session_locked

    @property
    def session_loss_usdt(self) -> float:
        """Total realised loss for this session (always ≥ 0)."""
        return max(0.0, -self._realised_pnl)

    @property
    def session_loss_pct(self) -> float:
        if self.session_start_balance == 0:
            return 0.0
        return (self.session_loss_usdt / self.session_start_balance) * 100

    @property
    def remaining_risk_usdt(self) -> float:
        return max(0.0, self.max_loss_usdt - self.session_loss_usdt)

    @property
    def open_trade(self) -> Optional[OpenTrade]:
        return self._open_trade

    # ── Trade lifecycle ───────────────────────────────────────────────────────

    def can_open_trade(self) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Checks:
          - Session not locked due to max loss
          - No open position already
          - Cooldown period respected
        """
        if self._session_locked:
            return False, (
                f"Session locked — cumulative loss {self.session_loss_pct:.1f}% "
                f"exceeds {self.max_loss_pct}%"
            )

        if self._open_trade is not None:
            return False, "Position already open"

        elapsed = time.time() - self._last_trade_at
        if elapsed < Config.TRADE_COOLDOWN_SECONDS:
            remaining = Config.TRADE_COOLDOWN_SECONDS - elapsed
            return False, f"Cooldown active ({remaining:.0f}s remaining)"

        return True, "OK"

    def register_open(self, trade: OpenTrade) -> None:
        """Call immediately after an order is filled."""
        if self._open_trade is not None:
            log.error(
                "register_open called while trade %s is already open — ignoring new trade %s",
                self._open_trade.trade_id, trade.trade_id,
            )
            return
        self._open_trade = trade
        log.info(
            "Trade opened | id=%s side=%s entry=%.4f contracts=%.6f TP=%.4f SL=%.4f",
            trade.trade_id, trade.side, trade.entry_price,
            trade.contracts, trade.take_profit, trade.stop_loss,
        )

    def set_order_ids(self, sl_order_id: str, tp_order_id: str) -> None:
        """
        Store exchange-native order IDs after placing SL/TP orders.
        Call this immediately after register_open() when native orders succeed.
        """
        if self._open_trade is None:
            log.warning("set_order_ids called with no open trade — ignoring")
            return
        self._open_trade.sl_order_id = sl_order_id
        self._open_trade.tp_order_id = tp_order_id
        log.info(
            "Native orders registered | SL_id=%s TP_id=%s",
            sl_order_id or "none", tp_order_id or "none",
        )

    def register_close(self, exit_price: float, reason: str) -> Optional[TradeRecord]:
        """
        Call when a position is closed.
        Returns the completed TradeRecord, or None if no trade was open.
        """
        if self._open_trade is None:
            log.warning("register_close called with no open trade")
            return None

        t = self._open_trade
        if t.side == "long":
            pnl = (exit_price - t.entry_price) * t.contracts
        else:
            pnl = (t.entry_price - exit_price) * t.contracts

        # Fee: 0.075% taker per side, applied to both entry and exit notional
        fee = (t.entry_price + exit_price) * t.contracts * 0.00075
        pnl -= fee

        record = TradeRecord(
            trade_id=t.trade_id,
            side=t.side,
            entry_price=t.entry_price,
            exit_price=exit_price,
            contracts=t.contracts,
            pnl_usdt=pnl,
            reason=reason,
            opened_at=t.opened_at,
        )

        self._realised_pnl += pnl
        self._trade_history.append(record)
        self._open_trade = None
        self._last_trade_at = time.time()

        log.info(
            "Trade closed | id=%s reason=%s exit=%.4f pnl=%.4f USDT | "
            "session_pnl=%.4f USDT (%.2f%%)",
            record.trade_id, reason, exit_price, pnl,
            self._realised_pnl, self.session_loss_pct,
        )

        self._check_session_limit()
        return record

    # ── Per-tick checks ───────────────────────────────────────────────────────

    def check_exit_conditions(self, current_price: float) -> Optional[str]:
        """
        Inspect the open trade against current price.
        Returns exit reason string ('take_profit' | 'stop_loss') or None.
        This acts as a software safety net alongside exchange-native orders.
        """
        if self._open_trade is None:
            return None

        t = self._open_trade
        if t.side == "long":
            if current_price >= t.take_profit:
                return "take_profit"
            if current_price <= t.stop_loss:
                return "stop_loss"
        else:
            if current_price <= t.take_profit:
                return "take_profit"
            if current_price >= t.stop_loss:
                return "stop_loss"

        return None

    def update_current_balance(self, current_balance: float) -> None:
        """
        Call periodically with the live account balance to catch unrealised
        losses that exceed the session limit (e.g. during a flash crash).
        """
        unrealised_loss = max(0.0, self.session_start_balance - current_balance)
        total_loss_pct = (
            (unrealised_loss / self.session_start_balance) * 100
            if self.session_start_balance else 0
        )

        if total_loss_pct >= self.max_loss_pct:
            log.warning(
                "Live balance check: total loss %.2f%% >= max %.2f%% — locking session",
                total_loss_pct, self.max_loss_pct,
            )
            self._session_locked = True

    # ── Reporting ─────────────────────────────────────────────────────────────

    def session_summary(self) -> dict:
        wins   = [t for t in self._trade_history if t.pnl_usdt > 0]
        losses = [t for t in self._trade_history if t.pnl_usdt <= 0]
        return {
            "start_balance":    self.session_start_balance,
            "total_trades":     len(self._trade_history),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate_pct":     (len(wins) / len(self._trade_history) * 100)
                                if self._trade_history else 0,
            "realised_pnl":     round(self._realised_pnl, 4),
            "session_loss_pct": round(self.session_loss_pct, 2),
            "session_locked":   self._session_locked,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _check_session_limit(self) -> None:
        if self.session_loss_pct >= self.max_loss_pct:
            self._session_locked = True
            log.error(
                "SESSION LIMIT REACHED — loss %.2f%% >= %.2f%%. "
                "No further trades will be opened this session.",
                self.session_loss_pct, self.max_loss_pct,
            )
