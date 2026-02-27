# Phemex Scalp Trading Bot

A Python scalp trading bot for Phemex perpetual futures. Uses an EMA crossover + RSI strategy with a hard **30 % maximum session loss** guard.

---

## Features

| Feature | Detail |
|---|---|
| Exchange | Phemex (mainnet + testnet) via CCXT |
| Strategy | EMA 9/21 crossover + RSI 14 filter |
| Timeframe | Configurable (default 1 m) |
| Direction | Long & Short |
| Risk guard | 30 % max loss per session (configurable) |
| Per-trade risk | TP 0.6 % / SL 0.35 % (configurable) |
| Position sizing | USDT notional × leverage |
| Logging | Console + rotating file |

---

## Quick Start

### 1. Clone & install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your Phemex API key and secret
```

> **Start with `PHEMEX_TESTNET=true`** to verify behaviour before risking real funds.

### 3. Run the bot

```bash
python bot.py
```

Press **Ctrl+C** to stop gracefully. All open positions are closed automatically on shutdown.

---

## Configuration Reference

See `.env.example` for all options. Key parameters:

| Variable | Default | Description |
|---|---|---|
| `PHEMEX_API_KEY` | — | API key (required) |
| `PHEMEX_API_SECRET` | — | API secret (required) |
| `PHEMEX_TESTNET` | `true` | Use testnet |
| `TRADING_SYMBOL` | `BTC/USDT:USDT` | Perpetual pair |
| `TRADING_TIMEFRAME` | `1m` | Candle interval |
| `TRADE_SIZE_USDT` | `100` | Notional per trade (before leverage) |
| `LEVERAGE` | `5` | Futures leverage |
| `MAX_SESSION_LOSS_PCT` | `30` | Session hard-stop % |
| `TAKE_PROFIT_PCT` | `0.6` | TP per trade % |
| `STOP_LOSS_PCT` | `0.35` | SL per trade % |
| `TRADE_COOLDOWN_SECONDS` | `30` | Min gap between trades |

---

## Strategy Logic

```
Signal LONG  → EMA9 crosses above EMA21  AND  30 < RSI < 60
Signal SHORT → EMA9 crosses below EMA21  AND  40 < RSI < 70

No trade if session loss ≥ 30 % of starting balance.
```

---

## Risk Warnings

- **This bot trades real money when `PHEMEX_TESTNET=false`.** Use at your own risk.
- Past performance of any strategy does not guarantee future results.
- Always test on testnet first and start with small position sizes.
- Crypto markets are highly volatile; scalp bots can lose money quickly.

---

## Project Structure

```
bot.py           Main entry point & loop
config.py        Configuration loader
exchange.py      Phemex CCXT wrapper
strategy.py      EMA + RSI scalp strategy
risk_manager.py  Session loss guard
trader.py        Order execution bridge
logger.py        Logging setup
requirements.txt Python dependencies
.env.example     Environment variable template
```
