# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Binance Futures bidirectional DCA trading bot. Runs Long and Short positions simultaneously per coin with GA-optimized parameters. All positions use Hedge Mode (`positionSide=LONG/SHORT`) with Cross margin. Managed via PM2 for always-on operation.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# First time: copy example configs and fill in API keys
cp config/.env.example config/.env
cp config/config.example.json config/config.json
cp config/optimize_config.example.json config/optimize_config.json

# Trading
python3 main_trading.py --testnet              # testnet mode
python3 main_trading.py --mainnet --confirm     # mainnet (requires --confirm)
python3 main_trading.py --config config/my.json # custom config

# GA Optimization
python3 main_optimize.py                        # runs for all tickers in optimize_config.json
python3 main_optimize.py --tickers BTC/USDT ETH/USDT  # specific tickers only

# Monitoring
python3 main_web_monitor.py                     # localhost:8080, auto-opens browser
python3 main_web_monitor.py --mainnet --port 3000

# Daily Report
python3 main_daily_report.py                    # sends Telegram daily report
python3 main_daily_report.py --test             # dry-run to stdout
python3 main_daily_report.py --listen           # long-poll for /report Telegram command

# PM2 (production)
pm2 start ecosystem.config.js
pm2 logs dca-bot
pm2 restart dca-bot
```

## Architecture

Three PM2-managed processes (see `ecosystem.config.js`):
- **dca-bot** (`main_trading.py`) — 24/7 live trading
- **dca-optimize** (`main_optimize.py`) — cron-triggered GA optimization (Wed/Sat 9am, `autorestart: false`)
- **dca-report** (`main_daily_report.py`) — Telegram daily PnL report (long-poll mode)

Plus one standalone process (not in PM2):
- **web monitor** (`main_web_monitor.py`) — browser dashboard, run manually

### src/trading/ — Live Trading

**Class hierarchy:** `TradingExecutor` → creates one `SymbolTrader` per coin → each owns a stateless `DCAStrategy` (calculates levels) and mutable `PositionState` objects (long/short).

**Data flow:**
```
TradingExecutor (single-threaded main loop)
  ├── PriceFeed (WebSocket mark prices, background thread)
  │   └── broadcasts to all SymbolTraders via callback
  ├── OrderUpdateFeed (User Data Stream, background thread)
  │   └── detects fills → triggers state updates
  ├── SymbolTrader[N] (per coin, protected by _lock)
  │   ├── checks entry/DCA/TP/SL conditions
  │   ├── places orders via APIClient
  │   └── saves state to disk after each event
  └── periodic_sync (every 5 min) — reconciles local state with exchange
```

**Concurrency:** WebSocket streams run in background threads. `SymbolTrader._lock` protects `_current_price`. Main loop is single-threaded. Only one bot instance runs per PM2 process (no file locking needed).

**Error recovery:** Failed orders tracked in `_pending_sl`, `_pending_tp`, `_pending_dca` sets and retried every iteration. No exceptions thrown in trading loop — logged and retried. OrderUpdateFeed reconnects with 5s backoff. Listen key auto-renewed every 30 min.

**Sync/Reconciliation:** `_sync_with_exchange()` on startup loads exchange positions and matches to local state. `periodic_sync()` every 5 min detects state drift (e.g., TP/SL filled while offline). `_reconcile_orders()` classifies orphaned orders and re-places missing SL/TP.

**State persistence (saved after each event):**
- `data/state/{SYMBOL}_state.json` — serialized PositionState (`.to_dict()` / `.from_dict()`)
- `data/margins/{SYMBOL}_margin.json` — per-coin capital (updated only when both sides inactive)
- `data/active_params/{SYMBOL}.json` — currently loaded params for power recovery
- `data/logs/trades/{SYMBOL}_YYYYMM.jsonl` — append-only trade event log

**BnbManager** (`bnb_manager.py`): auto-maintains BNB balance for fee discount via Binance Convert API (mainnet only, failures silent).

### src/optimization/ — GA Engine

`ga_engine.py` runs genetic algorithm with Numba JIT (`run_dual_simulation()`). 12-parameter genome per strategy (6 long + 6 short: `price_dev, take_profit, max_dca, dev_mult, vol_mult, sl_ratio`). Fitness = Sharpe ratio across 4 overlapping window phases to reduce overfitting. Output: `data/params/{SYMBOL}.json`.

`backtester.py` provides vectorized backtest computation.

### src/common/ — Shared

- `api_client.py` — Binance Futures REST wrapper (`binance-futures-connector`'s `UMFutures`)
- `trading_config.py` — loads `config/config.json` (symbols, weights)
- `config_loader.py` — loads GA params from `data/params/`; falls back from USDC to USDT variant if missing
- `data_manager.py` — OHLCV fetcher via CCXT (`ccxt` installed separately, not in requirements.txt) with `data/ohlcv_cache/` caching
- `logger.py` — per-component logger factory; outputs to `data/logs/{component}.log`

### Web Monitor & Daily Report

**Web monitor** (`main_web_monitor.py`): standalone HTTP server, reads state/params/margin files from `data/`. Records balance snapshots to `data/balance_snapshots/balance_YYYYMM.jsonl` every 10 min for PnL calculation.

**Daily report** (`main_daily_report.py`): imports snapshot/balance/trade data from `main_web_monitor.py`, sends HTML via Telegram Bot API.

**StatusDisplay** (`status_display.py`): ANSI box format, TTY-aware (outputs nothing under PM2).

## Config Files

- `config/.env` — API keys (`BINANCE_API_KEY`, `BINANCE_API_SECRET`, `USE_TESTNET`), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `config/config.json` — trading symbols with weight-based capital allocation
- `config/optimize_config.json` — GA settings, simulation parameters, ticker list

## Key Conventions

- Symbol format: `BTC_USDT` in config/filenames, `BTCUSDT` for Binance API, `BTC/USDT` for CCXT
- All positions use Cross margin mode; one leverage setting per symbol (shared Long/Short)
- Margin only updates when both Long and Short for a coin are inactive
- DCA uses geometric level spacing: `price_deviation`, `dev_multiplier`, `vol_multiplier`
- Orders: Base=MARKET, DCA=LIMIT, TP=LIMIT(reduceOnly), SL=STOP_MARKET(mark price trigger)
- `data/` directory is not git-tracked (runtime state, logs, params)
- Config files (`config/.env`, `config/config.json`, `config/optimize_config.json`) are gitignored; only `.example` templates are tracked
- The `.venv` virtualenv is specified as the PM2 interpreter in `ecosystem.config.js`
- All comments and docstrings are in English
- Capital allocation: `wallet_balance × weight[coin]`; weight sum > 1.0 is allowed (over-leverage)
- DCAStrategy is stateless (pure calculator); SymbolTrader holds and mutates state

## Runtime Data Layout

```
data/
├── state/              # PositionState per symbol
├── margins/            # Capital allocations (power recovery)
├── active_params/      # Currently loaded GA params
├── params/             # GA optimization output
├── logs/
│   ├── *.log           # Component logs
│   └── trades/         # Trade JSONL (per symbol, per month)
├── ohlcv_cache/        # CCXT OHLCV cache
└── balance_snapshots/  # Balance history (monthly JSONL)
```
