# DCA_auto

Binance Futures bidirectional DCA automated trading bot.

Operates Long/Short simultaneously, performing automatic entry/DCA/TP/SL based on GA-optimized parameters.

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Initialize config files
mv config/.env.example config/.env
mv config/config.example.json config/config.json
mv config/optimize_config.example.json config/optimize_config.json

# 3. Enter API keys in config/.env, then edit config/config.json / config/optimize_config.json

# 4. GA optimization -> generate parameters
python3 main_optimize.py

# 5. Start trading
python3 main_trading.py --testnet
```

## Project Structure

```
DCA_auto/
├── main_trading.py              # Live trading entry point (PM2 driven)
├── main_optimize.py             # GA optimization execution
├── main_monitor.py              # Standalone monitoring (terminal status dashboard)
├── main_web_monitor.py          # Web-based monitoring (browser dashboard)
├── ecosystem.config.js          # PM2 run config (venv interpreter specified)
├── requirements.txt
│
├── config/                      # Config files
│   ├── .env.example             # API key template
│   ├── config.example.json      # Trading config template
│   └── optimize_config.example.json # Optimization config template
│
├── src/
│   ├── common/
│   │   ├── api_client.py        # Binance Futures REST API (binance-futures-connector)
│   │   ├── trading_config.py    # config.json loader
│   │   ├── config_loader.py     # Optimization parameter loader (data/params/)
│   │   ├── data_manager.py      # OHLCV data fetcher
│   │   └── logger.py            # Logging setup
│   │
│   ├── trading/
│   │   ├── executor.py          # Main trading executor (SymbolTrader + TradingExecutor)
│   │   ├── strategy.py          # DCA strategy calculations (entry, DCA levels, TP/SL prices)
│   │   ├── price_feed.py        # Mark Price WebSocket + User Data Stream
│   │   ├── margin_manager.py    # Per-coin margin persistence (power recovery)
│   │   ├── state_manager.py     # Position state save/restore
│   │   └── status_display.py    # Terminal status dashboard (ANSI in-place update)
│   │
│   └── optimization/
│       ├── ga_engine.py         # GA optimization engine
│       └── backtester.py        # Backtest engine
│
└── data/                        # Not tracked by git
    ├── params/                  # GA optimization results (BTC_USDT.json, ...)
    ├── active_params/           # Currently active parameters (for power recovery)
    ├── margins/                 # Per-coin margin state files
    ├── state/                   # Position state files
    └── logs/                    # All log files
        └── trades/              # Trade JSONL logs
```

## Trading Logic

### Bidirectional DCA

Runs Long and Short positions **simultaneously** for each coin.

1. **Base Order** (MARKET) — immediate fill for fast entry
2. **DCA Orders** (LIMIT) — placed at geometric intervals from base fill price
3. **Take Profit** (LIMIT, reduceOnly) — based on average price, precise fill with no slippage
4. **Stop Loss** (STOP_MARKET, Mark Price trigger) — market close when mark price hits SL

### DCA Level Calculation

Uses `price_deviation`, `dev_multiplier`, and `vol_multiplier` determined by GA optimization.

- Each level's trigger price widens by `dev_multiplier` from the previous level
- Each level's margin increases by `vol_multiplier` from the previous level
- On DCA fill: recalculate average price and reposition SL/TP

### Order Fill Detection

- **User Data Stream** (WebSocket) — real-time detection via `ORDER_TRADE_UPDATE` events
- **Polling backup** — every 5 minutes, cross-check open orders to catch missed events
- Listen key auto-renewed every 30 minutes

### Position Lifecycle

```
Entry (MARKET) -> Place DCA (LIMIT) + SL (STOP_MARKET) + TP (LIMIT)
  │
  ├── DCA fill  -> recalculate avg -> reposition SL/TP
  ├── TP fill   -> reset state -> update margin -> re-enter immediately
  └── SL fill   -> reset state -> update margin -> wait cooldown
```

## Capital Management

### Capital Allocation (config/config.json)

On startup, the bot fetches the Binance total wallet balance (`totalWalletBalance`) and allocates capital per coin using `weight` ratios.

```json
{
  "cooldown_hours": 6,
  "symbols": {
    "BTC_USDT": { "weight": 0.40 },
    "ETH_USDT": { "weight": 0.35 },
    "SOL_USDT": { "weight": 0.25 }
  }
}
```

- weight sum < 1.0: reserve buffer (room for more coins)
- weight sum = 1.0: fully allocated
- weight sum > 1.0: over-leveraged (allowed)
- **No auto-normalization** — used as-is from config

### Cross Margin Mode

All positions run in Cross margin mode. Wallet balance is shared, so individual positions cannot be liquidated in isolation.

Binance Futures supports only one leverage per symbol, so Long and Short use the same leverage.

### Margin Persistence (MarginManager)

Each coin's allocated capital is saved to `data/margins/{SYMBOL}_margin.json`.

- **Power recovery**: on restart, reads margin file to restore previous capital
- **Update condition**: only updated when **both** Long and Short for that coin are inactive (after TP/SL)
- **Drawdown protection**: retains existing margin even if balance drops (recovery within cycle is expected for DCA)
- **Growth reflected**: new capital applied when balance increases

## Config Files

### config/.env — API Keys

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
USE_TESTNET=true
```

### config/config.json — Trading Config

Coins to trade and capital weights. Symbol names must match filenames in `data/params/`.

### config/optimize_config.json — Optimization Config

```json
{
  "tickers": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
  "simulation": {
    "timeframe": "1m",
    "data_years": 5,
    "initial_capital": 1000.0,
    "fixed_leverage": 25,
    "fixed_base_margin": 5.0,
    "fixed_dca_margin": 5.0,
    "fee_rate": 0.0005,
    "slip_rate": 0.0003,
    "cooldown_hours": 6,
    "sharpe_days": 14
  },
  "ga": {
    "min_pop_size": 2000,
    "max_pop_size": 5000,
    "elite_ratio": 0.05,
    "max_generations": 5000,
    "max_patience_limit": 50
  }
}
```

## Running

```bash
# Testnet
python3 main_trading.py --testnet

# Mainnet
python3 main_trading.py --mainnet

# Custom config file
python3 main_trading.py --config config/my_config.json
```

### Always-on with PM2

Uses `ecosystem.config.js` in the project root. Specifies the venv `.venv` Python as the interpreter for correct package resolution.

```bash
pm2 start ecosystem.config.js
pm2 save
```

To run on testnet, change `args` in `ecosystem.config.js` to `"--testnet"`.

### Monitoring

While the bot runs in the background via PM2, you can check live status in a separate terminal.

```bash
# Default (5s interval, uses config/config.json)
python3 main_monitor.py

# Custom refresh interval
python3 main_monitor.py --interval 10

# Mainnet display
python3 main_monitor.py --mainnet
```

`main_monitor.py` runs completely independently from the bot process. Reads `data/state/`, `data/params/`, and `data/margins/` files to display the following dashboard in the terminal:

```
╔════════════════════════════════════════════════════════════════╗
║  DCA Trading Bot       TESTNET     Uptime: 02:34              ║
╠════════════════════════════════════════════════════════════════╣
║  BTC/USDT   $97,231.50                     Capital: $400.00   ║
║    LONG  ▲  0.0041 @ 97,100.20  DCA 1/5  TP 97,850.0         ║
║    SHORT ▼  0.0040 @ 97,350.00  DCA 0/5  TP 96,800.0         ║
║                                                                ║
║  ETH/USDT   $3,412.80                      Capital: $350.00   ║
║    LONG  ▲  0.1050 @ 3,400.50   DCA 0/5  TP 3,440.0          ║
║    SHORT ▼  ── waiting (cooldown 2:15) ──                      ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  Capital: $750.00  |  Active: 3/4  |  14:32:01                ║
╚════════════════════════════════════════════════════════════════╝
```

### Web Monitoring

Browser-based dashboard for a more convenient view. Uses only Python standard library, no extra dependencies.

```bash
# Default (localhost:8080, auto-opens browser)
python3 main_web_monitor.py

# Custom port
python3 main_web_monitor.py --port 3000

# Mainnet mode
python3 main_web_monitor.py --mainnet

# Disable auto browser open
python3 main_web_monitor.py --no-open
```

## Tech Stack

- **Binance API**: `binance-futures-connector` (REST: `UMFutures`, WebSocket: `UMFuturesWebsocketClient`)
- **Optimization**: Genetic Algorithm (NumPy + Numba JIT)
- **Backtest**: Numba-accelerated vectorized computation
