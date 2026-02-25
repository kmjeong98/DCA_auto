module.exports = {
  apps: [
    {
      name: "dca-bot",
      script: "main_trading.py",
      interpreter: ".venv/bin/python3",
      args: "--mainnet --confirm",
      cwd: __dirname,
    },
    {
      name: "dca-optimize",
      script: "main_optimize.py",
      interpreter: ".venv/bin/python3",
      cwd: __dirname,
      cron_restart: "0 9 * * 3,6",
      autorestart: false,
      watch: false,
    },
  ],
};
