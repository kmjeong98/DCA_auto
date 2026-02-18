module.exports = {
  apps: [
    {
      name: "dca-bot",
      script: "main_trading.py",
      interpreter: ".venv/bin/python3",
      args: "--mainnet",
      cwd: __dirname,
    },
  ],
};
