# Environment Setup Guide

This guide explains how to set up environment variables for the trading platform.

## API Keys and Sensitive Credentials

For security reasons, API keys and other sensitive credentials are not stored in configuration files. Instead, they are loaded from environment variables through the `config/donotshare/donotshare.py` file.

## Required Environment Variables

### Broker API Keys

#### Binance
```bash
# For live trading
export BINANCE_KEY="your_binance_api_key"
export BINANCE_SECRET="your_binance_api_secret"

# For paper trading (optional, can use same keys)
export BINANCE_PAPER_KEY="your_binance_paper_api_key"
export BINANCE_PAPER_SECRET="your_binance_paper_api_secret"
```

#### IBKR (Interactive Brokers)
```bash
export IBKR_HOST="127.0.0.1"
export IBKR_PORT="7497"  # 7497 for TWS, 4001 for IB Gateway
export IBKR_PAPER_PORT="4002"  # For paper trading
export IBKR_CLIENT_ID="1"
export IBKR_KEY="your_ibkr_key"
export IBKR_SECRET="your_ibkr_secret"
export IBKR_PAPER_KEY="your_ibkr_paper_key"
export IBKR_PAPER_SECRET="your_ibkr_paper_secret"
```

### Notification Credentials

#### Telegram
```bash
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

#### Email (Optional)
```bash
export gmail_username="your_email@gmail.com"
export gmail_password="your_app_password"
export SENDGRID_API_KEY="your_sendgrid_api_key"
```

### Web Interface Credentials

#### Web GUI
```bash
export WEBGUI_PORT="5000"
export WEBGUI_LOGIN="admin"
export WEBGUI_PASSWORD="your_password"
```

#### API
```bash
export API_PORT="8000"
export API_LOGIN="api_user"
export API_PASSWORD="your_api_password"
```

#### Admin Access
```bash
export ADMIN_USERNAME="admin"
export ADMIN_PASSWORD="your_admin_password"
```

## Setting Up Environment Variables

### Windows (PowerShell)
```powershell
# Set environment variables for current session
$env:BINANCE_KEY="your_api_key"
$env:BINANCE_SECRET="your_api_secret"

# Set environment variables permanently
[Environment]::SetEnvironmentVariable("BINANCE_KEY", "your_api_key", "User")
[Environment]::SetEnvironmentVariable("BINANCE_SECRET", "your_api_secret", "User")
```

### Windows (Command Prompt)
```cmd
# Set environment variables for current session
set BINANCE_KEY=your_api_key
set BINANCE_SECRET=your_api_secret

# Set environment variables permanently
setx BINANCE_KEY "your_api_key"
setx BINANCE_SECRET "your_api_secret"
```

### Linux/macOS
```bash
# Set environment variables for current session
export BINANCE_KEY="your_api_key"
export BINANCE_SECRET="your_api_secret"

# Add to ~/.bashrc or ~/.zshrc for permanent setup
echo 'export BINANCE_KEY="your_api_key"' >> ~/.bashrc
echo 'export BINANCE_SECRET="your_api_secret"' >> ~/.bashrc
source ~/.bashrc
```

## Using .env Files (Alternative)

You can create a `.env` file in the `config/donotshare/` directory:

```bash
# config/donotshare/.env file
BINANCE_KEY=your_api_key
BINANCE_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

The system automatically loads this file through the `donotshare.py` module.

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use different API keys for paper trading and live trading**
3. **Set appropriate permissions on your API keys**
4. **Rotate API keys regularly**
5. **Use environment-specific configurations**
6. **Keep the `config/donotshare/` directory in `.gitignore**

## Broker-Specific Notes

### Binance
- Create API keys in your Binance account
- Set appropriate permissions (read info, spot trading)
- For paper trading, you can use the same keys or create separate ones
- Enable IP restrictions for additional security

### IBKR
- Requires TWS (Trader Workstation) or IB Gateway to be running
- Configure TWS to accept API connections
- Set appropriate permissions in TWS

### Telegram
- Create a bot using @BotFather
- Get your chat ID by messaging @userinfobot
- Add the bot to your chat/group

## Testing Environment Variables

You can test if your environment variables are set correctly:

```python
import os

# Test broker credentials
print(f"Binance API Key: {'SET' if os.getenv('BINANCE_KEY') else 'NOT SET'}")
print(f"Binance API Secret: {'SET' if os.getenv('BINANCE_SECRET') else 'NOT SET'}")

# Test notification credentials
print(f"Telegram Bot Token: {'SET' if os.getenv('TELEGRAM_BOT_TOKEN') else 'NOT SET'}")
print(f"Telegram Chat ID: {'SET' if os.getenv('TELEGRAM_CHAT_ID') else 'NOT SET'}")
```

## Troubleshooting

### Common Issues

1. **"API key not found" error**
   - Check if environment variable is set: `echo $BINANCE_KEY`
   - Restart your terminal/IDE after setting variables
   - Verify variable name spelling
   - Check if `.env` file is in the correct location

2. **"Invalid API key" error**
   - Verify API key is correct
   - Check if API key has appropriate permissions
   - Ensure IP restrictions allow your IP

3. **"Connection refused" (IBKR)**
   - Ensure TWS or IB Gateway is running
   - Check port number (7497 for TWS, 4001 for Gateway)
   - Verify API connections are enabled in TWS

### Debug Mode

Enable debug logging to see more details:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## File Structure

```
config/
├── donotshare/
│   ├── __init__.py
│   ├── donotshare.py      # Loads environment variables
│   └── .env              # Your environment variables (not in git)
└── trading/
    └── 0001.json         # Trading bot configuration (no sensitive data)
``` 