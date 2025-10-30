# Trading Bot Web UI Scripts

This directory contains scripts to run the trading web UI on different platforms.

## 🪟 Windows Development - `trading-bot-webui.bat`

**Purpose**: Start the trading web UI in development mode on Windows

### Usage:
```cmd
# Quick start (recommended)
bin\trading-bot-webui.bat

# Interactive menu
bin\trading-bot-webui.bat --menu

# Backend only
bin\trading-bot-webui.bat --backend

# Frontend only (run in separate window)
bin\trading-bot-webui.bat --frontend

# Setup only
bin\trading-bot-webui.bat --setup
```

### Features:
- ✅ Automatic dependency checking (Python, Node.js)
- ✅ Virtual environment setup
- ✅ Frontend dependency installation
- ✅ Configuration validation
- ✅ Development server with auto-reload
- ✅ Interactive menu system
- ✅ Color-coded output

### What it does:
1. Checks Python and Node.js installation
2. Creates/activates Python virtual environment
3. Installs Python dependencies from `requirements-webui.txt`
4. Installs frontend npm packages
5. Runs setup if configuration is missing
6. Starts backend server on port 8000
7. Provides instructions for starting frontend

### Access Points:
- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:5173 (separate window)
- **API Docs**: http://localhost:8000/docs

---

## 🐧 Ubuntu/Pi5 System Service - `svc-trading-bot-webui.sh`

**Purpose**: Install and manage the trading web UI as a system service on Ubuntu/Raspberry Pi 5

### Usage:
```bash
# Make executable first
chmod +x bin/svc-trading-bot-webui.sh

# Interactive menu
sudo ./bin/svc-trading-bot-webui.sh

# Direct commands
sudo ./bin/svc-trading-bot-webui.sh install
sudo ./bin/svc-trading-bot-webui.sh start
sudo ./bin/svc-trading-bot-webui.sh status
sudo ./bin/svc-trading-bot-webui.sh logs
```

### Features:
- ✅ Complete system service installation
- ✅ Nginx reverse proxy setup
- ✅ SSL/TLS ready configuration
- ✅ Automatic startup on boot
- ✅ Resource limits and security
- ✅ Log management
- ✅ Health monitoring
- ✅ Frontend build automation

### What it does:
1. **Installation**:
   - Creates service user (`pi`)
   - Installs Python/Node.js dependencies
   - Builds frontend for production
   - Creates systemd service file
   - Sets up Nginx reverse proxy
   - Configures security and resource limits

2. **Service Management**:
   - Start/stop/restart service
   - Enable/disable auto-start
   - View real-time logs
   - Monitor system resources
   - Update dependencies

### Service Details:
- **Service Name**: `trading-webui`
- **Service User**: `pi`
- **Web UI Port**: `8080` (internal)
- **Public Access**: Port `80` (via Nginx)
- **Logs**: `/var/log/trading-webui/`
- **Auto-start**: Enabled on boot

### Access Points:
- **Web UI**: http://your-pi-ip or http://localhost
- **Direct Backend**: http://your-pi-ip:8080 (if needed)

---

## 🚀 Quick Start Guide

### Windows Development:
```cmd
# 1. Open Command Prompt as Administrator (recommended)
# 2. Navigate to project directory
cd path\to\your\trading-project

# 3. Run the script
bin\trading-bot-webui.bat

# 4. In another Command Prompt window, start frontend:
bin\trading-bot-webui.bat --frontend
```

### Ubuntu/Pi5 Production:
```bash
# 1. SSH to your Pi/Ubuntu server
ssh pi@your-pi-ip

# 2. Navigate to project directory
cd /path/to/your/trading-project

# 3. Make script executable
chmod +x bin/svc-trading-bot-webui.sh

# 4. Install as system service
sudo ./bin/svc-trading-bot-webui.sh install

# 5. Start the service
sudo ./bin/svc-trading-bot-webui.sh start

# 6. Check status
sudo ./bin/svc-trading-bot-webui.sh status
```

---

## 🔧 Configuration

Both scripts will automatically:
- Create `.env` file from `.env.example` if missing
- Run `setup_enhanced_trading.py` if configuration is missing
- Install required dependencies
- Set up proper directory structure

### Required Files:
- `.env` - Environment variables (API keys, etc.)
- `config/enhanced_trading/raspberry_pi_multi_strategy.json` - Strategy configuration
- `requirements-webui.txt` - Python dependencies

---

## 📊 Monitoring and Logs

### Windows:
- Backend logs: Console output
- Frontend logs: Browser developer tools
- Configuration: `config/enhanced_trading/`

### Ubuntu/Pi5:
```bash
# View live logs
sudo ./bin/svc-trading-bot-webui.sh logs

# View recent logs
sudo ./bin/svc-trading-bot-webui.sh recent-logs

# System service status
sudo systemctl status trading-webui

# Nginx logs
sudo tail -f /var/log/trading-webui/nginx_access.log
sudo tail -f /var/log/trading-webui/nginx_error.log
```

---

## 🔐 Security

### Windows Development:
- Runs on localhost only
- Development mode (not for production)
- No authentication required for local development

### Ubuntu/Pi5 Production:
- Nginx reverse proxy with security headers
- Service runs as non-root user (`pi`)
- Resource limits and process isolation
- Systemd security features enabled
- Ready for SSL/TLS certificates

---

## 🛠️ Troubleshooting

### Common Issues:

1. **Python not found**:
   - Windows: Install from https://python.org
   - Ubuntu: `sudo apt install python3 python3-pip python3-venv`

2. **Node.js not found**:
   - Windows: Install from https://nodejs.org
   - Ubuntu: Script will auto-install

3. **Permission denied**:
   - Windows: Run Command Prompt as Administrator
   - Ubuntu: Use `sudo` for service operations

4. **Port already in use**:
   - Check if another service is using port 8000/8080
   - Kill existing processes or change port

5. **Frontend build fails**:
   - Check Node.js version (requires 16+)
   - Clear npm cache: `npm cache clean --force`
   - Delete `node_modules` and reinstall

### Getting Help:
```bash
# Windows
bin\trading-bot-webui.bat --help

# Ubuntu/Pi5
./bin/svc-trading-bot-webui.sh --help
```

---

## 📝 Default Login Credentials

For both platforms:
- **Username**: `admin` | **Password**: `admin`
- **Username**: `trader` | **Password**: `trader`

Change these in production by implementing proper authentication in the backend.