# Trading Web UI Startup Guide

This directory contains scripts and documentation for starting the Trading Web UI in both development and production environments.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Available Files](#available-files)
- [Development Mode](#development-mode)
- [Production Mode](#production-mode)
- [System Service Setup (Linux/Raspberry Pi)](#system-service-setup-linuxraspberry-pi)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### First Time Setup
```bash
# Initialize database with default users (run once)
python bin/web_ui/init_webui_database.py
```

### Development (Windows)
```batch
# From project root
bin\web_ui\start_webui_dev.bat
```

### Development (Linux/macOS)
```bash
# From project root
python src/web_ui/run_web_ui.py --dev
```

### Production (Linux/Raspberry Pi)
```bash
# From project root
./bin/web_ui/start_webui_prod.sh
```

## 📁 Available Files

This directory contains the following files:

### Core Startup Scripts
- **`start_webui_dev.bat`** - **RECOMMENDED** Windows development startup script
- **`start_webui_dev_simple.bat`** - Windows development startup script (simple, no colors)
- **`start_backend_only.bat`** - Windows backend only (for manual two-step startup)
- **`start_frontend_only.bat`** - Windows frontend only (for manual two-step startup)
- **`start_webui_prod.sh`** - Linux/Raspberry Pi production startup script
- **`trading-webui.service`** - Systemd service template file

### Database & Setup Scripts
- **`init_webui_database.py`** - Initialize database with default users (run once)

### Debugging Scripts
- **`test_environment.bat`** - Windows environment test script for troubleshooting
- **`test_frontend.bat`** - Test frontend build and dependencies
- **`test_imports.bat`** - Test Python imports to identify import issues

### Documentation
- **`README.md`** - This comprehensive guide

### Trading System Setup (Optional)
- **`setup_enhanced_trading.py`** - Sets up the enhanced trading system
- **`start_enhanced_trading.py`** - Interactive menu for trading system management

> **Note:** The trading system setup files are optional and used for configuring the broader trading system. The web UI can run independently using the main startup scripts.

## 🔧 Development Mode

Development mode provides:
- ✅ Auto-reload for both backend and frontend
- ✅ Hot module replacement (HMR)
- ✅ Source maps for debugging
- ✅ Detailed error messages
- ✅ Development server with proxy

### Windows Development

**Scripts:** 

```batch
# RECOMMENDED: Main development script
bin\web_ui\start_webui_dev.bat

# Alternative: Simple version (no colors)
bin\web_ui\start_webui_dev_simple.bat

# Manual two-step approach (if main runner fails):
# Step 1: Start backend in one terminal
bin\web_ui\start_backend_only.bat

# Step 2: Start frontend in another terminal  
bin\web_ui\start_frontend_only.bat

# If having issues, run the diagnostic tests
bin\web_ui\test_environment.bat     # General environment test
bin\web_ui\test_frontend.bat        # Frontend build test
bin\web_ui\test_imports.bat         # Python imports test
```

**What it does:**
1. Checks for virtual environment and dependencies
2. Installs frontend dependencies if needed
3. Starts backend API with auto-reload
4. Starts frontend dev server with HMR
5. Configures proxy for seamless API calls

### Linux/macOS Development

**Script:** `src/web_ui/run_web_ui.py`

```bash
# Basic development mode
python src/web_ui/run_web_ui.py --dev

# Custom host and port
python src/web_ui/run_web_ui.py --dev --host 0.0.0.0 --port 8000

# Help
python src/web_ui/run_web_ui.py --help
```

### Development URLs

- **Frontend UI:** http://localhost:5002
- **Backend API:** http://localhost:5003
- **API Documentation:** http://localhost:5003/docs
- **API Redoc:** http://localhost:5003/redoc

### Default Login Credentials

After running `init_webui_database.py`, you can log in with:

**Default Users:**
- **admin** / **admin** (Administrator)
- **trader** / **trader** (Trader)  
- **viewer** / **viewer** (Viewer)

**Existing Users (if any):**
- Users can log in with their username (part before @) and the same as password
- Example: `al.sa` / `al.sa` or `akossyrev` / `akossyrev`

## 🚀 Production Mode

Production mode provides:
- ✅ Optimized frontend build
- ✅ Static file serving
- ✅ Production logging
- ✅ Error handling
- ✅ Process management

### Linux/Raspberry Pi Production

**Script:** `bin/web_ui/start_webui_prod.sh`

```bash
# Make script executable (first time only)
chmod +x bin/web_ui/start_webui_prod.sh

# Basic usage
./bin/web_ui/start_webui_prod.sh

# Custom host and port
./bin/web_ui/start_webui_prod.sh --host 0.0.0.0 --port 8080

# Using environment variables
WEBUI_HOST=0.0.0.0 WEBUI_PORT=8080 ./bin/web_ui/start_webui_prod.sh

# Help
./bin/web_ui/start_webui_prod.sh --help
```

**What it does:**
1. Validates environment and dependencies
2. Builds frontend for production (if needed)
3. Starts backend in production mode
4. Serves static frontend files
5. Handles graceful shutdown

### Manual Production Start

```bash
# From project root
python src/web_ui/run_web_ui.py --host 0.0.0.0 --port 5003
```

## 🔧 System Service Setup (Linux/Raspberry Pi)

For production deployment on Raspberry Pi or Linux servers, set up a systemd service:

### 1. Create Service File

You can use the provided template or create your own:

```bash
# Option 1: Copy the provided template
sudo cp bin/web_ui/trading-webui.service /etc/systemd/system/

# Option 2: Create manually
sudo nano /etc/systemd/system/trading-webui.service
```

**Important:** Edit the service file to match your system paths:
- Replace `/home/pi/trading-system` with your actual project path
- Adjust `User` and `Group` if not using `pi` user
- Modify ports if using different configuration

### 2. Service File Content (Template)

```ini
[Unit]
Description=Trading Web UI Service
Documentation=https://github.com/your-repo/trading-system
After=network.target network-online.target
Wants=network-online.target
StartLimitIntervalSec=500
StartLimitBurst=5

[Service]
Type=exec
User=pi
Group=pi
WorkingDirectory=/home/pi/trading-system
Environment=PYTHONPATH=/home/pi/trading-system
Environment=PYTHONUNBUFFERED=1
Environment=WEBUI_HOST=0.0.0.0
Environment=WEBUI_PORT=5003

# Use the production startup script
ExecStart=/home/pi/trading-system/bin/web_ui/start_webui_prod.sh --host 0.0.0.0 --port 5003

# Restart policy
Restart=always
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=30

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/pi/trading-system/logs
ReadWritePaths=/home/pi/trading-system/db

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-webui

[Install]
WantedBy=multi-user.target
```

### 3. Install and Start Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable trading-webui.service

# Start the service
sudo systemctl start trading-webui.service

# Check service status
sudo systemctl status trading-webui.service

# View logs
sudo journalctl -u trading-webui.service -f

# Stop the service
sudo systemctl stop trading-webui.service

# Restart the service
sudo systemctl restart trading-webui.service
```

### 4. Service Management Commands

```bash
# Check if service is running
sudo systemctl is-active trading-webui.service

# Check if service is enabled
sudo systemctl is-enabled trading-webui.service

# View recent logs
sudo journalctl -u trading-webui.service --since "1 hour ago"

# View logs with follow
sudo journalctl -u trading-webui.service -f

# Reload service configuration (after editing service file)
sudo systemctl daemon-reload
sudo systemctl restart trading-webui.service
```

## 🗄️ Database Setup

### First Time Setup

Before using the web UI, you need to initialize the database:

```bash
# From project root
python bin/web_ui/init_webui_database.py
```

**What this does:**
- Creates database tables if they don't exist
- Creates default users (admin, trader, viewer) if no users exist
- Sets up proper foreign key relationships for audit logging

**Default Users Created:**
- `admin@trading-system.local` (admin role)
- `trader@trading-system.local` (trader role)  
- `viewer@trading-system.local` (viewer role)

**Login Credentials:**
- Username: `admin`, Password: `admin`
- Username: `trader`, Password: `trader`
- Username: `viewer`, Password: `viewer`

### Existing Users

If you already have users in the database, they can log in using:
- Username: The part before @ in their email
- Password: Same as username (temporary simple authentication)

Example: User with email `john.doe@example.com` logs in with `john.doe` / `john.doe`

## ⚙️ Configuration

### Port Configuration

The system uses these default ports (defined in `config/donotshare/donotshare.py`):

- **TRADING_API_PORT:** 5003 (Backend API)
- **TRADING_WEBGUI_PORT:** 5002 (Frontend UI)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WEBUI_HOST` | Host to bind to | `0.0.0.0` |
| `WEBUI_PORT` | Backend API port | `5003` |
| `PYTHONPATH` | Python path | Project root |
| `PYTHONUNBUFFERED` | Unbuffered Python output | `1` |

### Frontend Configuration

Frontend configuration is in `src/web_ui/frontend/vite.config.ts`:

```typescript
export default defineConfig({
  server: {
    port: 5002, // TRADING_WEBGUI_PORT
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:5003', // TRADING_API_PORT
        changeOrigin: true,
      },
      '/auth': {
        target: 'http://localhost:5003',
        changeOrigin: true,
      }
    }
  }
})
```

## 📋 Prerequisites

### Development Prerequisites

**Python:**
- Python 3.8+
- Virtual environment (`.venv`)
- Required Python packages (see `requirements.txt`)

**Node.js:**
- Node.js 18.0.0+
- npm (comes with Node.js)

**System:**
- Git (for development)
- Modern web browser

### Production Prerequisites (Raspberry Pi)

**System packages:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python development tools
sudo apt-get install -y python3-dev python3-venv python3-pip

# Install system dependencies
sudo apt-get install -y build-essential git
```

**Python environment:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
sudo netstat -tlnp | grep :5003

# Kill process using the port
sudo kill -9 <PID>

# Or use a different port
python src/web_ui/run_web_ui.py --port 8000
```

#### 2. Virtual Environment Not Found
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 3. Node.js/npm Not Found
```bash
# Install Node.js on Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

#### 4. Frontend Build Fails
```bash
# Clean and reinstall frontend dependencies
cd src/web_ui/frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

#### 5. Database Issues
```bash
# Check if database file exists
ls -la db/trading.db

# Check database permissions
chmod 664 db/trading.db

# View application logs
tail -f logs/web_ui/app.log
```

#### 6. Permission Denied (Linux)
```bash
# Make scripts executable
chmod +x bin/web_ui/start_webui_prod.sh

# Fix file ownership
sudo chown -R $USER:$USER /path/to/project

# Fix directory permissions
chmod -R 755 /path/to/project
```

#### 7. Windows Batch Script Issues
```batch
# If the batch script exits immediately, run diagnostic tests in order:
bin\web_ui\test_environment.bat     # Check overall environment
bin\web_ui\test_frontend.bat        # Test frontend build
bin\web_ui\test_imports.bat         # Test Python imports

# If colors don't display properly, use the simple version
bin\web_ui\start_webui_dev_simple.bat

# If you get "command not found" errors, check your PATH
echo %PATH%

# Make sure you're running from the project root directory
cd /d "C:\path\to\your\project"
bin\web_ui\start_webui_dev_simple.bat

# Common issues and solutions:
# - Config import fails: Check if config/donotshare/donotshare.py exists
# - Database import fails: Check if database models are properly installed
# - FastAPI import fails: Run "pip install fastapi uvicorn" in your venv
# - Authentication fails: Run "python bin/web_ui/init_webui_database.py"
```

#### 8. Authentication Issues
```bash
# If you can't log in, initialize the database
python bin/web_ui/init_webui_database.py

# Check if users exist
python -c "
from src.data.db.services.database_service import get_database_service
from src.data.db.models.model_users import User
db = get_database_service()
with db.uow() as r:
    users = r.s.query(User).all()
    for u in users: print(f'ID: {u.id}, Email: {u.email}, Role: {u.role}')
"

# Try logging in with default credentials:
# admin/admin, trader/trader, viewer/viewer
```

### Service Troubleshooting

#### Check Service Status
```bash
# Detailed service status
sudo systemctl status trading-webui.service -l

# View service logs
sudo journalctl -u trading-webui.service --no-pager

# View recent logs
sudo journalctl -u trading-webui.service --since "10 minutes ago"
```

#### Service Won't Start
```bash
# Check service file syntax
sudo systemd-analyze verify /etc/systemd/system/trading-webui.service

# Check if paths exist
ls -la /home/pi/trading-system/bin/web_ui/start_webui_prod.sh

# Check permissions
sudo -u pi /home/pi/trading-system/bin/web_ui/start_webui_prod.sh --help
```

### Log Locations

- **Application logs:** `logs/web_ui/`
- **System service logs:** `sudo journalctl -u trading-webui.service`
- **Frontend build logs:** `src/web_ui/frontend/dist/`

### Getting Help

1. **Check logs** first for error messages
2. **Verify prerequisites** are installed
3. **Test manually** before using service
4. **Check port availability**
5. **Verify file permissions**

## 📚 Additional Resources

- **API Documentation:** http://localhost:5003/docs (when running)
- **Frontend Source:** `src/web_ui/frontend/`
- **Backend Source:** `src/web_ui/backend/`
- **Configuration:** `config/donotshare/donotshare.py`
- **Main Runner:** `src/web_ui/run_web_ui.py`

## 🔄 Development Workflow

1. **Initialize database (first time only):**
   ```bash
   python bin/web_ui/init_webui_database.py
   ```

2. **Start development environment:**
   ```bash
   # Windows
   bin\web_ui\start_webui_dev.bat
   
   # Linux/macOS
   python src/web_ui/run_web_ui.py --dev
   ```

3. **Make changes to code** (auto-reload will handle updates)

4. **Test changes** in browser at http://localhost:5002

5. **Build for production:**
   ```bash
   cd src/web_ui/frontend
   npm run build
   ```

6. **Test production build:**
   ```bash
   python src/web_ui/run_web_ui.py --host 0.0.0.0 --port 5003
   ```

7. **Deploy to production** using systemd service

---

**Happy Trading! 🚀📈**