# Scheduler Service Setup on Raspberry Pi

## Current Issue
The scheduler service is failing to start because Python can't find the `src` module.

## Solution: Update the systemd service file

### Step 1: Test locally first (recommended)

Before fixing systemd, verify the scheduler works locally:

```bash
cd /opt/apps/e-trading
./bin/scheduler/test-scheduler.sh
```

If this works, proceed to Step 2. If not, there's a deeper issue with the installation.

---

### Step 2: Update the systemd service file

Edit the service file:

```bash
sudo nano /etc/systemd/system/scheduler.service
```

Make sure the **Environment variables** section looks exactly like this:

```ini
# Environment variables
Environment=PYTHONPATH=/opt/apps/e-trading
Environment=TRADING_ENV=production
Environment=LOG_LEVEL=INFO
Environment=SCHEDULER_MAX_WORKERS=10
Environment=DATA_CACHE_ENABLED=true
```

**IMPORTANT:** The `Environment=PYTHONPATH=/opt/apps/e-trading` line MUST be there!

Save and exit (Ctrl+X, then Y, then Enter).

---

### Step 3: Reload and restart the service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable scheduler.service

# Start the service
sudo systemctl start scheduler.service

# Check status
sudo systemctl status scheduler.service
```

---

### Step 4: View logs

```bash
# View recent logs
sudo journalctl -u scheduler.service -n 50

# Watch logs in real-time
sudo journalctl -u scheduler.service -f
```

---

## Alternative: Use the automated setup script

Copy the updated `scheduler.service` file to your Pi, then run:

```bash
cd /opt/apps/e-trading
sudo ./bin/scheduler/systemd-setup.sh /opt/apps/e-trading
```

This will:
1. Create the service file with correct paths
2. Enable the service
3. Start it
4. Show status and logs

---

## Troubleshooting

### If service still fails to start:

1. **Check the service file is correct:**
   ```bash
   cat /etc/systemd/system/scheduler.service | grep PYTHONPATH
   ```
   Should output: `Environment=PYTHONPATH=/opt/apps/e-trading`

2. **Check virtual environment exists:**
   ```bash
   ls -la /opt/apps/e-trading/.venv/bin/python
   ```

3. **Test Python imports manually:**
   ```bash
   cd /opt/apps/e-trading
   source .venv/bin/activate
   export PYTHONPATH=/opt/apps/e-trading
   python -c "from src.scheduler.main import main; print('Import successful!')"
   ```

4. **Check database connectivity:**
   ```bash
   cd /opt/apps/e-trading
   source .venv/bin/activate
   export PYTHONPATH=/opt/apps/e-trading
   python -c "from src.data.db.core.database import get_database_url; print(get_database_url())"
   ```

5. **View detailed error logs:**
   ```bash
   sudo journalctl -u scheduler.service --no-pager | tail -100
   ```

---

## Expected Output (Success)

When the service starts successfully, you should see:

```bash
$ sudo systemctl status scheduler.service
● scheduler.service - E-Trading Scheduler Service
     Loaded: loaded (/etc/systemd/system/scheduler.service; enabled; preset: enabled)
     Active: active (running) since Sun 2025-12-01 15:00:00 GMT; 5s ago
       Docs: https://github.com/iwozere/e-trading
   Main PID: 12345 (python)
      Tasks: 3 (limit: 4164)
     Memory: 50.0M
        CPU: 1.234s
     CGroup: /system.slice/scheduler.service
             └─12345 /opt/apps/e-trading/.venv/bin/python -m src.scheduler.main

Dec 01 15:00:00 raspberrypi e-trading-scheduler[12345]: INFO - Scheduler application initialized
Dec 01 15:00:00 raspberrypi e-trading-scheduler[12345]: INFO - Initializing scheduler services...
Dec 01 15:00:01 raspberrypi e-trading-scheduler[12345]: INFO - All scheduler services initialized successfully
Dec 01 15:00:01 raspberrypi e-trading-scheduler[12345]: INFO - Starting scheduler application...
Dec 01 15:00:02 raspberrypi e-trading-scheduler[12345]: INFO - Scheduler started successfully
```

---

## Useful Commands

```bash
# Start service
sudo systemctl start scheduler.service

# Stop service
sudo systemctl stop scheduler.service

# Restart service
sudo systemctl restart scheduler.service

# Check status
sudo systemctl status scheduler.service

# Enable on boot
sudo systemctl enable scheduler.service

# Disable on boot
sudo systemctl disable scheduler.service

# View logs (last 50 lines)
sudo journalctl -u scheduler.service -n 50

# Watch logs in real-time
sudo journalctl -u scheduler.service -f

# View all logs
sudo journalctl -u scheduler.service --no-pager

# Reload systemd after editing service file
sudo systemctl daemon-reload
```
