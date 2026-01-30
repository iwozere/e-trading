#!/bin/bash

# Define paths
SERVICE_NAME="ibkr_screener.service"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME"
SOURCE_SERVICE="$(pwd)/bin/screener/$SERVICE_NAME"
RUN_SCRIPT="$(pwd)/bin/screener/run_screener.sh"

echo "Installing IBKR Screener Service..."

# 1. Ensure run script is executable
chmod +x $RUN_SCRIPT

# 2. Copy service file to systemd directory
if [ -f "$SOURCE_SERVICE" ]; then
    sudo cp $SOURCE_SERVICE $SERVICE_FILE
    echo "Service file copied to $SERVICE_FILE"
else
    echo "Error: Service file not found at $SOURCE_SERVICE"
    exit 1
fi

# 3. Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "Service $SERVICE_NAME installed, enabled, and started."
echo "Check status with: systemctl status $SERVICE_NAME"
echo "Check logs with: journalctl -u $SERVICE_NAME -f"
