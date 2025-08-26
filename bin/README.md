# Script Launcher

This directory contains a convenient script launcher that allows you to easily run any of the available scripts in the Crypto Trading Platform.

## Files

- `run_script_launcher.bat` - Windows batch file launcher
- `run_script_launcher.sh` - Linux/Mac shell script launcher

## Usage

### Windows
```bash
# Double-click the file or run from command prompt
run_script_launcher.bat
```

### Linux/Mac
```bash
# Make sure the script is executable
chmod +x run_script_launcher.sh

# Run the launcher
./run_script_launcher.sh
```

## Available Scripts

The launcher provides access to the following scripts:

1. **Telegram Bot** - Main Telegram bot for user interactions
2. **Telegram Admin Panel** - Web-based admin interface (http://localhost:5001)
3. **Telegram Screener Bot** - Screener bot functionality
4. **Telegram Background Services** - Background alert and schedule processing
5. **JSON to CSV Converter** - Convert optimization results from JSON to CSV
6. **Optimizer** - Run strategy optimization
7. **Plotter** - Generate performance plots
8. **LSTM Optimizer** - Run LSTM model optimization

## Features

- **Virtual Environment Management**: Automatically activates the `.venv` environment
- **Error Handling**: Checks for virtual environment existence and provides helpful error messages
- **Interactive Menu**: Easy-to-use numbered menu system
- **Return to Menu**: After each script execution, returns to the main menu
- **Cross-Platform**: Works on Windows, Linux, and Mac

## Requirements

- Python virtual environment (`.venv`) must be set up
- All dependencies must be installed (`pip install -r requirements.txt`)
- Proper file permissions (for shell script on Unix systems)

## Troubleshooting

### Virtual Environment Not Found
If you get an error about the virtual environment not being found:
1. Create the virtual environment: `python -m venv .venv`
2. Install dependencies: `pip install -r requirements.txt`

### Permission Denied (Linux/Mac)
If you get a permission denied error on the shell script:
```bash
chmod +x run_script_launcher.sh
```

### Script Execution Fails
- Check that all dependencies are installed
- Verify that the virtual environment is properly set up
- Check the individual script logs for specific error messages

## Individual Scripts

If you prefer to run scripts individually, you can still use the original script files:

- `run_telegram_bot.bat/.sh`
- `run_telegram_admin_panel.bat/.sh`
- `run_telegram_screener_bot.bat/.sh`
- `run_telegram_screener_background.bat/.sh`
- `run_json2csv.bat/.sh`
- `run_optimizer.bat/.sh`
- `run_plotter.bat/.sh`
- `run_lstm_optimizer.bat/.sh`

All individual scripts also include virtual environment activation and error handling.

# Runner Scripts

This directory contains convenient batch files (Windows) and shell scripts (Linux/macOS) to run the Python scripts without needing to start an IDE or manually activate the virtual environment.

## Available Scripts

### Windows (.bat files)
- `run_optimizer.bat` - Runs the strategy optimizer
- `run_plotter.bat` - Runs the plotting system
- `run_json2csv.bat` - Converts optimization results from JSON to CSV
- `run_telegram_bot.bat` - Runs the Telegram screener bot

### Linux/macOS (.sh files)
- `run_optimizer.sh` - Runs the strategy optimizer
- `run_plotter.sh` - Runs the plotting system
- `run_json2csv.sh` - Converts optimization results from JSON to CSV
- `run_telegram_bot.sh` - Runs the Telegram screener bot

## How to Use

### Windows
1. Double-click any `.bat` file in the `bin/` directory
2. Or run from command prompt:
   ```cmd
   cd bin
   run_optimizer.bat
   ```

### Linux/macOS
1. Make scripts executable (first time only):
   ```bash
   cd bin
   chmod +x *.sh
   ```
2. Run any script:
   ```bash
   ./run_optimizer.sh
   ```

## What These Scripts Do

Each script automatically:

1. **Locates the project root** - Finds the correct directory structure
2. **Checks for virtual environment** - Verifies `.venv` exists
3. **Runs the Python script** - Executes the corresponding Python file in the background using absolute paths
4. **Logs output** - All output is written to the `logs/` directory in the project root
5. **Handles errors** - Provides clear error messages if something goes wrong
6. **Waits for user input** (Windows) - Keeps the window open to see results

## Prerequisites

Before using these scripts, ensure you have:

1. **Virtual environment created**:
   ```bash
   python -m venv .venv
   ```

2. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data files ready** (for optimizer and plotter):
   - Place CSV data files in the `data/` directory
   - Configure optimization settings in `config/optimizer/`

## Error Handling

The scripts include comprehensive error checking:

- **Virtual environment not found** - Provides setup instructions
- **Script execution failed** - Displays error details
- **Missing dependencies** - Guides through installation

## Troubleshooting

### Common Issues

1. **"Virtual environment not found"**
   - Run: `python -m venv .venv`
   - Then: `pip install -r requirements.txt`

2. **"Script failed"**
   - Check the Python script output for specific errors in the `logs/` directory
   - Ensure all required data files are present
   - Verify configuration files are correct

### Windows Specific
- Ensure you are running as Administrator if needed
- Check Windows Defender is not blocking the scripts
- Use Command Prompt or PowerShell

### Linux/macOS Specific
- Make sure scripts are executable: `chmod +x *.sh`
- Use bash shell: `bash run_optimizer.sh`
- Check file permissions and ownership

## Script Details

### run_optimizer.bat/sh
- Runs strategy optimization with configurable parameters
- Generates optimization results in JSON format
- Saves results to `results/` directory

### run_plotter.bat/sh
- Creates visual charts from optimization results
- Generates PNG plots with indicators and trade signals
- Saves plots to `results/` directory

### run_json2csv.bat/sh
- Converts optimization JSON results to CSV format
- Creates summary files for Excel analysis
- Useful for comparing multiple optimization runs

### run_telegram_bot.bat/sh
- Runs the Telegram screener bot for live alerts
- Logs output to `logs/telegram_bot.log`

## Environment Variables

The scripts automatically handle:
- **PYTHONPATH** - Adds project root to Python path
- **Virtual Environment** - Activates `.venv` with all dependencies
- **Working Directory** - Uses absolute paths for all execution

## Security Notes

- Scripts only activate the local `.venv` virtual environment
- No external network calls or installations
- All paths are absolute and relative to the project structure
- Error handling prevents execution with missing dependencies
