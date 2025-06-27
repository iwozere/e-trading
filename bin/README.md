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
