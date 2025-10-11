@echo off
REM Web UI Test Runner for Windows
REM =============================
REM 
REM This batch script provides an easy way to run all Web UI tests on Windows.
REM It supports both backend Python tests and frontend TypeScript tests.

echo.
echo ================================================================================
echo                           WEB UI TEST RUNNER
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if Node.js is available for frontend tests
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Node.js is not installed or not in PATH
    echo Frontend tests will be skipped
    set SKIP_FRONTEND=1
) else (
    set SKIP_FRONTEND=0
)

REM Parse command line arguments
set VERBOSE=0
set NO_COVERAGE=0
set BACKEND_ONLY=0
set FRONTEND_ONLY=0
set SEQUENTIAL=0

:parse_args
if "%1"=="" goto :run_tests
if "%1"=="-v" set VERBOSE=1
if "%1"=="--verbose" set VERBOSE=1
if "%1"=="--no-coverage" set NO_COVERAGE=1
if "%1"=="--backend-only" set BACKEND_ONLY=1
if "%1"=="--frontend-only" set FRONTEND_ONLY=1
if "%1"=="--sequential" set SEQUENTIAL=1
if "%1"=="--help" goto :show_help
shift
goto :parse_args

:show_help
echo Usage: run_tests.bat [options]
echo.
echo Options:
echo   -v, --verbose      Enable verbose output
echo   --no-coverage      Disable coverage reporting
echo   --backend-only     Run only backend tests
echo   --frontend-only    Run only frontend tests
echo   --sequential       Run tests sequentially (not in parallel)
echo   --help             Show this help message
echo.
echo Examples:
echo   run_tests.bat                    # Run all tests with coverage
echo   run_tests.bat --verbose          # Run with verbose output
echo   run_tests.bat --backend-only     # Run only Python backend tests
echo   run_tests.bat --frontend-only    # Run only TypeScript frontend tests
echo.
pause
exit /b 0

:run_tests
echo Starting Web UI test suite...
echo.

REM Build Python command arguments
set PYTHON_ARGS=
if %VERBOSE%==1 set PYTHON_ARGS=%PYTHON_ARGS% --verbose
if %NO_COVERAGE%==1 set PYTHON_ARGS=%PYTHON_ARGS% --no-coverage
if %BACKEND_ONLY%==1 set PYTHON_ARGS=%PYTHON_ARGS% --backend-only
if %FRONTEND_ONLY%==1 set PYTHON_ARGS=%PYTHON_ARGS% --frontend-only
if %SEQUENTIAL%==1 set PYTHON_ARGS=%PYTHON_ARGS% --sequential

REM Skip frontend if Node.js not available and not backend-only
if %SKIP_FRONTEND%==1 if %BACKEND_ONLY%==0 if %FRONTEND_ONLY%==0 (
    echo Node.js not available, running backend tests only...
    set PYTHON_ARGS=%PYTHON_ARGS% --backend-only
)

REM Run the Python test runner
echo Executing: python run_all_tests.py%PYTHON_ARGS%
echo.
python run_all_tests.py%PYTHON_ARGS%

REM Capture exit code
set TEST_EXIT_CODE=%errorlevel%

echo.
if %TEST_EXIT_CODE%==0 (
    echo ================================================================================
    echo                              ALL TESTS PASSED!
    echo ================================================================================
) else (
    echo ================================================================================
    echo                              SOME TESTS FAILED!
    echo ================================================================================
    echo Exit code: %TEST_EXIT_CODE%
)

REM Show coverage reports if available
if %NO_COVERAGE%==0 (
    echo.
    echo Coverage reports generated:
    if exist "backend\htmlcov\index.html" (
        echo   Backend: backend\htmlcov\index.html
    )
    if exist "frontend\coverage\index.html" (
        echo   Frontend: frontend\coverage\index.html
    )
)

echo.
echo Test run completed. Press any key to exit...
pause >nul

exit /b %TEST_EXIT_CODE%