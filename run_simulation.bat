@echo off
rem Ensure we are running from the script's directory (root)
cd /d "%~dp0"
echo Starting IoT Task Offloading Simulation...
echo Running from: %CD%
echo ------------------------------------------

rem Check if venv exists in src\venv
if not exist "src\venv\Scripts\python.exe" (
    echo Virtual Environment not found in src\venv!
    echo Current Directory Contents:
    dir src
    pause
    exit /b
)

rem Run simulation directly
"src\venv\Scripts\python.exe" src\simulation_env.py

if %errorlevel% neq 0 (
    echo.
    echo Simulation crashed or failed to start!
    echo Check the error message above.
)
pause
