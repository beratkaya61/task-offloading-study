@echo off
rem IoT Task Offloading Unified Run Script
cd /d "%~dp0"
echo ==============================================
echo   IOT TASK OFFLOADING: UNIFIED RUNNER
echo ==============================================

rem 1. Check Virtual Environment
if not exist "src\venv\Scripts\python.exe" (
    echo [ERROR] Virtual Environment not found in src\venv!
    echo Lutfen kurulum adimlarini takip edin.
    pause
    exit /b
)

rem 2. AI Model Check
if not exist "src\models\ppo_offloading_agent.zip" (
    echo [WARNING] Trained PPO model not found at src\models\ppo_offloading_agent.zip
    echo [INFO] Simulation will run using Semantic Rule-based logic.
    echo ------------------------------------------
) else (
    echo [SUCCESS] PPO Model detected. AI Core will be active.
    echo ------------------------------------------
)

rem 3. Launch Simulation
echo [INFO] Starting Dashboard...
"src\venv\Scripts\python.exe" src\simulation_env.py

if %errorlevel% neq 0 (
    echo.
    echo [CRASH] Simulation exited with error.
    echo [TIP] Dependencies eksik olabilir. Lutfen 'pip install -r src/requirements.txt' calistirdiginizdan emin olun.
)
pause
