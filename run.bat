@echo off
echo ============================================
echo  PlantGuard AI - Smart Farming Assistant
echo ============================================
echo.
echo Starting the application...
echo.

REM Check if virtual environment exists
set "PROJECT_DIR=%~dp0"

REM Prefer the workspace virtual environment (.venv)
set "PYEXE=%PROJECT_DIR%.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    REM Fallback to a legacy venv folder if present
    set "PYEXE=%PROJECT_DIR%venv\Scripts\python.exe"
)
if not exist "%PYEXE%" (
    echo No virtual environment found. Using system Python.
    set "PYEXE=python"
) else (
    echo Using Python: %PYEXE%
)

echo.
echo Installing/updating dependencies...
"%PYEXE%" -m pip install -r requirements.txt

echo.
echo Starting FastAPI server on http://localhost:5000
echo.
echo Open your browser and navigate to:
echo    http://localhost:5000/static/index.html
echo.
echo Press Ctrl+C to stop the server.
echo ============================================
echo.

"%PYEXE%" -m uvicorn main:app --host 127.0.0.1 --port 5000

pause
