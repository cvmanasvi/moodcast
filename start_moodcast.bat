@echo off
rem Starts the MoodCast Flask app. Run from anywhere; paths resolve to this folder.
cd /d "%~dp0"

echo.
echo ========================================
echo   Starting MoodCast Server
echo ========================================
echo.
echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. See the messages above.
    pause
    exit /b 1
)

echo.
echo Starting Flask server...
echo Open your browser to: http://127.0.0.1:5000
echo.
python flask_app.py
pause
