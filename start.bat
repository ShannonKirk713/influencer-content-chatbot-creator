
@echo off
echo ========================================
echo Starting Fanvue Chatbot
echo ========================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Fanvue Chatbot...
echo The application will be available at: http://127.0.0.1:7861
echo Press Ctrl+C to stop the application
echo.

python main.py

echo.
echo Application stopped.
pause
