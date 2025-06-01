@echo off
echo ========================================
echo Fanvue Chatbot Update Script
echo ========================================
echo.

echo Pulling latest updates from GitHub...
git pull origin main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error: Failed to pull updates from GitHub
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ✅ Successfully pulled updates from GitHub
echo.

echo Installing/updating Python dependencies...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error: Failed to install dependencies
    echo Please check your Python installation and try again.
    pause
    exit /b 1
)

echo.
echo ✅ Successfully updated dependencies
echo.

echo ========================================
echo Update completed successfully!
echo ========================================
echo.
echo You can now run the chatbot with start.bat
echo.
pause