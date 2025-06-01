@echo off
setlocal enabledelayedexpansion

REM Enable logging
set "LOGFILE=%~dp0install_log.txt"
echo ======================================== > "%LOGFILE%"
echo Influencer Chatbot Installation Log >> "%LOGFILE%"
echo Started at: %date% %time% >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"

echo ========================================
echo Influencer Chatbot Installation Script
echo ========================================
echo.
echo Installation log: %LOGFILE%
echo.

REM Clear various caches before installation
echo Clearing caches to prevent conflicts...
echo Clearing caches... >> "%LOGFILE%" 2>&1

REM Clear pip cache
if exist "%LOCALAPPDATA%\pip\Cache" (
    echo Clearing pip cache... >> "%LOGFILE%" 2>&1
    rmdir /s /q "%LOCALAPPDATA%\pip\Cache" >> "%LOGFILE%" 2>&1
)

REM Clear Python cache files
echo Clearing Python cache files... >> "%LOGFILE%" 2>&1
for /r . %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d" >> "%LOGFILE%" 2>&1
    )
)

REM Clear .pyc files
echo Clearing .pyc files... >> "%LOGFILE%" 2>&1
del /s /q *.pyc >> "%LOGFILE%" 2>&1

REM Clear Gradio cache
if exist "%USERPROFILE%\.cache\gradio" (
    echo Clearing Gradio cache... >> "%LOGFILE%" 2>&1
    rmdir /s /q "%USERPROFILE%\.cache\gradio" >> "%LOGFILE%" 2>&1
)

REM Clear temporary files
if exist "%TEMP%\gradio" (
    echo Clearing temporary Gradio files... >> "%LOGFILE%" 2>&1
    rmdir /s /q "%TEMP%\gradio" >> "%LOGFILE%" 2>&1
)

REM Check if Python is installed
echo Checking Python installation...
echo Checking Python installation... >> "%LOGFILE%" 2>&1
python --version >> "%LOGFILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error: Python is not installed or not in PATH
    echo ❌ Error: Python is not installed or not in PATH >> "%LOGFILE%"
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo Check the log file for details: %LOGFILE%
    pause
    exit /b 1
)

echo ✅ Python is installed
echo ✅ Python is installed >> "%LOGFILE%"
echo.

REM Create virtual environment
echo Creating virtual environment...
echo Creating virtual environment... >> "%LOGFILE%" 2>&1
python -m venv venv >> "%LOGFILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error: Failed to create virtual environment
    echo ❌ Error: Failed to create virtual environment >> "%LOGFILE%"
    echo Please check your Python installation and try again.
    echo Check the log file for details: %LOGFILE%
    pause
    exit /b 1
)

echo ✅ Virtual environment created
echo ✅ Virtual environment created >> "%LOGFILE%"
echo.

REM Activate virtual environment
echo Activating virtual environment...
echo Activating virtual environment... >> "%LOGFILE%" 2>&1
call venv\Scripts\activate.bat

REM Upgrade pip first
echo Upgrading pip...
echo Upgrading pip... >> "%LOGFILE%" 2>&1
python -m pip install --upgrade pip >> "%LOGFILE%" 2>&1

REM Clear pip cache in virtual environment
echo Clearing virtual environment pip cache...
echo Clearing virtual environment pip cache... >> "%LOGFILE%" 2>&1
python -m pip cache purge >> "%LOGFILE%" 2>&1

REM Install requirements with no cache
echo Installing Python dependencies...
echo Installing Python dependencies... >> "%LOGFILE%" 2>&1
python -m pip install --no-cache-dir -r requirements.txt >> "%LOGFILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error: Failed to install dependencies
    echo ❌ Error: Failed to install dependencies >> "%LOGFILE%"
    echo Please check your internet connection and try again.
    echo Check the log file for details: %LOGFILE%
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully
echo ✅ Dependencies installed successfully >> "%LOGFILE%"
echo.

REM Create necessary directories
echo Creating necessary directories...
echo Creating necessary directories... >> "%LOGFILE%" 2>&1
if not exist "models" mkdir models >> "%LOGFILE%" 2>&1
if not exist "exports" mkdir exports >> "%LOGFILE%" 2>&1
if not exist "logs" mkdir logs >> "%LOGFILE%" 2>&1
if not exist "conversation_logs" mkdir conversation_logs >> "%LOGFILE%" 2>&1

echo ✅ Directories created
echo ✅ Directories created >> "%LOGFILE%"
echo.

echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo Installation completed successfully! >> "%LOGFILE%"
echo Completed at: %date% %time% >> "%LOGFILE%"
echo.
echo You can now run the chatbot with start.bat
echo The application will be available at: http://localhost:7861
echo.
echo Next steps:
echo 1. Run start.bat to launch the application
echo 2. Open http://localhost:7861 in your browser
echo 3. Go to Model Management tab to load a model
echo 4. Start generating content!
echo.
pause