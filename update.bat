@echo off
setlocal enabledelayedexpansion

REM Enable logging
set "LOGFILE=%~dp0update_log.txt"
echo ======================================== > "%LOGFILE%"
echo Influencer Chatbot Update Log >> "%LOGFILE%"
echo Started at: %date% %time% >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"

echo ========================================
echo Influencer Chatbot Update Script
echo ========================================
echo.
echo Update log: %LOGFILE%
echo.

REM Clear various caches before update
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

REM Clear virtual environment pip cache if it exists
if exist "venv\Scripts\activate.bat" (
    echo Clearing virtual environment pip cache...
    echo Clearing virtual environment pip cache... >> "%LOGFILE%" 2>&1
    call venv\Scripts\activate.bat
    python -m pip cache purge >> "%LOGFILE%" 2>&1
    deactivate
)

echo Pulling latest updates from GitHub...
echo Pulling latest updates from GitHub... >> "%LOGFILE%" 2>&1
git pull origin main >> "%LOGFILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error: Failed to pull updates from GitHub
    echo ❌ Error: Failed to pull updates from GitHub >> "%LOGFILE%"
    echo Please check your internet connection and try again.
    echo Check the log file for details: %LOGFILE%
    pause
    exit /b 1
)

echo.
echo ✅ Successfully pulled updates from GitHub
echo ✅ Successfully pulled updates from GitHub >> "%LOGFILE%"
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    echo Activating virtual environment... >> "%LOGFILE%" 2>&1
    call venv\Scripts\activate.bat
    
    echo Installing/updating Python dependencies...
    echo Installing/updating Python dependencies... >> "%LOGFILE%" 2>&1
    python -m pip install --upgrade pip >> "%LOGFILE%" 2>&1
    python -m pip install --no-cache-dir --upgrade -r requirements.txt >> "%LOGFILE%" 2>&1
    
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo ❌ Error: Failed to install dependencies
        echo ❌ Error: Failed to install dependencies >> "%LOGFILE%"
        echo Please check your Python installation and try again.
        echo Check the log file for details: %LOGFILE%
        pause
        exit /b 1
    )
) else (
    echo Virtual environment not found. Installing dependencies globally...
    echo Virtual environment not found. Installing dependencies globally... >> "%LOGFILE%" 2>&1
    python -m pip install --upgrade pip >> "%LOGFILE%" 2>&1
    python -m pip install --no-cache-dir --upgrade -r requirements.txt >> "%LOGFILE%" 2>&1
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ❌ Error: Failed to install dependencies
        echo ❌ Error: Failed to install dependencies >> "%LOGFILE%"
        echo Please check your Python installation and try again.
        echo Check the log file for details: %LOGFILE%
        pause
        exit /b 1
    )
)

echo.
echo ✅ Successfully updated dependencies
echo ✅ Successfully updated dependencies >> "%LOGFILE%"
echo.

echo ========================================
echo Update completed successfully!
echo ========================================
echo.
echo Update completed successfully! >> "%LOGFILE%"
echo Completed at: %date% %time% >> "%LOGFILE%"
echo.
echo You can now run the chatbot with start.bat
echo The application will be available at: http://localhost:7861
echo.
pause