@echo off
echo ========================================
echo Fanvue Chatbot - Windows Installation
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found. Checking version...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.8 or higher is required
    python --version
    pause
    exit /b 1
)

echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo Creating necessary directories...
if not exist models mkdir models
if not exist exports mkdir exports
if not exist logs mkdir logs

echo Downloading AI model (this may take several minutes)...
python setup.py --download-model Luna-AI-Llama2-Uncensored
if errorlevel 1 (
    echo WARNING: Model download failed. You can download it manually later.
    echo The application will still work, but you'll need to load a model first.
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To start the application, run: start.bat
echo Or manually run: venv\Scripts\activate.bat && python main.py
echo.
echo The application will be available at: http://localhost:7860
echo.
pause