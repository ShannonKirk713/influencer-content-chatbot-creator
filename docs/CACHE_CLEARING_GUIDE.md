# Comprehensive Cache Clearing Guide for Fanvue Chatbot

This guide provides detailed instructions for clearing various types of cached files that might cause issues with the Fanvue Chatbot application. If you're experiencing errors like "'enable_queue' parameter not found" or other compatibility issues, clearing these caches can often resolve the problems.

## 🔍 Why Clear Caches?

Caches can become outdated when:
- Dependencies are updated (like Gradio 4.0+)
- Python packages are upgraded
- Virtual environments become corrupted
- Model files become corrupted
- Application settings change

## 🪟 Windows Instructions

### 1. Python Package Cache (pip cache)

**Clear pip cache:**
```cmd
# Open Command Prompt as Administrator
pip cache purge

# Alternative: Clear specific package cache
pip cache remove gradio
pip cache remove transformers
pip cache remove torch
```

**Clear pip user cache:**
```cmd
# Navigate to user cache directory
cd %LOCALAPPDATA%\pip\cache
# Delete all contents
rmdir /s /q .
```

### 2. Python Bytecode Cache (__pycache__)

**Method 1: Using Command Prompt**
```cmd
# Navigate to your project directory
cd C:\path\to\fanvue_chatbot

# Remove all __pycache__ directories recursively
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"

# Remove all .pyc files
del /s /q *.pyc
```

**Method 2: Using PowerShell**
```powershell
# Navigate to project directory
cd C:\path\to\fanvue_chatbot

# Remove __pycache__ directories
Get-ChildItem -Path . -Recurse -Directory -Name "__pycache__" | Remove-Item -Recurse -Force

# Remove .pyc files
Get-ChildItem -Path . -Recurse -File -Name "*.pyc" | Remove-Item -Force
```

### 3. Virtual Environment Cache

**Complete virtual environment reset:**
```cmd
# Navigate to project directory
cd C:\path\to\fanvue_chatbot

# Deactivate current environment
deactivate

# Remove entire virtual environment
rmdir /s /q venv

# Recreate virtual environment
python -m venv venv

# Activate new environment
venv\Scripts\activate.bat

# Reinstall requirements
pip install --no-cache-dir -r requirements.txt
```

### 4. Gradio Cache

**Clear Gradio temporary files:**
```cmd
# Clear Gradio cache directory
rmdir /s /q %TEMP%\gradio

# Clear user-specific Gradio cache
rmdir /s /q %USERPROFILE%\.gradio
```

### 5. Hugging Face Cache

**Clear downloaded models and tokenizers:**
```cmd
# Clear Hugging Face cache
rmdir /s /q %USERPROFILE%\.cache\huggingface

# Alternative location
rmdir /s /q %LOCALAPPDATA%\huggingface
```

### 6. Application-Specific Cache

**Clear Fanvue Chatbot specific cache:**
```cmd
# Navigate to project directory
cd C:\path\to\fanvue_chatbot

# Remove log files
del /q *.log

# Remove temporary conversation files (optional)
rmdir /s /q conversation_logs

# Remove any cached model files (if you want to re-download)
rmdir /s /q models
```

### 7. System Temporary Files

**Clear Windows temp directory:**
```cmd
# Clear system temp
rmdir /s /q %TEMP%

# Recreate temp directory
mkdir %TEMP%
```

## 🐧 Linux/macOS Instructions

### 1. Python Package Cache (pip cache)

**Clear pip cache:**
```bash
# Clear all pip cache
pip cache purge

# Clear specific package cache
pip cache remove gradio
pip cache remove transformers
pip cache remove torch
```

**Clear pip user cache manually:**
```bash
# Remove pip cache directory
rm -rf ~/.cache/pip

# For macOS, also check:
rm -rf ~/Library/Caches/pip
```

### 2. Python Bytecode Cache (__pycache__)

**Remove all __pycache__ directories and .pyc files:**
```bash
# Navigate to project directory
cd /path/to/fanvue_chatbot

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove .pyc files
find . -name "*.pyc" -delete

# Remove .pyo files (if any)
find . -name "*.pyo" -delete
```

### 3. Virtual Environment Cache

**Complete virtual environment reset:**
```bash
# Navigate to project directory
cd /path/to/fanvue_chatbot

# Deactivate current environment
deactivate

# Remove entire virtual environment
rm -rf venv

# Recreate virtual environment
python3 -m venv venv

# Activate new environment
source venv/bin/activate

# Reinstall requirements without cache
pip install --no-cache-dir -r requirements.txt
```

### 4. Gradio Cache

**Clear Gradio temporary files:**
```bash
# Clear Gradio cache
rm -rf ~/.gradio
rm -rf /tmp/gradio*

# For macOS:
rm -rf ~/Library/Caches/gradio
```

### 5. Hugging Face Cache

**Clear downloaded models and tokenizers:**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface

# For macOS:
rm -rf ~/Library/Caches/huggingface
```

### 6. Application-Specific Cache

**Clear Fanvue Chatbot specific cache:**
```bash
# Navigate to project directory
cd /path/to/fanvue_chatbot

# Remove log files
rm -f *.log

# Remove temporary conversation files (optional)
rm -rf conversation_logs

# Remove any cached model files (if you want to re-download)
rm -rf models
```

### 7. System Temporary Files

**Clear system temp directories:**
```bash
# Clear /tmp (be careful with this)
sudo rm -rf /tmp/*

# Clear user-specific temp
rm -rf /tmp/tmp*
```

## 🔧 Advanced Troubleshooting

### Complete Clean Installation

If you're still experiencing issues after clearing caches, try a complete clean installation:

**Windows:**
```cmd
# 1. Backup your conversation logs (if needed)
copy conversation_logs\*.txt backup_conversations\

# 2. Remove entire project directory
cd ..
rmdir /s /q fanvue_chatbot

# 3. Fresh clone from GitHub
git clone https://github.com/Valorking6/fanvue-content-chatbot.git fanvue_chatbot
cd fanvue_chatbot

# 4. Create fresh virtual environment
python -m venv venv
venv\Scripts\activate.bat

# 5. Install requirements without cache
pip install --no-cache-dir -r requirements.txt

# 6. Restore conversation logs (if needed)
copy backup_conversations\*.txt conversation_logs\
```

**Linux/macOS:**
```bash
# 1. Backup your conversation logs (if needed)
cp -r conversation_logs backup_conversations

# 2. Remove entire project directory
cd ..
rm -rf fanvue_chatbot

# 3. Fresh clone from GitHub
git clone https://github.com/Valorking6/fanvue-content-chatbot.git fanvue_chatbot
cd fanvue_chatbot

# 4. Create fresh virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install requirements without cache
pip install --no-cache-dir -r requirements.txt

# 6. Restore conversation logs (if needed)
cp -r ../backup_conversations/* conversation_logs/
```

### Verify Clean Installation

After clearing caches or doing a clean installation, verify everything is working:

```bash
# Check Python version
python --version

# Check pip version
pip --version

# Check Gradio version (should be 4.0+)
pip show gradio

# Check if main dependencies are installed correctly
pip list | grep -E "gradio|transformers|torch|llama-cpp-python"

# Test the application
python main.py
```

## 🚨 Common Issues and Solutions

### Issue: "'enable_queue' parameter not found"
**Solution:** This indicates an old Gradio version. Clear pip cache and reinstall:
```bash
pip cache purge
pip uninstall gradio
pip install --no-cache-dir "gradio>=4.0.0"
```

### Issue: "Module not found" errors
**Solution:** Virtual environment corruption. Reset the virtual environment:
```bash
deactivate
rm -rf venv  # or rmdir /s /q venv on Windows
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows
pip install --no-cache-dir -r requirements.txt
```

### Issue: Model loading failures
**Solution:** Clear Hugging Face cache and re-download models:
```bash
rm -rf ~/.cache/huggingface
python main.py  # Models will be re-downloaded
```

### Issue: Permission errors (Windows)
**Solution:** Run Command Prompt as Administrator and try again.

### Issue: Permission errors (Linux/macOS)
**Solution:** Use `sudo` for system directories, but avoid using `sudo pip`:
```bash
# Don't do this: sudo pip install
# Instead, use virtual environments or user installs:
pip install --user package_name
```

## 📝 Automation Scripts

### Windows Batch Script (clear_cache.bat)

Create a batch file to automate cache clearing:

```batch
@echo off
echo Clearing Fanvue Chatbot caches...

echo Clearing pip cache...
pip cache purge

echo Clearing Python bytecode cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc

echo Clearing Gradio cache...
rmdir /s /q %TEMP%\gradio 2>nul
rmdir /s /q %USERPROFILE%\.gradio 2>nul

echo Clearing Hugging Face cache...
rmdir /s /q %USERPROFILE%\.cache\huggingface 2>nul

echo Clearing application logs...
del /q *.log 2>nul

echo Cache clearing complete!
pause
```

### Linux/macOS Shell Script (clear_cache.sh)

Create a shell script to automate cache clearing:

```bash
#!/bin/bash
echo "Clearing Fanvue Chatbot caches..."

echo "Clearing pip cache..."
pip cache purge

echo "Clearing Python bytecode cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "Clearing Gradio cache..."
rm -rf ~/.gradio 2>/dev/null
rm -rf /tmp/gradio* 2>/dev/null

echo "Clearing Hugging Face cache..."
rm -rf ~/.cache/huggingface 2>/dev/null

echo "Clearing application logs..."
rm -f *.log 2>/dev/null

echo "Cache clearing complete!"
```

Make the script executable:
```bash
chmod +x clear_cache.sh
./clear_cache.sh
```

## 🔄 Regular Maintenance

To prevent cache-related issues:

1. **Weekly:** Clear Python bytecode cache
2. **Monthly:** Clear pip cache
3. **When updating dependencies:** Clear all caches
4. **When experiencing issues:** Follow this complete guide

## 📞 Still Having Issues?

If you're still experiencing problems after following this guide:

1. Check the GitHub Issues page for similar problems
2. Create a new issue with:
   - Your operating system and version
   - Python version (`python --version`)
   - Pip version (`pip --version`)
   - Gradio version (`pip show gradio`)
   - Complete error message
   - Steps you've already tried

## 📚 Additional Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Pip Cache Documentation](https://pip.pypa.io/en/stable/topics/caching/)
- [Hugging Face Cache Management](https://huggingface.co/docs/transformers/installation#cache-setup)

---

**Note:** Always backup important data (like conversation logs) before clearing caches or doing clean installations.
