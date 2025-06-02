"""
Auto-open browser functionality for the enhanced chatbot system.
"""

import webbrowser
import time
import os

def auto_open_browser(url="http://127.0.0.1:7861", delay=1.5):
    """
    Automatically open the web browser to the specified URL after a delay.
    
    Args:
        url (str): The URL to open in the browser
        delay (float): Delay in seconds before opening the browser
    """
    try:
        time.sleep(delay)
        webbrowser.open(url)
        print(f"🌐 Browser opened automatically at {url}")
    except Exception as e:
        print(f"⚠️ Could not auto-open browser: {e}")

def check_browser_availability():
    """
    Check if a web browser is available on the system.
    
    Returns:
        bool: True if browser is available, False otherwise
    """
    try:
        # Try to get the default browser
        browser = webbrowser.get()
        return browser is not None
    except Exception:
        return False

def open_browser_with_fallback(url="http://127.0.0.1:7861"):
    """
    Open browser with fallback options if default browser fails.
    
    Args:
        url (str): The URL to open in the browser
    """
    try:
        # Try default browser first
        webbrowser.open(url)
        print(f"🌐 Browser opened at {url}")
        return True
    except Exception as e:
        print(f"⚠️ Default browser failed: {e}")
        
        # Try specific browsers as fallback
        browsers = ['firefox', 'chrome', 'chromium', 'safari', 'opera']
        
        for browser_name in browsers:
            try:
                browser = webbrowser.get(browser_name)
                browser.open(url)
                print(f"🌐 Opened {browser_name} at {url}")
                return True
            except Exception:
                continue
        
        print(f"❌ Could not open any browser. Please manually navigate to {url}")
        return False