"""Shared browser discovery for installation and runtime; never downloads on import."""
from __future__ import annotations

import os
from pathlib import Path
import shutil


def installed_browser() -> str | None:
    candidates = []
    for product in ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe"):
        for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
            if os.environ.get(variable):
                candidates.append(Path(os.environ[variable]) / product)
    for name in ("msedge", "microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
        executable = shutil.which(name)
        if executable:
            candidates.append(Path(executable))
    return next((str(path) for path in candidates if path.is_file()), None)


def browser_info() -> dict:
    installed = installed_browser()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"available": False, "component": False, "executable": installed,
                "reason": "Run setup-windows.bat to install the browser component with MeloMate."}
    if installed:
        return {"available": True, "component": True, "executable": installed, "source": "installed"}
    with sync_playwright() as driver:
        executable = driver.chromium.executable_path
    return {"available": Path(executable).is_file(), "component": True, "executable": executable,
            "source": "playwright", "reason": "Run setup-windows.bat if the browser is missing."}


if __name__ == "__main__":
    import json
    import sys
    result = browser_info()
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["available"] else 1)
