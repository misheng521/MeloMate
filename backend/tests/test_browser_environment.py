"""Browser setup discovery without browser downloads or application dependencies."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import browser_environment


class BrowserEnvironmentTests(unittest.TestCase):
    def test_existing_edge_is_preferred_over_chrome(self):
        with patch.dict(os.environ, {"PROGRAMFILES": "C:/Programs"}, clear=True), \
             patch.object(browser_environment.shutil, "which", return_value=None), \
             patch.object(Path, "is_file", return_value=True):
            self.assertEqual(browser_environment.installed_browser(), str(Path("C:/Programs/Microsoft/Edge/Application/msedge.exe")))

    def test_chrome_can_be_reused_if_edge_is_missing(self):
        with patch.dict(os.environ, {"LOCALAPPDATA": "C:/Local"}, clear=True), \
             patch.object(browser_environment.shutil, "which", return_value=None), \
             patch.object(Path, "is_file", lambda p: p.name == "chrome.exe"):
            self.assertEqual(browser_environment.installed_browser(), str(Path("C:/Local/Google/Chrome/Application/chrome.exe")))

    def test_missing_installation_is_not_reported_as_ready(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(browser_environment.shutil, "which", return_value=None):
            self.assertIsNone(browser_environment.installed_browser())

    def test_component_missing_is_distinct_from_browser_executable(self):
        with patch.object(browser_environment, "installed_browser", return_value="edge.exe"), \
             patch.dict(sys.modules, {"playwright.sync_api": None}):
            result = browser_environment.browser_info()
            self.assertFalse(result["available"])
            self.assertFalse(result["component"])
            self.assertEqual(result["executable"], "edge.exe")


if __name__ == "__main__":
    unittest.main()
