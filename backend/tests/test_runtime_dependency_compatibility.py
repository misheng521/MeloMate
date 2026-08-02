import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

import check_runtime_dependencies


class RuntimeDependencyCompatibilityTests(unittest.TestCase):
    def test_supported_mcp_v1_is_accepted(self):
        with patch.object(check_runtime_dependencies, "version", return_value="1.28.2"):
            valid, message = check_runtime_dependencies.validate_mcp_version()

        self.assertTrue(valid)
        self.assertIn("verified", message)

    def test_mcp_v2_is_rejected_with_actionable_message(self):
        with patch.object(check_runtime_dependencies, "version", return_value="2.0.0"):
            valid, message = check_runtime_dependencies.validate_mcp_version()

        self.assertFalse(valid)
        self.assertIn(">=1.28,<2", message)

    def test_requirements_prevent_an_incompatible_major_upgrade(self):
        requirements = (BACKEND_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("mcp>=1.28,<2", requirements.splitlines())

    def test_start_scripts_run_the_version_validator(self):
        for script_name in ("setup-windows.bat", "start.bat"):
            script = (PROJECT_ROOT / script_name).read_text(encoding="utf-8")
            with self.subTest(script=script_name):
                self.assertIn("check_runtime_dependencies.py", script)


if __name__ == "__main__":
    unittest.main()
