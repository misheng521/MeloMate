"""Real local process tests: installed Python/Node only, no downloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workspace_core as workspace
import local_execution as local


class LocalExecutionTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[2] / ".tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="local test ", dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scope = patch.multiple(workspace, ROOT=self.root, WORKSPACE_ROOT=self.root / "workspace")
        scope.start()
        self.addCleanup(scope.stop)
        self.project = workspace.workspace_path("小可", "项目_work")
        self.project.mkdir(parents=True)

    def run_file(self, code, **kwargs):
        workspace.write_workspace_file("小可", "项目_work", "task.py", code)
        return local.run_command("小可", ["python", "task.py"], "项目_work", **kwargs)

    def test_saved_original_file_runs_and_can_be_repaired(self):
        result = self.run_file("raise ValueError('fixture failure')")
        self.assertFalse(result["ok"], result)
        self.assertIn("fixture failure", result["output"])
        result = self.run_file("from pathlib import Path; Path('result.txt').write_text('ok'); print('运行完成')")
        self.assertTrue(result["ok"], result)
        self.assertEqual((self.project / "result.txt").read_text(), "ok")
        self.assertIn("运行完成", result["output"])
        self.assertEqual([p.name for p in self.project.glob('*.py')], ['task.py'])
        self.assertFalse((self.project / '.venv').exists())

    def test_node_and_npm_reuse_installed_runtime(self):
        if not shutil.which("node"):
            self.skipTest("Node is not installed")
        workspace.write_workspace_file("小可", "项目_work", "task.js", "require('fs').writeFileSync('node.txt','ok'); console.log('node ready')")
        result = local.run_command("小可", ["node", "task.js"], "项目_work")
        self.assertTrue(result["ok"], result)
        self.assertTrue((self.project / "node.txt").is_file())
        result = local.run_command("小可", ["npm", "--version"], "项目_work")
        self.assertTrue(result["ok"], result)
        result = local.run_command("小可", ["npm", "prefix"], "项目_work")
        self.assertTrue(result["ok"], result)
        self.assertEqual(Path(result["output"].strip()), self.project)
        package = json.dumps({"private": True, "type": "module", "scripts": {"check": "node task.js"}})
        workspace.write_workspace_file("小可", "项目_work", "package.json", package)
        workspace.write_workspace_file("小可", "项目_work", "task.js", "import fs from 'node:fs'; console.log(fs.existsSync('node.txt'))")
        result = local.run_command("小可", ["npm", "run", "check"], "项目_work")
        self.assertTrue(result["ok"], result)
        self.assertEqual((self.project / "package.json").read_text(), package)

    def test_pip_prepares_offline_venv_and_python_reuses_it(self):
        result = local.run_command("小可", ["python", "-m", "pip", "--version"], "项目_work")
        self.assertTrue(result["ok"], result)
        self.assertIn(str(self.project / '.venv').lower(), result["output"].lower())
        result = self.run_file("import sys; print(sys.prefix)")
        self.assertTrue(result["ok"], result)
        self.assertIn(str(self.project / '.venv'), result["output"])

    def test_existing_incompatible_venv_is_not_overwritten(self):
        target = self.project / '.venv'
        target.mkdir()
        (target / 'keep.txt').write_text('keep')
        result = self.run_file("print('should not run')")
        self.assertFalse(result["executed"])
        self.assertEqual((target / 'keep.txt').read_text(), 'keep')

    def test_timeout_kills_descendants(self):
        result = self.run_file("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',\"import time; from pathlib import Path; time.sleep(2); Path('escaped.txt').write_text('bad')\"]); time.sleep(60)", timeout_seconds=1)
        self.assertTrue(result["timed_out"], result)
        time.sleep(1.3)
        self.assertFalse((self.project / 'escaped.txt').exists())

    def test_successful_parent_also_cleans_up_background_children(self):
        result = self.run_file("import subprocess,sys; subprocess.Popen([sys.executable,'-c',\"import time; from pathlib import Path; time.sleep(1); Path('orphan.txt').write_text('bad')\"]); print('done')")
        self.assertTrue(result["ok"], result)
        time.sleep(1.2)
        self.assertFalse((self.project / 'orphan.txt').exists())

    def test_mcp_cancellation_stops_running_code(self):
        import anyio
        workspace.write_workspace_file("小可", "项目_work", "task.py", "import time; from pathlib import Path; Path('ready').write_text('yes'); time.sleep(2); Path('cancelled-write').write_text('bad')")
        async def scenario():
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(local.run_command_async, "小可", ["python", "task.py"], "项目_work")
                with anyio.fail_after(5):
                    while not (self.project / 'ready').exists():
                        await anyio.sleep(0.02)
                tasks.cancel_scope.cancel()
        anyio.run(scenario)
        time.sleep(2.1)
        self.assertFalse((self.project / 'cancelled-write').exists())

    def test_pre_cancelled_command_does_not_start(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(local.subprocess, 'Popen') as start:
            result = local.run_command('小可', ['python', 'task.py'], '项目_work', cancel_event=cancel)
        self.assertTrue(result['cancelled'])
        start.assert_not_called()

    def test_output_is_bounded_and_preserves_final_error(self):
        result = self.run_file("print('x'*100000); raise RuntimeError('final failure')")
        self.assertTrue(result['output_truncated'])
        self.assertLessEqual(len(result['output'].encode()), local.OUTPUT_LIMIT)
        self.assertIn('final failure', result['output'])

    def test_scope_checks_and_no_false_sandbox_claim(self):
        result = local.run_command('小可', ['python'], '../outside')
        self.assertFalse(result['executed'])
        info = local.runtime_info()
        self.assertFalse(info['filesystem_isolated'])
        self.assertFalse(info['network_isolated'])
        outside = self.root / 'disposable.txt'
        result = self.run_file(f"from pathlib import Path; Path({str(outside)!r}).write_text('allowed')")
        self.assertTrue(result['ok'], result)
        self.assertEqual(outside.read_text(), 'allowed')

    def test_missing_executable_is_reported_without_execution(self):
        result = local.run_command('小可', ['nonexistent-melomate-fixture-943'], '项目_work')
        self.assertFalse(result['executed'])
        self.assertIn('not found', result['error'])

    @unittest.skipUnless(os.name == 'nt', 'Windows PowerShell')
    def test_existing_powershell_runs_saved_script(self):
        workspace.write_workspace_file('小可', '项目_work', 'task.ps1', "[System.IO.File]::WriteAllText((Join-Path (Get-Location) 'shell.txt'), 'ok')")
        result = local.run_command('小可', ['powershell', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', 'task.ps1'], '项目_work')
        self.assertTrue(result['ok'], result)
        self.assertEqual((self.project / 'shell.txt').read_text(), 'ok')

    def test_environment_does_not_inherit_backend_keys_or_hooks(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'hidden', 'NODE_OPTIONS': 'bad', 'PYTHONPATH': 'bad'}):
            env = local.project_environment(self.project)
        for key in ('OPENAI_API_KEY', 'NODE_OPTIONS', 'PYTHONPATH'):
            self.assertNotIn(key, env)
        self.assertTrue(Path(env['TEMP']).is_relative_to(self.project))

    @unittest.skipUnless(os.name == 'nt', 'Windows console and Job Object')
    def test_console_stays_hidden_and_ownership_failure_never_runs_code(self):
        result = self.run_file("import ctypes; print(ctypes.windll.kernel32.GetConsoleWindow())")
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['output'].strip(), '0')
        with patch('windows_process_job.WindowsJob.assign', side_effect=OSError('fixture ownership failure')):
            result = self.run_file("from pathlib import Path; Path('must-not-run').write_text('bad')")
        self.assertFalse(result['executed'])
        self.assertFalse((self.project / 'must-not-run').exists())


if __name__ == '__main__':
    unittest.main()
