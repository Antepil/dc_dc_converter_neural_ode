import os
import subprocess
import sys
import tempfile
import unittest

from training.run_lock import exclusive_run


class RunLockTests(unittest.TestCase):
    def test_second_process_rejected_and_released_lock_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "from training.run_lock import exclusive_run; import sys\nwith exclusive_run(sys.argv[1]): print('acquired')"
            env = dict(os.environ, PYTHONPATH=os.getcwd())
            with exclusive_run(directory):
                result = subprocess.run([sys.executable, "-c", code, directory],
                                        capture_output=True, text=True, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Another process owns", result.stderr)
            result = subprocess.run([sys.executable, "-c", code, directory],
                                    capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
