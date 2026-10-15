import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import load_checkpoint


class V2RejectionTests(unittest.TestCase):
    def test_v2_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"schema_version": "known_pwm_matlab_v2", "model_state": {}}, path)
            with self.assertRaisesRegex(ValueError, "拒绝混用 V2"):
                load_checkpoint(str(path))


if __name__ == "__main__":
    unittest.main()
