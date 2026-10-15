import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.data import build_mode_pairs, collate, load_data, make_windows, normalize
from tests.helpers import load_config, write_npz


class HybridDataSchemaTests(unittest.TestCase):
    def test_v3_load_normalize_windows_and_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            write_npz(Path(directory) / "train.npz")
            split = normalize(load_data(directory), load_config()["normalization"])["train"]
        self.assertEqual(split["states_norm"].shape, (1, 101, 2))
        batch = collate(split, make_windows(split, 1), 1, "cpu", torch.float32)
        self.assertEqual(batch["target"].shape, (101, 1, 2))
        self.assertEqual(batch["interval_mode"].shape, (100, 1))
        pairs = build_mode_pairs(split)
        self.assertTrue(all(values["x"].shape[0] > 0 for values in pairs.values()))

    def test_wrong_load_and_illegal_transition_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            write_npz(path, load_ohm=40.0)
            with self.assertRaisesRegex(ValueError, "R=80"):
                load_data(directory)
            write_npz(path, invalid_transition=True)
            with self.assertRaisesRegex(ValueError, "模式顺序"):
                load_data(directory)

    def test_v2_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "train.npz", states=np.zeros((1, 101, 2)))
            with self.assertRaisesRegex(ValueError, "缺少 v3 字段"):
                load_data(directory)


if __name__ == "__main__":
    unittest.main()
