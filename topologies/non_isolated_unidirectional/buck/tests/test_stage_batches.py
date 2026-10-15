import argparse
import unittest

from tests.helpers import load_config
from training.batching import parse_stage_batch, resolve_training_batch_sizes


class StageBatchTests(unittest.TestCase):
    def setUp(self):
        self.stages = load_config()["training"]["stages"]

    def test_legacy_checkpoint_and_future_stage_overrides(self):
        checkpoint = {"stage_report": {"five_cycle": {"training_batch_size": 64},
                                        "ten_cycle": {"training_batch_size": 64}}}
        result = resolve_training_batch_sizes(self.stages, checkpoint,
                                              stage_batches=[("ten_cycle", 16), ("twenty_cycle", 4)])
        self.assertEqual(result["five_cycle"], 64)
        self.assertEqual(result["ten_cycle"], 16)
        self.assertEqual(result["twenty_cycle"], 4)
        # Persist the plan before the twenty-cycle stage starts.
        checkpoint["training_batch_sizes"] = result
        self.assertEqual(resolve_training_batch_sizes(self.stages, checkpoint), result)

    def test_cli_precedence_and_config_unchanged(self):
        checkpoint = {"training_batch_sizes": {"ten_cycle": 16, "twenty_cycle": 4}}
        result = resolve_training_batch_sizes(self.stages, checkpoint, 32, [("ten_cycle", 8)])
        self.assertEqual(result["ten_cycle"], 8)
        self.assertEqual(result["twenty_cycle"], 32)
        self.assertEqual(next(s for s in self.stages if s["name"] == "ten_cycle")["batch_size"], 8)
        self.assertNotIn("event_pretrain", result)

    def test_invalid_overrides_rejected(self):
        for value in ("ten_cycle", "ten_cycle=0", "ten_cycle=-1", "ten_cycle=a", "=16"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_stage_batch(value)
        for overrides in ([("unknown", 4)], [("event_pretrain", 4)],
                          [("ten_cycle", 4), ("ten_cycle", 8)]):
            with self.assertRaises(ValueError):
                resolve_training_batch_sizes(self.stages, stage_batches=overrides)
