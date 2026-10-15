import tempfile
import unittest
import torch

from training.best_weights import BestValidationWeights


class BestWeightsTests(unittest.TestCase):
    def test_selection_survives_restart_and_keeps_best_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(1, 1)
            tracker = BestValidationWeights(directory, "ten_cycle", "config")
            expected = model.weight.detach().clone()
            self.assertTrue(tracker.consider(model, 1, 0.1))
            with torch.no_grad():
                model.weight.add_(10)
            tracker = BestValidationWeights(directory, "ten_cycle", "config")
            self.assertFalse(tracker.consider(model, 2, 0.2))
            self.assertFalse(tracker.consider(model, 3, float("nan")))
            torch.testing.assert_close(tracker.load()["weights"]["weight"], expected)
            self.assertTrue(tracker.consider(model, 4, 0.05))
            self.assertEqual(tracker.load()["epoch"], 4)
            other = BestValidationWeights(directory, "twenty_cycle", "config")
            self.assertTrue(other.consider(model, 1, 0.5))
            with self.assertRaises(ValueError):
                BestValidationWeights(directory, "ten_cycle", "different config")
