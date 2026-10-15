import unittest
import json

import torch

from training.loops import event_pretrain_gate
from training.losses import event_pretrain_loss
from training.model import NeuralEventHybridODE
from tests.helpers import CONFIG_PATH, LinearEventModel, load_config, synthetic_split


class EventNetworkTests(unittest.TestCase):
    def test_architecture_signature(self):
        model = NeuralEventHybridODE(load_config()["model"])
        signature = model.architecture_signature()
        self.assertEqual(signature["on"], [(3, 64), (64, 64), (64, 2)])
        self.assertEqual(signature["off"], [(3, 64), (64, 64), (64, 2)])
        self.assertEqual(signature["idle"], [(2, 64), (64, 64), (64, 2)])
        self.assertEqual(signature["event"], [(3, 64), (64, 64), (64, 1)])
        self.assertEqual(signature["event_update"], "identity")

    def test_only_soft_and_hard_profiles_are_configured(self):
        config = load_config()
        self.assertEqual(config["model"]["supported_idle_profiles"], ["soft_idle_loss", "hard_idle_constraint"])
        experiment_dir = CONFIG_PATH.parent / "experiments"
        profiles = {
            json.loads(path.read_text(encoding="utf-8"))["idle_profile"]
            for path in experiment_dir.glob("*.json")
        }
        self.assertEqual(profiles, {"soft_idle_loss", "hard_idle_constraint"})

    def test_pretrain_objective_decreases(self):
        torch.manual_seed(3)
        model = NeuralEventHybridODE(load_config()["model"])
        optimizer = torch.optim.Adam(model.event_network.parameters(), lr=1e-2)
        vs = torch.ones(16, 1) * 0.75
        surface = torch.cat((torch.zeros(16, 1), torch.ones(16, 1) * 0.3), dim=1)
        pre = surface.clone(); pre[:, 0] = 0.2
        counterfactual = surface.clone(); counterfactual[:, 0] = -0.2

        def objective():
            return event_pretrain_loss(
                model.event_value(surface, vs),
                model.event_value(pre, vs),
                model.event_value(counterfactual, vs),
                0.1,
            )["total"]

        initial = float(objective().detach())
        for _ in range(50):
            loss = objective()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(float(objective().detach()), initial * 0.1)

    def test_vectorized_gate_matches_known_linear_event_times(self):
        config = load_config()
        data = synthetic_split((0.60, 0.65, 0.70))
        report = event_pretrain_gate(
            LinearEventModel(root=0.25), data, config, "cpu", torch.float32
        )
        self.assertTrue(report["passed"])
        self.assertLess(report["event_time_mae_tau"], 1e-6)
        self.assertEqual(report["event_detection_rate"], 1.0)
        self.assertEqual(report["exactly_one_root_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
