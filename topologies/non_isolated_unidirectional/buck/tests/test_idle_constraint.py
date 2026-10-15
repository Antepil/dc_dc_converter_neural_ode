import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from training.model import NeuralEventHybridODE
from training.losses import rollout_losses
from tests.helpers import load_config


def set_idle_constant(model, current_rate, voltage_rate):
    for parameter in model.f_idle.parameters():
        nn.init.zeros_(parameter)
    final = [layer for layer in model.f_idle.modules() if isinstance(layer, nn.Linear)][-1]
    final.bias.data.copy_(torch.tensor([current_rate, voltage_rate], dtype=final.bias.dtype))


class IdleConstraintTests(unittest.TestCase):
    def test_soft_and_hard_idle_share_raw_network(self):
        cfg = load_config()["model"]
        soft = NeuralEventHybridODE(cfg, "soft_idle_loss")
        hard = NeuralEventHybridODE(cfg, "hard_idle_constraint")
        set_idle_constant(soft, 2.0, 3.0)
        hard.f_idle.load_state_dict(soft.f_idle.state_dict())
        x = torch.tensor([[0.2, 0.3]])
        torch.testing.assert_close(soft.idle_vector_field(x), torch.tensor([[2.0, 3.0]]))
        torch.testing.assert_close(hard.idle_vector_field(x), torch.tensor([[0.0, 3.0]]))

    def test_event_map_is_identity_and_keeps_gradient(self):
        model = NeuralEventHybridODE(load_config()["model"])
        state = torch.tensor([[0.1, 0.2]], requires_grad=True)
        mapped = model.apply_event(state)
        torch.testing.assert_close(mapped, state)
        mapped.sum().backward()
        torch.testing.assert_close(state.grad, torch.ones_like(state))

    def test_four_current_losses_use_cycle_masks(self):
        class LossModel:
            def vector_field(self, mode, x, vs):
                return torch.zeros_like(x)

            def idle_vector_field(self, x, vs=None):
                return torch.stack((torch.full_like(x[:, 0], 3.0), torch.zeros_like(x[:, 1])), dim=1)

            def event_value(self, x, vs):
                return torch.ones(x.shape[0], dtype=x.dtype)

        states = torch.zeros(101, 1, 2, requires_grad=True)
        states.data[76:, 0, 0] = 2.0
        rollout = SimpleNamespace(
            states=states,
            event_time_tau=torch.tensor([[0.75]]),
            event_current_norm=torch.tensor([[0.5]], requires_grad=True),
        )
        mode = torch.zeros(100, 1, dtype=torch.long)
        mode[50:75] = 1
        mode[75:] = 2
        batch = {
            "target": torch.zeros_like(states),
            "event_time_tau": torch.tensor([[0.75]]),
            "event_state": torch.zeros(1, 1, 2),
            "event_mask": torch.tensor([[True]]),
            "interval_mode": mode,
            "vs": torch.ones(1, 1),
        }
        terms = rollout_losses(LossModel(), rollout, batch, {
            "state": 0.0, "event_time": 0.0, "guard": 0.0,
            "event_current": 1.0, "idle_state": 1.0,
            "idle_tangent": 1.0, "idle_end": 1.0, "ripple": 0.0,
        })
        self.assertAlmostEqual(float(terms["event_current"].detach()), 0.25)
        self.assertAlmostEqual(float(terms["idle_state"].detach()), 4.0)
        self.assertAlmostEqual(float(terms["idle_tangent"].detach()), 9.0)
        self.assertAlmostEqual(float(terms["idle_end"].detach()), 4.0)
        terms["total"].backward()
        self.assertIsNotNone(rollout.event_current_norm.grad)


if __name__ == "__main__":
    unittest.main()
