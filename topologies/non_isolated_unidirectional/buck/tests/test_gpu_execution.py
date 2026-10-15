"""Execution optimizations must preserve derivatives and event diagnostics."""
import copy
import unittest

import torch

from tests.helpers import LinearEventModel, load_config
from training.hybrid_rk4 import hybrid_rk4
from training.model import NeuralEventHybridODE


class GPUExecutionTests(unittest.TestCase):
    def test_uniform_modes_match_dense_outputs_and_gradients(self):
        for normalization in ("fixed_reference", "input_voltage_relative"):
            for profile in ("soft_idle_loss", "hard_idle_constraint"):
                cfg = copy.deepcopy(load_config()["model"])
                cfg["network_normalization"] = normalization
                for mode in (0, 1, 2):
                    model = NeuralEventHybridODE(cfg, profile).double()
                    x = torch.tensor([[0.4, 0.2], [0.7, 0.3]], dtype=torch.float64, requires_grad=True)
                    vs = torch.tensor([[0.5], [1.0]], dtype=torch.float64, requires_grad=True)
                    fast = model.vector_field(mode, x, vs)
                    dense = model.vector_field(torch.full((2,), mode), x, vs)
                    torch.testing.assert_close(fast, dense)
                    inputs = (x, vs, *model.parameters())
                    a = torch.autograd.grad(fast.square().sum(), inputs, allow_unused=True)
                    b = torch.autograd.grad(dense.square().sum(), inputs, allow_unused=True)
                    for parameter, ga, gb in zip(inputs, a, b):
                        torch.testing.assert_close(
                            torch.zeros_like(parameter) if ga is None else ga,
                            torch.zeros_like(parameter) if gb is None else gb,
                        )

    def test_deferred_checks_preserve_events_and_gradients(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for root in (0.25, -1.0, 0.8):
                model = LinearEventModel(root=root).to(device)
                x = torch.tensor([[0.5, 0.1], [0.6, 0.1]], device=device, requires_grad=True)
                vs = torch.ones(2, 1, device=device)
                fast = hybrid_rk4(model, x, vs, 1)
                checked = hybrid_rk4(model, x, vs, 1, check_finite_every_step=True)
                for field in vars(fast):
                    torch.testing.assert_close(getattr(fast, field), getattr(checked, field))
                inputs = (x, *model.parameters())
                a = torch.autograd.grad(fast.states.sum() + fast.event_time_tau.sum(), inputs, allow_unused=True)
                b = torch.autograd.grad(checked.states.sum() + checked.event_time_tau.sum(), inputs, allow_unused=True)
                for ga, gb in zip(a, b):
                    if ga is None or gb is None:
                        self.assertIs(ga, gb)
                    else:
                        torch.testing.assert_close(ga, gb)

    def test_nonfinite_values_still_rejected(self):
        for strict in (False, True):
            model = LinearEventModel()
            x = torch.tensor([[0.5, float("nan")]])
            with self.assertRaises(FloatingPointError):
                hybrid_rk4(model, x, torch.ones(1, 1), 1, check_finite_every_step=strict)
            with torch.no_grad():
                model.event_bias.fill_(float("nan"))
            with self.assertRaises(FloatingPointError):
                hybrid_rk4(model, torch.tensor([[0.5, 0.1]]), torch.ones(1, 1), 1,
                           check_finite_every_step=strict)

    def test_replaced_trial_state_is_still_checked(self):
        class InvalidTrial(LinearEventModel):
            def vector_field(self, mode, x, vs):
                dx = super().vector_field(mode, x, vs)
                mode = torch.as_tensor(mode, device=x.device)
                bad = (mode == 1) & (x[:, 0] < 0.5)
                return torch.stack((dx[:, 0], torch.where(bad, float("nan"), dx[:, 1])), dim=1)

        # Invalid initial guard replaces the nonfinite full OFF trial with a
        # finite alpha=0 event state. Checking only returned states would miss it.
        for strict in (False, True):
            with self.assertRaises(FloatingPointError):
                hybrid_rk4(InvalidTrial(root=0.8), torch.tensor([[0.5, 0.1]]),
                           torch.ones(1, 1), 1, check_finite_every_step=strict)
