import unittest

import torch

from training.hybrid_rk4 import hybrid_rk4
from training.model import MODE_IDLE, MODE_OFF_DIODE
from tests.helpers import LinearEventModel


class HybridRK4Tests(unittest.TestCase):
    def test_analytic_and_asynchronous_events(self):
        model = LinearEventModel()
        y0 = torch.tensor([[0.50, 0.1], [0.60, 0.1]])
        result = hybrid_rk4(model, y0, torch.ones(2, 1), 1)
        torch.testing.assert_close(result.event_time_tau[:, 0], torch.tensor([0.75, 0.85]), atol=2e-6, rtol=0)
        torch.testing.assert_close(result.event_state[:, 0, 0], torch.full((2,), 0.25), atol=2e-6, rtol=0)
        self.assertTrue(result.event_mask.all())
        self.assertTrue((result.interval_mode[50:75, 0] == MODE_OFF_DIODE).all())
        self.assertTrue((result.interval_mode[76:, 0] == MODE_IDLE).all())

    def test_missed_event_stays_off_until_pwm_rise(self):
        model = LinearEventModel(root=-1.0)
        result = hybrid_rk4(model, torch.tensor([[0.5, 0.1]]), torch.ones(1, 1), 1)
        self.assertFalse(result.event_mask.item())
        self.assertTrue(result.missed_event.item())
        self.assertTrue((result.interval_mode[50:, 0] == MODE_OFF_DIODE).all())
        self.assertTrue(torch.isfinite(result.states).all())

    def test_event_time_gradient(self):
        model = LinearEventModel()
        y0 = torch.tensor([[0.50, 0.1]], requires_grad=True)
        result = hybrid_rk4(model, y0, torch.ones(1, 1), 1)
        result.event_time_tau.sum().backward()
        self.assertAlmostEqual(float(y0.grad[0, 0]), 1.0, places=3)
        self.assertAlmostEqual(float(model.event_weight.grad), 0.25, places=3)
        self.assertAlmostEqual(float(model.event_bias.grad), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
