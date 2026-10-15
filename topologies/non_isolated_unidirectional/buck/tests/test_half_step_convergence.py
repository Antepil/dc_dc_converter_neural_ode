import unittest

import torch

from training.hybrid_rk4 import hybrid_rk4
from tests.helpers import LinearEventModel


class HalfStepConvergenceTests(unittest.TestCase):
    def test_linear_event_matches_at_half_step(self):
        model = LinearEventModel()
        y0 = torch.tensor([[0.50, 0.1], [0.60, 0.1]])
        vs = torch.ones(2, 1)
        coarse = hybrid_rk4(model, y0, vs, 1, step_size_tau=0.01, samples_per_cycle=100)
        fine = hybrid_rk4(model, y0, vs, 1, step_size_tau=0.005, samples_per_cycle=200)
        torch.testing.assert_close(coarse.states, fine.states[::2], atol=3e-6, rtol=0)
        torch.testing.assert_close(coarse.event_time_tau, fine.event_time_tau, atol=3e-6, rtol=0)
        self.assertTrue(torch.equal(coarse.event_mask, fine.event_mask))


if __name__ == "__main__":
    unittest.main()
