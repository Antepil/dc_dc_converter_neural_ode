import unittest
import torch
from tests.helpers import synthetic_split, load_config
from training.data import central_derivative_labels, collate
from training.losses import voltage_derivative_loss
from training.model import NeuralEventHybridODE
from training.loops import run_mode_dynamics, run_rollout_stage


class DerivativeTests(unittest.TestCase):
    def test_central_training_and_rollout_report(self):
        torch.set_num_threads(1)
        cfg = load_config()
        cfg['training']['derivative_method'] = 'central'
        cfg['training']['loss_weights']['voltage_derivative'] = 0.01
        model = NeuralEventHybridODE(cfg['model'])
        data = synthetic_split()
        pretrain = next(s for s in cfg['training']['stages'] if s['name'] == 'mode_dynamics')
        run_mode_dynamics(model, data, data, pretrain, cfg, 'cpu', torch.float32, max_batches=1, max_epochs=1)
        stage = next(s for s in cfg['training']['stages'] if s['name'] == 'one_cycle')
        report = run_rollout_stage(model, data, data, stage, cfg, 'cpu', torch.float32, max_windows=3, max_epochs=1)
        self.assertGreater(report['last_val_loss_terms']['voltage_derivative'], 0)

    def test_quadratic_derivative_units_and_event_mask(self):
        data = synthetic_split()
        tau = data['time_s'] / 5e-5
        data['states_norm'][:, :, 1] = tau.square()
        dx, mask = central_derivative_labels(data)
        expected = (2*tau).expand(3, -1)
        torch.testing.assert_close(dx[:, :, 1][mask], expected[mask], atol=1e-12, rtol=1e-12)
        for row, event in enumerate((70, 75, 80)):
            self.assertFalse(mask[row, event-2:event+3].any())
        self.assertFalse(mask[:, 48:53].any())
        self.assertFalse(mask[:, [0, 100]].any())

    def test_loss_reaches_vector_fields_and_zero_for_exact_prediction(self):
        data = synthetic_split()
        batch = collate(data, [(0, 0), (1, 0)], 1, 'cpu', torch.float32, derivative_supervision=True)
        model = NeuralEventHybridODE(load_config()['model'])
        loss = voltage_derivative_loss(model, batch)
        loss.backward()
        for field in (model.f_on, model.f_off, model.f_idle):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in field.parameters()))
        self.assertTrue(all(p.grad is None for p in model.event_network.parameters()))
        class Exact:
            def vector_field(self, mode, x, vs):
                return torch.zeros_like(x)
        batch['derivative_target'].zero_()
        self.assertEqual(float(voltage_derivative_loss(Exact(), batch)), 0)


if __name__ == '__main__':
    unittest.main()
