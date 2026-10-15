import unittest
from copy import deepcopy
import torch
from training.model import NeuralEventHybridODE
from tests.helpers import load_config

class RelativeNormalizationTests(unittest.TestCase):
    def model(self):
        cfg=deepcopy(load_config()['model'])
        cfg['network_normalization']='input_voltage_relative'
        return NeuralEventHybridODE(cfg)

    def test_scaling_and_gradients_all_modes(self):
        model=self.model().double()
        x=torch.tensor([[.2,.4],[.1,.6],[.01,.5]],dtype=torch.float64,requires_grad=True)
        vs=torch.tensor([[.5],[.75],[1.]],dtype=torch.float64)
        modes=torch.tensor([0,1,2])
        dx=model.vector_field(modes,x,vs)
        inp=torch.cat([x/vs,vs],dim=1)
        torch.testing.assert_close(model._on_off_input(x,vs),inp)
        scaled=model._on_off_input(2.5*x,2.5*vs)
        torch.testing.assert_close(scaled[:,:2],inp[:,:2])
        torch.testing.assert_close(scaled[:,2:],2.5*inp[:,2:])
        expected=vs*torch.cat([model.f_on(inp[:1]),model.f_off(inp[1:2]),model.f_idle((x/vs)[2:])])
        torch.testing.assert_close(dx,expected)
        torch.testing.assert_close(model.event_value(x,vs),model.event_network(inp).squeeze(1))
        dx.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for net in [model.f_on,model.f_off,model.f_idle]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.parameters()))

    def test_weights_cannot_silently_cross_coordinate_systems(self):
        fixed=NeuralEventHybridODE(load_config()['model'])
        relative=self.model()
        with self.assertRaises(RuntimeError): relative.load_state_dict(fixed.state_dict())
        with self.assertRaises(RuntimeError): fixed.load_state_dict(relative.state_dict())
        legacy=dict(relative.state_dict())
        legacy['_input_voltage_relative_version']=legacy.pop('_input_voltage_relative_conditioned_v2')
        with self.assertRaises(RuntimeError): relative.load_state_dict(legacy)
        relative.load_state_dict(self.model().state_dict())

