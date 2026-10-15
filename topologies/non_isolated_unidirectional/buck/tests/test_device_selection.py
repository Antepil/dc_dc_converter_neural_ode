"""设备路由与跨平台指纹检查；不进行训练。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from train import _device, _config_sha256

class DeviceSelectionTests(unittest.TestCase):
    def test_auto_prefers_cuda_and_validates_index(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(_device('auto'), 'cuda:0')
            self.assertEqual(_device('cuda'), 'cuda:0')
            self.assertEqual(_device('cuda:1'), 'cuda:1')
            with self.assertRaises(RuntimeError): _device('cuda:2')

    def test_mps_and_explicit_cpu(self):
        with patch('torch.cuda.is_available', return_value=False), patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(_device('auto'), 'mps')
            self.assertEqual(_device('mps'), 'mps')
            with self.assertRaises(RuntimeError): _device('cuda')
        with patch('torch.cuda.is_available', return_value=False), patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaises(RuntimeError): _device('auto')
            with self.assertRaises(RuntimeError): _device('mps')
            self.assertEqual(_device('cpu'), 'cpu')
        with self.assertRaises(ValueError): _device('cuda:-1')

    def test_config_line_endings(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.json',Path(d)/'b.json'
            a.write_bytes(b'{\n "a": 1\n}\n')
            b.write_bytes(b'{\r\n "a": 1\r\n}\r\n')
            self.assertEqual(_config_sha256(str(a)),_config_sha256(str(b)))
