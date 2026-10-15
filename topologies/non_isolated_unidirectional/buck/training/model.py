#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Buck DCM Neural Event ODE 模型。

数据和配置校验由上游入口负责。本模块只保存可学习网络及其确定的前向
语义；固定网格积分和事件状态机位于 ``hybrid_rk4.py``。
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn


MODE_ON = 0
MODE_OFF_DIODE = 1
MODE_IDLE = 2
MODE_NAMES = {
    MODE_ON: "ON",
    MODE_OFF_DIODE: "OFF_DIODE",
    MODE_IDLE: "IDLE",
}


class MLP(nn.Module):
    """Tanh MLP，采用设计规定的 Xavier normal/零偏置初始化。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, out_dim: int):
        super().__init__()
        modules = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        modules.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*modules)
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mlp(spec: dict) -> MLP:
    return MLP(
        spec["input_dim"],
        spec["hidden_width"],
        spec["hidden_layers"],
        spec["output_dim"],
    )


class NeuralEventHybridODE(nn.Module):
    """三模式 Buck DCM 向量场、神经事件面和恒等事件映射。

    ``vector_field(mode, x, vs)`` 和 ``event_value(x, vs)`` 是
    ``hybrid_rk4.py`` 的正式调用接口。输入约定：``x`` 为 ``(B, 2)``，
    ``mode`` 为 ``(B,)`` 的模式编码，``vs`` 为 ``(B, 1)`` 的归一化输入电压。
    ``mode`` 也可以是标量，便于局部预训练和单样本测试。
    """

    def __init__(
        self,
        cfg_model: dict,
        idle_profile: str = "soft_idle_loss",
    ):
        super().__init__()
        fields = cfg_model["vector_fields"]
        self.f_on = _mlp(fields["on"])
        self.f_off = _mlp(fields["off"])
        self.f_idle = _mlp(fields["idle"])
        self.event_network = _mlp(cfg_model["event_network"])
        self.idle_profile = idle_profile
        self.network_normalization = cfg_model.get("network_normalization", "fixed_reference")
        if self.network_normalization not in ("fixed_reference", "input_voltage_relative"):
            raise ValueError(f"未知网络归一化: {self.network_normalization}")
        if self.network_normalization == "input_voltage_relative":
            # 标记纯权重文件的坐标语义，阻止与旧权重静默混用。
            self.register_buffer("_input_voltage_relative_conditioned_v2", torch.tensor(2))

    @property
    def mode_names(self) -> dict:
        return MODE_NAMES.copy()

    def architecture_signature(self) -> dict:
        """返回 checkpoint 使用的结构签名。"""
        def layers(module: nn.Module) -> list[tuple[int, int]]:
            return [
                (layer.in_features, layer.out_features)
                for layer in module.modules()
                if isinstance(layer, nn.Linear)
            ]

        signature = {
            "architecture": "neural_event_hybrid_ode",
            "modes": {name: code for code, name in MODE_NAMES.items()},
            "on": layers(self.f_on),
            "off": layers(self.f_off),
            "idle": layers(self.f_idle),
            "event": layers(self.event_network),
            "idle_profile": self.idle_profile,
            "event_update": "identity",
        }
        if self.network_normalization != "fixed_reference":
            signature["network_normalization"] = self.network_normalization
            signature["voltage_conditioning"] = "preserved_on_off_event_v2"
        return signature

    @staticmethod
    def _vs_batch(vs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        vs = vs.to(device=x.device, dtype=x.dtype)
        if vs.ndim == 1:
            vs = vs.unsqueeze(1)
        if vs.shape[0] == 1:
            vs = vs.expand(x.shape[0], -1)
        return vs

    @staticmethod
    def _mode_batch(
        mode: Union[int, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        mode = torch.as_tensor(mode, device=device, dtype=torch.long)
        if mode.ndim == 0:
            mode = mode.expand(batch_size)
        return mode

    def _on_off_input(self, x: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        vs = self._vs_batch(vs, x)
        if self.network_normalization == "input_voltage_relative":
            return torch.cat([x / vs, vs], dim=1)
        return torch.cat([x, vs], dim=1)

    def idle_raw(self, x: torch.Tensor, vs: torch.Tensor | None = None) -> torch.Tensor:
        """IDLE 神经导数，返回积分器的固定参考坐标。"""
        if self.network_normalization == "input_voltage_relative":
            if vs is None:
                raise ValueError("input_voltage_relative 的 IDLE 导数需要输入电压")
            scale = self._vs_batch(vs, x)
            return scale * self.f_idle(x / scale)
        return self.f_idle(x)

    def idle_vector_field(self, x: torch.Tensor, vs: torch.Tensor | None = None) -> torch.Tensor:
        """返回 soft 或 hard IDLE profile 的实际向量场。"""
        raw = self.idle_raw(x, vs)
        if self.idle_profile == "hard_idle_constraint":
            return torch.cat([torch.zeros_like(raw[:, :1]), raw[:, 1:2]], dim=1)
        return raw

    def vector_field(
        self,
        mode: Union[int, torch.Tensor],
        x: torch.Tensor,
        vs: torch.Tensor,
    ) -> torch.Tensor:
        """按逐样本离散模式选择状态导数。"""
        # 调用方已知整个 batch 的模式时，不运行另外两个 MLP。
        # 混合模式仍使用稠密计算，避免 GPU nonzero/动态索引同步。
        if isinstance(mode, int):
            if mode == MODE_IDLE:
                return self.idle_vector_field(x, vs)
            if mode in (MODE_ON, MODE_OFF_DIODE):
                network = self.f_on if mode == MODE_ON else self.f_off
                dx = network(self._on_off_input(x, vs))
                if self.network_normalization == "input_voltage_relative":
                    dx = self._vs_batch(vs, x) * dx
                return dx
        mode = self._mode_batch(mode, x.shape[0], x.device)
        inp = self._on_off_input(x, vs)
        dx_on = self.f_on(inp)
        dx_off = self.f_off(inp)
        if self.network_normalization == "input_voltage_relative":
            scale = self._vs_batch(vs, x)
            dx_on, dx_off = scale * dx_on, scale * dx_off
        dx_idle = self.idle_vector_field(x, vs)
        return torch.where(
            (mode == MODE_ON).unsqueeze(1),
            dx_on,
            torch.where((mode == MODE_OFF_DIODE).unsqueeze(1), dx_off, dx_idle),
        )

    def event_value(self, x: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        """计算神经事件分数 ``g_phi(iL_norm, vC_norm, vS_norm)``。"""
        return self.event_network(self._on_off_input(x, vs)).squeeze(1)

    def apply_event(self, x_event: torch.Tensor) -> torch.Tensor:
        """自然截止事件的恒等状态映射。"""
        return x_event

    def forward(
        self,
        mode: Union[int, torch.Tensor],
        x: torch.Tensor,
        vs: torch.Tensor,
    ) -> torch.Tensor:
        """``nn.Module`` 调用形式，等价于 ``vector_field``。"""
        return self.vector_field(mode, x, vs)
