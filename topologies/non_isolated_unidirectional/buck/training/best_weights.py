"""Keep genuinely best observed validation weights separately for each stage.

Bundles here are selection artifacts, not resumable training checkpoints.
"""
import math
import os
from pathlib import Path

import torch


class BestValidationWeights:
    def __init__(self, directory, stage, config_sha256):
        self.path = Path(directory) / "best_validation" / f"{stage}.pt"
        self.stage = stage
        self.config_sha256 = config_sha256
        self.metric = float("inf")
        if self.path.is_file():
            self.metric = self.load()["metric"]

    def load(self):
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if payload["config_sha256"] != self.config_sha256 or payload["stage"] != self.stage:
            raise ValueError("Best-validation weights do not match this stage/config")
        return payload

    def consider(self, model, epoch, metric):
        if not math.isfinite(metric) or metric >= self.metric:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "epoch": epoch, "metric": metric, "stage": self.stage,
                   "config_sha256": self.config_sha256}
        temporary = self.path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, self.path)
        self.metric = metric
        return True
