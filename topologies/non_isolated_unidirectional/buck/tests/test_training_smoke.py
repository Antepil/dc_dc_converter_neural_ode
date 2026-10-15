import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from training.checkpoint import load_checkpoint, save_checkpoint
from training.loops import run_event_pretrain, run_mode_dynamics, run_rollout_stage
from training.model import NeuralEventHybridODE
from tests.helpers import load_config, synthetic_split


class TrainingSmokeTests(unittest.TestCase):
    def test_training_batch_override_records_actual_updates(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cfg = load_config()
        stage = dict(next(s for s in cfg["training"]["stages"] if s["name"] == "one_cycle"))
        stage.update(batch_size=3, training_batch_size=2)
        model = NeuralEventHybridODE(cfg["model"])
        data = synthetic_split()
        observed = []
        report = run_rollout_stage(model, data, data, stage, cfg, "cpu", torch.float32,
                                   max_windows=3, max_epochs=1,
                                   best_callback=lambda m, epoch, loss: observed.append((epoch, loss)))
        self.assertEqual(report["training_batch_size"], 2)
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual(observed[0][0], 1)
        self.assertEqual(observed[0][1], report["last_val_loss_terms"]["total"])
        self.assertEqual(stage["batch_size"], 3)
        self.assertTrue(all(int(s["step"]) == 2 for s in report["_optimizer_state"]["state"].values()))

    def test_event_pretrain_resume_resets_legacy_plateau_learning_rate(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        cfg = load_config()
        model = NeuralEventHybridODE(cfg["model"])
        data = synthetic_split()
        stage = next(
            item for item in cfg["training"]["stages"]
            if item["name"] == "event_pretrain"
        )
        optimizer = torch.optim.AdamW(
            model.event_network.parameters(),
            lr=stage["event_network_learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        loss = sum(parameter.square().sum() for parameter in model.event_network.parameters())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer_state = deepcopy(optimizer.state_dict())
        optimizer_state["param_groups"][0]["lr"] = 1e-12
        legacy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer).state_dict()

        report = run_event_pretrain(
            model, data, data, stage, cfg, "cpu", torch.float32,
            max_batches=1, max_epochs=2,
            resume={
                "epoch": 1,
                "best": 0.1,
                "stall": 20,
                "optimizer_state": optimizer_state,
                "scheduler_state": legacy_scheduler,
            },
        )

        self.assertEqual(report["epochs"], 2)
        self.assertIsNone(report["_scheduler_state"])
        self.assertEqual(
            report["_optimizer_state"]["param_groups"][0]["lr"],
            stage["event_network_learning_rate"],
        )

    def test_checkpoint_every_two_epochs_and_resume_current_stage(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        cfg = load_config()
        self.assertEqual(cfg["training"]["checkpoint_interval_epochs"], 2)
        model = NeuralEventHybridODE(cfg["model"])
        data = synthetic_split()
        stage = next(item for item in cfg["training"]["stages"] if item["name"] == "mode_dynamics")
        saved_epochs = []
        saved = {}

        def capture(epoch, best, stall, report, optimizer_state, scheduler_state):
            saved_epochs.append(epoch)
            saved.update({
                "epoch": epoch,
                "best": best,
                "stall": stall,
                "optimizer_state": optimizer_state,
                "scheduler_state": scheduler_state,
            })

        first = run_mode_dynamics(
            model, data, data, stage, cfg, "cpu", torch.float32,
            max_batches=1, max_epochs=2,
            checkpoint_every=2, checkpoint_callback=capture,
        )
        self.assertEqual(first["epochs"], 2)
        resumed = run_mode_dynamics(
            model, data, data, stage, cfg, "cpu", torch.float32,
            max_batches=1, max_epochs=4, resume=dict(saved),
            checkpoint_every=2, checkpoint_callback=capture,
        )
        self.assertEqual(resumed["epochs"], 4)
        self.assertEqual(saved_epochs, [2, 4])

    def test_three_training_phases_and_checkpoint(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        cfg = load_config()
        model = NeuralEventHybridODE(cfg["model"])
        data = synthetic_split()
        stages = {stage["name"]: stage for stage in cfg["training"]["stages"]}
        mode_report = run_mode_dynamics(
            model, data, data, stages["mode_dynamics"], cfg, "cpu", torch.float32,
            max_batches=1, max_epochs=1,
        )
        event_report = run_event_pretrain(
            model, data, data, stages["event_pretrain"], cfg, "cpu", torch.float32,
            max_batches=1, max_epochs=1,
        )
        rollout_report = run_rollout_stage(
            model, data, data, stages["one_cycle"], cfg, "cpu", torch.float32,
            max_windows=3, max_epochs=1,
        )
        self.assertTrue(torch.isfinite(torch.tensor(mode_report["val_mode_derivative_mse"])))
        self.assertTrue(torch.isfinite(torch.tensor(event_report["val_event_pretrain_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(rollout_report["val_rollout_loss"])))

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pt")
            save_checkpoint(
                path,
                model=model,
                stage="one_cycle",
                epoch=1,
                best=rollout_report["val_rollout_loss"],
                stall=0,
                stage_report={"one_cycle": {"val_rollout_loss": rollout_report["val_rollout_loss"]}},
                completed_stages=["mode_dynamics", "event_pretrain", "one_cycle"],
                config_sha256="test",
                training_batch_sizes={"ten_cycle": 16, "twenty_cycle": 4},
                normalization=cfg["normalization"],
                loss_weights=cfg["training"]["loss_weights"],
                hybrid_settings={"step_size_tau": 0.01, "secant_refinement_iterations": 2},
                idle_profile="soft_idle_loss",
                rng_state={"torch_cpu": torch.get_rng_state()},
                optimizer_state=rollout_report["_optimizer_state"],
                scheduler_state=rollout_report["_scheduler_state"],
            )
            restored = load_checkpoint(path)
        self.assertEqual(restored["schema_version"], "hybrid_event_matlab_v3")
        self.assertEqual(restored["training_batch_sizes"], {"ten_cycle": 16, "twenty_cycle": 4})
        self.assertIsNotNone(restored["optimizer_state"])
        self.assertEqual(restored["model_signature"], model.architecture_signature())

    def test_hard_profile_joint_rollout_backpropagates(self):
        torch.manual_seed(9)
        cfg = load_config()
        model = NeuralEventHybridODE(cfg["model"], "hard_idle_constraint")
        data = synthetic_split()
        stage = next(item for item in cfg["training"]["stages"] if item["name"] == "one_cycle")
        report = run_rollout_stage(
            model, data, data, stage, cfg, "cpu", torch.float32,
            max_windows=3, max_epochs=1,
        )
        self.assertTrue(torch.isfinite(torch.tensor(report["val_rollout_loss"])))
        x = torch.tensor([[0.2, 0.3]])
        self.assertEqual(model.idle_vector_field(x)[0, 0].detach().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
