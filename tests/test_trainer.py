import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf


def _tiny_cfg(max_train_steps=10, eval_every_steps=5, patience=100):
    """Minimal config for exercising the training loop."""
    return OmegaConf.create({
        "train": {
            "lr": 1e-3, "weight_decay": 0.0, "warmup_ratio": 0.0,
            "max_train_steps": max_train_steps, "eval_every_steps": eval_every_steps,
            "patience": patience, "log_every": 1000, "train_auc_batches": 2,
        },
        "data": {"batch_size": 4},
        "study": {"results_path": None},
    })


class _TinyModel(nn.Module):
    """A 2-class classifier over flat features, HF-style output interface."""
    def __init__(self, in_dim=8):
        super().__init__()
        self.classifier = nn.Linear(in_dim, 2)

    def forward(self, pixel_values=None, labels=None, **kw):
        logits = self.classifier(pixel_values)
        out = type("O", (), {})()          # simple namespace with .logits/.loss
        out.logits = logits
        if labels is not None:
            out.loss = nn.functional.cross_entropy(logits, labels)
        return out


def _fake_loader(n=16, in_dim=8):
    """Synthetic loader with BOTH classes guaranteed present (so roc_auc_score
    never sees a single-class batch)."""
    x = torch.randn(n, in_dim)
    y = torch.zeros(n, dtype=torch.long)
    y[::2] = 1                              # alternate 0/1 -> both classes present
    ds = TensorDataset(x, y)

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return {"pixel_values": xs, "labels": ys}

    return DataLoader(ds, batch_size=4, collate_fn=collate)


def test_returns_three_values(tmp_path):
    """run_training returns (model, best_auc, best_step) with valid types."""
    torch.manual_seed(0)
    from vit_histo.trainer import run_training
    cfg = _tiny_cfg(max_train_steps=6, eval_every_steps=3)
    cfg.study.results_path = str(tmp_path / "study_results.jsonl")

    result = run_training(cfg, _TinyModel(), _fake_loader(), _fake_loader(), "cpu")

    assert len(result) == 3
    model, best_auc, best_step = result
    assert isinstance(best_auc, float)
    assert isinstance(best_step, int)


def test_training_runs_end_to_end(tmp_path):
    """Smoke test: the loop runs to completion on tiny data and selects a
    real best checkpoint within the training horizon."""
    torch.manual_seed(0)
    from vit_histo.trainer import run_training
    cfg = _tiny_cfg(max_train_steps=6, eval_every_steps=3)
    cfg.study.results_path = str(tmp_path / "study_results.jsonl")

    model, best_auc, best_step = run_training(
        cfg, _TinyModel(), _fake_loader(), _fake_loader(), "cpu"
    )

    # validation fires at steps 3 and 6, so a real checkpoint is selected
    assert best_step in range(1, 7)
    assert 0.0 <= best_auc <= 1.0


def test_trial_record_written(tmp_path):
    """The per-trial JSONL record is written with the expected keys."""
    import json
    torch.manual_seed(0)
    from vit_histo.trainer import run_training
    results_path = tmp_path / "study_results.jsonl"
    cfg = _tiny_cfg(max_train_steps=6, eval_every_steps=3)
    cfg.study.results_path = str(results_path)

    run_training(cfg, _TinyModel(), _fake_loader(), _fake_loader(), "cpu")

    assert results_path.exists(), "trial record JSONL was not written"
    record = json.loads(results_path.read_text().strip().splitlines()[-1])
    for key in ("best_auc", "best_step", "max_train_steps", "lr"):
        assert key in record