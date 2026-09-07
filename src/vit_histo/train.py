from __future__ import annotations

import os
import random
import time

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from vit_histo.logging_setup import setup_logging


def final_run(cfg, device):
    """Phase-2 final run: train the (best) config, then report AUC + Brier on
    BOTH the validation and test sets. Writes val_predictions.csv and
    test_predictions.csv (prob,label) under the Hydra run dir for later
    calibration analysis (e.g. CORP in R).

    NOTE: this trains a model itself, so it must NOT be called after main() has
    already trained — it is invoked as an early branch in main() instead.
    """
    from vit_histo.data import build_dataloaders
    from vit_histo.model import build_model
    from vit_histo.trainer import run_training, collect_predictions
    from vit_histo.evaluate import evaluate

    run_dir = os.getcwd()  # Hydra has chdir'd into the per-run output dir

    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg)
    model, best_auc, best_step = run_training(cfg, model, train_loader, val_loader, device)

    # val
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    val_m = evaluate(val_probs, val_labels, csv_path=os.path.join(run_dir, "val_predictions.csv"))

    # test (touched once, here)
    test_probs, test_labels = collect_predictions(model, test_loader, device)
    test_m = evaluate(test_probs, test_labels, csv_path=os.path.join(run_dir, "test_predictions.csv"))

    logger.info("val  | AUC {:.4f} | Brier {:.4f}", val_m["auc"], val_m["brier"])
    logger.info("test | AUC {:.4f} | Brier {:.4f}", test_m["auc"], test_m["brier"])
    logger.success("Predictions written to {}/val_predictions.csv and test_predictions.csv", run_dir)
    return {"val": val_m, "test": test_m}


def _resolve_device(requested: str) -> str:
    """Pick the compute device (auto -> cuda if available, else cpu)."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    """Seed stdlib, numpy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU


def _dummy_loop(cfg: DictConfig) -> None:
    logger.info("Running DUMMY loop (dry_run) — no model, no data, no gradients")
    steps = 3
    for i in range(steps):
        fake_loss = 1.0 / (i + 1)
        logger.info(
            "step {}/{} | fake_loss={:.4f} | lr={:.1e}",
            i + 1, steps, fake_loss, cfg.train.lr,
        )
        time.sleep(0.2)
    logger.success("Dummy loop finished cleanly.")

# python -m vit_histo.train                    # real run (sweep-style, val only)
# python -m vit_histo.train dry_run=true       # config / plumbing check
# python -m vit_histo.train eval_on_test=true  # Phase-2 final run (val + test)
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float | None:
    # Hydra has already chdir'd into the per-run output dir.
    run_dir = os.getcwd()
    setup_logging(run_dir)

    logger.info("Resolved configuration:\n{}", OmegaConf.to_yaml(cfg))
    _set_seed(cfg.seed)

    device = _resolve_device(cfg.device)
    logger.info("Using device: {}", device)

    if bool(cfg.get("dry_run", False)):
        _dummy_loop(cfg)
        return

    # --- Phase-2 final run: train best config, eval on BOTH val and test ---
    # Early branch so we train exactly once (final_run does its own training).
    if bool(cfg.get("eval_on_test", False)):
        results = final_run(cfg, device)
        return results["test"]["auc"]

    # --- otherwise: normal sweep run (train once, eval val only) ---
    # Imported here (not at top) so dry_run=true works without ML deps loaded.
    from vit_histo.data import build_dataloaders
    from vit_histo.model import build_model
    from vit_histo.trainer import collect_predictions, run_training
    from vit_histo.evaluate import evaluate

    train_loader, val_loader, test_loader, processor = build_dataloaders(cfg)
    model = build_model(cfg)

    model, best_auc, best_step = run_training(cfg, model, train_loader, val_loader, device)

    # evaluation (sweep path: validate on val_loader) ---
    probs, labels = collect_predictions(model, val_loader, device)
    metrics = evaluate(probs, labels, csv_path=os.path.join(run_dir, "val_predictions.csv"))
    logger.info("val metrics: {}", metrics)
    logger.success("Predictions written to {}/val_predictions.csv", run_dir)

    # objective for the Optuna sweeper (maximize)
    return metrics["auc"]


if __name__ == "__main__":
    main()