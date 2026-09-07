from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig

import json
import os

def _write_trial_record(cfg, run_dir, record):
    """Append one trial's result to a study-level JSONL file."""
    results_path = cfg.get("study", {}).get("results_path", None)
    if results_path is None:
        results_path = os.path.join(
            os.path.dirname(run_dir.rstrip("/")), "study_results.jsonl"
        )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Appended trial record to {}", results_path)

@torch.no_grad()
def collect_predictions(model, val_loader, device: str, max_batches: int = 0):
    """Run the model over a loader; return (probs, labels) as numpy arrays.

    probs: predicted probability of the positive class (class 1).
    Model-agnostic: passes the whole batch to the model, so it works for any
    architecture whose batch dict matches its forward signature (ViT:
    pixel_values, text models: input_ids/attention_mask, etc.). Labels are
    pulled out for the metric and not fed to the forward pass.
    """
    model.eval()
    all_probs, all_labels = [], []
    for i, batch in enumerate(val_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")                 # remove labels before forward

        outputs = model(**batch)                     # pass whatever inputs remain
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[:, 1]      # P(class 1)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

        if max_batches and (i + 1) >= max_batches:
            break

    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def _validate(model, val_loader, device: str, max_batches: int = 0) -> float:
    """Return validation AUC (threshold-independent, robust to class imbalance).

    max_batches: if >0, evaluate only that many batches (used to estimate
    train AUC cheaply for the diagnostic log line).
    """
    from sklearn.metrics import roc_auc_score

    probs, labels = collect_predictions(model, val_loader, device, max_batches=max_batches)
    return float(roc_auc_score(labels, probs))


def run_training(cfg: DictConfig, model, train_loader, val_loader, device: str):
    """Fine-tune `model` with early stopping on validation AUC. Returns the best model."""
    from transformers import get_cosine_schedule_with_warmup

    # --- setup ---
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # max_train_steps is a fixed number of steps that's identical for every trial
    # not an epoch-based loop; according to the Google book
    max_train_steps = int(cfg.train.max_train_steps)
    warmup_steps = int(cfg.train.warmup_ratio * max_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # early-stopping state
    best_auc = -float("inf")
    best_state = None
    # Update 8/29/26: validate on a step cadence, not epoch cadence, so we can stop early mid-epoch.
    eval_every = int(cfg.train.get("eval_every_steps", len(train_loader)))
    steps_since_improve = 0
    patience = cfg.train.patience

    # training loops
    # Step-based training loop in liue of the epoch loop
    # A fixed number of steps that's identical for every trial.
    global_step = 0
    #epoch = 0
    stop = False
    best_step = -1 
    while global_step < max_train_steps and not stop:
        model.train()
        for i, batch in enumerate(train_loader):
            if global_step >= max_train_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % cfg.train.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "step {}/{} | loss {:.4f} | lr {:.2e}",
                    global_step, max_train_steps, loss.item(), current_lr,
                )

            # validation + retrospective checkpoint selection, on a step cadence
            if global_step % eval_every == 0:
                val_auc = _validate(model, val_loader, device)
                # diagnostic log line: also estimate TRAIN AUC cheaply (max_batches=10)
                # configurable in case the train set is huge and we don't want to waste time on a full pass.
                train_auc = _validate(model, train_loader, device, max_batches=cfg.train.get("train_auc_batches", 10))
                logger.info(
                    "step {} | train AUC {:.4f} | val AUC {:.4f}",
                    global_step, train_auc, val_auc,
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_step = global_step
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
                    if steps_since_improve >= patience:
                        logger.info(
                            "Early stopping at step {} (best AUC {:.4f})",
                            global_step, best_auc,
                        )
                        stop = True
                        break

                model.train()   # restore train mode after _validate switched to eval

        #epoch += 1

    # restore the best-AUC weights, not the (possibly overfit) final ones
    if best_state is not None:
        model.load_state_dict(best_state)

    logger.success(
        "Training finished. Best val AUC {:.4f} @ step {}/{}",
        best_auc, best_step, max_train_steps,)

    run_dir = os.getcwd()
    record = {
        "best_auc": best_auc,
        "best_step": best_step,
        "max_train_steps": max_train_steps,
        "lr": float(cfg.train.lr),
        "warmup_ratio": float(cfg.train.warmup_ratio),
        "weight_decay": float(cfg.train.weight_decay),
        "batch_size": int(cfg.data.batch_size),
        "run_dir": run_dir,
    }
    _write_trial_record(cfg, run_dir, record)

    return model, best_auc, best_step