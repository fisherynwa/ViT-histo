import numpy as np


def evaluate(probs: np.ndarray, labels: np.ndarray, csv_path: str | None = None) -> dict:
    """Compute headline metrics and (optionally) export probs+labels to CSV.

    Args:
        probs: predicted probability of class 1, shape (N,).
        labels: true binary labels, shape (N,).
        csv_path: if given, write a two-column CSV (prob,label) for R analysis.

    Returns:
        dict of accuracy, auc, and the Brier score 
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

    labels = np.asarray(labels)
    probs = np.asarray(probs)
  
    metrics = {
        "auc": float(roc_auc_score(labels, probs)),
        "brier": float(np.mean((probs - labels) ** 2)),
    }

    if csv_path is not None:
        import pandas as pd
        pd.DataFrame({"prob": probs, "label": labels}).to_csv(csv_path, index=False)

    return metrics