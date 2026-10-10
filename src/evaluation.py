"""Métricas e persistência de resultados."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from src.config import METRICS

def score_percent(y_true, y_pred, y_proba=None) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out = {
        "Accuracy_%": accuracy_score(y_true, y_pred) * 100,
        "Precision_%": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall_%": recall_score(y_true, y_pred, zero_division=0) * 100,
        "F1_%": f1_score(y_true, y_pred, zero_division=0) * 100,
        "Especificidade_%": tn/(tn+fp) * 100,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if y_proba is not None:
        out["ROC_AUC_%"] = roc_auc_score(y_true, y_proba) * 100
    return out

def save_metrics(df: pd.DataFrame, filename: str) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS / filename, index=False)
