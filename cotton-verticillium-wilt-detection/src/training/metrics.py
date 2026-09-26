"""
Evaluation metrics for Cotton Verticillium Wilt Detection.
Calculates diagnostic accuracy, class-wise F1, early-detection recall, and lead-time MAE.
"""

from typing import List, Dict, Any, Tuple
import math


def calculate_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int = 4) -> List[List[int]]:
    """Compute NxN confusion matrix using pure Python."""
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def evaluate_diagnostic_metrics(
    y_true: List[int],
    y_pred: List[int],
    days_true: Optional[List[float]] = None,
    days_pred: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive diagnostic evaluation.
    """
    if class_names is None:
        class_names = ["Healthy", "Early-Stage Wilt (Pre-Visual)", "Moderate Wilt", "Severe Wilt"]

    total = len(y_true)
    if total == 0:
        return {}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = (correct / total) * 100.0

    cm = calculate_confusion_matrix(y_true, y_pred, num_classes=len(class_names))

    per_class = {}
    for c_idx, c_name in enumerate(class_names):
        tp = cm[c_idx][c_idx]
        fp = sum(cm[r][c_idx] for r in range(len(class_names)) if r != c_idx)
        fn = sum(cm[c_idx][col] for col in range(len(class_names)) if col != c_idx)
        total_class = tp + fn

        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[c_name] = {
            "samples": total_class,
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "f1_score": round(f1 / 100.0, 4)
        }

    # Early detection lead time metrics (for Class 1 samples)
    lead_time_metrics = {}
    if days_true is not None and days_pred is not None:
        early_errors = []
        for t_cls, d_t, d_p in zip(y_true, days_true, days_pred):
            if t_cls == 1 and d_t < 0: # Early-stage wilt
                early_errors.append(abs(d_t - d_p))
        if early_errors:
            lead_time_metrics["mean_absolute_error_days"] = round(sum(early_errors) / len(early_errors), 2)
            lead_time_metrics["early_stage_sample_count"] = len(early_errors)

    return {
        "overall_accuracy_pct": round(accuracy, 2),
        "total_samples": total,
        "class_breakdown": per_class,
        "confusion_matrix": cm,
        "lead_time_metrics": lead_time_metrics
    }
