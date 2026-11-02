import sys
import numpy as np
from scipy import ndimage


def detection_counts(prediction, reference, min_dice):
    """TP/FP/FN for 1 study, matching connected components by their Dice.

    prediction, reference: binary foreground masks of the same shape
    min_dice: a predicted and a reference lesion match when their Dice exceeds this; 0 counts any
        overlapping voxel as a match
    -> (tp, fp, fn) ints
    """
    pred_labels, n_pred = ndimage.label(prediction)
    ref_labels, n_ref = ndimage.label(reference)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)
    ref_sizes = np.bincount(ref_labels.ravel(), minlength=n_ref + 1)

    tp = 0
    matched_pred_ids = set()
    for ref_id in range(1, n_ref + 1):
        pred_ids_under_ref = pred_labels[ref_labels == ref_id]
        intersections = np.bincount(pred_ids_under_ref, minlength=n_pred + 1)
        ref_hit = False
        for pred_id in np.flatnonzero(intersections[1:]) + 1:
            lesion_dice = 2 * intersections[pred_id] / (ref_sizes[ref_id] + pred_sizes[pred_id])
            if lesion_dice > min_dice:
                matched_pred_ids.add(pred_id)
                ref_hit = True
        tp += ref_hit

    fn = n_ref - tp
    fp = n_pred - len(matched_pred_ids)
    return tp, fp, fn


def score_case(prediction, reference, min_dice):
    """Dice and lesion detection counts for one study.

    prediction, reference: binary lesion masks of the same shape
    min_dice: lesion Dice a predicted lesion needs for a hit; the `*_any` counts use any overlap
    -> dict: positive, dice (NaN for a lesion-free reference), tp/fp/fn, tp/fp/fn_any
    """
    prediction = prediction > 0
    reference = reference > 0
    tp, fp, fn = detection_counts(prediction, reference, min_dice)
    tp_any, fp_any, fn_any = detection_counts(prediction, reference, 0.0)

    positive = bool(reference.any())
    overlap_voxels = np.logical_and(prediction, reference).sum()
    mask_voxels = prediction.sum() + reference.sum()
    dice = 2 * overlap_voxels / mask_voxels if positive else np.nan
    return {
        "positive": positive,
        "dice": dice,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_any": tp_any,
        "fp_any": fp_any,
        "fn_any": fn_any,
    }


def cc_dice(prediction, reference):
    """Mean Dice over ground-truth lesions (26-connected), each scored inside its Voronoi region.

    prediction, reference: binary lesion masks of the same shape
    -> CC-Dice, NaN for a lesion-free reference; CCMetrics imported here as its import disables autograd globally,
        with CuPy hidden so its GPU module, which crashes without a usable CUDA driver, is skipped
    Source: alexanderjaus/CC-Metrics
    """
    cupy_module = sys.modules.get("cupy")
    sys.modules["cupy"] = None
    from CCMetrics import CCDiceMetric
    if cupy_module is None:
        del sys.modules["cupy"]
    else:
        sys.modules["cupy"] = cupy_module

    prediction = prediction > 0
    reference = reference > 0
    if not reference.any():
        return np.nan
    metric = CCDiceMetric(use_caching=False)
    prediction_onehot = np.stack([~prediction, prediction])[None]
    reference_onehot = np.stack([~reference, reference])[None]
    metric(y_pred=prediction_onehot, y=reference_onehot)
    return float(metric.cc_aggregate(mode="patient").mean())


def detection_scores(tp, fp, fn):
    """-> (precision, recall, f1), 0 where undefined."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def cohort_metrics(per_case):
    """Dice over lesion-positive studies and pooled lesion detection scores over one group.

    per_case: `score_case` rows for this group
    -> metrics dict; `*_any` detection scores count any overlapping voxel as a hit; with a cc_dice column its
        mean over lesion-positive studies
    """
    precision, recall, f1 = detection_scores(*per_case[["tp", "fp", "fn"]].sum())
    precision_any, recall_any, f1_any = detection_scores(*per_case[["tp_any", "fp_any", "fn_any"]].sum())
    positive_dice = per_case["dice"].dropna()
    metrics = {
        "n": len(per_case),
        "n_positive": int(per_case["positive"].sum()),
        "dice": float(positive_dice.mean()),
        "dice_std": float(positive_dice.std()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_any": precision_any,
        "recall_any": recall_any,
        "f1_any": f1_any,
    }
    if "cc_dice" in per_case:
        metrics["cc_dice"] = float(per_case["cc_dice"].dropna().mean())
    return metrics


def grouped_metrics(per_case):
    """`cohort_metrics` over all studies ("pooled") and over each cohort.

    per_case: `score_case` rows with a `cohort` column
    -> {group name: metrics dict}
    """
    metrics = {"pooled": cohort_metrics(per_case)}
    for cohort, cohort_cases in per_case.groupby("cohort"):
        metrics[cohort] = cohort_metrics(cohort_cases)
    return metrics
