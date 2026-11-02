import argparse
import json
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from eval.lesion_metrics import cc_dice, grouped_metrics, score_case


def score_nifti_case(pred_path, label_path, lesion_label, min_dice):
    """`score_case` plus CC-Dice for one saved nnU-Net prediction.

    pred_path, label_path: label-map NIfTIs on the same grid
    lesion_label: lesion value in both label maps
    min_dice: lesion Dice a predicted lesion needs for a hit
    -> scores dict with cc_dice
    """
    prediction = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))) == lesion_label
    reference = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))) == lesion_label
    scores = score_case(prediction, reference, min_dice)
    scores["cc_dice"] = cc_dice(prediction, reference)
    return scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--min-dice", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    labels_dir = dataset_dir / "labelsTs"
    pred_dir = Path(args.pred_dir)
    cases = pd.read_csv(dataset_dir / "cases.csv")
    lesion_label = json.loads((dataset_dir / "dataset.json").read_text())["labels"]["lesion"]
    test_cases = cases[cases["split"] == "test"]

    rows = []
    for case in tqdm(test_cases.itertuples(), total=len(test_cases), desc="scoring"):
        scores = score_nifti_case(
            pred_dir / f"{case.case_id}.nii.gz", labels_dir / f"{case.case_id}.nii.gz", lesion_label, args.min_dice
        )
        rows.append({"case_id": case.case_id, "cohort": case.cohort, **scores})
    per_case = pd.DataFrame(rows)
    per_case.to_csv(pred_dir / "per_case.csv", index=False)

    groups = grouped_metrics(per_case)
    for cohort, metrics in groups.items():
        print(f"{cohort:>10}  " + "  ".join(f"{name}={value:.3f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
