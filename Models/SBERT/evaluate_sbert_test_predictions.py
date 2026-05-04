import argparse
import json
import os
from typing import List

from sklearn.metrics import accuracy_score, classification_report, f1_score

from _constants import (
    sbert_predictions_dir,
    sbert_checkpoints_dir,
)

from Models.SBERT.config_sbert_model import TARGET_LABEL


def resolve_targets(target: str) -> List[str]:
    if target == "all":
        return ["occupation", "gender", "birthyear"]
    return [target]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_test_metrics_dir() -> str:
    # parallel zu predictions/
    base_dir = os.path.dirname(sbert_predictions_dir)
    return os.path.join(base_dir, "test_metrics")


def evaluate_target(target_label: str):
    pred_path = os.path.join(
        sbert_predictions_dir,
        f"{target_label}_test_predictions.json",
    )

    ckpt_path = os.path.join(
        sbert_checkpoints_dir,
        f"{target_label}_best.pt",
    )

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    predictions = load_json(pred_path)

    if not predictions:
        raise ValueError(f"No predictions found in: {pred_path}")

    if "true_label" not in predictions[0]:
        raise ValueError(
            f"{pred_path} enthält keine true_label. "
            "Metrics können nur berechnet werden, wenn Testlabels vorhanden sind."
        )

    y_true = [str(row["true_label"]) for row in predictions]
    y_pred = [str(row["pred_label"]) for row in predictions]

    labels = sorted(set(y_true) | set(y_pred))

    # stabile Reihenfolge wie im Training, falls Checkpoint/Predictions diese Labels enthalten
    if target_label == "gender":
        ordered = ["male", "female"]
        labels = [x for x in ordered if x in labels]
    elif target_label == "occupation":
        ordered = ["sports", "performer", "creator", "politics"]
        labels = [x for x in ordered if x in labels]
    elif target_label == "birthyear":
        # funktioniert sowohl für 60 Jahre als auch für eure 5 Buckets
        labels = sorted(labels, key=lambda x: int(x))

    label_to_id = {label: idx for idx, label in enumerate(labels)}

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "target_label": target_label,
        "num_samples": len(predictions),
        "num_test": len(predictions),
        "num_labels": len(labels),
        "label_to_id": label_to_id,
        "test_macro_f1": macro_f1,
        "test_accuracy": acc,
        "classification_report": report,
    }

    # optionale Modellinfos aus Predictions übernehmen, falls vorhanden
    for key in ["pooling_strategy", "classifier_type", "normalize_inputs", "input_dim"]:
        if key in predictions[0]:
            metrics[key] = predictions[0][key]

    out_dir = get_test_metrics_dir()
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{target_label}_test_metrics.json")
    save_json(metrics, out_path)

    print(f"[OK] Saved test metrics for {target_label}: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SBERT test predictions")
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
        default=TARGET_LABEL,
    )

    args = parser.parse_args()

    for target in resolve_targets(args.target):
        evaluate_target(target)


if __name__ == "__main__":
    main()