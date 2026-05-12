import argparse
import json
import os
import pickle
import sys
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from _constants import (
    train_label_path,
    train_feeds_path,
    test_label_path,
    test_feeds_path,
    hybrid_v4_feature_models_dir,
    hybrid_v4_feature_predictions_dir,
    hybrid_v4_feature_metrics_dir,
)

from Models.HybridV4.feature.config_features import (
    RANDOM_SEED,
    CLASS_WEIGHT,
    C,
    MAX_ITER,
)

from Models.HybridV4.feature.train_feature_models import (
    build_examples,
    build_feature_matrix,
    save_json,
    save_pickle,
)


VAL_RATIO = 0.1


GATE_CONFIGS = {
    "creator_binary": {
        "labels": ["not_creator", "creator"],
        "filter_occupations": None,
    },
    "creator_vs_performer": {
        "labels": ["performer", "creator"],
        "filter_occupations": {"performer", "creator"},
    },
}


def ensure_dirs():
    os.makedirs(hybrid_v4_feature_models_dir, exist_ok=True)
    os.makedirs(hybrid_v4_feature_predictions_dir, exist_ok=True)
    os.makedirs(hybrid_v4_feature_metrics_dir, exist_ok=True)


def get_gate_label(example: dict, gate: str) -> str:
    occupation = str(example["occupation"])

    if gate == "creator_binary":
        return "creator" if occupation == "creator" else "not_creator"

    if gate == "creator_vs_performer":
        if occupation not in {"creator", "performer"}:
            raise ValueError(
                f"creator_vs_performer received invalid occupation={occupation}"
            )
        return occupation

    raise ValueError(f"Unknown gate: {gate}")


def filter_examples_for_gate(examples: List[dict], gate: str) -> List[dict]:
    filter_occupations = GATE_CONFIGS[gate]["filter_occupations"]

    if filter_occupations is None:
        return examples

    return [
        ex for ex in examples
        if str(ex["occupation"]) in filter_occupations
    ]


def split_train_val_for_gate(examples: List[dict], gate: str):
    labels = GATE_CONFIGS[gate]["labels"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    celebrity_ids = [ex["celebrity_id"] for ex in examples]
    example_by_id = {ex["celebrity_id"]: ex for ex in examples}
    y = [label_to_id[get_gate_label(ex, gate)] for ex in examples]

    train_ids, val_ids = train_test_split(
        celebrity_ids,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_examples = [example_by_id[cid] for cid in train_ids]
    val_examples = [example_by_id[cid] for cid in val_ids]

    return train_examples, val_examples


def predict_gate_examples(
    examples: List[dict],
    gate: str,
    labels: List[str],
    clf,
    vectorizer,
    scaler,
    split: str,
):
    x, _, _ = build_feature_matrix(
        examples,
        vectorizer=vectorizer,
        scaler=scaler,
        fit=False,
    )

    probs = clf.predict_proba(x)
    pred_ids = np.argmax(probs, axis=1)
    id_to_label = {idx: label for idx, label in enumerate(labels)}
    pred_labels = [id_to_label[int(i)] for i in pred_ids]

    predictions = []

    for ex, prob, pred_label in zip(examples, probs, pred_labels):
        predictions.append({
            "celebrity_id": ex["celebrity_id"],
            "target": gate,
            "split": split,
            "true_label": get_gate_label(ex, gate),
            "occupation": str(ex["occupation"]),
            "labels": labels,
            "feature_probabilities": prob.tolist(),
            "feature_pred_label": pred_label,
            "feature_model": "creator_gate_tfidf_word_char_style_logreg",
        })

    return predictions


def evaluate_gate_predictions(predictions: List[dict], gate: str, split: str):
    labels = GATE_CONFIGS[gate]["labels"]

    y_true = [row["true_label"] for row in predictions]
    y_pred = [row["feature_pred_label"] for row in predictions]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "target": gate,
        "split": split,
        "model": "creator_gate_tfidf_word_char_style_logreg",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "classification_report": report,
        "num_celebrities": len(predictions),
    }


def train_one_gate(gate: str, train_all_examples: List[dict], test_examples_all: List[dict]):
    print(f"\n========== HybridV4 creator gate: {gate} ==========")

    labels = GATE_CONFIGS[gate]["labels"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    train_all_filtered = filter_examples_for_gate(train_all_examples, gate)
    test_filtered = filter_examples_for_gate(test_examples_all, gate)

    train_examples, val_examples = split_train_val_for_gate(
        train_all_filtered,
        gate,
    )

    y_train_labels = [get_gate_label(ex, gate) for ex in train_examples]
    y_train = np.asarray(
        [label_to_id[label] for label in y_train_labels],
        dtype=np.int64,
    )

    print(f"[INFO] Train celebrities: {len(train_examples)}")
    print(f"[INFO] Val celebrities:   {len(val_examples)}")
    print(f"[INFO] Test celebrities:  {len(test_filtered)}")
    print(f"[INFO] Labels:            {label_to_id}")

    x_train, vectorizer, scaler = build_feature_matrix(
        train_examples,
        fit=True,
    )

    print(f"[INFO] Feature shape:     {x_train.shape}")

    clf = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        solver="liblinear",
        verbose=0,
    )

    clf.fit(x_train, y_train)

    val_predictions = predict_gate_examples(
        examples=val_examples,
        gate=gate,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        split="val",
    )

    test_predictions = predict_gate_examples(
        examples=test_filtered,
        gate=gate,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        split="test",
    )

    val_metrics = evaluate_gate_predictions(val_predictions, gate, "val")
    test_metrics = evaluate_gate_predictions(test_predictions, gate, "test")

    model_payload = {
        "target": gate,
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "classifier": clf,
    }

    model_path = os.path.join(
        hybrid_v4_feature_models_dir,
        f"{gate}_feature_model.pkl",
    )
    val_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{gate}_val_feature_probs.json",
    )
    test_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{gate}_test_feature_probs.json",
    )
    val_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{gate}_val_feature_metrics.json",
    )
    test_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{gate}_test_feature_metrics.json",
    )

    save_pickle(model_payload, model_path)
    save_json(val_predictions, val_pred_path)
    save_json(test_predictions, test_pred_path)
    save_json(val_metrics, val_metrics_path)
    save_json(test_metrics, test_metrics_path)

    print(
        f"[RESULT] {gate} "
        f"val_acc={val_metrics['accuracy']:.4f} "
        f"val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"[OK] Saved model:      {model_path}")
    print(f"[OK] Saved val preds:  {val_pred_path}")
    print(f"[OK] Saved test preds: {test_pred_path}")


def resolve_gates(gate: str) -> List[str]:
    if gate == "all":
        return ["creator_binary", "creator_vs_performer"]
    return [gate]


def main():
    parser = argparse.ArgumentParser(
        description="Train HybridV4 creator gate models."
    )
    parser.add_argument(
        "--gate",
        choices=["creator_binary", "creator_vs_performer", "all"],
        default="all",
    )

    args = parser.parse_args()

    ensure_dirs()

    train_all_examples = build_examples(
        label_path=train_label_path,
        feeds_path=train_feeds_path,
    )

    test_examples = build_examples(
        label_path=test_label_path,
        feeds_path=test_feeds_path,
    )

    for gate in resolve_gates(args.gate):
        train_one_gate(
            gate=gate,
            train_all_examples=train_all_examples,
            test_examples_all=test_examples,
        )


if __name__ == "__main__":
    main()