import argparse
import json
import os
import sys
from typing import Iterable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


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
    hybrid_v4_splits_dir,
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


def iter_ndjson(path: str) -> Iterable[dict]:
    print(f"[INFO] Streaming: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid NDJSON in {path} at line {line_idx}: {e}"
                ) from e


def load_fusion_split_ids_for_occupation():
    path = os.path.join(
        hybrid_v4_splits_dir,
        "occupation_fusion_split.ndjson",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing fusion split file: {path}")

    fusion_train_ids = set()
    fusion_val_ids = set()

    for row in iter_ndjson(path):
        if row["target"] != "occupation":
            continue

        cid = str(row["celebrity_id"])

        if row["split"] == "fusion_train":
            fusion_train_ids.add(cid)
        elif row["split"] == "fusion_val":
            fusion_val_ids.add(cid)
        else:
            raise ValueError(f"Unknown split value: {row['split']}")

    print("[INFO] Loaded occupation fusion split:")
    print(f"[INFO]   fusion_train IDs: {len(fusion_train_ids)}")
    print(f"[INFO]   fusion_val IDs:   {len(fusion_val_ids)}")

    return fusion_train_ids, fusion_val_ids


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


def filter_examples_by_ids(examples: List[dict], ids: set) -> List[dict]:
    return [
        ex for ex in examples
        if str(ex["celebrity_id"]) in ids
    ]


def filter_examples_for_gate(examples: List[dict], gate: str) -> List[dict]:
    filter_occupations = GATE_CONFIGS[gate]["filter_occupations"]

    if filter_occupations is None:
        return examples

    return [
        ex for ex in examples
        if str(ex["occupation"]) in filter_occupations
    ]


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
            "celebrity_id": str(ex["celebrity_id"]),
            "target": gate,
            "split": split,
            "true_label": get_gate_label(ex, gate),
            "occupation": str(ex["occupation"]),
            "labels": labels,
            "feature_probabilities": prob.tolist(),
            "feature_pred_label": pred_label,
            "feature_model": "creator_gate_tfidf_word_char_style_logreg_fusion_split",
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
        "model": "creator_gate_tfidf_word_char_style_logreg_fusion_split",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "classification_report": report,
        "num_celebrities": len(predictions),
    }


def train_one_gate(
    gate: str,
    train_all_examples: List[dict],
    test_examples_all: List[dict],
):
    print(f"\n========== HybridV4 creator gate on fusion split: {gate} ==========")

    labels = GATE_CONFIGS[gate]["labels"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    fusion_train_ids, fusion_val_ids = load_fusion_split_ids_for_occupation()

    fusion_train_examples_all = filter_examples_by_ids(
        train_all_examples,
        fusion_train_ids,
    )
    fusion_val_examples_all = filter_examples_by_ids(
        train_all_examples,
        fusion_val_ids,
    )

    fusion_train_examples = filter_examples_for_gate(
        fusion_train_examples_all,
        gate,
    )
    fusion_val_examples = filter_examples_for_gate(
        fusion_val_examples_all,
        gate,
    )
    test_examples = filter_examples_for_gate(
        test_examples_all,
        gate,
    )

    y_train_labels = [get_gate_label(ex, gate) for ex in fusion_train_examples]
    y_train = np.asarray(
        [label_to_id[label] for label in y_train_labels],
        dtype=np.int64,
    )

    print(f"[INFO] Fusion train celebrities: {len(fusion_train_examples)}")
    print(f"[INFO] Fusion val celebrities:   {len(fusion_val_examples)}")
    print(f"[INFO] Official test celebrities:{len(test_examples)}")
    print(f"[INFO] Labels:                   {label_to_id}")

    x_train, vectorizer, scaler = build_feature_matrix(
        fusion_train_examples,
        fit=True,
    )

    print(f"[INFO] Feature shape:            {x_train.shape}")

    clf = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        solver="liblinear",
        verbose=0,
    )

    clf.fit(x_train, y_train)

    fusion_val_predictions = predict_gate_examples(
        examples=fusion_val_examples,
        gate=gate,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        split="fusion_val",
    )

    test_predictions = predict_gate_examples(
        examples=test_examples,
        gate=gate,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        split="test",
    )

    fusion_val_metrics = evaluate_gate_predictions(
        fusion_val_predictions,
        gate,
        "fusion_val",
    )
    test_metrics = evaluate_gate_predictions(
        test_predictions,
        gate,
        "test",
    )

    model_payload = {
        "target": gate,
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "classifier": clf,
        "split_source": "occupation_fusion_split.ndjson",
    }

    model_path = os.path.join(
        hybrid_v4_feature_models_dir,
        f"{gate}_feature_model_fusion_split.pkl",
    )

    fusion_val_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{gate}_fusion_val_feature_probs.json",
    )
    test_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{gate}_test_feature_probs.json",
    )

    fusion_val_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{gate}_fusion_val_feature_metrics.json",
    )
    test_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{gate}_test_feature_metrics.json",
    )

    save_pickle(model_payload, model_path)
    save_json(fusion_val_predictions, fusion_val_pred_path)
    save_json(test_predictions, test_pred_path)
    save_json(fusion_val_metrics, fusion_val_metrics_path)
    save_json(test_metrics, test_metrics_path)

    print(
        f"[RESULT] {gate} "
        f"fusion_val_acc={fusion_val_metrics['accuracy']:.4f} "
        f"fusion_val_macro_f1={fusion_val_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"[OK] Saved model:            {model_path}")
    print(f"[OK] Saved fusion val preds: {fusion_val_pred_path}")
    print(f"[OK] Saved test preds:       {test_pred_path}")


def resolve_gates(gate: str) -> List[str]:
    if gate == "all":
        return ["creator_binary", "creator_vs_performer"]
    return [gate]


def main():
    parser = argparse.ArgumentParser(
        description="Train HybridV4 creator gates using shared fusion split."
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