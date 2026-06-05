import argparse
import json
import os
import pickle
import sys
from typing import Dict, Iterable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier


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
    TARGETS,
    LABEL_ORDERS,
    CLASS_WEIGHT,
    C,
    MAX_ITER,
)

from Models.HybridV4.feature.train_feature_models import (
    build_examples,
    build_feature_matrix,
    get_label,
    save_json,
    save_pickle,
)


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


def load_split_ids(target: str):
    path = os.path.join(
        hybrid_v4_splits_dir,
        f"{target}_fusion_split.ndjson",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing fusion split file: {path}")

    fusion_train_ids = set()
    fusion_val_ids = set()

    for row in iter_ndjson(path):
        if row["target"] != target:
            continue

        cid = str(row["celebrity_id"])

        if row["split"] == "fusion_train":
            fusion_train_ids.add(cid)
        elif row["split"] == "fusion_val":
            fusion_val_ids.add(cid)
        else:
            raise ValueError(f"Unknown split value: {row['split']}")

    if not fusion_train_ids:
        raise ValueError(f"No fusion_train IDs found for target={target}")

    if not fusion_val_ids:
        raise ValueError(f"No fusion_val IDs found for target={target}")

    print(f"[INFO] Loaded split for {target}:")
    print(f"[INFO]   fusion_train IDs: {len(fusion_train_ids)}")
    print(f"[INFO]   fusion_val IDs:   {len(fusion_val_ids)}")

    return fusion_train_ids, fusion_val_ids


def filter_examples_by_ids(examples: List[dict], ids: set) -> List[dict]:
    return [
        ex for ex in examples
        if str(ex["celebrity_id"]) in ids
    ]


def predict_examples(
    examples: List[dict],
    target: str,
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
            "target": target,
            "split": split,
            "true_label": get_label(ex, target),
            "labels": labels,
            "feature_probabilities": prob.tolist(),
            "feature_pred_label": pred_label,
            "feature_model": "tfidf_word_char_style_logreg_fusion_split",
        })

    return predictions


def evaluate_predictions(predictions: List[dict], target: str, split: str):
    labels = LABEL_ORDERS[target]

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
        "target": target,
        "split": split,
        "model": "tfidf_word_char_style_logreg_fusion_split",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "classification_report": report,
        "num_celebrities": len(predictions),
    }


def train_one_target(
    target: str,
    train_all_examples: List[dict],
    test_examples: List[dict],
):
    print(f"\n========== HybridV4 feature model on fusion split: target={target} ==========")

    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    fusion_train_ids, fusion_val_ids = load_split_ids(target)

    fusion_train_examples = filter_examples_by_ids(
        train_all_examples,
        fusion_train_ids,
    )
    fusion_val_examples = filter_examples_by_ids(
        train_all_examples,
        fusion_val_ids,
    )

    if len(fusion_train_examples) != len(fusion_train_ids):
        print(
            f"[WARN] fusion_train examples mismatch: "
            f"examples={len(fusion_train_examples)} ids={len(fusion_train_ids)}"
        )

    if len(fusion_val_examples) != len(fusion_val_ids):
        print(
            f"[WARN] fusion_val examples mismatch: "
            f"examples={len(fusion_val_examples)} ids={len(fusion_val_ids)}"
        )

    y_train_labels = [get_label(ex, target) for ex in fusion_train_examples]
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

    base_clf = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        solver="liblinear",
        verbose=0,
    )

    clf = OneVsRestClassifier(base_clf)
    clf.fit(x_train, y_train)

    fusion_val_predictions = predict_examples(
        examples=fusion_val_examples,
        target=target,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        split="fusion_val",
    )

    test_predictions = predict_examples(
        examples=test_examples,
        target=target,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        split="test",
    )

    fusion_val_metrics = evaluate_predictions(
        fusion_val_predictions,
        target,
        "fusion_val",
    )
    test_metrics = evaluate_predictions(
        test_predictions,
        target,
        "test",
    )

    model_payload = {
        "target": target,
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "classifier": clf,
        "split_source": f"{target}_fusion_split.ndjson",
    }

    model_path = os.path.join(
        hybrid_v4_feature_models_dir,
        f"{target}_feature_model_fusion_split.pkl",
    )

    fusion_val_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_fusion_val_feature_probs.json",
    )
    test_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_test_feature_probs.json",
    )

    fusion_val_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{target}_fusion_val_feature_metrics.json",
    )
    test_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{target}_test_feature_metrics.json",
    )

    save_pickle(model_payload, model_path)
    save_json(fusion_val_predictions, fusion_val_pred_path)
    save_json(test_predictions, test_pred_path)
    save_json(fusion_val_metrics, fusion_val_metrics_path)
    save_json(test_metrics, test_metrics_path)

    print(
        f"[RESULT] {target} feature "
        f"fusion_val_acc={fusion_val_metrics['accuracy']:.4f} "
        f"fusion_val_macro_f1={fusion_val_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"[OK] Saved model:            {model_path}")
    print(f"[OK] Saved fusion val preds: {fusion_val_pred_path}")
    print(f"[OK] Saved test preds:       {test_pred_path}")


def resolve_targets(target: str) -> List[str]:
    if target == "all":
        return TARGETS
    return [target]


def main():
    parser = argparse.ArgumentParser(
        description="Train HybridV4 feature models using shared fusion split."
    )
    parser.add_argument(
        "--target",
        choices=["occupation", "gender", "birthyear", "all"],
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

    for target in resolve_targets(args.target):
        train_one_target(
            target=target,
            train_all_examples=train_all_examples,
            test_examples=test_examples,
        )


if __name__ == "__main__":
    main()