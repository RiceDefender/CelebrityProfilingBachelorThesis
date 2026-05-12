import argparse
import json
import os
import pickle
import sys
from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


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
    TARGETS,
    LABEL_ORDERS,
    CLASS_WEIGHT,
    SVD_N_COMPONENTS,
    SVD_RANDOM_SEED,
    SVD_C,
    SVD_MAX_ITER,
)

from Models.HybridV4.feature.train_feature_models import (
    build_examples,
    build_feature_matrix,
    get_label,
    save_json,
    save_pickle,
)


VAL_RATIO = 0.1


def ensure_dirs():
    os.makedirs(hybrid_v4_feature_models_dir, exist_ok=True)
    os.makedirs(hybrid_v4_feature_predictions_dir, exist_ok=True)
    os.makedirs(hybrid_v4_feature_metrics_dir, exist_ok=True)


def split_train_val(examples: List[dict], target: str):
    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    celebrity_ids = [ex["celebrity_id"] for ex in examples]
    example_by_id = {ex["celebrity_id"]: ex for ex in examples}
    y = [label_to_id[get_label(ex, target)] for ex in examples]

    train_ids, val_ids = train_test_split(
        celebrity_ids,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_examples = [example_by_id[cid] for cid in train_ids]
    val_examples = [example_by_id[cid] for cid in val_ids]

    return train_examples, val_examples


def fit_svd_branch(train_examples: List[dict]):
    """
    Builds sparse TF-IDF/style features using the existing HybridV4 feature pipeline,
    then compresses them with TruncatedSVD and scales the dense representation.
    """

    x_train_sparse, vectorizer, style_scaler = build_feature_matrix(
        train_examples,
        fit=True,
    )

    n_components = min(
        SVD_N_COMPONENTS,
        x_train_sparse.shape[0] - 1,
        x_train_sparse.shape[1] - 1,
    )

    if n_components < 2:
        raise ValueError(
            f"Too few dimensions for SVD: requested={SVD_N_COMPONENTS}, "
            f"available_shape={x_train_sparse.shape}"
        )

    print(f"[INFO] Sparse feature shape: {x_train_sparse.shape}")
    print(f"[INFO] SVD components:       {n_components}")

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=SVD_RANDOM_SEED,
    )

    x_train_svd = svd.fit_transform(x_train_sparse)

    dense_scaler = StandardScaler()
    x_train_dense = dense_scaler.fit_transform(x_train_svd)

    explained = float(np.sum(svd.explained_variance_ratio_))
    print(f"[INFO] SVD explained variance ratio sum: {explained:.4f}")

    return x_train_dense, vectorizer, style_scaler, svd, dense_scaler, explained


def transform_svd_branch(
    examples: List[dict],
    vectorizer,
    style_scaler,
    svd,
    dense_scaler,
):
    x_sparse, _, _ = build_feature_matrix(
        examples,
        vectorizer=vectorizer,
        scaler=style_scaler,
        fit=False,
    )

    x_svd = svd.transform(x_sparse)
    x_dense = dense_scaler.transform(x_svd)

    return x_dense


def predict_examples(
    examples: List[dict],
    target: str,
    labels: List[str],
    clf,
    vectorizer,
    style_scaler,
    svd,
    dense_scaler,
    split: str,
):
    x = transform_svd_branch(
        examples=examples,
        vectorizer=vectorizer,
        style_scaler=style_scaler,
        svd=svd,
        dense_scaler=dense_scaler,
    )

    probs = clf.predict_proba(x)
    pred_ids = np.argmax(probs, axis=1)

    id_to_label = {idx: label for idx, label in enumerate(labels)}
    pred_labels = [id_to_label[int(i)] for i in pred_ids]

    predictions = []

    for ex, prob, pred_label in zip(examples, probs, pred_labels):
        predictions.append({
            "celebrity_id": ex["celebrity_id"],
            "target": target,
            "split": split,
            "true_label": get_label(ex, target),
            "labels": labels,
            "svd_feature_probabilities": prob.tolist(),
            "svd_feature_pred_label": pred_label,
            "feature_model": "tfidf_word_char_style_svd_logreg",
        })

    return predictions


def evaluate_predictions(predictions: List[dict], target: str, split: str):
    labels = LABEL_ORDERS[target]

    y_true = [row["true_label"] for row in predictions]
    y_pred = [row["svd_feature_pred_label"] for row in predictions]

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
        "model": "tfidf_word_char_style_svd_logreg",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "classification_report": report,
        "num_celebrities": len(predictions),
    }


def train_one_target(target: str, train_all_examples: List[dict], test_examples: List[dict]):
    print(f"\n========== HybridV4 SVD feature model: target={target} ==========")

    labels = LABEL_ORDERS[target]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    train_examples, val_examples = split_train_val(train_all_examples, target)

    y_train_labels = [get_label(ex, target) for ex in train_examples]
    y_train = np.asarray(
        [label_to_id[label] for label in y_train_labels],
        dtype=np.int64,
    )

    print(f"[INFO] Train celebrities: {len(train_examples)}")
    print(f"[INFO] Val celebrities:   {len(val_examples)}")
    print(f"[INFO] Test celebrities:  {len(test_examples)}")
    print(f"[INFO] Labels:            {label_to_id}")

    (
        x_train,
        vectorizer,
        style_scaler,
        svd,
        dense_scaler,
        explained_variance,
    ) = fit_svd_branch(train_examples)

    base_clf = LogisticRegression(
        C=SVD_C,
        max_iter=SVD_MAX_ITER,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        solver="liblinear",
        verbose=0,
    )

    clf = OneVsRestClassifier(base_clf)

    clf.fit(x_train, y_train)

    val_predictions = predict_examples(
        examples=val_examples,
        target=target,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        style_scaler=style_scaler,
        svd=svd,
        dense_scaler=dense_scaler,
        split="val",
    )

    test_predictions = predict_examples(
        examples=test_examples,
        target=target,
        labels=labels,
        clf=clf,
        vectorizer=vectorizer,
        style_scaler=style_scaler,
        svd=svd,
        dense_scaler=dense_scaler,
        split="test",
    )

    val_metrics = evaluate_predictions(val_predictions, target, "val")
    test_metrics = evaluate_predictions(test_predictions, target, "test")

    for metrics in [val_metrics, test_metrics]:
        metrics["svd_n_components"] = int(svd.n_components)
        metrics["svd_explained_variance_ratio_sum"] = explained_variance

    model_payload = {
        "target": target,
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "vectorizer": vectorizer,
        "style_scaler": style_scaler,
        "svd": svd,
        "dense_scaler": dense_scaler,
        "classifier": clf,
        "svd_explained_variance_ratio_sum": explained_variance,
    }

    model_path = os.path.join(
        hybrid_v4_feature_models_dir,
        f"{target}_svd_feature_model.pkl",
    )
    val_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_val_svd_feature_probs.json",
    )
    test_pred_path = os.path.join(
        hybrid_v4_feature_predictions_dir,
        f"{target}_test_svd_feature_probs.json",
    )
    val_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{target}_val_svd_feature_metrics.json",
    )
    test_metrics_path = os.path.join(
        hybrid_v4_feature_metrics_dir,
        f"{target}_test_svd_feature_metrics.json",
    )

    save_pickle(model_payload, model_path)
    save_json(val_predictions, val_pred_path)
    save_json(test_predictions, test_pred_path)
    save_json(val_metrics, val_metrics_path)
    save_json(test_metrics, test_metrics_path)

    print(
        f"[RESULT] {target} SVD "
        f"val_acc={val_metrics['accuracy']:.4f} "
        f"val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"[OK] Saved model:      {model_path}")
    print(f"[OK] Saved val preds:  {val_pred_path}")
    print(f"[OK] Saved test preds: {test_pred_path}")


def resolve_targets(target: str) -> List[str]:
    if target == "all":
        return TARGETS
    return [target]


def main():
    parser = argparse.ArgumentParser(
        description="Train HybridV4 Koloski-inspired SVD feature models."
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