import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from _constants import (
    sbert_train_vectors_path,
    sbert_v2_checkpoints_dir,
    sbert_v2_predictions_dir,
    sbert_v2_metrics_dir,
)

from Models.SBERT.config_sbert_model import RANDOM_SEED, VAL_RATIO


BIRTHYEAR_BUCKETS = ["1994", "1985", "1975", "1963", "1947"]
BUCKET_VALUES = np.array([1994, 1985, 1975, 1963, 1947], dtype=np.float32)

BIRTHYEAR_REGRESSION_SAMPLE_WEIGHT_BY_BUCKET = {
    "1994": 1.5,
    "1985": 0.9,
    "1975": 0.95,
    "1963": 1.0,
    "1947": 1.8,
}

def build_regression_sample_weights(y_train_years):
    weights = []

    for year in y_train_years:
        bucket = year_to_bucket(year)
        weights.append(
            BIRTHYEAR_REGRESSION_SAMPLE_WEIGHT_BY_BUCKET.get(bucket, 1.0)
        )

    return np.asarray(weights, dtype=np.float32)

def ensure_dirs():
    os.makedirs(sbert_v2_checkpoints_dir, exist_ok=True)
    os.makedirs(sbert_v2_predictions_dir, exist_ok=True)
    os.makedirs(sbert_v2_metrics_dir, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def year_to_bucket(year):
    year = float(year)
    idx = int(np.argmin(np.abs(BUCKET_VALUES - year)))
    return BIRTHYEAR_BUCKETS[idx]


def load_chunk_data():
    rows = load_json(sbert_train_vectors_path)

    X = []
    y_year = []
    celebrity_ids = []

    for row in rows:
        X.append(np.asarray(row["embedding"], dtype=np.float32))
        y_year.append(float(row["birthyear"]))
        celebrity_ids.append(str(row["celebrity_id"]))

    return np.stack(X), np.asarray(y_year, dtype=np.float32), celebrity_ids


def aggregate_predictions(celebrity_ids, true_years, pred_years):
    grouped_pred = defaultdict(list)
    grouped_true = {}

    for cid, true_y, pred_y in zip(celebrity_ids, true_years, pred_years):
        grouped_pred[cid].append(float(pred_y))
        grouped_true[cid] = float(true_y)

    predictions = []
    y_true_bucket = []
    y_pred_bucket = []
    y_true_year = []
    y_pred_year = []

    for cid in sorted(grouped_pred.keys()):
        pred_year = float(np.mean(grouped_pred[cid]))
        true_year = grouped_true[cid]

        pred_bucket = year_to_bucket(pred_year)
        true_bucket = year_to_bucket(true_year)

        predictions.append({
            "celebrity_id": cid,
            "true_birthyear": true_year,
            "pred_birthyear": pred_year,
            "true_label": true_bucket,
            "pred_label": pred_bucket,
            "num_chunks": len(grouped_pred[cid]),
            "version": "sbert_v2_birthyear_regression",
            "regressor_type": "ridge",
        })

        y_true_year.append(true_year)
        y_pred_year.append(pred_year)
        y_true_bucket.append(true_bucket)
        y_pred_bucket.append(pred_bucket)

    return predictions, y_true_year, y_pred_year, y_true_bucket, y_pred_bucket


def main():
    ensure_dirs()

    X, y_year, celebrity_ids = load_chunk_data()

    unique_celebrities = sorted(set(celebrity_ids))
    celeb_to_year = {}

    for cid, year in zip(celebrity_ids, y_year):
        celeb_to_year[cid] = year

    celeb_bucket_labels = [year_to_bucket(celeb_to_year[cid]) for cid in unique_celebrities]

    train_cids, val_cids = train_test_split(
        unique_celebrities,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=celeb_bucket_labels,
    )

    train_cids = set(train_cids)
    val_cids = set(val_cids)

    train_mask = np.asarray([cid in train_cids for cid in celebrity_ids])
    val_mask = np.asarray([cid in val_cids for cid in celebrity_ids])

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y_year[train_mask], y_year[val_mask]

    val_ids = [cid for cid, keep in zip(celebrity_ids, val_mask) if keep]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    sample_weights = build_regression_sample_weights(y_train)

    reg = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    reg.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    pred_val_years = reg.predict(X_val)

    predictions, y_true_year, y_pred_year, y_true_bucket, y_pred_bucket = aggregate_predictions(
        celebrity_ids=val_ids,
        true_years=y_val,
        pred_years=pred_val_years,
    )

    labels = BIRTHYEAR_BUCKETS

    metrics = {
        "target_label": "birthyear",
        "version": "sbert_v2_birthyear_regression",
        "regressor_type": "random_forest",
        "alpha": 10.0,
        "num_train_chunks": int(len(X_train)),
        "num_val_chunks": int(len(X_val)),
        "num_train_celebrities": int(len(train_cids)),
        "num_val_celebrities": int(len(val_cids)),
        "year_mae": float(mean_absolute_error(y_true_year, y_pred_year)),
        "year_rmse": float(np.sqrt(mean_squared_error(y_true_year, y_pred_year))),
        "bucket_accuracy": float(accuracy_score(y_true_bucket, y_pred_bucket)),
        "bucket_macro_f1": float(f1_score(
            y_true_bucket,
            y_pred_bucket,
            labels=labels,
            average="macro",
            zero_division=0,
        )),
        "classification_report": classification_report(
            y_true_bucket,
            y_pred_bucket,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    with open(os.path.join(sbert_v2_checkpoints_dir, "birthyear_regression_random_forest.pkl"), "wb") as f:
        pickle.dump({
            "regressor": reg,
            "scaler": scaler,
            "bucket_values": BUCKET_VALUES,
            "bucket_labels": BIRTHYEAR_BUCKETS,
        }, f)

    with open(os.path.join(sbert_v2_predictions_dir, "birthyear_regression_rf_val_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(os.path.join(sbert_v2_metrics_dir, "birthyear_regression_rf_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[RESULT] val_bucket_acc={metrics['bucket_accuracy']:.4f} val_bucket_macro_f1={metrics['bucket_macro_f1']:.4f}")
    print(f"[RESULT] year_mae={metrics['year_mae']:.2f} year_rmse={metrics['year_rmse']:.2f}")


if __name__ == "__main__":
    main()