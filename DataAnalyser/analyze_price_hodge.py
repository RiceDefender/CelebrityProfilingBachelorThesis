import ast
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

import _constants as C


# --------------------------------------------------
# Config
# --------------------------------------------------
# Official PAN uses a lenient age score, but the exact m-value is not
# explicit in the loaded snippets. Keep it configurable.
AGE_TOLERANCE = 3

GENDER_LABELS = ["female", "male"]
OCCUPATION_LABELS = ["sports", "performer", "creator", "politics"]


# --------------------------------------------------
# IO helpers
# --------------------------------------------------
def ensure_dirs() -> None:
    os.makedirs(C.comparison_dir, exist_ok=True)
    os.makedirs(C.comparison_plots_dir, exist_ok=True)
    os.makedirs(C.comparison_tables_dir, exist_ok=True)


def load_ndjson(path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    if "id" not in df.columns:
        raise ValueError(f"'id' column missing in {path}")

    df["id"] = df["id"].astype(str)
    if "birthyear" in df.columns:
        df["birthyear"] = pd.to_numeric(df["birthyear"], errors="coerce").astype("Int64")

    return df


def load_diff_pairs(path: str) -> pd.DataFrame:
    """
    Parses lines like:
    {true_json} | {pred_json}
    Returns one row per mismatch.
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "|" not in line:
                continue

            left, right = line.split("|", 1)
            left = left.strip()
            right = right.strip()

            true_obj = ast.literal_eval(left)
            pred_obj = ast.literal_eval(right)

            rows.append(
                {
                    "id": str(true_obj["id"]),
                    "true_gender": true_obj["gender"],
                    "pred_gender": pred_obj["gender"],
                    "true_occupation": true_obj["occupation"],
                    "pred_occupation": pred_obj["occupation"],
                    "true_birthyear": int(true_obj["birthyear"]),
                    "pred_birthyear": int(pred_obj["birthyear"]),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------
# Merge / validation
# --------------------------------------------------
def build_comparison_df(
    gt_path: str,
    pred_path: str,
) -> pd.DataFrame:
    gt = load_ndjson(gt_path).rename(
        columns={
            "gender": "true_gender",
            "occupation": "true_occupation",
            "birthyear": "true_birthyear",
        }
    )

    pred = load_ndjson(pred_path).rename(
        columns={
            "gender": "pred_gender",
            "occupation": "pred_occupation",
            "birthyear": "pred_birthyear",
        }
    )

    df = gt.merge(pred, on="id", how="inner", validate="one_to_one")

    missing_gt = set(gt["id"]) - set(df["id"])
    missing_pred = set(pred["id"]) - set(df["id"])

    if missing_gt:
        print(f"[WARN] Missing predictions for {len(missing_gt)} ground-truth ids")
    if missing_pred:
        print(f"[WARN] Extra prediction ids not in ground truth: {len(missing_pred)}")

    return df


def validate_diff_against_full_predictions(df: pd.DataFrame, diff_df: pd.DataFrame) -> None:
    """
    Sanity check:
    - every diff row should be a mismatch in the merged full dataframe
    - number of diff rows should equal number of rows where at least one task mismatches
    """
    merged = df.copy()

    merged["any_mismatch"] = (
        (merged["true_gender"] != merged["pred_gender"])
        | (merged["true_occupation"] != merged["pred_occupation"])
        | (merged["true_birthyear"] != merged["pred_birthyear"])
    )

    mismatch_ids = set(merged.loc[merged["any_mismatch"], "id"])
    diff_ids = set(diff_df["id"].astype(str))

    missing_in_diff = mismatch_ids - diff_ids
    extra_in_diff = diff_ids - mismatch_ids

    print("\n[DIFF VALIDATION]")
    print(f"  mismatching ids in full predictions: {len(mismatch_ids)}")
    print(f"  ids in diff.txt:                    {len(diff_ids)}")
    print(f"  missing in diff.txt:               {len(missing_in_diff)}")
    print(f"  extra in diff.txt:                 {len(extra_in_diff)}")


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def age_lenient_hits(y_true: pd.Series, y_pred: pd.Series, tolerance: int) -> pd.Series:
    return (y_true - y_pred).abs() <= tolerance


def age_lenient_f1(y_true: pd.Series, y_pred: pd.Series, tolerance: int) -> float:
    """
    Lenient age metric:
    prediction counts as correct if |pred - true| <= tolerance.
    Then compute binary F1 over hit vs miss.
    """
    hits = age_lenient_hits(y_true, y_pred, tolerance).astype(int)
    y_ref = pd.Series([1] * len(hits))
    return f1_score(y_ref, hits, average="binary", zero_division=0)


def harmonic_mean(values: List[float]) -> float:
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def compute_metrics(df: pd.DataFrame, tolerance: int) -> Dict[str, float]:
    gender_acc = accuracy_score(df["true_gender"], df["pred_gender"])
    gender_f1_macro = f1_score(
        df["true_gender"], df["pred_gender"], average="macro", zero_division=0
    )

    occupation_acc = accuracy_score(df["true_occupation"], df["pred_occupation"])
    occupation_f1_macro = f1_score(
        df["true_occupation"], df["pred_occupation"], average="macro", zero_division=0
    )

    age_acc_exact = accuracy_score(df["true_birthyear"], df["pred_birthyear"])
    age_mae = (df["true_birthyear"] - df["pred_birthyear"]).abs().mean()
    age_f1_lenient = age_lenient_f1(df["true_birthyear"], df["pred_birthyear"], tolerance)

    c_rank = harmonic_mean([gender_f1_macro, occupation_f1_macro, age_f1_lenient])

    return {
        "n_samples": float(len(df)),
        "gender_accuracy": gender_acc,
        "gender_macro_f1": gender_f1_macro,
        "occupation_accuracy": occupation_acc,
        "occupation_macro_f1": occupation_f1_macro,
        "age_accuracy_exact": age_acc_exact,
        "age_mae": age_mae,
        "age_lenient_f1": age_f1_lenient,
        "c_rank": c_rank,
        "age_tolerance": float(tolerance),
    }


# --------------------------------------------------
# Reports / tables
# --------------------------------------------------
def save_classification_reports(df: pd.DataFrame) -> None:
    gender_report = classification_report(
        df["true_gender"],
        df["pred_gender"],
        labels=GENDER_LABELS,
        output_dict=True,
        zero_division=0,
    )
    occupation_report = classification_report(
        df["true_occupation"],
        df["pred_occupation"],
        labels=OCCUPATION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(gender_report).T.to_csv(
        os.path.join(C.comparison_tables_dir, "gender_classification_report.csv")
    )
    pd.DataFrame(occupation_report).T.to_csv(
        os.path.join(C.comparison_tables_dir, "occupation_classification_report.csv")
    )


def save_metrics(metrics: Dict[str, float]) -> None:
    pd.DataFrame([metrics]).to_csv(
        os.path.join(C.comparison_tables_dir, "summary_metrics.csv"),
        index=False,
    )


def save_error_tables(df: pd.DataFrame, tolerance: int) -> None:
    out = df.copy()
    out["gender_correct"] = out["true_gender"] == out["pred_gender"]
    out["occupation_correct"] = out["true_occupation"] == out["pred_occupation"]
    out["birthyear_correct_exact"] = out["true_birthyear"] == out["pred_birthyear"]
    out["birthyear_abs_error"] = (out["true_birthyear"] - out["pred_birthyear"]).abs()
    out["birthyear_correct_lenient"] = out["birthyear_abs_error"] <= tolerance
    out["any_error"] = ~(
        out["gender_correct"] & out["occupation_correct"] & out["birthyear_correct_exact"]
    )

    out.to_csv(os.path.join(C.comparison_tables_dir, "merged_predictions.csv"), index=False)
    out.loc[out["any_error"]].to_csv(
        os.path.join(C.comparison_tables_dir, "error_cases.csv"), index=False
    )

    age_error_dist = (
        out["birthyear_abs_error"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("abs_error_years")
        .reset_index(name="count")
    )
    age_error_dist.to_csv(
        os.path.join(C.comparison_tables_dir, "birthyear_absolute_error_distribution.csv"),
        index=False,
    )


# --------------------------------------------------
# Plots
# --------------------------------------------------
def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: List[str],
    title: str,
    filename: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(C.comparison_plots_dir, filename), dpi=200)
    plt.close(fig)


def save_birthyear_error_histogram(df: pd.DataFrame, filename: str) -> None:
    errors = (df["true_birthyear"] - df["pred_birthyear"]).abs()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(errors, bins=30)
    ax.set_title("Absolute Birthyear Error Distribution")
    ax.set_xlabel("Absolute error in years")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(C.comparison_plots_dir, filename), dpi=200)
    plt.close(fig)


def save_prediction_distribution_plot(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    title: str,
    filename: str,
    order: List[str],
) -> None:
    true_counts = df[true_col].value_counts().reindex(order, fill_value=0)
    pred_counts = df[pred_col].value_counts().reindex(order, fill_value=0)

    plot_df = pd.DataFrame({
        "True": true_counts,
        "Predicted": pred_counts,
    })

    ax = plot_df.plot(kind="bar", figsize=(8, 4.5))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(C.comparison_plots_dir, filename), dpi=200)
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    ensure_dirs()

    print("[INFO] Loading ground truth and Price/Hodge predictions...")
    df = build_comparison_df(
        gt_path=C.test_label_path,
        pred_path=C.price_hodge_label_path,
    )

    print(f"[INFO] Merged rows: {len(df)}")

    print("[INFO] Loading diff file for sanity check...")
    diff_df = load_diff_pairs(C.price_hodge_diff_path)
    validate_diff_against_full_predictions(df, diff_df)

    print("[INFO] Computing metrics...")
    metrics = compute_metrics(df, tolerance=AGE_TOLERANCE)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("[INFO] Saving tables...")
    save_metrics(metrics)
    save_classification_reports(df)
    save_error_tables(df, tolerance=AGE_TOLERANCE)

    print("[INFO] Saving plots...")
    save_confusion_matrix(
        df["true_gender"],
        df["pred_gender"],
        GENDER_LABELS,
        "Gender Confusion Matrix (Price & Hodge)",
        "gender_confusion_matrix.png",
    )
    save_confusion_matrix(
        df["true_occupation"],
        df["pred_occupation"],
        OCCUPATION_LABELS,
        "Occupation Confusion Matrix (Price & Hodge)",
        "occupation_confusion_matrix.png",
    )
    save_birthyear_error_histogram(df, "birthyear_absolute_error_histogram.png")

    save_prediction_distribution_plot(
        df,
        "true_gender",
        "pred_gender",
        "True vs Predicted Gender Distribution",
        "gender_distribution_true_vs_pred.png",
        GENDER_LABELS,
    )
    save_prediction_distribution_plot(
        df,
        "true_occupation",
        "pred_occupation",
        "True vs Predicted Occupation Distribution",
        "occupation_distribution_true_vs_pred.png",
        OCCUPATION_LABELS,
    )

    print("[DONE] Comparison finished.")


if __name__ == "__main__":
    main()