"""
plot_profile_relationships.py

Visualize relationships between PAN celebrity profile dimensions:
occupation, gender, and birthyear.

Typical usage from project root:

    python -m DataAnalyser.plot_profile_relationships --source labels --split test

    python -m DataAnalyser.plot_profile_relationships --source predictions --split test

    python -m DataAnalyser.plot_profile_relationships --source compare --split test

    python -m DataAnalyser.plot_profile_relationships --source labels --split train

Outputs:
    plots/profile_relationships/<source>/<split>/

This script is intentionally conservative: it does not claim to explain BERTweet internals.
It visualizes dataset structure, prediction structure, and error slices across targets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


# ---------------------------------------------------------------------
# Import project constants robustly
# ---------------------------------------------------------------------
try:
    from _constants import (  # type: ignore
        root_dir,
        plots_dir,
        train_label_path,
        test_label_path,
        bertweet_v3_predictions_dir,
    )
except Exception:
    # fallback when run outside the project package
    CURRENT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT)
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    try:
        from _constants import (  # type: ignore
            root_dir,
            plots_dir,
            train_label_path,
            test_label_path,
            bertweet_v3_predictions_dir,
        )
    except Exception:
        root_dir = os.getcwd()
        plots_dir = os.path.join(root_dir, "plots")
        train_label_path = os.path.join(root_dir, "data", "train", "labels.ndjson")
        test_label_path = os.path.join(root_dir, "data", "test", "labels.ndjson")
        bertweet_v3_predictions_dir = os.path.join(root_dir, "outputs", "bertweet_v3", "predictions")


TARGETS = ["occupation", "gender", "birthyear"]
DEFAULT_CLASSES = {
    "occupation": ["sports", "performer", "creator", "politics"],
    "gender": ["male", "female"],
    "birthyear": ["1994", "1985", "1975", "1963", "1947"],
}


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_ndjson(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Expected JSON list in {path}")


def labels_to_df(path: str) -> pd.DataFrame:
    rows = read_ndjson(path)
    out = []
    for r in rows:
        cid = str(r.get("id") or r.get("celebrity_id") or r.get("author_id") or r.get("user_id"))
        if not cid or cid == "None":
            continue
        out.append(
            {
                "celebrity_id": cid,
                "occupation": str(r.get("occupation", "")),
                "gender": str(r.get("gender", "")),
                "birthyear": str(r.get("birthyear", "")),
            }
        )
    df = pd.DataFrame(out).drop_duplicates("celebrity_id")
    return df


def prediction_path(target: str, split: str, pred_dir: Optional[str] = None) -> str:
    base = pred_dir or bertweet_v3_predictions_dir
    candidates = [
        os.path.join(base, f"{target}_{split}_predictions.json"),
        os.path.join(base, f"{target}_{split}_all_predictions.json"),
        os.path.join(base, f"{target}_{split}.json"),
        # common v3_1 / gated names can be passed via --*-pred-path if not found
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find prediction file for "
        f"target={target}, split={split}. Tried:\n" + "\n".join(candidates)
    )


def predictions_to_df(
    occupation_path: str,
    gender_path: str,
    birthyear_path: str,
    prefix: str = "pred",
) -> pd.DataFrame:
    frames = []
    for target, path in [
        ("occupation", occupation_path),
        ("gender", gender_path),
        ("birthyear", birthyear_path),
    ]:
        rows = read_json(path)
        out = []
        for r in rows:
            cid = str(r.get("celebrity_id") or r.get("id"))
            pred = str(r.get("pred_label", ""))
            true = str(r.get("true_label", ""))
            probs = r.get("probabilities")
            max_prob = None
            if isinstance(probs, list) and probs:
                try:
                    max_prob = float(max(probs))
                except Exception:
                    max_prob = None
            out.append(
                {
                    "celebrity_id": cid,
                    f"{prefix}_{target}": pred,
                    f"true_from_{target}_file": true,
                    f"{target}_confidence": max_prob,
                    f"{target}_correct": pred == true if true else None,
                }
            )
        frames.append(pd.DataFrame(out).drop_duplicates("celebrity_id"))

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="celebrity_id", how="outer")
    return df


def load_source_df(args: argparse.Namespace) -> pd.DataFrame:
    label_path = args.label_path
    if label_path is None:
        label_path = test_label_path if args.split.startswith("test") else train_label_path

    labels_df = labels_to_df(label_path)

    if args.source == "labels":
        return labels_df

    occ_path = args.occupation_pred_path or prediction_path("occupation", args.split, args.pred_dir)
    gen_path = args.gender_pred_path or prediction_path("gender", args.split, args.pred_dir)
    age_path = args.birthyear_pred_path or prediction_path("birthyear", args.split, args.pred_dir)
    pred_df = predictions_to_df(occ_path, gen_path, age_path, prefix="pred")

    if args.source == "predictions":
        return pred_df.rename(
            columns={
                "pred_occupation": "occupation",
                "pred_gender": "gender",
                "pred_birthyear": "birthyear",
            }
        )

    # compare: include both true and pred columns
    comp = labels_df.merge(pred_df, on="celebrity_id", how="inner")
    return comp


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------
def save_crosstab_tables(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_dir: str,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(df[x], df[y], dropna=False)
    row_pct = pd.crosstab(df[x], df[y], normalize="index", dropna=False) * 100
    col_pct = pd.crosstab(df[x], df[y], normalize="columns", dropna=False) * 100

    counts.to_csv(os.path.join(out_dir, f"{prefix}_{x}_x_{y}_counts.csv"))
    row_pct.to_csv(os.path.join(out_dir, f"{prefix}_{x}_x_{y}_row_percent.csv"))
    col_pct.to_csv(os.path.join(out_dir, f"{prefix}_{x}_x_{y}_col_percent.csv"))
    return counts, row_pct, col_pct


def profile_combo_table(df: pd.DataFrame, cols: List[str], out_dir: str, prefix: str) -> pd.DataFrame:
    combo = (
        df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    combo["percent"] = combo["count"] / combo["count"].sum() * 100
    combo.to_csv(os.path.join(out_dir, f"{prefix}_profile_combinations.csv"), index=False)
    return combo


def error_slice_tables(df: pd.DataFrame, out_dir: str) -> None:
    """Create error-rate tables for compare source."""
    if not all(c in df.columns for c in ["occupation", "gender", "birthyear", "pred_occupation", "pred_gender", "pred_birthyear"]):
        return

    for target in TARGETS:
        correct_col = f"{target}_is_correct"
        df[correct_col] = df[target].astype(str) == df[f"pred_{target}"].astype(str)

    slices = [
        ["occupation"],
        ["gender"],
        ["birthyear"],
        ["occupation", "gender"],
        ["occupation", "birthyear"],
        ["gender", "birthyear"],
        ["occupation", "gender", "birthyear"],
    ]
    for target in TARGETS:
        correct_col = f"{target}_is_correct"
        for cols in slices:
            agg = (
                df.groupby(cols, dropna=False)[correct_col]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"mean": "accuracy"})
            )
            agg["error_rate"] = 1.0 - agg["accuracy"]
            agg.to_csv(
                os.path.join(out_dir, f"error_rate_{target}_by_{'_'.join(cols)}.csv"),
                index=False,
            )

    # Combined exact profile correctness
    df["all_targets_correct"] = (
        (df["occupation"].astype(str) == df["pred_occupation"].astype(str))
        & (df["gender"].astype(str) == df["pred_gender"].astype(str))
        & (df["birthyear"].astype(str) == df["pred_birthyear"].astype(str))
    )
    combo_acc = (
        df.groupby(["occupation", "gender", "birthyear"], dropna=False)["all_targets_correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "all_target_accuracy"})
        .sort_values("count", ascending=False)
    )
    combo_acc.to_csv(os.path.join(out_dir, "error_rate_all_targets_by_true_profile.csv"), index=False)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def _safe_title(text: str) -> str:
    return text.replace("_", " ").title()


def plot_heatmap(table: pd.DataFrame, out_path: str, title: str, fmt: str = ".1f") -> None:
    plt.figure(figsize=(10, 7))
    if HAS_SEABORN:
        sns.heatmap(table, annot=True, fmt=fmt, cmap="Blues", cbar=True)
    else:
        plt.imshow(table.values, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
        plt.yticks(range(len(table.index)), table.index)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = table.iloc[i, j]
                text = f"{val:{fmt}}" if isinstance(val, float) else str(val)
                plt.text(j, i, text, ha="center", va="center")
    plt.title(title)
    plt.xlabel(table.columns.name or "")
    plt.ylabel(table.index.name or "")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_stacked_bar(row_pct: pd.DataFrame, out_path: str, title: str) -> None:
    ax = row_pct.plot(kind="bar", stacked=True, figsize=(11, 7), width=0.82)
    ax.set_ylabel("Percent within row group")
    ax.set_xlabel(row_pct.index.name or "")
    ax.set_title(title)
    ax.legend(title=row_pct.columns.name or "", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_crosstab_suite(df: pd.DataFrame, x: str, y: str, out_dir: str, prefix: str) -> None:
    counts, row_pct, col_pct = save_crosstab_tables(df, x, y, out_dir, prefix)

    plot_heatmap(
        counts,
        os.path.join(out_dir, f"{prefix}_{x}_x_{y}_counts_heatmap.png"),
        f"{_safe_title(prefix)}: {x} x {y} counts",
        fmt=".0f",
    )
    plot_heatmap(
        row_pct,
        os.path.join(out_dir, f"{prefix}_{x}_x_{y}_row_percent_heatmap.png"),
        f"{_safe_title(prefix)}: P({y} | {x}) in percent",
        fmt=".1f",
    )
    plot_stacked_bar(
        row_pct,
        os.path.join(out_dir, f"{prefix}_{x}_to_{y}_stacked_percent.png"),
        f"{_safe_title(prefix)}: distribution of {y} within {x}",
    )


def plot_profile_combinations(combo: pd.DataFrame, out_dir: str, prefix: str, top_k: int = 25) -> None:
    top = combo.head(top_k).copy()
    top["profile"] = top[["occupation", "gender", "birthyear"]].astype(str).agg(" | ".join, axis=1)
    plt.figure(figsize=(12, max(6, top_k * 0.35)))
    plt.barh(top["profile"][::-1], top["percent"][::-1])
    plt.xlabel("Percent")
    plt.title(f"{_safe_title(prefix)}: top profile combinations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_top_profile_combinations.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_compare_true_vs_pred_combos(df: pd.DataFrame, out_dir: str, top_k: int = 25) -> None:
    true_combo = profile_combo_table(df, ["occupation", "gender", "birthyear"], out_dir, "true")
    # Keep only prediction columns before renaming to avoid duplicate column names.
    pred_df = df[["celebrity_id", "pred_occupation", "pred_gender", "pred_birthyear"]].rename(
        columns={
            "pred_occupation": "occupation",
            "pred_gender": "gender",
            "pred_birthyear": "birthyear",
        }
    )
    pred_combo = profile_combo_table(pred_df, ["occupation", "gender", "birthyear"], out_dir, "pred")

    true_combo["profile"] = true_combo[["occupation", "gender", "birthyear"]].astype(str).agg(" | ".join, axis=1)
    pred_combo["profile"] = pred_combo[["occupation", "gender", "birthyear"]].astype(str).agg(" | ".join, axis=1)

    merged = true_combo[["profile", "percent"]].rename(columns={"percent": "true_percent"}).merge(
        pred_combo[["profile", "percent"]].rename(columns={"percent": "pred_percent"}),
        on="profile",
        how="outer",
    ).fillna(0.0)
    merged["abs_diff"] = (merged["pred_percent"] - merged["true_percent"]).abs()
    merged = merged.sort_values("abs_diff", ascending=False)
    merged.to_csv(os.path.join(out_dir, "true_vs_pred_profile_combo_percent_diff.csv"), index=False)

    top = merged.head(top_k).iloc[::-1]
    y = range(len(top))
    plt.figure(figsize=(13, max(6, top_k * 0.35)))
    plt.barh([i - 0.18 for i in y], top["true_percent"], height=0.35, label="true")
    plt.barh([i + 0.18 for i in y], top["pred_percent"], height=0.35, label="pred")
    plt.yticks(list(y), top["profile"])
    plt.xlabel("Percent")
    plt.title("True vs predicted profile combinations: largest distribution differences")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "true_vs_pred_profile_combinations_largest_diffs.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_error_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    for target in TARGETS:
        correct_col = f"{target}_is_correct"
        if correct_col not in df.columns:
            df[correct_col] = df[target].astype(str) == df[f"pred_{target}"].astype(str)
        for x, y in [("occupation", "gender"), ("occupation", "birthyear"), ("gender", "birthyear")]:
            pivot = df.pivot_table(index=x, columns=y, values=correct_col, aggfunc=lambda s: (1 - s.mean()) * 100)
            pivot = pivot.fillna(0.0)
            plot_heatmap(
                pivot,
                os.path.join(out_dir, f"error_rate_{target}_by_{x}_x_{y}.png"),
                f"{target} error rate by {x} x {y} (%)",
                fmt=".1f",
            )


def plot_confidence_correlations(df: pd.DataFrame, out_dir: str) -> None:
    conf_cols = [c for c in ["occupation_confidence", "gender_confidence", "birthyear_confidence"] if c in df.columns]
    if len(conf_cols) < 2:
        return
    corr = df[conf_cols].corr()
    corr.to_csv(os.path.join(out_dir, "prediction_confidence_correlation.csv"))
    plot_heatmap(
        corr,
        os.path.join(out_dir, "prediction_confidence_correlation.png"),
        "Prediction confidence correlations across targets",
        fmt=".2f",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot relationships between occupation, gender, and birthyear labels/predictions."
    )
    parser.add_argument("--source", choices=["labels", "predictions", "compare"], default="labels")
    parser.add_argument("--split", default="test", help="test, val, val_all, etc.")
    parser.add_argument("--label-path", default=None, help="Optional labels.ndjson path.")
    parser.add_argument("--pred-dir", default=None, help="Optional directory containing prediction JSON files.")
    parser.add_argument("--occupation-pred-path", default=None)
    parser.add_argument("--gender-pred-path", default=None)
    parser.add_argument("--birthyear-pred-path", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--top-k-combos", type=int, default=25)
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["crosstabs", "combos", "errors", "confidence"],
        choices=["crosstabs", "combos", "errors", "confidence"],
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(plots_dir, "profile_relationships", args.source, args.split)
    ensure_dir(out_dir)

    df = load_source_df(args)
    df.to_csv(os.path.join(out_dir, f"{args.source}_{args.split}_joined_data.csv"), index=False)

    # Determine which columns represent the active profile dimensions
    if args.source in ["labels", "predictions"]:
        active_cols = ["occupation", "gender", "birthyear"]
        prefix = args.source
    else:
        active_cols = ["occupation", "gender", "birthyear"]
        prefix = "true"

    # Drop rows with missing profile dimensions for relationship plots
    relationship_df = df.dropna(subset=active_cols).copy()
    for c in active_cols:
        relationship_df[c] = relationship_df[c].astype(str)

    if "crosstabs" in args.plots:
        plot_crosstab_suite(relationship_df, "occupation", "gender", out_dir, prefix)
        plot_crosstab_suite(relationship_df, "occupation", "birthyear", out_dir, prefix)
        plot_crosstab_suite(relationship_df, "gender", "birthyear", out_dir, prefix)

        if args.source == "compare":
            # Important: do not rename on the full df, because that would create
            # duplicate column names (true occupation + pred_occupation -> occupation).
            # Duplicate names make df["occupation"] a DataFrame, and pd.crosstab fails.
            pred_relationship_df = df[["celebrity_id", "pred_occupation", "pred_gender", "pred_birthyear"]].rename(
                columns={
                    "pred_occupation": "occupation",
                    "pred_gender": "gender",
                    "pred_birthyear": "birthyear",
                }
            ).dropna(subset=active_cols).copy()
            for c in active_cols:
                pred_relationship_df[c] = pred_relationship_df[c].astype(str)
            plot_crosstab_suite(pred_relationship_df, "occupation", "gender", out_dir, "pred")
            plot_crosstab_suite(pred_relationship_df, "occupation", "birthyear", out_dir, "pred")
            plot_crosstab_suite(pred_relationship_df, "gender", "birthyear", out_dir, "pred")

    if "combos" in args.plots:
        combo = profile_combo_table(relationship_df, active_cols, out_dir, prefix)
        plot_profile_combinations(combo, out_dir, prefix, args.top_k_combos)
        if args.source == "compare":
            plot_compare_true_vs_pred_combos(df, out_dir, args.top_k_combos)

    if args.source == "compare":
        error_slice_tables(df, out_dir)
        if "errors" in args.plots:
            plot_error_heatmaps(df, out_dir)
        if "confidence" in args.plots:
            plot_confidence_correlations(df, out_dir)

    summary = {
        "source": args.source,
        "split": args.split,
        "num_rows": int(len(df)),
        "out_dir": out_dir,
        "columns": list(df.columns),
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved profile relationship analysis to: {out_dir}")


if __name__ == "__main__":
    main()
