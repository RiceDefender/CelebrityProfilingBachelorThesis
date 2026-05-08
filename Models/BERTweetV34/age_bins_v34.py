import json
import os
from typing import Dict, Iterable, List

import numpy as np

N_AGE_BINS = 8
AGE_BIN_LABELS = [f"age_bin_{i}" for i in range(N_AGE_BINS)]


def _as_year(value) -> int:
    return int(float(value))


def compute_quantile_age_bins(years: Iterable[int], n_bins: int = N_AGE_BINS) -> List[Dict]:
    """Compute train-only quantile bins over birthyear values.

    Older birth years receive lower bin ids. Boundaries are inclusive.
    Duplicate boundaries are allowed in extremely small data, but the PAN train
    set is large enough that this should not happen in practice.
    """
    years = sorted(_as_year(y) for y in years)
    if not years:
        raise ValueError("Cannot compute age bins without birthyear values.")

    quantiles = np.linspace(0, 1, n_bins + 1)
    raw_edges = np.quantile(years, quantiles, method="nearest").astype(int).tolist()

    bins = []
    for i in range(n_bins):
        min_year = int(raw_edges[i])
        max_year = int(raw_edges[i + 1])

        # Make bins non-overlapping and inclusive by nudging interior starts.
        if i > 0:
            min_year = bins[-1]["max_year"] + 1

        # If many identical years cause an invalid interval, keep a narrow bin.
        if max_year < min_year:
            max_year = min_year

        bins.append({
            "label": f"age_bin_{i}",
            "min_year": min_year,
            "max_year": max_year,
            "description": f"{min_year}-{max_year}",
        })

    # Extend endpoints so unseen test years outside train range still map.
    bins[0]["min_year"] = -10**9
    bins[-1]["max_year"] = 10**9
    bins[0]["description"] = f"<= {bins[0]['max_year']}"
    bins[-1]["description"] = f">= {bins[-1]['min_year']}"

    return bins


def save_age_bins(bins: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "target": "birthyear_8range",
        "method": "train_quantiles",
        "n_bins": len(bins),
        "bins": bins,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_age_bins(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing age-bin file: {path}. Train birthyear_8range first, "
            "or generate bins from train labels before prediction/evaluation."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["bins"]


def map_year_to_age_bin(year, bins: List[Dict]) -> str:
    year = _as_year(year)
    for item in bins:
        if int(item["min_year"]) <= year <= int(item["max_year"]):
            return item["label"]
    # Should be unreachable because endpoints are extended.
    raise ValueError(f"Birthyear {year} did not fit any age bin: {bins}")


def age_bin_display_name(label: str, bins: List[Dict]) -> str:
    for item in bins:
        if item["label"] == label:
            return f"{label} ({item['description']})"
    return label
