"""
Feature engineering + model for the 30-day-ahead wildfire danger outlook.

One row of weekly data per forecast area becomes one training example:
features describe the last ~4 weeks of weather/soil plus the current CPC
monthly outlook; the label is whether a "fire week" (>= FIRE_MIN_DETECTIONS
satellite fire detections in that area that week) happens anywhere in the
4 weeks starting ~30 days later. That's the "long lead" part: the model
only sees information that would have actually been available 30 days
before the thing it's predicting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

# How many satellite fire detections in an area in one week counts as a
# "fire week" for labeling purposes. See NOTE below on calibrating this.
FIRE_MIN_DETECTIONS = 3

LEAD_WEEKS = 4  # ~30 days
LOOKBACK_WEEKS = 4

CAT_MAP = {"below": -1, "normal": 0, "above": 1}

FEATURE_COLUMNS = [
    "t2m_max_4wk_avg", "precip_4wk_sum", "gwetroot_latest", "gwetroot_trend",
    "cpc_temp_prob", "cpc_temp_cat_num", "cpc_precip_prob", "cpc_precip_cat_num",
    "week_of_year_sin", "week_of_year_cos", "area_climatology",
]


def _fire_week_rate_by_month(df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Per-area, per-calendar-month historical rate of fire weeks. Used both
    as a feature (does this area/month usually see fire activity?) and as the
    climatology baseline we compare the model against (Brier skill score)."""
    tmp = df.copy()
    tmp["fire_week"] = (tmp["fire_detections"] >= FIRE_MIN_DETECTIONS).astype(int)
    tmp["month"] = tmp["week_start"].dt.month
    rates = tmp.groupby(["area_id", "month"])["fire_week"].mean()
    return rates.to_dict()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: raw weekly per-area data (area_id, week_start, t2m_max, precip,
    gwetroot, cpc_*, fire_detections). Returns one row per (area, week) with
    engineered features + the forward-looking label, sorted chronologically
    within each area."""
    df = df.sort_values(["area_id", "week_start"]).reset_index(drop=True)
    df["fire_week"] = (df["fire_detections"] >= FIRE_MIN_DETECTIONS).astype(int)
    climatology = _fire_week_rate_by_month(df)

    rows = []
    for area_id, g in df.groupby("area_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        for i in range(n):
            lo = max(0, i - LOOKBACK_WEEKS + 1)
            window = g.iloc[lo : i + 1]
            if len(window) < 2:
                continue  # not enough history yet
            future = g.iloc[i + 1 : i + 1 + LEAD_WEEKS]
            if len(future) < LEAD_WEEKS:
                continue  # not enough future data to know the label (end of series)

            month = g.loc[i, "week_start"].month
            woy = g.loc[i, "week_start"].isocalendar().week

            rows.append(
                {
                    "area_id": area_id,
                    "week_start": g.loc[i, "week_start"],
                    "t2m_max_4wk_avg": window["t2m_max"].mean(),
                    "precip_4wk_sum": window["precip"].sum(),
                    "gwetroot_latest": g.loc[i, "gwetroot"],
                    "gwetroot_trend": g.loc[i, "gwetroot"] - window["gwetroot"].iloc[0],
                    "cpc_temp_prob": g.loc[i, "cpc_temp_prob"] or 40.0,
                    "cpc_temp_cat_num": CAT_MAP.get(g.loc[i, "cpc_temp_cat"], 0),
                    "cpc_precip_prob": g.loc[i, "cpc_precip_prob"] or 40.0,
                    "cpc_precip_cat_num": CAT_MAP.get(g.loc[i, "cpc_precip_cat"], 0),
                    "week_of_year_sin": np.sin(2 * np.pi * woy / 52),
                    "week_of_year_cos": np.cos(2 * np.pi * woy / 52),
                    "area_climatology": climatology.get((area_id, month), 0.0),
                    "label": int(future["fire_detections"].ge(FIRE_MIN_DETECTIONS).any()),
                }
            )
    return pd.DataFrame(rows)


def fire_week_frequency(df: pd.DataFrame) -> float:
    """Fraction of (area, week) rows that qualify as a fire week. Used to
    sanity-check FIRE_MIN_DETECTIONS - if this is near 1.0, the threshold is
    too low (everything looks dangerous); near 0 with zero positive weeks
    anywhere, it's too high."""
    fire_week = df["fire_detections"] >= FIRE_MIN_DETECTIONS
    return float(fire_week.mean())


def train_test_split_chronological(features: pd.DataFrame, test_fraction: float = 0.2):
    """Splits by week_start (not randomly) so the test set is always later in
    time than training - the realistic way to evaluate a forecast model."""
    cutoff = features["week_start"].quantile(1 - test_fraction)
    train = features[features["week_start"] < cutoff]
    test = features[features["week_start"] >= cutoff]
    return train, test


def train_model(features: pd.DataFrame) -> dict:
    train, test = train_test_split_chronological(features)
    if train["label"].nunique() < 2 or len(test) == 0:
        raise ValueError("Not enough label variety or test data to train/evaluate a model.")

    X_train, y_train = train[FEATURE_COLUMNS], train["label"]
    X_test, y_test = test[FEATURE_COLUMNS], test["label"]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else float("nan")
    brier = brier_score_loss(y_test, probs)

    baseline_probs = test["area_climatology"].fillna(y_train.mean())
    brier_baseline = brier_score_loss(y_test, baseline_probs)
    brier_skill = 1 - (brier / brier_baseline) if brier_baseline > 0 else float("nan")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "model": model,
        "metrics": {
            "roc_auc": None if np.isnan(roc_auc) else round(float(roc_auc), 3),
            "brier_score": round(float(brier), 4),
            "brier_score_baseline": round(float(brier_baseline), 4),
            "brier_skill_score": None if np.isnan(brier_skill) else round(float(brier_skill), 3),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "positive_rate": round(float(features["label"].mean()), 4),
        },
        "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in top_features],
    }


def danger_class(prob: float) -> str:
    if prob < 0.1:
        return "Low"
    if prob < 0.25:
        return "Moderate"
    if prob < 0.45:
        return "High"
    if prob < 0.65:
        return "Very High"
    return "Extreme"


def predict_latest(model, features: pd.DataFrame) -> pd.DataFrame:
    """One row per area: its most recent feature row + predicted 30-day-ahead
    fire-danger probability and class."""
    latest = features.sort_values("week_start").groupby("area_id").tail(1).reset_index(drop=True)
    if latest.empty:
        return latest
    probs = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest = latest.copy()
    latest["danger_prob"] = probs
    latest["danger_class"] = latest["danger_prob"].apply(danger_class)
    return latest
