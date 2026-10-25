"""
Feature engineering + model for the 30-day-ahead wildfire danger outlook.

Each row is one (area, week): features describe the last ~4 weeks of
weather/soil plus the current CPC outlook; the label is whether a "fire
week" happens anywhere in the 4 weeks starting ~30 days later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

# "Fire week" = an area's own 90th-percentile weekly detection count,
# floored at 20. Adaptive per area because a flat threshold doesn't
# generalize (e.g. Texas's routine agricultural burning alone exceeds
# Colorado's wildfire-season counts).
FIRE_PERCENTILE = 0.90
FIRE_MIN_DETECTIONS = 20

LEAD_WEEKS = 4  # ~30 days
LOOKBACK_WEEKS = 4

CAT_MAP = {"below": -1, "normal": 0, "above": 1}

FEATURE_COLUMNS = [
    "t2m_max_4wk_avg", "precip_4wk_sum", "gwetroot_latest", "gwetroot_trend",
    "cpc_temp_prob", "cpc_temp_cat_num", "cpc_precip_prob", "cpc_precip_cat_num",
    "week_of_year_sin", "week_of_year_cos", "area_climatology",
]


def area_fire_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """One fire-week threshold per area, from its own detection history."""
    p = df.groupby("area_id")["fire_detections"].quantile(FIRE_PERCENTILE)
    return p.clip(lower=FIRE_MIN_DETECTIONS).to_dict()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (area, week): engineered features + forward-looking
    label. area_climatology is NOT included here - it's attached after the
    train/test split (see train_model) so it can't leak test-period labels
    into training features."""
    df = df.sort_values(["area_id", "week_start"]).reset_index(drop=True)
    thresholds = area_fire_thresholds(df)

    rows = []
    for area_id, g in df.groupby("area_id"):
        g = g.reset_index(drop=True)
        threshold = thresholds[area_id]
        for i in range(len(g)):
            lo = max(0, i - LOOKBACK_WEEKS + 1)
            window = g.iloc[lo : i + 1]
            if len(window) < 2:
                continue  # not enough history yet
            future = g.iloc[i + 1 : i + 1 + LEAD_WEEKS]
            if len(future) < LEAD_WEEKS:
                continue  # end of series, no label available

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
                    "label": int(future["fire_detections"].ge(threshold).any()),
                }
            )
    return pd.DataFrame(rows)


def _month_rate_map(features: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Per-area, per-month historical rate of the fire-week label."""
    tmp = features.copy()
    tmp["month"] = tmp["week_start"].dt.month
    return tmp.groupby(["area_id", "month"])["label"].mean().to_dict()


def _attach_climatology(features: pd.DataFrame, clim_map: dict, default: float) -> pd.DataFrame:
    features = features.copy()
    months = features["week_start"].dt.month
    features["area_climatology"] = [clim_map.get((a, m), default) for a, m in zip(features["area_id"], months)]
    return features


def fire_week_frequency(df: pd.DataFrame) -> float:
    """Fraction of (area, week) rows that qualify as a fire week."""
    thresholds = area_fire_thresholds(df)
    threshold_per_row = df["area_id"].map(thresholds)
    return float((df["fire_detections"] >= threshold_per_row).mean())


def train_test_split_chronological(features: pd.DataFrame, test_fraction: float = 0.2):
    """Splits by week_start (not randomly), so test is always later than train."""
    cutoff = features["week_start"].quantile(1 - test_fraction)
    train = features[features["week_start"] < cutoff]
    test = features[features["week_start"] >= cutoff]
    return train, test


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42)


def train_model(features: pd.DataFrame) -> dict:
    train, test = train_test_split_chronological(features)
    if train["label"].nunique() < 2 or len(test) == 0:
        # Too little fire activity to train on - fall back to climatology.
        return _climatology_fallback(features, train, test)

    # Leak-free evaluation: climatology fit on train rows only, applied to both splits.
    train_clim_map = _month_rate_map(train)
    train_default = float(train["label"].mean())
    train_eval = _attach_climatology(train, train_clim_map, train_default)
    test_eval = _attach_climatology(test, train_clim_map, train_default)

    eval_model = _make_rf()
    eval_model.fit(train_eval[FEATURE_COLUMNS], train_eval["label"])

    probs = eval_model.predict_proba(test_eval[FEATURE_COLUMNS])[:, 1]
    y_test = test_eval["label"]
    roc_auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else float("nan")
    brier = brier_score_loss(y_test, probs)
    brier_baseline = brier_score_loss(y_test, test_eval["area_climatology"])
    brier_skill = 1 - (brier / brier_baseline) if brier_baseline > 0 else float("nan")

    # Deployed model: refit on everything with full-history climatology
    # (legitimate - no future to leak from at serving time).
    full_clim_map = _month_rate_map(features)
    features_full = _attach_climatology(features, full_clim_map, float(features["label"].mean()))
    final_model = _make_rf()
    final_model.fit(features_full[FEATURE_COLUMNS], features_full["label"])

    importances = dict(zip(FEATURE_COLUMNS, final_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "model": final_model,
        "features": features_full,
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


class ClimatologyModel:
    """Fallback for states with too little fire activity to train on:
    returns each row's area_climatology as its probability."""

    feature_importances_ = np.array([1.0 if f == "area_climatology" else 0.0 for f in FEATURE_COLUMNS])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["area_climatology"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        return np.column_stack([1 - p, p])


def _climatology_fallback(features: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = ClimatologyModel()
    metrics = {
        "roc_auc": None,
        "brier_score": None,
        "brier_score_baseline": None,
        "brier_skill_score": None,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positive_rate": round(float(features["label"].mean()), 4) if len(features) else 0.0,
        "note": "too little real fire activity in this window to train a model - showing area/season climatology instead",
    }
    if len(train) > 0 and len(test) > 0 and test["label"].nunique() > 1:
        train_clim_map = _month_rate_map(train)
        test_eval = _attach_climatology(test, train_clim_map, float(train["label"].mean()))
        probs = model.predict_proba(test_eval[FEATURE_COLUMNS])[:, 1]
        metrics["brier_score"] = round(float(brier_score_loss(test_eval["label"], probs)), 4)
        metrics["roc_auc"] = round(float(roc_auc_score(test_eval["label"], probs)), 3)

    full_clim_map = _month_rate_map(features) if len(features) else {}
    full_default = float(features["label"].mean()) if len(features) else 0.0
    features_full = _attach_climatology(features, full_clim_map, full_default)

    return {
        "model": model,
        "features": features_full,
        "metrics": metrics,
        "feature_importance": [{"feature": "area_climatology", "importance": 1.0}],
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
    """One row per area: its most recent feature row + predicted probability/class."""
    latest = features.sort_values("week_start").groupby("area_id").tail(1).reset_index(drop=True)
    if latest.empty:
        return latest
    latest = latest.copy()
    latest["danger_prob"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["danger_class"] = latest["danger_prob"].apply(danger_class)
    return latest
