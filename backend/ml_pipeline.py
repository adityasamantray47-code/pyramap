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

# "Fire week" threshold: an area's own FIRE_PERCENTILE of its weekly VIIRS
# detection counts, floored at FIRE_MIN_DETECTIONS.
#
# A single global absolute count doesn't fit every state. Calibrated against
# real Colorado data, a flat threshold of 3 flagged 50% of all area-weeks as
# "fire weeks" (meaningless - barely better than a coin flip); 20 brought
# that to ~11%, a reasonable Colorado-specific number. But Texas's routine
# winter/spring agricultural and rangeland burning alone produces far more
# VIIRS detections per week than Colorado's wildfire season does (confirmed
# live: Texas forecast areas have a 90th-percentile weekly count of 60-680,
# vs. Colorado's 5-100) - so the flat threshold of 20 flagged ~96% of all
# Texas weeks as "dangerous," which is just as uninformative as the original
# threshold of 3 was for Colorado.
#
# Comparing each area against its OWN typical activity instead asks "is this
# an unusually active week for this specific place," which is the thing that
# actually matters for a danger signal, and adapts automatically per region
# without needing a per-state constant maintained by hand. FIRE_MIN_DETECTIONS
# is now just a floor, so a near-zero-activity area (e.g. coastal New England)
# doesn't flag on single-digit background noise.
FIRE_PERCENTILE = 0.90
FIRE_MIN_DETECTIONS = 20


def area_fire_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """One threshold per area_id, from that area's own historical
    fire_detections distribution. Uses the whole history passed in (not
    split by train/test) - this defines what "fire week" MEANS for an area,
    which is a label-definition choice, not a per-row feature value, so it
    doesn't leak test-period information into training features the way
    area_climatology did (see train_model). A test-period extreme can still
    nudge the threshold itself slightly; that's an accepted simplification,
    not the severe feature-level leak that was fixed there."""
    p = df.groupby("area_id")["fire_detections"].quantile(FIRE_PERCENTILE)
    return p.clip(lower=FIRE_MIN_DETECTIONS).to_dict()

LEAD_WEEKS = 4  # ~30 days
LOOKBACK_WEEKS = 4

CAT_MAP = {"below": -1, "normal": 0, "above": 1}

FEATURE_COLUMNS = [
    "t2m_max_4wk_avg", "precip_4wk_sum", "gwetroot_latest", "gwetroot_trend",
    "cpc_temp_prob", "cpc_temp_cat_num", "cpc_precip_prob", "cpc_precip_cat_num",
    "week_of_year_sin", "week_of_year_cos", "area_climatology",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: raw weekly per-area data (area_id, week_start, t2m_max, precip,
    gwetroot, cpc_*, fire_detections). Returns one row per (area, week) with
    engineered features + the forward-looking label, sorted chronologically
    within each area.

    Does NOT include area_climatology here - see _month_rate_map /
    _attach_climatology. It used to be computed once up front from the
    entire dataset (train+test combined), which let a test-period row's
    climatology value partially encode that same row's own label (each
    area/month combination only occurs ~2x across 2 years of history, so a
    test instance could be ~50% of its own "historical" rate). Climatology
    must be fit after the train/test split - train-only for evaluation,
    full-history for the deployed model - so it's attached by the caller
    (train_model) instead of baked in here."""
    df = df.sort_values(["area_id", "week_start"]).reset_index(drop=True)
    thresholds = area_fire_thresholds(df)

    rows = []
    for area_id, g in df.groupby("area_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        threshold = thresholds[area_id]
        for i in range(n):
            lo = max(0, i - LOOKBACK_WEEKS + 1)
            window = g.iloc[lo : i + 1]
            if len(window) < 2:
                continue  # not enough history yet
            future = g.iloc[i + 1 : i + 1 + LEAD_WEEKS]
            if len(future) < LEAD_WEEKS:
                continue  # not enough future data to know the label (end of series)

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
    """Per-area, per-calendar-month historical rate of the forward-looking
    fire-week label, from whichever rows are passed in. Callers decide
    whether that's train-only rows (leak-free evaluation) or every row
    (best available climatology for the deployed model)."""
    tmp = features.copy()
    tmp["month"] = tmp["week_start"].dt.month
    return tmp.groupby(["area_id", "month"])["label"].mean().to_dict()


def _attach_climatology(features: pd.DataFrame, clim_map: dict, default: float) -> pd.DataFrame:
    features = features.copy()
    months = features["week_start"].dt.month
    features["area_climatology"] = [clim_map.get((a, m), default) for a, m in zip(features["area_id"], months)]
    return features


def fire_week_frequency(df: pd.DataFrame) -> float:
    """Fraction of (area, week) rows that qualify as a fire week under each
    area's own adaptive threshold (area_fire_thresholds). Used to sanity-
    check that threshold - if this is near 1.0, it's too low (everything
    looks dangerous); near 0 with zero positive weeks anywhere, too high."""
    thresholds = area_fire_thresholds(df)
    threshold_per_row = df["area_id"].map(thresholds)
    fire_week = df["fire_detections"] >= threshold_per_row
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
        # Low/no-fire-activity states (e.g. GA, LA, ME, NH, SC, VT over this
        # 2-year window) never have enough fire weeks for RandomForest to
        # learn anything - area_climatology (this area's own history for the
        # time of year) is a reasonable fallback that still varies
        # meaningfully by area and season instead of refusing to answer.
        return _climatology_fallback(features, train, test)

    # Leak-free evaluation: climatology fit on TRAIN rows only, then applied
    # to both splits, so a test row's climatology value can never be derived
    # from that same row's own (or any other test-period) label.
    train_clim_map = _month_rate_map(train)
    train_default = float(train["label"].mean())
    train_eval = _attach_climatology(train, train_clim_map, train_default)
    test_eval = _attach_climatology(test, train_clim_map, train_default)

    X_train, y_train = train_eval[FEATURE_COLUMNS], train_eval["label"]
    X_test, y_test = test_eval[FEATURE_COLUMNS], test_eval["label"]

    eval_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
    eval_model.fit(X_train, y_train)

    probs = eval_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else float("nan")
    brier = brier_score_loss(y_test, probs)

    brier_baseline = brier_score_loss(y_test, test_eval["area_climatology"])
    brier_skill = 1 - (brier / brier_baseline) if brier_baseline > 0 else float("nan")

    # Deployed model: refit on ALL data using the best-available climatology
    # (fit on the full history, not just the train split) - standard practice
    # once a holdout split has given an honest performance estimate above.
    # There's no leakage risk here: "now" has no future to leak from.
    full_clim_map = _month_rate_map(features)
    full_default = float(features["label"].mean())
    features_full = _attach_climatology(features, full_clim_map, full_default)

    final_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
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
    """Fallback "model" for states/areas with too little real fire activity
    to train a RandomForest on (see train_model). Just returns each row's
    own area_climatology feature as its probability - not learned, but still
    a real, area- and season-specific number rather than a constant."""

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
        # Same leak-free pattern as train_model: fit on train rows only, apply to test.
        train_clim_map = _month_rate_map(train)
        train_default = float(train["label"].mean())
        test_eval = _attach_climatology(test, train_clim_map, train_default)
        probs = model.predict_proba(test_eval[FEATURE_COLUMNS])[:, 1]
        metrics["brier_score"] = round(float(brier_score_loss(test_eval["label"], probs)), 4)
        metrics["roc_auc"] = round(float(roc_auc_score(test_eval["label"], probs)), 3)

    # Deployed "model": full-history climatology, same as the main path.
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
