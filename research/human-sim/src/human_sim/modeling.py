from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .context import CONTEXT_FEATURES

FEATURES = list(CONTEXT_FEATURES)
LEGACY_FEATURES = ["distance", "interval_ms", "strain", "clock_rate", "hidden", "context", "object_kind"]
TARGETS = ["hit_error_ms", "aim_offset_x", "aim_offset_y", "hold_duration_ms"]


def fit_models(data_path: str, output_path: str, seed: int = 1) -> dict[str, Any]:
    import joblib
    import polars as pl
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    frame = pl.read_parquet(data_path).filter(pl.col("hit"))
    if frame.height < 100:
        raise ValueError("At least 100 matched hit objects are required to fit a model")
    available_features = [name for name in FEATURES if name in frame.columns]
    if not available_features:
        available_features = [name for name in LEGACY_FEATURES if name in frame.columns]
    if not available_features:
        raise ValueError("Feature data contains none of the supported context features")
    x = frame.select(available_features).to_pandas()
    categorical = [name for name in available_features if name in {"context", "object_kind"}]
    numeric = [name for name in available_features if name not in categorical]
    bundle: dict[str, Any] = {"schema_version": 2, "features": available_features, "models": {}}

    def make_transform() -> ColumnTransformer:
        transformers = []
        if numeric:
            transformers.append(("numeric", StandardScaler(), numeric))
        if categorical:
            transformers.append(("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
        return ColumnTransformer(transformers)

    for target in TARGETS:
        if target not in frame.columns:
            continue
        valid = frame[target].is_not_null().to_numpy()
        if not bool(valid.any()):
            continue
        bundle["models"][target] = {}
        for quantile in (0.1, 0.5, 0.9):
            model = Pipeline(
                [("features", make_transform()), ("regressor", GradientBoostingRegressor(loss="quantile", alpha=quantile, random_state=seed))]
            )
            model.fit(x.loc[valid], frame.filter(pl.Series(valid))[target].to_numpy())
            bundle["models"][target][str(quantile)] = model

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "model_sha256": digest,
        "training_rows": frame.height,
        "targets": sorted(bundle["models"]),
        "features": available_features,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def evaluate(human_path: str, synthetic_path: str, output_path: str, seed: int = 1) -> dict[str, Any]:
    import numpy as np
    import polars as pl
    from scipy.stats import ks_2samp
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    human = pl.read_parquet(human_path).with_columns(pl.lit(0).alias("synthetic"))
    synthetic = pl.read_parquet(synthetic_path).with_columns(pl.lit(1).alias("synthetic"))
    common = [name for name in FEATURES + TARGETS if name in human.columns and name in synthetic.columns]
    data = pl.concat([human.select(common + ["synthetic"]), synthetic.select(common + ["synthetic"])], how="vertical")
    x = data.select(common).to_pandas()
    y = data["synthetic"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
    categorical = [name for name in common if name in {"context", "object_kind"}]
    numeric = [name for name in common if name not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    prep = ColumnTransformer(transformers)
    results: dict[str, Any] = {"schema_version": 1, "models": {}, "ks": {}}
    for name, classifier in {
        "logistic": LogisticRegression(max_iter=2000, random_state=seed),
        "gradient_boosted": HistGradientBoostingClassifier(random_state=seed),
    }.items():
        model = Pipeline([("features", prep), ("classifier", classifier)])
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        results["models"][name] = {
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "pr_auc": float(average_precision_score(y_test, probability)),
            "brier": float(brier_score_loss(y_test, probability)),
        }
        importance = permutation_importance(model, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed)
        results["models"][name]["feature_importance"] = {
            feature: float(value) for feature, value in sorted(zip(common, importance.importances_mean), key=lambda item: abs(item[1]), reverse=True)
        }
    for column in numeric:
        left = human[column].drop_nulls().to_numpy()
        right = synthetic[column].drop_nulls().to_numpy()
        if len(left) and len(right):
            results["ks"][column] = float(ks_2samp(left, right).statistic)
    results["core_distributions_at_or_below_0_15"] = float(np.mean([value <= 0.15 for value in results["ks"].values()]))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results
