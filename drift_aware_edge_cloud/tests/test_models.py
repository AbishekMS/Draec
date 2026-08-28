"""Phase 2 Unit and Integration Tests -- Edge (River) and Cloud (XGBoost) models.

Covers the 15 required checks:
1. Edge model creation
2. Cloud model creation
3. Edge training
4. Cloud training
5. Edge prediction
6. Cloud prediction
7. Probability prediction
8. Edge incremental update
9. Prediction shape
10. Reproducibility
11. Target exclusion
12. Leakage-feature exclusion
13. Same feature representation for Edge and Cloud
14. No validation/test data used during training
15. Small end-to-end smoke test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import CausalityError
from src.models import (
    BaseModel,
    CloudModel,
    CloudXGBoost,
    EdgeHoeffdingTree,
    EdgeModel,
    InputDimensionError,
    NotTrainedError,
    evaluate_model,
    evaluate_predictions,
    extract_partition_labels,
    load_causal_eval_data,
    load_causal_train_data,
    train_cloud_model,
    train_edge_model,
)
from src.models.trainer import LEAKAGE_COLUMNS
from src.utils import config as cfgmod


@pytest.fixture(scope="module")
def toy_data():
    """Deterministic synthetic toy dataset with exactly 37 features matching Phase 1."""
    rng = np.random.default_rng(42)
    feature_names = [f"feat_{i}" for i in range(37)]
    X = pd.DataFrame(rng.standard_normal((120, 37)), columns=feature_names)
    # Binary labels with minority class
    y = np.zeros(120, dtype=int)
    y[rng.choice(120, size=15, replace=False)] = 1
    return X, y, feature_names


# -----------------------------------------------------------------------------
# 1 & 2: Model creation
# -----------------------------------------------------------------------------


def test_1_edge_model_creation(cfg):
    edge = EdgeHoeffdingTree(cfg)
    assert isinstance(edge, BaseModel)
    assert edge.model_name == "RiverHoeffdingTreeClassifier"
    assert not edge.is_trained
    assert edge.n_features is None
    info = edge.get_info()
    assert info["model_name"] == "RiverHoeffdingTreeClassifier"
    assert info["is_trained"] is False
    assert "grace_period" in info["hyperparameters"]


def test_2_cloud_model_creation(cfg):
    cloud = CloudXGBoost(cfg)
    assert isinstance(cloud, BaseModel)
    assert cloud.model_name == "CloudXGBoostClassifier"
    assert not cloud.is_trained
    assert cloud.n_features is None
    info = cloud.get_info()
    assert info["model_name"] == "CloudXGBoostClassifier"
    assert info["is_trained"] is False
    assert "n_estimators" in info["hyperparameters"]


# -----------------------------------------------------------------------------
# 3 & 4: Model training
# -----------------------------------------------------------------------------


def test_3_edge_training(toy_data):
    X, y, _ = toy_data
    edge = EdgeModel()
    with pytest.raises(NotTrainedError):
        edge.predict(X)

    edge.fit(X, y)
    assert edge.is_trained
    assert edge.n_features == 37
    assert edge.n_samples_trained == len(X)
    assert edge.feature_names == tuple(X.columns)


def test_4_cloud_training(toy_data):
    X, y, _ = toy_data
    cloud = CloudModel()
    with pytest.raises(NotTrainedError):
        cloud.predict(X)

    cloud.fit(X, y)
    assert cloud.is_trained
    assert cloud.n_features == 37
    assert cloud.n_samples_trained == len(X)
    assert cloud.feature_names == tuple(X.columns)


# -----------------------------------------------------------------------------
# 5 & 6: Prediction
# -----------------------------------------------------------------------------


def test_5_edge_prediction(toy_data):
    X, y, _ = toy_data
    edge = EdgeModel().fit(X, y)
    preds = edge.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

    # Single-observation prediction
    single_pred = edge.predict_one(X.iloc[0])
    assert single_pred in {0, 1}
    assert edge.last_inference_time_s is not None
    assert edge.last_inference_time_s > 0


def test_6_cloud_prediction(toy_data):
    X, y, _ = toy_data
    cloud = CloudModel().fit(X, y)
    preds = cloud.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

    # Single-observation prediction
    single_pred = cloud.predict_one(X.iloc[0])
    assert single_pred in {0, 1}
    assert cloud.last_inference_time_s is not None
    assert cloud.last_inference_time_s > 0


# -----------------------------------------------------------------------------
# 7: Probability prediction
# -----------------------------------------------------------------------------


def test_7_probability_prediction(toy_data):
    X, y, _ = toy_data
    edge = EdgeModel().fit(X, y)
    cloud = CloudModel().fit(X, y)

    # Batch probabilities
    p_edge = edge.predict_proba(X)
    p_cloud = cloud.predict_proba(X)

    assert p_edge.shape == (len(X), 2)
    assert p_cloud.shape == (len(X), 2)
    np.testing.assert_allclose(p_edge.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(p_cloud.sum(axis=1), 1.0, atol=1e-5)

    # Single-observation probabilities
    sp_edge = edge.predict_proba_one(X.iloc[0])
    sp_cloud = cloud.predict_proba_one(X.iloc[0])

    assert set(sp_edge.keys()) == {0, 1}
    assert set(sp_cloud.keys()) == {0, 1}
    assert pytest.approx(sum(sp_edge.values()), abs=1e-5) == 1.0
    assert pytest.approx(sum(sp_cloud.values()), abs=1e-5) == 1.0


# -----------------------------------------------------------------------------
# 8: Edge incremental update
# -----------------------------------------------------------------------------


def test_8_edge_incremental_update(toy_data):
    X, y, _ = toy_data
    edge = EdgeModel()

    # Learn first 50 observations
    edge.learn_many(X.iloc[:50], y[:50])
    assert edge.n_samples_trained == 50
    assert edge.is_trained

    # Incremental update on next single sample
    edge.learn_one(X.iloc[50], y[50])
    assert edge.n_samples_trained == 51

    # Incremental update on subsequent batch
    edge.learn_many(X.iloc[51:100], y[51:100])
    assert edge.n_samples_trained == 100


# -----------------------------------------------------------------------------
# 9: Prediction shape and dimension guards
# -----------------------------------------------------------------------------


def test_9_prediction_shape_and_dimension_guards(toy_data):
    X, y, _ = toy_data
    edge = EdgeModel().fit(X, y)
    cloud = CloudModel().fit(X, y)

    assert edge.predict(X).shape == (len(X),)
    assert cloud.predict(X).shape == (len(X),)

    # Wrong feature count raises InputDimensionError
    bad_X = X.iloc[:, :20]
    with pytest.raises(InputDimensionError):
        edge.predict(bad_X)
    with pytest.raises(InputDimensionError):
        cloud.predict(bad_X)


# -----------------------------------------------------------------------------
# 10: Reproducibility
# -----------------------------------------------------------------------------


def test_10_reproducibility(toy_data, cfg):
    X, y, _ = toy_data
    # Cloud XGBoost reproducibility
    c1 = CloudModel(cfg).fit(X, y)
    c2 = CloudModel(cfg).fit(X, y)
    np.testing.assert_array_equal(c1.predict(X), c2.predict(X))
    np.testing.assert_allclose(c1.predict_proba(X), c2.predict_proba(X), atol=1e-6)

    # Edge Hoeffding Tree reproducibility under identical stream order
    e1 = EdgeModel(cfg).fit(X, y)
    e2 = EdgeModel(cfg).fit(X, y)
    np.testing.assert_array_equal(e1.predict(X), e2.predict(X))


# -----------------------------------------------------------------------------
# 11 & 12: Target and leakage feature exclusion
# -----------------------------------------------------------------------------


def test_11_and_12_target_and_leakage_exclusion(cfg):
    # Verify causal loader strictly strips Target and all 11 leakage fields
    X_train, y_train, stats, profile = load_causal_train_data(cfg, max_rows=100)
    assert "Target" not in X_train.columns
    for col in LEAKAGE_COLUMNS:
        assert col not in X_train.columns
        assert col not in profile.feature_names
        assert col not in stats.columns

    assert len(X_train.columns) == 37
    assert len(y_train) == len(X_train)


# -----------------------------------------------------------------------------
# 13: Identical 37-feature representation for Edge and Cloud
# -----------------------------------------------------------------------------


def test_13_same_feature_representation(cfg):
    X_train, y_train, _, _ = load_causal_train_data(cfg, max_rows=200)
    assert X_train.shape[1] == 37

    edge = train_edge_model(cfg, X_train, y_train)
    cloud = train_cloud_model(cfg, X_train, y_train)

    assert edge.n_features == 37
    assert cloud.n_features == 37
    assert edge.feature_names == cloud.feature_names
    assert tuple(X_train.columns) == edge.feature_names


# -----------------------------------------------------------------------------
# 14: No validation/test data used during training
# -----------------------------------------------------------------------------


def test_14_no_validation_or_test_used_during_training(cfg):
    # Training loader only permits baseline_train
    with pytest.raises(CausalityError):
        load_causal_eval_data(cfg, role="baseline_train", stats=None)

    # Evaluation loader requires frozen baseline stats
    with pytest.raises(CausalityError, match="without frozen BaselineStatistics"):
        load_causal_eval_data(cfg, role="baseline_validation", stats=None)


# -----------------------------------------------------------------------------
# 15: Small end-to-end smoke test
# -----------------------------------------------------------------------------


def test_15_small_end_to_end_smoke_test(cfg):
    # 1. Load small causal train subset
    X_train, y_train, stats, _ = load_causal_train_data(cfg, max_rows=300)
    assert len(X_train) == 300
    assert X_train.shape[1] == 37

    # 2. Train both models
    edge = train_edge_model(cfg, X_train, y_train)
    cloud = train_cloud_model(cfg, X_train, y_train)

    # 3. Load small causal validation subset
    X_val, y_val = load_causal_eval_data(cfg, "baseline_validation", stats, max_rows=100)
    assert len(X_val) == 100
    assert X_val.shape[1] == 37

    # 4. Evaluate on validation subset
    res_edge = evaluate_model(edge, X_val, y_val)
    res_cloud = evaluate_model(cloud, X_val, y_val)

    assert 0.0 <= res_edge["macro_f1"] <= 1.0
    assert 0.0 <= res_cloud["macro_f1"] <= 1.0
    assert res_edge["inference_time_s"] is not None
    assert res_cloud["inference_time_s"] is not None
    assert res_edge["n_samples"] == 100
    assert res_cloud["n_samples"] == 100
