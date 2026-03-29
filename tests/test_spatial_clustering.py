"""Tests for spatial clustering and hotspot detection."""

import numpy as np
import pandas as pd
import pytest

from src.spatial_clustering import (
    EARTH_RADIUS_KM,
    HotspotDetector,
    haversine_distance_matrix,
)
from src.risk_scorer import RiskScorer, RISK_LEVELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def riyadh_cluster_coords():
    """Synthetic crash coordinates forming a tight cluster near Riyadh."""
    rng = np.random.default_rng(99)
    center_lat, center_lon = 24.7136, 46.6753
    n = 15
    lats = center_lat + rng.normal(0, 0.005, n)
    lons = center_lon + rng.normal(0, 0.005, n)
    return np.column_stack([lats, lons])


@pytest.fixture
def two_cluster_df():
    """DataFrame with two well-separated spatial clusters and some noise.

    :return: DataFrame with lat, lon, timestamp, severity, highway_id,
             visibility_km columns.
    """
    rng = np.random.default_rng(42)
    base_time = pd.Timestamp("2025-06-01 08:00:00")

    rows = []
    # cluster A  -- Riyadh area
    for i in range(10):
        rows.append({
            "lat": 24.71 + rng.normal(0, 0.003),
            "lon": 46.68 + rng.normal(0, 0.003),
            "timestamp": base_time + pd.Timedelta(minutes=int(rng.integers(0, 60))),
            "severity": int(rng.choice([2, 3])),
            "highway_id": "RRD",
            "visibility_km": round(float(rng.uniform(2, 8)), 2),
        })
    # cluster B  -- Jeddah area, ~900 km away
    for i in range(8):
        rows.append({
            "lat": 21.54 + rng.normal(0, 0.003),
            "lon": 39.17 + rng.normal(0, 0.003),
            "timestamp": base_time + pd.Timedelta(minutes=int(rng.integers(0, 90))),
            "severity": int(rng.choice([1, 2, 3])),
            "highway_id": "E30",
            "visibility_km": round(float(rng.uniform(3, 10)), 2),
        })
    # noise points scattered far apart
    for i in range(4):
        rows.append({
            "lat": 20.0 + i * 2,
            "lon": 40.0 + i * 2,
            "timestamp": base_time + pd.Timedelta(hours=int(rng.integers(5, 20))),
            "severity": 1,
            "highway_id": "E11",
            "visibility_km": 10.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def incident_df_for_risk():
    """Incidents suitable for risk scoring (includes road_segment_id, weather)."""
    rng = np.random.default_rng(7)
    base_time = pd.Timestamp("2025-03-15 07:30:00")
    rows = []
    for i in range(20):
        rows.append({
            "timestamp": base_time + pd.Timedelta(minutes=int(rng.integers(0, 300))),
            "lat": 24.7 + rng.normal(0, 0.01),
            "lon": 46.7 + rng.normal(0, 0.01),
            "highway_id": "RRD",
            "road_segment_id": f"RRD-S{rng.integers(0, 5):03d}",
            "severity": int(rng.choice([1, 2, 3, 4])),
            "weather": rng.choice(["clear", "rain", "sandstorm"]),
            "visibility_km": round(float(rng.uniform(0.5, 10)), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# haversine_distance_matrix
# ---------------------------------------------------------------------------

class TestHaversineDistanceMatrix:
    def test_self_distance_is_zero(self, riyadh_cluster_coords):
        """Points should have zero distance to themselves."""
        mat = haversine_distance_matrix(riyadh_cluster_coords)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-10)

    def test_symmetry(self, riyadh_cluster_coords):
        mat = haversine_distance_matrix(riyadh_cluster_coords)
        np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_known_distance(self):
        # Riyadh to Jeddah ~950 km
        coords = np.array([[24.7136, 46.6753], [21.5433, 39.1728]])
        mat = haversine_distance_matrix(coords)
        assert 800 < mat[0, 1] < 1100, f"Expected ~950 km, got {mat[0, 1]}"

    def test_shape(self, riyadh_cluster_coords):
        n = len(riyadh_cluster_coords)
        mat = haversine_distance_matrix(riyadh_cluster_coords)
        assert mat.shape == (n, n)

    def test_nonnegative(self, riyadh_cluster_coords):
        mat = haversine_distance_matrix(riyadh_cluster_coords)
        assert (mat >= 0).all()


# ---------------------------------------------------------------------------
# HotspotDetector — spatial clustering
# ---------------------------------------------------------------------------

class TestDetectSpatial:
    """Test DBSCAN spatial clustering."""

    def test_finds_clusters(self, two_cluster_df):
        detector = HotspotDetector(eps_km=5.0, min_samples=3)
        result = detector.detect_spatial(two_cluster_df)
        assert "cluster_id" in result.columns
        labels = set(result["cluster_id"]) - {-1}
        assert len(labels) >= 2, "Should find at least two spatial clusters"

    def test_noise_points_labelled_minus_one(self, two_cluster_df):
        detector = HotspotDetector(eps_km=5.0, min_samples=3)
        result = detector.detect_spatial(two_cluster_df)
        assert -1 in result["cluster_id"].values

    def test_too_few_points_returns_all_noise(self):
        """Fewer points than min_samples -> everything labelled noise."""
        df = pd.DataFrame({"lat": [24.7, 24.8], "lon": [46.6, 46.7]})
        detector = HotspotDetector(min_samples=5)
        result = detector.detect_spatial(df)
        assert (result["cluster_id"] == -1).all()

    def test_output_length_matches_input(self, two_cluster_df):
        detector = HotspotDetector()
        result = detector.detect_spatial(two_cluster_df)
        assert len(result) == len(two_cluster_df)


# ---------------------------------------------------------------------------
# HotspotDetector — ST-DBSCAN
# ---------------------------------------------------------------------------

class TestDetectSTDBSCAN:
    def test_st_cluster_column_present(self, two_cluster_df):
        detector = HotspotDetector(eps_km=5.0, eps_minutes=120, min_samples=3)
        result = detector.detect_st_dbscan(two_cluster_df)
        assert "st_cluster_id" in result.columns

    def test_temporally_separated_points_split(self):
        """Points in same location but >eps_minutes apart should be in different clusters."""
        base = pd.Timestamp("2025-01-01 00:00")
        rows = []
        # group A: 10 points at t=0..30 min
        for i in range(10):
            rows.append({"lat": 24.71, "lon": 46.68,
                         "timestamp": base + pd.Timedelta(minutes=i * 3),
                         "severity": 2, "highway_id": "RRD", "visibility_km": 5.0})
        # group B: 10 points at t=300..330 min (well beyond eps_minutes=60)
        for i in range(10):
            rows.append({"lat": 24.71, "lon": 46.68,
                         "timestamp": base + pd.Timedelta(minutes=300 + i * 3),
                         "severity": 2, "highway_id": "RRD", "visibility_km": 5.0})
        df = pd.DataFrame(rows)
        detector = HotspotDetector(eps_km=5.0, eps_minutes=60, min_samples=3)
        result = detector.detect_st_dbscan(df)
        ids = set(result["st_cluster_id"]) - {-1}
        assert len(ids) >= 2, "Temporally separated groups should form separate ST clusters"

    def test_stores_clusters(self, two_cluster_df):
        detector = HotspotDetector(eps_km=5.0, min_samples=3)
        detector.detect_st_dbscan(two_cluster_df)
        stored = detector.get_clusters()
        assert stored is not None
        assert len(stored) == len(two_cluster_df)


# ---------------------------------------------------------------------------
# Cluster metadata
# ---------------------------------------------------------------------------

class TestClusterMetadata:
    def test_metadata_columns(self, two_cluster_df):
        detector = HotspotDetector(eps_km=5.0, min_samples=3)
        clustered = detector.detect_st_dbscan(two_cluster_df)
        meta = detector.get_cluster_metadata(clustered)
        expected_cols = {"cluster_id", "center_lat", "center_lon",
                         "incident_count", "avg_severity", "max_severity",
                         "time_span_minutes", "dominant_highway", "avg_visibility"}
        assert expected_cols.issubset(set(meta.columns))

    def test_empty_clusters_return_empty_df(self):
        df = pd.DataFrame({
            "lat": [24.7], "lon": [46.6], "st_cluster_id": [-1],
            "timestamp": [pd.Timestamp("2025-01-01")], "severity": [1],
            "highway_id": ["E30"], "visibility_km": [5.0],
        })
        detector = HotspotDetector()
        meta = detector.get_cluster_metadata(df)
        assert len(meta) == 0

    def test_severity_bounds(self, two_cluster_df):
        """avg_severity must be within the range of input severities."""
        detector = HotspotDetector(eps_km=5.0, min_samples=3)
        clustered = detector.detect_st_dbscan(two_cluster_df)
        meta = detector.get_cluster_metadata(clustered)
        if len(meta) > 0:
            assert meta["avg_severity"].min() >= 1.0
            assert meta["max_severity"].max() <= 4


# ---------------------------------------------------------------------------
# Risk scorer — score bounds
# ---------------------------------------------------------------------------

class TestRiskScorer:
    def test_scores_between_0_and_100(self, incident_df_for_risk):
        scorer = RiskScorer()
        result = scorer.compute_scores(incident_df_for_risk)
        assert (result["risk_score"] >= 0).all()
        assert (result["risk_score"] <= 100).all()

    def test_risk_levels_are_valid(self, incident_df_for_risk):
        scorer = RiskScorer()
        result = scorer.compute_scores(incident_df_for_risk)
        valid_levels = set(RISK_LEVELS.keys())
        assert set(result["risk_level"]).issubset(valid_levels)

    def test_empty_df_returns_empty(self):
        scorer = RiskScorer()
        empty = pd.DataFrame(columns=[
            "timestamp", "road_segment_id", "severity", "weather",
            "highway_id", "lat", "lon",
        ])
        result = scorer.compute_scores(empty)
        assert len(result) == 0

    def test_get_segment_score_default(self):
        scorer = RiskScorer()
        assert scorer.get_segment_score("nonexistent") == 0.0

    def test_time_multiplier_rush_hour(self):
        assert RiskScorer._get_time_multiplier(7) == 1.4

    def test_time_multiplier_late_night(self):
        assert RiskScorer._get_time_multiplier(2) == 1.3

    def test_time_multiplier_normal(self):
        assert RiskScorer._get_time_multiplier(12) == 1.0

    def test_classify_risk_boundaries(self):
        assert RiskScorer._classify_risk(0) == "low"
        assert RiskScorer._classify_risk(24.9) == "low"
        assert RiskScorer._classify_risk(25) == "moderate"
        assert RiskScorer._classify_risk(50) == "high"
        assert RiskScorer._classify_risk(75) == "critical"
        assert RiskScorer._classify_risk(100) == "critical"


# ---------------------------------------------------------------------------
# Emerging hotspots
# ---------------------------------------------------------------------------

class TestEmergingHotspots:
    def test_no_baseline_marks_all_emerging(self, two_cluster_df):
        """Without historical baseline every current cluster is emerging."""
        detector = HotspotDetector(eps_km=5.0, min_samples=3)
        result = detector.detect_emerging_hotspots(two_cluster_df)
        if len(result) > 0:
            assert result["is_emerging"].all()
            assert (result["anomaly_score"] == 1.0).all()

    def test_anomaly_score_is_positive(self, two_cluster_df):
        detector = HotspotDetector(eps_km=5.0, min_samples=3)
        result = detector.detect_emerging_hotspots(two_cluster_df)
        if len(result) > 0:
            assert (result["anomaly_score"] > 0).all()
