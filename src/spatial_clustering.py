import logging
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger = logging.getLogger("hotspot_tracker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(_handler)

EPS_SPATIAL_KM = 5.0
EPS_TEMPORAL_MIN = 120
MIN_SAMPLES = 3
EARTH_RADIUS_KM = 6371.0


def haversine_distance_matrix(coords):
    """Compute pairwise haversine distance matrix using sklearn.

    Args:
        coords: Nx2 array of (lat, lon) in degrees.

    Returns:
        NxN distance matrix in kilometers.
    """
    coords_rad = np.radians(coords)
    return haversine_distances(coords_rad) * EARTH_RADIUS_KM


class HotspotDetector:
    """Spatio-temporal crash hotspot detection using DBSCAN and ST-DBSCAN.

    Identifies clusters of crash incidents based on geographic proximity
    and temporal co-occurrence, with emerging hotspot detection against
    historical baselines.
    """

    def __init__(self, eps_km=EPS_SPATIAL_KM, eps_minutes=EPS_TEMPORAL_MIN,
                 min_samples=MIN_SAMPLES):
        self.eps_km = eps_km
        self.eps_minutes = eps_minutes
        self.min_samples = min_samples
        self._clusters = None
        self._historical_baseline = None

    def detect_spatial(self, df):
        """Run DBSCAN clustering on GPS coordinates."""
        if len(df) < self.min_samples:
            df = df.copy()
            df["cluster_id"] = -1
            return df

        coords = df[["lat", "lon"]].values
        coords_rad = np.radians(coords)

        eps_rad = self.eps_km / EARTH_RADIUS_KM

        clustering = DBSCAN(
            eps=eps_rad,
            min_samples=self.min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )
        labels = clustering.fit_predict(coords_rad)

        result = df.copy()
        result["cluster_id"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info("Spatial DBSCAN: %d clusters, %d noise points", n_clusters, n_noise)
        print(f"Spatial DBSCAN: {n_clusters} clusters, {n_noise} noise points")

        return result

    def detect_st_dbscan(self, df):
        """Run ST-DBSCAN clustering by both space and time.

        Applies spatial DBSCAN first, then refines clusters by checking
        temporal proximity within each spatial cluster.

        Args:
            df: DataFrame with 'lat', 'lon', and 'timestamp' columns.

        Returns:
            DataFrame with 'st_cluster_id' column added.
        """
        if len(df) < self.min_samples:
            df = df.copy()
            df["st_cluster_id"] = -1
            return df

        spatial_result = self.detect_spatial(df)

        st_labels = np.full(len(df), -1)
        next_cluster_id = 0

        spatial_clusters = set(spatial_result["cluster_id"].unique()) - {-1}

        for sc_id in spatial_clusters:
            mask = spatial_result["cluster_id"] == sc_id
            subset = spatial_result[mask].copy()
            indices = subset.index.values

            if len(subset) < self.min_samples:
                continue

            timestamps = pd.to_datetime(subset["timestamp"])
            time_minutes = (timestamps - timestamps.min()).dt.total_seconds() / 60.0
            time_values = time_minutes.values.reshape(-1, 1)

            if time_values.max() - time_values.min() <= self.eps_minutes:
                st_labels[indices] = next_cluster_id
                next_cluster_id += 1
            else:
                time_clustering = DBSCAN(
                    eps=self.eps_minutes,
                    min_samples=self.min_samples,
                    metric="euclidean",
                )
                time_labels = time_clustering.fit_predict(time_values)

                for tl in set(time_labels) - {-1}:
                    t_mask = time_labels == tl
                    st_labels[indices[t_mask]] = next_cluster_id
                    next_cluster_id += 1

        result = df.copy()
        result["st_cluster_id"] = st_labels

        n_clusters = len(set(st_labels)) - (1 if -1 in st_labels else 0)
        logger.info("ST-DBSCAN: %d spatio-temporal clusters", n_clusters)
        print(f"ST-DBSCAN: {n_clusters} spatio-temporal clusters")

        self._clusters = result
        return result

    def get_cluster_metadata(self, df, cluster_col="st_cluster_id"):
        """Compute metadata for each detected cluster.

        Args:
            df: DataFrame with cluster labels.
            cluster_col: Name of the cluster ID column.

        Returns:
            DataFrame with cluster center, count, severity, and time span.
        """
        clusters = df[df[cluster_col] != -1].copy()

        if len(clusters) == 0:
            return pd.DataFrame(columns=[
                "cluster_id", "center_lat", "center_lon", "incident_count",
                "avg_severity", "max_severity", "time_span_minutes",
                "dominant_highway", "avg_visibility",
            ])

        metadata = []
        for cid in sorted(clusters[cluster_col].unique()):
            group = clusters[clusters[cluster_col] == cid]
            timestamps = pd.to_datetime(group["timestamp"])
            time_span = (timestamps.max() - timestamps.min()).total_seconds() / 60.0

            metadata.append({
                "cluster_id": cid,
                "center_lat": round(group["lat"].mean(), 4),
                "center_lon": round(group["lon"].mean(), 4),
                "incident_count": len(group),
                "avg_severity": round(group["severity"].mean(), 2),
                "max_severity": int(group["severity"].max()),
                "time_span_minutes": round(time_span, 1),
                "dominant_highway": group["highway_id"].mode().iloc[0] if len(group) > 0 else "unknown",
                "avg_visibility": round(group["visibility_km"].mean(), 2),
            })

        return pd.DataFrame(metadata).sort_values("incident_count", ascending=False).reset_index(drop=True)

    def set_historical_baseline(self, df):
        """Compute historical baseline cluster statistics.

        Args:
            df: Full historical incident DataFrame.
        """
        result = self.detect_st_dbscan(df)
        self._historical_baseline = self.get_cluster_metadata(result)
        print(f"Historical baseline set: {len(self._historical_baseline)} clusters")

    def detect_emerging_hotspots(self, current_df, baseline_metadata=None):
        """Compare current window clusters to historical baseline.

        Args:
            current_df: Current time window incident DataFrame.
            baseline_metadata: Historical cluster metadata. Uses stored baseline if None.

        Returns:
            DataFrame of emerging hotspots with anomaly scores.
        """
        baseline = baseline_metadata if baseline_metadata is not None else self._historical_baseline

        current_result = self.detect_st_dbscan(current_df)
        current_meta = self.get_cluster_metadata(current_result)

        if len(current_meta) == 0:
            print("No current clusters to compare.")
            return pd.DataFrame()

        if baseline is None or len(baseline) == 0:
            current_meta["anomaly_score"] = 1.0
            current_meta["is_emerging"] = True
            return current_meta

        avg_baseline_count = baseline["incident_count"].mean()
        avg_baseline_severity = baseline["avg_severity"].mean()

        emerging = []
        for _, cluster in current_meta.iterrows():
            count_ratio = cluster["incident_count"] / max(avg_baseline_count, 1)
            severity_ratio = cluster["avg_severity"] / max(avg_baseline_severity, 1)
            anomaly_score = round((count_ratio + severity_ratio) / 2, 3)

            row = cluster.to_dict()
            row["anomaly_score"] = anomaly_score
            row["is_emerging"] = anomaly_score > 1.5
            emerging.append(row)

        result = pd.DataFrame(emerging).sort_values("anomaly_score", ascending=False)
        n_emerging = result["is_emerging"].sum()
        logger.info("Emerging hotspots: %d / %d clusters", n_emerging, len(result))
        print(f"Emerging hotspots: {n_emerging} / {len(result)} clusters")

        return result.reset_index(drop=True)

    def get_clusters(self):
        return self._clusters

