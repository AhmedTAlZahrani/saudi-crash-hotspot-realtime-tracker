import pandas as pd
import numpy as np
from pathlib import Path

DEFAULT_ALERT_LIMIT = 50
DEFAULT_HOTSPOT_LIMIT = 20
KPI_WINDOW_HOURS = 1


class DashboardDataManager:
    """Aggregation layer providing data feeds for the Streamlit dashboard.

    Consolidates hotspot clusters, risk scores, alert feeds, time series,
    and KPIs into dashboard-ready formats.
    """

    def __init__(self, alert_limit=DEFAULT_ALERT_LIMIT,
                 hotspot_limit=DEFAULT_HOTSPOT_LIMIT):
        self.alert_limit = alert_limit
        self.hotspot_limit = hotspot_limit
        self._current_hotspots = pd.DataFrame()
        self._risk_scores = pd.DataFrame()
        self._alert_feed = []
        self._incident_series = pd.DataFrame()
        self._kpis = {}

    def update_hotspots(self, cluster_metadata):
        """Update the current hotspot list.

        Args:
            cluster_metadata: DataFrame from HotspotDetector.get_cluster_metadata().
        """
        if len(cluster_metadata) == 0:
            self._current_hotspots = pd.DataFrame()
            return

        self._current_hotspots = cluster_metadata.head(self.hotspot_limit).copy()
        print(f"Dashboard hotspots updated: {len(self._current_hotspots)} clusters")

    def update_risk_scores(self, risk_df):
        """Update the risk score map data.

        Args:
            risk_df: DataFrame from RiskScorer.compute_scores() with
                road_segment_id, risk_score, risk_level columns.
        """
        self._risk_scores = risk_df.copy()
        print(f"Dashboard risk scores updated: {len(risk_df)} segments")

    def update_alert_feed(self, alerts):
        """Append new alerts to the feed.

        Args:
            alerts: List of alert dictionaries from AlertDispatcher.
        """
        self._alert_feed.extend(alerts)
        self._alert_feed = self._alert_feed[-self.alert_limit:]

    def compute_incident_time_series(self, df, freq="1h"):
        """Compute incident count time series at the given frequency.

        Args:
            df: Incident DataFrame with timestamp column.
            freq: Resampling frequency string (e.g., '1h', '30min').

        Returns:
            DataFrame with timestamp index and incident_count column.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        series = df.resample(freq).size().reset_index(name="incident_count")
        series.columns = ["timestamp", "incident_count"]

        self._incident_series = series
        return series

    def compute_severity_time_series(self, df, freq="1h"):
        """Compute average severity over time.

        Args:
            df: Incident DataFrame with timestamp and severity columns.
            freq: Resampling frequency string.

        Returns:
            DataFrame with timestamp and avg_severity columns.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        series = df["severity"].resample(freq).mean().reset_index()
        series.columns = ["timestamp", "avg_severity"]
        series["avg_severity"] = series["avg_severity"].fillna(0).round(2)

        return series

    def compute_kpis(self, df, alerts, reference_time=None):
        """Compute summary KPIs for the dashboard.

        Args:
            df: Current incident DataFrame.
            alerts: List of alert dictionaries.
            reference_time: Current reference time. Uses max timestamp if None.

        Returns:
            Dict of KPI values.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if reference_time is None:
            reference_time = df["timestamp"].max()
        else:
            reference_time = pd.Timestamp(reference_time)

        last_hour = df[df["timestamp"] >= reference_time - pd.Timedelta(hours=KPI_WINDOW_HOURS)]
        last_24h = df[df["timestamp"] >= reference_time - pd.Timedelta(hours=24)]

        active_hotspots = len(self._current_hotspots)
        incidents_last_hour = len(last_hour)
        incidents_last_24h = len(last_24h)

        active_alerts = [a for a in alerts if a.get("status") == "active"]
        avg_distance_to_zone = 0.0
        if active_alerts:
            distances = [a.get("distance_to_zone_km", 0) for a in active_alerts]
            avg_distance_to_zone = round(np.mean(distances), 1)

        avg_severity_last_hour = round(last_hour["severity"].mean(), 2) if len(last_hour) > 0 else 0.0
        sandstorm_pct = round(100.0 * last_24h["is_sandstorm"].mean(), 1) if len(last_24h) > 0 else 0.0

        critical_segments = 0
        if len(self._risk_scores) > 0:
            critical_segments = (self._risk_scores["risk_level"] == "critical").sum()

        self._kpis = {
            "active_hotspots": active_hotspots,
            "incidents_last_hour": incidents_last_hour,
            "incidents_last_24h": incidents_last_24h,
            "active_alerts": len(active_alerts),
            "avg_response_distance_km": avg_distance_to_zone,
            "avg_severity_last_hour": avg_severity_last_hour,
            "sandstorm_incident_pct": sandstorm_pct,
            "critical_segments": critical_segments,
            "total_segments_monitored": len(self._risk_scores),
        }

        return self._kpis

    def get_hotspots(self):
        """Get the current hotspot list with metadata.

        Returns:
            DataFrame of active hotspot clusters.
        """
        return self._current_hotspots

    def get_risk_map_data(self):
        """Get risk score data for map visualization.

        Returns:
            DataFrame with segment coordinates, scores, and levels.
        """
        return self._risk_scores

    def get_alert_feed(self, last_n=None):
        """Get the alert feed.

        Args:
            last_n: Number of recent alerts. Returns all if None.

        Returns:
            List of alert dictionaries.
        """
        if last_n is None:
            return list(self._alert_feed)
        return self._alert_feed[-last_n:]

    def get_incident_series(self):
        """Get the incident count time series.

        Returns:
            DataFrame with timestamp and incident_count columns.
        """
        return self._incident_series

    def get_kpis(self):
        """Get the latest computed KPIs.

        Returns:
            Dict of KPI values.
        """
        return self._kpis

    def get_highway_summary(self, df):
        """Get per-highway incident summary.

        Args:
            df: Incident DataFrame.

        Returns:
            DataFrame with highway stats.
        """
        df = df.copy()
        summary = df.groupby("highway_id").agg(
            incident_count=("incident_id", "count"),
            avg_severity=("severity", "mean"),
            sandstorm_incidents=("is_sandstorm", "sum"),
            avg_visibility=("visibility_km", "mean"),
        ).reset_index()

        summary["avg_severity"] = summary["avg_severity"].round(2)
        summary["avg_visibility"] = summary["avg_visibility"].round(2)
        summary = summary.sort_values("incident_count", ascending=False).reset_index(drop=True)

        return summary

    def get_weather_breakdown(self, df):
        """Get incident breakdown by weather condition.

        Args:
            df: Incident DataFrame.

        Returns:
            DataFrame with weather condition counts and percentages.
        """
        df = df.copy()
        counts = df["weather"].value_counts().reset_index()
        counts.columns = ["weather", "count"]
        counts["percentage"] = round(100.0 * counts["count"] / counts["count"].sum(), 1)
        return counts

    def export_snapshot(self, output_dir="output"):
        """Export current dashboard state to CSV files.

        Args:
            output_dir: Directory for output files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if len(self._current_hotspots) > 0:
            self._current_hotspots.to_csv(out / "hotspots.csv", index=False)

        if len(self._risk_scores) > 0:
            self._risk_scores.to_csv(out / "risk_scores.csv", index=False)

        if self._alert_feed:
            pd.DataFrame(self._alert_feed).to_csv(out / "alerts.csv", index=False)

        if len(self._incident_series) > 0:
            self._incident_series.to_csv(out / "incident_series.csv", index=False)

        print(f"Dashboard snapshot exported to {out}/")
