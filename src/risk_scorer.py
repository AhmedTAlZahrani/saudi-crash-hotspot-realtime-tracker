import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

DECAY_HALF_LIFE_HOURS = 12.0
RISK_LEVELS = {
    "low": (0, 25),
    "moderate": (25, 50),
    "high": (50, 75),
    "critical": (75, 100),
}

WEATHER_RISK_MULTIPLIER = {
    "clear": 1.0,
    "cloudy": 1.1,
    "rain": 1.5,
    "fog": 1.8,
    "sandstorm": 2.5,
    "dust": 1.6,
}

HIGHWAY_TYPE_WEIGHT = {
    "E30": 1.3,
    "E45": 1.2,
    "E11": 1.1,
    "RRD": 1.4,
    "MKM": 1.2,
}

RUSH_HOUR_RANGES = [(6, 9), (16, 19)]
RUSH_HOUR_MULTIPLIER = 1.4
LATE_NIGHT_MULTIPLIER = 1.3
BASE_SCORE_PER_INCIDENT = 5.0


class RiskScorer:
    """Dynamic risk scoring for road segments based on crash patterns.

    Computes a 0-100 risk score per segment using historical crash rate,
    recent incident density, weather conditions, time-of-day factors,
    and road type with exponential decay weighting.
    """

    def __init__(self, decay_half_life=DECAY_HALF_LIFE_HOURS):
        self.decay_half_life = decay_half_life
        self._segment_scores = {}
        self._segment_history = {}

    def compute_scores(self, df, reference_time=None):
        """Compute risk scores for all road segments.

        Args:
            df: DataFrame of incidents with timestamp, road_segment_id,
                severity, weather, highway_id columns.
            reference_time: Current time for decay calculation. Uses max
                timestamp if None.

        Returns:
            DataFrame with segment risk scores and levels.
        """
        if len(df) == 0:
            return pd.DataFrame(columns=[
                "road_segment_id", "risk_score", "risk_level", "incident_count",
                "avg_severity", "latest_weather", "highway_id",
            ])

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if reference_time is None:
            reference_time = df["timestamp"].max()
        else:
            reference_time = pd.Timestamp(reference_time)

        segments = df["road_segment_id"].unique()
        results = []

        for seg_id in segments:
            seg_data = df[df["road_segment_id"] == seg_id]
            score = self._compute_segment_score(seg_data, reference_time)
            results.append(score)

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

        self._segment_scores = dict(zip(result_df["road_segment_id"], result_df["risk_score"]))

        n_critical = (result_df["risk_level"] == "critical").sum()
        n_high = (result_df["risk_level"] == "high").sum()
        print(f"Risk scores computed for {len(result_df)} segments | Critical: {n_critical} | High: {n_high}")

        return result_df

    def _compute_segment_score(self, seg_data, reference_time):
        """Compute risk score for a single road segment.

        Args:
            seg_data: DataFrame of incidents for one segment.
            reference_time: Current reference time.

        Returns:
            Dict with segment score and metadata.
        """
        seg_id = seg_data["road_segment_id"].iloc[0]
        hw_id = seg_data["highway_id"].iloc[0]

        decay_lambda = np.log(2) / (self.decay_half_life * 3600)

        weighted_score = 0.0
        for _, incident in seg_data.iterrows():
            age_seconds = (reference_time - incident["timestamp"]).total_seconds()
            age_seconds = max(age_seconds, 0)

            decay_weight = np.exp(-decay_lambda * age_seconds)

            severity_factor = incident["severity"] ** 1.5

            weather_mult = WEATHER_RISK_MULTIPLIER.get(incident["weather"], 1.0)

            hour = incident["timestamp"].hour
            time_mult = self._get_time_multiplier(hour)

            incident_score = BASE_SCORE_PER_INCIDENT * severity_factor * weather_mult * time_mult * decay_weight
            weighted_score += incident_score

        road_weight = HIGHWAY_TYPE_WEIGHT.get(hw_id, 1.0)
        weighted_score *= road_weight

        historical_rate = len(seg_data) / max(1, (reference_time - seg_data["timestamp"].min()).days)
        history_bonus = min(historical_rate * 2.0, 15.0)
        weighted_score += history_bonus

        risk_score = min(round(weighted_score, 2), 100.0)
        risk_level = self._classify_risk(risk_score)

        latest_weather = seg_data.sort_values("timestamp").iloc[-1]["weather"]

        return {
            "road_segment_id": seg_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "incident_count": len(seg_data),
            "avg_severity": round(seg_data["severity"].mean(), 2),
            "latest_weather": latest_weather,
            "highway_id": hw_id,
            "historical_rate_per_day": round(historical_rate, 3),
        }

    @staticmethod
    def _get_time_multiplier(hour):
        """Get the time-of-day risk multiplier.

        Args:
            hour: Hour of day (0-23).

        Returns:
            Float multiplier for the given hour.
        """
        for start, end in RUSH_HOUR_RANGES:
            if start <= hour < end:
                return RUSH_HOUR_MULTIPLIER

        if 0 <= hour < 5:
            return LATE_NIGHT_MULTIPLIER

        return 1.0

    @staticmethod
    def _classify_risk(score):
        """Classify a risk score into a named level.

        Args:
            score: Numeric risk score (0-100).

        Returns:
            String risk level name.
        """
        for level, (low, high) in RISK_LEVELS.items():
            if low <= score < high:
                return level
        return "critical"

    def get_segment_score(self, segment_id):
        """Get the current risk score for a specific segment.

        Args:
            segment_id: Road segment identifier.

        Returns:
            Float risk score, or 0.0 if segment not scored.
        """
        return self._segment_scores.get(segment_id, 0.0)

    def get_top_segments(self, df, reference_time=None, top_n=20):
        """Get the top N highest-risk road segments.

        Args:
            df: Incident DataFrame.
            reference_time: Reference time for decay calculation.
            top_n: Number of top segments to return.

        Returns:
            DataFrame of top-risk segments.
        """
        scores = self.compute_scores(df, reference_time)
        return scores.head(top_n)

    def get_risk_summary(self, df, reference_time=None):
        """Get aggregated risk statistics across all segments.

        Args:
            df: Incident DataFrame.
            reference_time: Reference time for scoring.

        Returns:
            Dict with risk distribution and summary stats.
        """
        scores = self.compute_scores(df, reference_time)

        summary = {
            "total_segments": len(scores),
            "avg_risk_score": round(scores["risk_score"].mean(), 2),
            "max_risk_score": round(scores["risk_score"].max(), 2),
            "risk_distribution": {},
        }

        for level in RISK_LEVELS:
            count = (scores["risk_level"] == level).sum()
            summary["risk_distribution"][level] = count

        return summary

    def compute_24h_scores(self, df, reference_time=None):
        """Compute risk scores using only the last 24 hours of data.

        Args:
            df: Full incident DataFrame.
            reference_time: Current time reference.

        Returns:
            DataFrame of segment risk scores for the last 24 hours.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if reference_time is None:
            reference_time = df["timestamp"].max()
        else:
            reference_time = pd.Timestamp(reference_time)

        cutoff = reference_time - pd.Timedelta(hours=24)
        recent = df[df["timestamp"] >= cutoff]

        if len(recent) == 0:
            print("No incidents in the last 24 hours.")
            return pd.DataFrame()

        print(f"Computing 24h risk scores ({len(recent)} incidents)...")
        return self.compute_scores(recent, reference_time)

