import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
NUM_INCIDENTS = 100_000
NUM_HOTSPOT_EVENTS = 50
YEAR_START = "2025-01-01"
YEAR_END = "2025-12-31"

HIGHWAYS = {
    "E30": {
        "name": "Jeddah-Riyadh",
        "lat_range": (21.5, 24.7),
        "lon_range": (39.2, 46.7),
        "segments": 80,
        "weight": 0.30,
    },
    "E45": {
        "name": "Riyadh-Dammam",
        "lat_range": (24.7, 26.4),
        "lon_range": (46.7, 50.1),
        "segments": 50,
        "weight": 0.20,
    },
    "E11": {
        "name": "Coastal Highway",
        "lat_range": (21.5, 26.4),
        "lon_range": (39.1, 50.2),
        "segments": 70,
        "weight": 0.20,
    },
    "RRD": {
        "name": "Riyadh Ring Road",
        "lat_range": (24.6, 24.8),
        "lon_range": (46.5, 46.9),
        "segments": 30,
        "weight": 0.15,
    },
    "MKM": {
        "name": "Makkah-Madinah",
        "lat_range": (21.4, 24.5),
        "lon_range": (39.6, 39.9),
        "segments": 40,
        "weight": 0.15,
    },
}

WEATHER_OPTIONS = ["clear", "cloudy", "rain", "fog", "sandstorm", "dust"]
WEATHER_WEIGHTS = [0.45, 0.20, 0.08, 0.07, 0.12, 0.08]

HAJJ_START = "2025-06-05"
HAJJ_END = "2025-06-15"
NATIONAL_DAY = "2025-09-23"
SCHOOL_BREAK_SUMMER_START = "2025-06-20"
SCHOOL_BREAK_SUMMER_END = "2025-09-10"
SCHOOL_BREAK_WINTER_START = "2025-12-20"
SCHOOL_BREAK_WINTER_END = "2025-12-31"

RUSH_HOURS = [(6, 9), (16, 19)]
LATE_NIGHT_HOURS = (0, 5)


class CrashDataGenerator:
    """Generate synthetic crash incident data across Saudi Arabia's highway network.

    Creates realistic crash patterns with spatial clustering, temporal patterns,
    seasonal events, and weather-related hotspot injections.
    """

    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(SEED)

    def generate(self):
        """Generate the full crash incident dataset.

        Returns:
            DataFrame with 100,000 crash incidents including GPS, severity,
            weather, and temporal fields.
        """
        print("Generating crash incident dataset...")

        timestamps = self._generate_timestamps()
        highways = self._assign_highways()
        coords = self._generate_coordinates(highways)
        segments = self._assign_segments(highways)
        severities = self._generate_severities(timestamps, highways)
        weather = self._generate_weather(timestamps)
        vehicles = self._generate_vehicle_counts(severities)
        sandstorm_flags = np.array([w == "sandstorm" for w in weather]).astype(int)
        visibility = self._generate_visibility(weather)

        df = pd.DataFrame({
            "incident_id": [f"INC-{i:06d}" for i in range(NUM_INCIDENTS)],
            "timestamp": timestamps,
            "lat": coords[:, 0],
            "lon": coords[:, 1],
            "highway_id": highways,
            "severity": severities,
            "vehicle_count": vehicles,
            "weather": weather,
            "road_segment_id": segments,
            "is_sandstorm": sandstorm_flags,
            "visibility_km": visibility,
        })

        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"  Base incidents generated: {len(df)}")

        df = self._inject_hotspot_events(df)

        print(f"  Total incidents after hotspot injection: {len(df)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Highways: {df['highway_id'].nunique()}")

        return df

    def _generate_timestamps(self):
        """Generate timestamps with rush hour and seasonal patterns.

        Returns:
            Array of datetime timestamps.
        """
        start = pd.Timestamp(YEAR_START)
        end = pd.Timestamp(YEAR_END)
        total_seconds = int((end - start).total_seconds())

        base_timestamps = []
        for _ in range(NUM_INCIDENTS):
            offset = self.rng.integers(0, total_seconds)
            ts = start + pd.Timedelta(seconds=int(offset))
            base_timestamps.append(ts)

        timestamps = pd.DatetimeIndex(base_timestamps)

        hour_weights = np.ones(24) * 0.5
        for start_h, end_h in RUSH_HOURS:
            for h in range(start_h, end_h):
                hour_weights[h] = 2.0
        for h in range(LATE_NIGHT_HOURS[0], LATE_NIGHT_HOURS[1]):
            hour_weights[h] = 1.5

        hour_weights /= hour_weights.sum()
        adjusted_hours = self.rng.choice(24, size=NUM_INCIDENTS, p=hour_weights)

        adjusted = []
        for i, ts in enumerate(timestamps):
            new_ts = ts.replace(hour=int(adjusted_hours[i]),
                                minute=self.rng.integers(0, 60),
                                second=self.rng.integers(0, 60))
            adjusted.append(new_ts)

        return np.array(adjusted)

    def _assign_highways(self):
        """Assign highway IDs based on traffic volume weights.

        Returns:
            Array of highway ID strings.
        """
        ids = list(HIGHWAYS.keys())
        weights = [HIGHWAYS[h]["weight"] for h in ids]
        return self.rng.choice(ids, size=NUM_INCIDENTS, p=weights)

    def _generate_coordinates(self, highways):
        """Generate GPS coordinates along highway corridors.

        Args:
            highways: Array of highway ID strings.

        Returns:
            Nx2 array of (lat, lon) coordinates.
        """
        coords = np.zeros((NUM_INCIDENTS, 2))

        for i, hw_id in enumerate(highways):
            hw = HIGHWAYS[hw_id]
            t = self.rng.random()
            lat = hw["lat_range"][0] + t * (hw["lat_range"][1] - hw["lat_range"][0])
            lon = hw["lon_range"][0] + t * (hw["lon_range"][1] - hw["lon_range"][0])
            lat += self.rng.normal(0, 0.02)
            lon += self.rng.normal(0, 0.02)
            coords[i] = [lat, lon]

        return coords

    def _assign_segments(self, highways):
        """Assign road segment IDs within each highway.

        Args:
            highways: Array of highway ID strings.

        Returns:
            Array of segment ID strings.
        """
        segments = []
        for hw_id in highways:
            num_seg = HIGHWAYS[hw_id]["segments"]
            seg_num = self.rng.integers(0, num_seg)
            segments.append(f"{hw_id}-S{seg_num:03d}")
        return np.array(segments)

    def _generate_severities(self, timestamps, highways):
        """Generate severity levels with time and highway adjustments.

        Args:
            timestamps: Array of timestamps.
            highways: Array of highway IDs.

        Returns:
            Array of severity values (1-4).
        """
        base_probs = [0.40, 0.30, 0.20, 0.10]
        severities = []

        for ts, hw in zip(timestamps, highways):
            probs = list(base_probs)

            hour = pd.Timestamp(ts).hour
            if LATE_NIGHT_HOURS[0] <= hour <= LATE_NIGHT_HOURS[1]:
                probs[2] += 0.05
                probs[3] += 0.05
                probs[0] -= 0.10

            if hw == "E30":
                probs[3] += 0.03
                probs[0] -= 0.03

            probs = np.clip(probs, 0.01, None)
            probs = np.array(probs) / sum(probs)
            severities.append(self.rng.choice([1, 2, 3, 4], p=probs))

        return np.array(severities)

    def _generate_weather(self, timestamps):
        """Generate weather conditions with seasonal sandstorm patterns.

        Args:
            timestamps: Array of timestamps.

        Returns:
            Array of weather condition strings.
        """
        weather = []

        for ts in timestamps:
            month = pd.Timestamp(ts).month
            weights = list(WEATHER_WEIGHTS)

            if month in [3, 4, 5]:
                weights[4] *= 2.5
                weights[5] *= 1.5

            if month in [11, 12, 1]:
                weights[2] *= 1.5
                weights[3] *= 2.0

            weights = np.array(weights) / sum(weights)
            weather.append(self.rng.choice(WEATHER_OPTIONS, p=weights))

        return np.array(weather)

    def _generate_vehicle_counts(self, severities):
        """Generate vehicle counts correlated with severity.

        Args:
            severities: Array of severity levels.

        Returns:
            Array of vehicle counts.
        """
        counts = []
        for sev in severities:
            if sev == 1:
                count = self.rng.integers(1, 3)
            elif sev == 2:
                count = self.rng.integers(2, 5)
            elif sev == 3:
                count = self.rng.integers(2, 7)
            else:
                count = self.rng.integers(3, 12)
            counts.append(count)
        return np.array(counts)

    def _generate_visibility(self, weather):
        """Generate visibility distances based on weather.

        Args:
            weather: Array of weather condition strings.

        Returns:
            Array of visibility in kilometers.
        """
        visibility = []
        for w in weather:
            if w == "clear":
                vis = self.rng.uniform(8.0, 15.0)
            elif w == "cloudy":
                vis = self.rng.uniform(5.0, 12.0)
            elif w == "rain":
                vis = self.rng.uniform(2.0, 6.0)
            elif w == "fog":
                vis = self.rng.uniform(0.2, 2.0)
            elif w == "sandstorm":
                vis = self.rng.uniform(0.05, 1.0)
            else:
                vis = self.rng.uniform(1.0, 4.0)
            visibility.append(round(vis, 2))
        return np.array(visibility)

    def _inject_hotspot_events(self, df):
        """Inject clustered hotspot events simulating sandstorm pile-ups.

        Creates 50 hotspot events with 10+ crashes in a 2-hour window on
        a 20km stretch of highway.

        Args:
            df: Base incident DataFrame.

        Returns:
            DataFrame with hotspot incidents appended.
        """
        print(f"  Injecting {NUM_HOTSPOT_EVENTS} hotspot events...")
        hotspot_rows = []

        for h in range(NUM_HOTSPOT_EVENTS):
            hw_id = self.rng.choice(list(HIGHWAYS.keys()))
            hw = HIGHWAYS[hw_id]

            t_param = self.rng.random()
            center_lat = hw["lat_range"][0] + t_param * (hw["lat_range"][1] - hw["lat_range"][0])
            center_lon = hw["lon_range"][0] + t_param * (hw["lon_range"][1] - hw["lon_range"][0])

            day_offset = self.rng.integers(0, 365)
            hour = self.rng.integers(0, 24)
            base_time = pd.Timestamp(YEAR_START) + pd.Timedelta(days=int(day_offset), hours=int(hour))

            num_crashes = self.rng.integers(10, 20)
            seg_base = self.rng.integers(0, HIGHWAYS[hw_id]["segments"])

            for c in range(num_crashes):
                time_offset = self.rng.integers(0, 120)
                ts = base_time + pd.Timedelta(minutes=int(time_offset))
                lat = center_lat + self.rng.normal(0, 0.05)
                lon = center_lon + self.rng.normal(0, 0.05)
                seg_offset = self.rng.integers(-2, 3)
                seg_id = f"{hw_id}-S{max(0, seg_base + seg_offset):03d}"

                hotspot_rows.append({
                    "incident_id": f"INC-H{h:03d}-{c:03d}",
                    "timestamp": ts,
                    "lat": lat,
                    "lon": lon,
                    "highway_id": hw_id,
                    "severity": int(self.rng.choice([3, 4], p=[0.4, 0.6])),
                    "vehicle_count": int(self.rng.integers(3, 15)),
                    "weather": "sandstorm",
                    "road_segment_id": seg_id,
                    "is_sandstorm": 1,
                    "visibility_km": round(float(self.rng.uniform(0.05, 0.5)), 2),
                })

        hotspot_df = pd.DataFrame(hotspot_rows)
        combined = pd.concat([df, hotspot_df], ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        return combined

    def save(self, df, filename="crash_incidents.csv"):
        """Save the dataset to CSV.

        Args:
            df: Incident DataFrame.
            filename: Output filename.
        """
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Dataset saved to {path} ({len(df)} records)")


if __name__ == "__main__":
    generator = CrashDataGenerator()
    data = generator.generate()
    generator.save(data)

