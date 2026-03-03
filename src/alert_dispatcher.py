import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import uuid
from math import radians, cos, sin, asin, sqrt

SEVERITY_THRESHOLD = 3
INCIDENT_COUNT_THRESHOLD = 5
COOLDOWN_MINUTES = 30
ALERT_RADIUS_KM = 10.0
EARTH_RADIUS_KM = 6371.0

RESPONSE_ZONES = {
    "RZ-RYD": {"name": "Riyadh Emergency", "lat": 24.7136, "lon": 46.6753},
    "RZ-JED": {"name": "Jeddah Emergency", "lat": 21.4858, "lon": 39.1925},
    "RZ-DAM": {"name": "Dammam Emergency", "lat": 26.3927, "lon": 49.9777},
    "RZ-MKH": {"name": "Makkah Emergency", "lat": 21.3891, "lon": 39.8579},
    "RZ-MDN": {"name": "Madinah Emergency", "lat": 24.5247, "lon": 39.5692},
    "RZ-TAB": {"name": "Tabuk Emergency", "lat": 28.3998, "lon": 36.5716},
    "RZ-ABH": {"name": "Abha Emergency", "lat": 18.2164, "lon": 42.5053},
    "RZ-BRD": {"name": "Buraydah Emergency", "lat": 26.3260, "lon": 43.9750},
}

RECOMMENDED_ACTIONS = {
    1: "Monitor situation. Standard patrol response.",
    2: "Dispatch traffic patrol. Set up warning signs.",
    3: "Dispatch ambulance and traffic control. Divert traffic.",
    4: "Full emergency response. Close road segment. Helicopter standby.",
}


def _haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in kilometers.

    Args:
        lat1: Latitude of point 1.
        lon1: Longitude of point 1.
        lat2: Latitude of point 2.
        lon2: Longitude of point 2.

    Returns:
        Distance in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return EARTH_RADIUS_KM * c


class AlertDispatcher:
    """Alert dispatch system for detected crash hotspots.

    Generates, deduplicates, and routes alerts to emergency response zones
    with geofencing, cooldown management, and notification routing.
    """

    def __init__(self, severity_threshold=SEVERITY_THRESHOLD,
                 count_threshold=INCIDENT_COUNT_THRESHOLD,
                 cooldown_minutes=COOLDOWN_MINUTES,
                 alert_radius=ALERT_RADIUS_KM):
        self.severity_threshold = severity_threshold
        self.count_threshold = count_threshold
        self.cooldown_minutes = cooldown_minutes
        self.alert_radius = alert_radius
        self._alert_history = []
        self._active_alerts = {}
        self._stats = {
            "total_alerts": 0,
            "critical_alerts": 0,
            "deduplicated": 0,
            "dispatched": 0,
        }

    def evaluate_hotspots(self, cluster_metadata):
        """Evaluate detected clusters and generate alerts for qualifying hotspots.

        Args:
            cluster_metadata: DataFrame from HotspotDetector.get_cluster_metadata()
                with center_lat, center_lon, incident_count, avg_severity columns.

        Returns:
            List of alert dictionaries.
        """
        if len(cluster_metadata) == 0:
            return []

        new_alerts = []

        for _, cluster in cluster_metadata.iterrows():
            if cluster["incident_count"] < self.count_threshold:
                continue

            if cluster["avg_severity"] < self.severity_threshold:
                continue

            if self._is_duplicate(cluster["center_lat"], cluster["center_lon"]):
                self._stats["deduplicated"] += 1
                continue

            alert = self._create_alert(cluster)
            new_alerts.append(alert)

            self._alert_history.append(alert)
            self._active_alerts[alert["alert_id"]] = alert
            self._stats["total_alerts"] += 1

            if cluster["avg_severity"] >= 3.5:
                self._stats["critical_alerts"] += 1

        if new_alerts:
            print(f"Dispatched {len(new_alerts)} new alerts")

        return new_alerts

    def _create_alert(self, cluster):
        """Create an alert from a cluster.

        Args:
            cluster: Series with cluster metadata.

        Returns:
            Dict with alert fields.
        """
        severity = min(4, max(1, int(round(cluster["avg_severity"]))))
        nearest_zone = self._find_nearest_zone(cluster["center_lat"], cluster["center_lon"])

        radius = self.alert_radius
        if cluster["incident_count"] > 15:
            radius *= 1.5
        elif cluster["incident_count"] > 10:
            radius *= 1.2

        alert = {
            "alert_id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": pd.Timestamp.now().isoformat(),
            "lat": round(cluster["center_lat"], 4),
            "lon": round(cluster["center_lon"], 4),
            "radius_km": round(radius, 1),
            "severity": severity,
            "incident_count": int(cluster["incident_count"]),
            "avg_severity": cluster["avg_severity"],
            "recommended_action": RECOMMENDED_ACTIONS.get(severity, RECOMMENDED_ACTIONS[1]),
            "assigned_zone": nearest_zone["zone_id"],
            "assigned_zone_name": nearest_zone["zone_name"],
            "distance_to_zone_km": nearest_zone["distance_km"],
            "status": "active",
        }

        if "dominant_highway" in cluster.index:
            alert["highway_id"] = cluster["dominant_highway"]

        self._stats["dispatched"] += 1
        return alert

    def _is_duplicate(self, lat, lon):
        """Check if an alert already exists in the same area within cooldown.

        Args:
            lat: Latitude of the new alert.
            lon: Longitude of the new alert.

        Returns:
            Boolean indicating if this is a duplicate.
        """
        cutoff = pd.Timestamp.now() - pd.Timedelta(minutes=self.cooldown_minutes)

        for alert in self._alert_history[-100:]:
            alert_time = pd.Timestamp(alert["timestamp"])
            if alert_time < cutoff:
                continue

            dist = _haversine(lat, lon, alert["lat"], alert["lon"])
            if dist < self.alert_radius:
                return True

        return False

    def _find_nearest_zone(self, lat, lon):
        """Find the nearest emergency response zone.

        Args:
            lat: Latitude of the alert location.
            lon: Longitude of the alert location.

        Returns:
            Dict with zone_id, zone_name, and distance_km.
        """
        nearest = None
        min_dist = float("inf")

        for zone_id, zone in RESPONSE_ZONES.items():
            dist = _haversine(lat, lon, zone["lat"], zone["lon"])
            if dist < min_dist:
                min_dist = dist
                nearest = {
                    "zone_id": zone_id,
                    "zone_name": zone["name"],
                    "distance_km": round(dist, 1),
                }

        return nearest

    def check_geofence(self, lat, lon):
        """Check if a point falls within any active alert geofence.

        Args:
            lat: Latitude to check.
            lon: Longitude to check.

        Returns:
            List of alert IDs whose geofence contains the point.
        """
        matching = []

        for alert_id, alert in self._active_alerts.items():
            if alert["status"] != "active":
                continue

            dist = _haversine(lat, lon, alert["lat"], alert["lon"])
            if dist <= alert["radius_km"]:
                matching.append(alert_id)

        return matching

    def resolve_alert(self, alert_id):
        """Mark an alert as resolved.

        Args:
            alert_id: ID of the alert to resolve.

        Returns:
            Boolean indicating success.
        """
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id]["status"] = "resolved"
            print(f"Alert {alert_id} resolved")
            return True
        return False

    def get_active_alerts(self):
        """Get all currently active alerts.

        Returns:
            List of active alert dictionaries.
        """
        return [a for a in self._active_alerts.values() if a["status"] == "active"]

    def get_alert_history(self, last_n=50):
        """Get recent alert history.

        Args:
            last_n: Number of recent alerts to return.

        Returns:
            List of alert dictionaries.
        """
        return self._alert_history[-last_n:]

    def get_stats(self):
        """Get alert dispatch statistics.

        Returns:
            Dict with alert counts and rates.
        """
        active = len([a for a in self._active_alerts.values() if a["status"] == "active"])
        return {
            "total_alerts": self._stats["total_alerts"],
            "critical_alerts": self._stats["critical_alerts"],
            "deduplicated": self._stats["deduplicated"],
            "dispatched": self._stats["dispatched"],
            "currently_active": active,
        }

    def get_zone_workload(self):
        """Get the number of active alerts per response zone.

        Returns:
            Dict mapping zone ID to active alert count.
        """
        workload = {zone_id: 0 for zone_id in RESPONSE_ZONES}

        for alert in self._active_alerts.values():
            if alert["status"] == "active":
                zone_id = alert["assigned_zone"]
                workload[zone_id] = workload.get(zone_id, 0) + 1

        return workload

    def clear_expired_alerts(self, max_age_hours=24):
        """Remove alerts older than the specified age.

        Args:
            max_age_hours: Maximum alert age in hours.

        Returns:
            Number of alerts cleared.
        """
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=max_age_hours)
        cleared = 0

        expired_ids = []
        for alert_id, alert in self._active_alerts.items():
            if pd.Timestamp(alert["timestamp"]) < cutoff:
                expired_ids.append(alert_id)

        for aid in expired_ids:
            self._active_alerts[aid]["status"] = "expired"
            cleared += 1

        if cleared:
            print(f"Cleared {cleared} expired alerts")

        return cleared
