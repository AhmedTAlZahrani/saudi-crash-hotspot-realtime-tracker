import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.crash_stream_sim import CrashDataGenerator
from src.stream_ingestor import StreamIngestor
from src.spatial_clustering import HotspotDetector
from src.risk_scorer import RiskScorer
from src.alert_dispatcher import AlertDispatcher
from src.dashboard_data import DashboardDataManager

PAGE_TITLE = "Saudi Crash Hotspot Tracker"
PAGE_ICON = "🚨"
MAP_CENTER_LAT = 24.0
MAP_CENTER_LON = 44.0
DEFAULT_ZOOM = 6

SEVERITY_COLORS = {
    1: "#2ecc71",
    2: "#f1c40f",
    3: "#e67e22",
    4: "#e74c3c",
}

RISK_COLORS = {
    "low": "#2ecc71",
    "moderate": "#f1c40f",
    "high": "#e67e22",
    "critical": "#e74c3c",
}


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def load_data():
    """Load or generate the crash incident dataset.

    Returns:
        DataFrame of crash incidents.
    """
    data_path = Path("data/crash_incidents.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    else:
        generator = CrashDataGenerator()
        df = generator.generate()
        generator.save(df)
    return df


def create_sidebar(df):
    """Build sidebar controls for filtering and configuration.

    Args:
        df: Full incident DataFrame.

    Returns:
        Dict of filter parameters.
    """
    st.sidebar.header("Controls")

    time_window = st.sidebar.slider(
        "Time Window (hours)", min_value=1, max_value=72, value=24, step=1,
    )

    severity_filter = st.sidebar.multiselect(
        "Severity Filter", options=[1, 2, 3, 4], default=[1, 2, 3, 4],
    )

    highway_filter = st.sidebar.multiselect(
        "Highway Filter",
        options=sorted(df["highway_id"].unique()),
        default=sorted(df["highway_id"].unique()),
    )

    eps_km = st.sidebar.slider(
        "DBSCAN eps (km)", min_value=1.0, max_value=20.0, value=5.0, step=0.5,
    )

    eps_min = st.sidebar.slider(
        "DBSCAN eps (minutes)", min_value=30, max_value=360, value=120, step=30,
    )

    min_samples = st.sidebar.slider(
        "Min Samples", min_value=2, max_value=10, value=3, step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Saudi Crash Hotspot Tracker**")
    st.sidebar.markdown("Real-time spatio-temporal detection")

    return {
        "time_window": time_window,
        "severity_filter": severity_filter,
        "highway_filter": highway_filter,
        "eps_km": eps_km,
        "eps_min": eps_min,
        "min_samples": min_samples,
    }


def filter_data(df, params, reference_time=None):
    """Apply sidebar filters to the dataset.

    Args:
        df: Full incident DataFrame.
        params: Dict of filter parameters from sidebar.
        reference_time: End time for the window.

    Returns:
        Filtered DataFrame.
    """
    filtered = df.copy()

    if reference_time is None:
        reference_time = filtered["timestamp"].max()

    cutoff = reference_time - pd.Timedelta(hours=params["time_window"])
    filtered = filtered[filtered["timestamp"] >= cutoff]
    filtered = filtered[filtered["severity"].isin(params["severity_filter"])]
    filtered = filtered[filtered["highway_id"].isin(params["highway_filter"])]

    return filtered


def render_live_map(df, hotspot_meta):
    """Render the live crash incident map with hotspot overlays.

    Args:
        df: Filtered incident DataFrame.
        hotspot_meta: DataFrame of cluster metadata.
    """
    st.subheader("Live Crash Incident Map")

    m = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=DEFAULT_ZOOM,
        tiles="CartoDB dark_matter",
    )

    sample = df.sample(n=min(2000, len(df)), random_state=42) if len(df) > 2000 else df

    for _, row in sample.iterrows():
        color = SEVERITY_COLORS.get(row["severity"], "#95a5a6")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2 + row["severity"],
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"ID: {row['incident_id']}<br>"
                  f"Severity: {row['severity']}<br>"
                  f"Highway: {row['highway_id']}<br>"
                  f"Weather: {row['weather']}",
        ).add_to(m)

    if len(hotspot_meta) > 0:
        for _, cluster in hotspot_meta.iterrows():
            severity_color = SEVERITY_COLORS.get(
                min(4, max(1, int(round(cluster["avg_severity"])))), "#e74c3c"
            )
            folium.Circle(
                location=[cluster["center_lat"], cluster["center_lon"]],
                radius=5000,
                color=severity_color,
                fill=True,
                fill_opacity=0.2,
                popup=f"Cluster {cluster['cluster_id']}<br>"
                      f"Incidents: {cluster['incident_count']}<br>"
                      f"Avg Severity: {cluster['avg_severity']}",
            ).add_to(m)

    st_folium(m, width=None, height=500)

    col1, col2, col3 = st.columns(3)
    col1.metric("Incidents Displayed", len(sample))
    col2.metric("Active Hotspots", len(hotspot_meta))
    col3.metric("Avg Severity", round(df["severity"].mean(), 2) if len(df) > 0 else 0)


def render_risk_heatmap(risk_df, df):
    """Render the road segment risk heatmap.

    Args:
        risk_df: DataFrame of segment risk scores.
        df: Incident DataFrame for coordinate lookup.
    """
    st.subheader("Road Segment Risk Heatmap")

    if len(risk_df) == 0:
        st.info("No risk data available for the current window.")
        return

    m = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=DEFAULT_ZOOM,
        tiles="CartoDB dark_matter",
    )

    seg_coords = df.groupby("road_segment_id").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    ).reset_index()

    merged = risk_df.merge(seg_coords, on="road_segment_id", how="left").dropna(subset=["lat", "lon"])

    top_segments = merged.head(200)

    for _, row in top_segments.iterrows():
        color = RISK_COLORS.get(row["risk_level"], "#95a5a6")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3 + row["risk_score"] / 20,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Segment: {row['road_segment_id']}<br>"
                  f"Risk: {row['risk_score']}<br>"
                  f"Level: {row['risk_level']}",
        ).add_to(m)

    st_folium(m, width=None, height=500)

    col1, col2, col3, col4 = st.columns(4)
    risk_dist = risk_df["risk_level"].value_counts()
    col1.metric("Critical", risk_dist.get("critical", 0))
    col2.metric("High", risk_dist.get("high", 0))
    col3.metric("Moderate", risk_dist.get("moderate", 0))
    col4.metric("Low", risk_dist.get("low", 0))

    fig = px.histogram(
        risk_df, x="risk_score", nbins=30,
        title="Risk Score Distribution",
        labels={"risk_score": "Risk Score", "count": "Segments"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_hotspot_analysis(df, hotspot_meta, params):
    """Render the hotspot analysis tab.

    Args:
        df: Filtered incident DataFrame.
        hotspot_meta: DataFrame of cluster metadata.
        params: Sidebar parameters.
    """
    st.subheader("Hotspot Analysis")

    st.markdown(f"**ST-DBSCAN Parameters:** eps_spatial={params['eps_km']}km, "
                f"eps_temporal={params['eps_min']}min, min_samples={params['min_samples']}")

    if len(hotspot_meta) > 0:
        st.dataframe(hotspot_meta, use_container_width=True)

        fig_count = px.bar(
            hotspot_meta.head(15),
            x="cluster_id", y="incident_count",
            color="avg_severity",
            title="Top Hotspot Clusters by Incident Count",
            color_continuous_scale="YlOrRd",
        )
        fig_count.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_count, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_sev = px.scatter(
                hotspot_meta, x="incident_count", y="avg_severity",
                size="time_span_minutes", color="dominant_highway",
                title="Incident Count vs Severity",
            )
            fig_sev.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_sev, use_container_width=True)

        with col2:
            hw_counts = hotspot_meta["dominant_highway"].value_counts().reset_index()
            hw_counts.columns = ["highway", "clusters"]
            fig_hw = px.pie(
                hw_counts, values="clusters", names="highway",
                title="Hotspots by Highway",
            )
            fig_hw.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_hw, use_container_width=True)
    else:
        st.info("No hotspot clusters detected in the current window.")

    st.markdown("---")
    st.subheader("Incident Time Series")

    dm = DashboardDataManager()
    ts = dm.compute_incident_time_series(df, freq="1h")

    if len(ts) > 0:
        fig_ts = px.line(
            ts, x="timestamp", y="incident_count",
            title="Hourly Incident Rate",
        )
        fig_ts.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_ts, use_container_width=True)


def render_alert_feed(alerts, alert_stats):
    """Render the alert feed tab.

    Args:
        alerts: List of alert dictionaries.
        alert_stats: Dict of alert statistics.
    """
    st.subheader("Alert Feed")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Alerts", alert_stats.get("total_alerts", 0))
    col2.metric("Critical Alerts", alert_stats.get("critical_alerts", 0))
    col3.metric("Currently Active", alert_stats.get("currently_active", 0))
    col4.metric("Deduplicated", alert_stats.get("deduplicated", 0))

    st.markdown("---")

    if alerts:
        alert_df = pd.DataFrame(alerts)
        display_cols = [
            "alert_id", "severity", "incident_count", "lat", "lon",
            "radius_km", "assigned_zone_name", "distance_to_zone_km",
            "recommended_action", "status",
        ]
        available_cols = [c for c in display_cols if c in alert_df.columns]
        st.dataframe(alert_df[available_cols], use_container_width=True)

        st.markdown("### Dispatch Recommendations")
        for alert in alerts[-5:]:
            severity = alert.get("severity", 1)
            icon = "🔴" if severity >= 4 else "🟠" if severity >= 3 else "🟡" if severity >= 2 else "🟢"
            st.markdown(
                f"{icon} **{alert['alert_id']}** | "
                f"Severity {severity} | "
                f"{alert['incident_count']} incidents | "
                f"→ {alert.get('assigned_zone_name', 'N/A')} "
                f"({alert.get('distance_to_zone_km', 0)} km)"
            )
            st.caption(alert.get("recommended_action", ""))

        st.markdown("---")
        st.subheader("Alert Statistics")

        if len(alert_df) > 1:
            sev_counts = alert_df["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            fig = px.bar(
                sev_counts, x="severity", y="count",
                title="Alerts by Severity Level",
                color="severity",
                color_continuous_scale="YlOrRd",
            )
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alerts dispatched yet.")


def main():
    """Main dashboard entry point."""
    setup_page()
    st.title("Saudi Crash Hotspot Real-Time Tracker")
    st.caption("Spatio-temporal crash hotspot detection and emergency dispatch")

    df = load_data()
    params = create_sidebar(df)
    filtered = filter_data(df, params)

    detector = HotspotDetector(
        eps_km=params["eps_km"],
        eps_minutes=params["eps_min"],
        min_samples=params["min_samples"],
    )

    clustered = detector.detect_st_dbscan(filtered)
    hotspot_meta = detector.get_cluster_metadata(clustered)

    scorer = RiskScorer()
    risk_df = scorer.compute_scores(filtered)

    dispatcher = AlertDispatcher()
    new_alerts = dispatcher.evaluate_hotspots(hotspot_meta)
    all_alerts = dispatcher.get_alert_history()
    alert_stats = dispatcher.get_stats()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Map", "Risk Heatmap", "Hotspot Analysis", "Alert Feed",
    ])

    with tab1:
        render_live_map(filtered, hotspot_meta)

    with tab2:
        render_risk_heatmap(risk_df, filtered)

    with tab3:
        render_hotspot_analysis(filtered, hotspot_meta, params)

    with tab4:
        render_alert_feed(all_alerts, alert_stats)


if __name__ == "__main__":
    main()

