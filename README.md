# Saudi Crash Hotspot Real-Time Tracker

![CI](https://github.com/AhmedTAlZahrani/saudi-crash-hotspot-realtime-tracker/actions/workflows/ci.yml/badge.svg)

Real-time crash hotspot detection and emergency dispatch for Saudi Arabia's highway network. The system ingests a continuous stream of crash incidents, clusters them using ST-DBSCAN (spatial + temporal), scores road segments by dynamic risk, and dispatches alerts to the nearest emergency response zone.

## Streaming Architecture

```
crash event stream
       |
       v
StreamIngestor (sliding window buffer, configurable throughput)
       |
       v
HotspotDetector.detect_st_dbscan()
       |   spatial DBSCAN on GPS coords (haversine metric)
       |   temporal DBSCAN within each spatial cluster
       |
       +---> cluster metadata (center, count, severity, time span)
       |
       v
RiskScorer.compute_scores()
       |   per-segment scoring with exponential decay
       |   weather / time-of-day / road-type multipliers
       |
       v
AlertDispatcher.evaluate_hotspots()
       |   severity + count thresholds
       |   geofence deduplication + cooldown
       |   nearest response zone routing
       |
       v
Streamlit dashboard (Folium maps, Plotly charts, alert feed)
```

## Streaming Endpoints & Data Flow

The `StreamIngestor` replays time-sorted incidents at a configurable rate (`throughput` events/min) and maintains a sliding time window. Downstream consumers call:

- `ingest_batch(batch_size)` -- pull the next batch of events
- `get_window_events()` -- all events in the current sliding window
- `consume_events(count)` -- pop events from the queue
- `seek_to_time(target)` -- jump the stream to a specific timestamp
- `set_throughput(n)` / `set_window(minutes)` -- tune at runtime

The `AlertDispatcher` exposes:

- `evaluate_hotspots(cluster_metadata)` -- generate & route alerts
- `get_active_alerts()` / `get_alert_history(last_n)` -- alert feeds
- `check_geofence(lat, lon)` -- test if a point is inside an active alert zone
- `resolve_alert(alert_id)` -- mark an alert resolved

## Quick Start

```bash
pip install -r requirements.txt

# generate crash data
python -m src.crash_stream_sim

# run dashboard
streamlit run app.py
```

### Docker

```bash
docker build -t crash-hotspot-tracker .
docker run -p 8501:8501 crash-hotspot-tracker
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps_km` | 5.0 | DBSCAN spatial radius (km) |
| `eps_minutes` | 120 | DBSCAN temporal window (min) |
| `min_samples` | 3 | Minimum cluster size |
| `throughput` | 500 | Stream replay rate (events/min) |
| `window_minutes` | 60 | Sliding window size |
| `decay_half_life` | 12h | Risk score decay |

## License

MIT License -- see [LICENSE](LICENSE) for details.
