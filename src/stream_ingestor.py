import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
import time

DEFAULT_THROUGHPUT = 500
DEFAULT_WINDOW_MINUTES = 60
DEFAULT_DATA_PATH = "data/crash_incidents.csv"


class StreamIngestor:
    """Simulated real-time event ingestion from a pre-generated crash dataset.

    Replays time-sorted incidents at configurable throughput, maintaining a
    sliding time window buffer for downstream consumers.
    """

    def __init__(self, data_path=DEFAULT_DATA_PATH, throughput=DEFAULT_THROUGHPUT,
                 window_minutes=DEFAULT_WINDOW_MINUTES):
        self.data_path = Path(data_path)
        self.throughput = throughput
        self.window_minutes = window_minutes
        self._events = None
        self._event_queue = deque()
        self._window_buffer = deque()
        self._current_index = 0
        self._current_time = None
        self._running = False
        self._total_ingested = 0

    def load_data(self):
        """Load and sort the crash dataset by timestamp."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        self._events = df
        self._current_time = df["timestamp"].iloc[0]
        print(f"  Loaded {len(df)} events")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return len(df)

    def start(self):
        if self._events is None:
            self.load_data()
        self._running = True
        self._current_index = 0
        self._total_ingested = 0
        print(f"Stream started | Throughput: {self.throughput} events/min | Window: {self.window_minutes} min")

    def stop(self):
        self._running = False
        print(f"Stream stopped | Total ingested: {self._total_ingested}")

    def is_running(self):
        """Check if the stream is active.

        Returns:
            Boolean indicating stream status.
        """
        return self._running and self._current_index < len(self._events)

    def ingest_batch(self, batch_size=None):
        """Ingest the next batch of events from the dataset.

        Args:
            batch_size: Number of events to ingest. Defaults to throughput setting.

        Returns:
            DataFrame of newly ingested events.
        """
        if not self._running:
            print("Stream is not running. Call start() first.")
            return pd.DataFrame()

        if batch_size is None:
            batch_size = self.throughput

        end_index = min(self._current_index + batch_size, len(self._events))
        batch = self._events.iloc[self._current_index:end_index].copy()

        if len(batch) == 0:
            self._running = False
            print("End of dataset reached.")
            return pd.DataFrame()

        self._current_time = batch["timestamp"].iloc[-1]

        for _, row in batch.iterrows():
            event = row.to_dict()
            self._event_queue.append(event)
            self._window_buffer.append(event)

        self._trim_window()
        self._current_index = end_index
        self._total_ingested += len(batch)

        # hack: sleep to avoid rate limiting
        import time as _t
        _t.sleep(0.01)

        return batch

    def _trim_window(self):
        if self._current_time is None:
            return

        cutoff = self._current_time - pd.Timedelta(minutes=self.window_minutes)

        while self._window_buffer and pd.Timestamp(self._window_buffer[0]["timestamp"]) < cutoff:
            self._window_buffer.popleft()

    def get_window_events(self):
        """Get all events in the current sliding time window.

        Returns:
            DataFrame of events within the window.
        """
        if not self._window_buffer:
            return pd.DataFrame()
        return pd.DataFrame(list(self._window_buffer))

    def consume_events(self, count=None):
        """Consume events from the event queue.

        Args:
            count: Number of events to consume. None consumes all.

        Returns:
            List of event dictionaries.
        """
        if count is None:
            count = len(self._event_queue)

        consumed = []
        for _ in range(min(count, len(self._event_queue))):
            consumed.append(self._event_queue.popleft())

        return consumed

    def get_current_time(self):
        """Get the current replay timestamp.

        Returns:
            Current timestamp in the replay stream.
        """
        return self._current_time

    def get_queue_size(self):
        """Get the number of unconsumed events in the queue.

        Returns:
            Integer count of queued events.
        """
        return len(self._event_queue)

    def get_window_size(self):
        """Get the number of events in the sliding window.

        Returns:
            Integer count of windowed events.
        """
        return len(self._window_buffer)

    def get_progress(self):
        """Get ingestion progress as a percentage.

        Returns:
            Float between 0 and 100.
        """
        if self._events is None or len(self._events) == 0:
            return 0.0
        return round(100.0 * self._current_index / len(self._events), 2)

    def get_stats(self):
        """Get current stream statistics.

        Returns:
            Dict with ingestion metrics.
        """
        return {
            "total_events": len(self._events) if self._events is not None else 0,
            "ingested": self._total_ingested,
            "progress_pct": self.get_progress(),
            "queue_size": self.get_queue_size(),
            "window_size": self.get_window_size(),
            "current_time": str(self._current_time),
            "running": self._running,
        }

    def seek_to_time(self, target_time):
        """Jump the stream to a specific timestamp.

        Args:
            target_time: Timestamp to seek to.

        Returns:
            Number of events skipped.
        """
        if self._events is None:
            self.load_data()

        target = pd.Timestamp(target_time)
        mask = self._events["timestamp"] <= target
        new_index = mask.sum()
        skipped = new_index - self._current_index

        self._current_index = new_index
        self._current_time = target
        self._window_buffer.clear()
        self._event_queue.clear()

        window_start = max(0, new_index - self.throughput)
        window_events = self._events.iloc[window_start:new_index]
        for _, row in window_events.iterrows():
            self._window_buffer.append(row.to_dict())

        self._trim_window()
        print(f"Seeked to {target_time} | Skipped {skipped} events")
        return skipped

    def set_throughput(self, events_per_minute):
        """Update the ingestion throughput.

        Args:
            events_per_minute: New throughput rate.
        """
        self.throughput = events_per_minute
        print(f"Throughput set to {events_per_minute} events/min")

    def set_window(self, minutes):
        """Update the sliding window duration.

        Args:
            minutes: New window size in minutes.
        """
        self.window_minutes = minutes
        self._trim_window()
        print(f"Window set to {minutes} minutes")
