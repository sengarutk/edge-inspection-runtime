"""Disk spooler stress, capacity boundaries, and FIFO purge tests."""

import json
from pathlib import Path
import pytest

from src.config import SpoolerConfig
from src.spooler import DiskSpooler


def test_spooler_capacity_and_fifo_purge(tmp_path: Path) -> None:
    """Verify FIFO queue purging when exceeding maximum record capacity."""
    db_file = tmp_path / "spool_cap_test.db"
    cfg = SpoolerConfig(db_path=str(db_file), max_spool_records=50)
    spooler = DiskSpooler(config=cfg)

    # Enqueue 75 records (capacity 50)
    for i in range(75):
        spooler.enqueue("test/topic", json.dumps({"seq": i}), qos=1)

    # Peak depth must be capped at 50
    depth = spooler.get_queue_depth()
    assert depth == 50

    # Oldest 25 records (0..24) must have been purged, oldest retained should be seq 25
    batch = spooler.peek_batch(limit=1)
    assert len(batch) == 1
    # batch tuple format: (id, topic, payload, qos)
    oldest_data = json.loads(batch[0][2])
    assert oldest_data["seq"] == 25

    spooler.close()


def test_spooler_drain_reconnection(tmp_path: Path) -> None:
    """Verify full drain and zero loss under nominal capacity buffer."""
    db_file = tmp_path / "spool_drain_test.db"
    cfg = SpoolerConfig(db_path=str(db_file), max_spool_records=1000)
    spooler = DiskSpooler(config=cfg)

    for i in range(150):
        spooler.enqueue("test/topic", json.dumps({"val": i}), qos=1)

    assert spooler.get_queue_depth() == 150

    drained_count = 0
    while True:
        batch = spooler.peek_batch(limit=50)
        if not batch:
            break
        # batch tuple: (id, topic, payload, qos)
        ids = [b[0] for b in batch]
        spooler.delete_acknowledged(ids)
        drained_count += len(batch)

    assert drained_count == 150
    assert spooler.get_queue_depth() == 0

    spooler.close()
