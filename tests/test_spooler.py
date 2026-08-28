"""Unit tests for local SQLite disk spooler fallback subsystem."""

from pathlib import Path
import pytest

from src.config import SpoolerConfig
from src.spooler import DiskSpooler


@pytest.fixture
def spooler(tmp_path: Path) -> DiskSpooler:
    """Fixture providing an isolated DiskSpooler instance in a temp directory."""
    db_file = tmp_path / "spooler.db"
    cfg = SpoolerConfig(db_path=str(db_file), max_spool_records=10)
    sp = DiskSpooler(config=cfg)
    yield sp
    sp.close()


def test_spooler_direct_path_initialization(tmp_path: Path) -> None:
    """Test initializing DiskSpooler directly with db_path string."""
    db_file = tmp_path / "direct_spool.db"
    sp = DiskSpooler(db_path=str(db_file), max_spool_records=25)
    assert sp.max_records == 25
    sp.close()


def test_enqueue_and_queue_depth(spooler: DiskSpooler) -> None:
    """Test enqueuing messages and tracking queue depth."""
    assert spooler.get_queue_depth() == 0

    spooler.enqueue("inspection/test", '{"msg": 1}', qos=1)
    spooler.enqueue("inspection/test", '{"msg": 2}', qos=1)
    spooler.enqueue("inspection/test", '{"msg": 3}', qos=0)

    assert spooler.get_queue_depth() == 3


def test_peek_and_delete_fifo_ordering(spooler: DiskSpooler) -> None:
    """Test that peek_batch retrieves records in FIFO order and delete_acknowledged drops them."""
    spooler.enqueue("inspection/test", '{"val": "first"}', qos=1)
    spooler.enqueue("inspection/test", '{"val": "second"}', qos=1)
    spooler.enqueue("inspection/test", '{"val": "third"}', qos=1)

    batch = spooler.peek_batch(limit=2)
    assert len(batch) == 2
    rec1_id, topic1, payload1, qos1 = batch[0]
    rec2_id, topic2, payload2, qos2 = batch[1]

    assert "first" in payload1
    assert "second" in payload2
    assert qos1 == 1

    deleted = spooler.delete_acknowledged([rec1_id])
    assert deleted == 1
    assert spooler.get_queue_depth() == 2

    next_batch = spooler.peek_batch(limit=10)
    assert len(next_batch) == 2
    assert "second" in next_batch[0][2]
    assert "third" in next_batch[1][2]

    spooler.delete_acknowledged([next_batch[0][0], next_batch[1][0]])
    assert spooler.get_queue_depth() == 0
    assert spooler.delete_acknowledged([]) == 0


def test_purge_expired_capacity_cap(spooler: DiskSpooler) -> None:
    """Test that max_records capacity is enforced by dropping oldest records."""
    for i in range(15):
        spooler.enqueue("inspection/test", f'{{"index": {i}}}', qos=1)

    assert spooler.get_queue_depth() == 10
    batch = spooler.peek_batch(limit=1)
    assert '"index": 5' in batch[0][2]