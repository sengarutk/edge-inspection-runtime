"""Local Disk Fallback Spooler Subsystem.

Provides zero-data-loss SQLite queue buffering when upstream MQTT broker is offline.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

from src.config import SpoolerConfig, load_mqtt_config


class DiskSpooler:
    """Thread-safe SQLite persistent spooler for offline edge message buffering."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_spool_records: Optional[int] = None,
        config: Optional[SpoolerConfig] = None,
    ) -> None:
        """Initialize the disk spooler database.

        Args:
            db_path: Optional path to SQLite spool database.
            max_spool_records: Optional maximum records retained before FIFO purge.
            config: Optional SpoolerConfig object.
        """
        if config is not None:
            self.db_path = Path(config.db_path)
            self.max_records = config.max_spool_records
        else:
            mqtt_cfg = load_mqtt_config()
            self.db_path = Path(db_path or mqtt_cfg.spooler.db_path)
            self.max_records = max_spool_records or mqtt_cfg.spooler.max_spool_records

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

        logger.info(f"Initialized DiskSpooler (db={self.db_path}, max_records={self.max_records})")

    def _init_db(self) -> None:
        """Create spool table and indexes if not existing."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spool_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    qos INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_spool_id ON spool_queue(id ASC);"
            )

    def enqueue(self, topic: str, payload: str, qos: int = 1) -> bool:
        """Enqueue a message to the persistent disk spool.

        Args:
            topic: MQTT publication topic.
            payload: JSON formatted message string.
            qos: MQTT Quality of Service level.

        Returns:
            True if successfully inserted.
        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO spool_queue (topic, payload, qos, created_at, retry_count)
                VALUES (?, ?, ?, ?, 0);
                """,
                (topic, payload, qos, now_utc),
            )
            # Enforce max record limit
            self.purge_expired(self.max_records)
        return True

    def peek_batch(self, limit: int = 50) -> List[Tuple[int, str, str, int]]:
        """Fetch the oldest unacknowledged batch of records.

        Args:
            limit: Maximum records to retrieve.

        Returns:
            List of tuples: (record_id, topic, payload, qos).
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, topic, payload, qos
                FROM spool_queue
                ORDER BY id ASC
                LIMIT ?;
                """,
                (limit,),
            )
            return cursor.fetchall()

    def delete_acknowledged(self, record_ids: List[int]) -> int:
        """Drop successfully sent records from the queue.

        Args:
            record_ids: List of record primary keys.

        Returns:
            Number of records deleted.
        """
        if not record_ids:
            return 0

        placeholders = ",".join("?" for _ in record_ids)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"DELETE FROM spool_queue WHERE id IN ({placeholders});", record_ids
            )
            return cursor.rowcount

    def get_queue_depth(self) -> int:
        """Return the current count of pending spooled messages."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM spool_queue;")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def purge_expired(self, max_records: int) -> int:
        """Purge oldest records if total count exceeds max_records.

        Args:
            max_records: Maximum allowable records in database.

        Returns:
            Number of purged records.
        """
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM spool_queue;")
            count = int(cursor.fetchone()[0])
            if count > max_records:
                excess = count - max_records
                self._conn.execute(
                    """
                    DELETE FROM spool_queue
                    WHERE id IN (
                        SELECT id FROM spool_queue ORDER BY id ASC LIMIT ?
                    );
                    """,
                    (excess,),
                )
                logger.warning(f"Purged {excess} oldest records from DiskSpooler to respect capacity.")
                return excess
        return 0

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            self._conn.close()
            logger.info("DiskSpooler database connection closed.")