"""Resilient MQTT Publisher with Local Disk Fallback Spooling.

Provides dual-path event publishing and an automatic background spool draining daemon.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional, Union
from loguru import logger
import paho.mqtt.client as mqtt

from src.config import MQTTConfig, load_mqtt_config
from src.spooler import DiskSpooler


class ResilientMQTTPublisher:
    """Industrial MQTT publisher with automatic disk fallback and drain recovery."""

    def __init__(
        self,
        config: Optional[MQTTConfig] = None,
        spooler: Optional[DiskSpooler] = None,
    ) -> None:
        """Initialize publisher with MQTT and spooler configurations.

        Args:
            config: Optional MQTTConfig instance.
            spooler: Optional DiskSpooler instance.
        """
        self.config = config or load_mqtt_config()
        self.spooler = spooler or DiskSpooler(
            db_path=self.config.spooler.db_path,
            max_spool_records=self.config.spooler.max_spool_records,
        )

        client_id = f"{self.config.broker.client_id_prefix}_{int(time.time())}"
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        self._is_connected = False
        self._running = False
        self._drain_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()

        self._setup_callbacks()
        logger.info(f"Initialized ResilientMQTTPublisher (client_id={client_id})")

    @property
    def is_connected(self) -> bool:
        """True if MQTT client is actively connected to broker."""
        with self._state_lock:
            return self._is_connected

    def _setup_callbacks(self) -> None:
        """Attach Paho MQTT event callbacks."""
        def on_connect(client: mqtt.Client, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
            code = getattr(rc, "value", rc)
            if code == 0:
                with self._state_lock:
                    self._is_connected = True
                logger.info(f"MQTT Publisher connected successfully to {self.config.broker.host}:{self.config.broker.port}")
            else:
                with self._state_lock:
                    self._is_connected = False
                logger.warning(f"MQTT connection refused with result code: {code}")

        def on_disconnect(client: mqtt.Client, userdata: Any, disconnect_flags: Any, rc: Any, properties: Any = None) -> None:
            with self._state_lock:
                self._is_connected = False
            logger.warning(f"MQTT Publisher disconnected from broker (rc={rc})")

        self._client.on_connect = on_connect
        self._client.on_disconnect = on_disconnect

    def start(self) -> None:
        """Connect to broker and start background network and spool drain workers."""
        with self._state_lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()

        try:
            self._client.reconnect_delay_set(
                min_delay=int(self.config.broker.reconnect_delay_min_s),
                max_delay=int(self.config.broker.reconnect_delay_max_s),
            )
            self._client.connect_async(
                host=self.config.broker.host,
                port=self.config.broker.port,
                keepalive=self.config.broker.keepalive,
            )
            self._client.loop_start()
        except Exception as exc:
            logger.warning(f"Initial async MQTT broker connection failed: {exc}. Spooler active.")

        # Start drain worker
        self._drain_thread = threading.Thread(target=self._drain_worker, daemon=True, name="mqtt_spool_drainer")
        self._drain_thread.start()
        logger.info("ResilientMQTTPublisher started.")

    def stop(self) -> None:
        """Stop background worker threads and disconnect cleanly."""
        with self._state_lock:
            if not self._running:
                return
            self._running = False

        self._stop_event.set()
        if self._drain_thread and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=2.0)

        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception as exc:
            logger.debug(f"Disconnect cleanup exception: {exc}")

        logger.info("ResilientMQTTPublisher stopped.")

    def publish_event(
        self,
        topic: str,
        payload: Union[Dict[str, Any], str],
        qos: Optional[int] = None,
    ) -> bool:
        """Publish message via dual-path dispatch (immediate MQTT or disk spool fallback).

        Args:
            topic: Destination MQTT topic.
            payload: Dictionary or serialized JSON string.
            qos: Optional QoS level. Defaults to topic configuration.

        Returns:
            True if sent or safely persisted to disk.
        """
        payload_str = payload if isinstance(payload, str) else json.dumps(payload)
        effective_qos = qos if qos is not None else self.config.qos.risk_events

        if self.is_connected:
            try:
                info = self._client.publish(topic, payload_str, qos=effective_qos)
                if info.rc == mqtt.MQTT_ERR_SUCCESS:
                    return True
                logger.warning(f"MQTT publish failed with rc={info.rc}. Committing to DiskSpooler.")
            except Exception as exc:
                logger.warning(f"Exception during MQTT publish: {exc}. Committing to DiskSpooler.")

        # Offline / Failure fallback path
        self.spooler.enqueue(topic, payload_str, effective_qos)
        return True

    def publish(self, topic: str, payload: Union[Dict[str, Any], str], qos: Optional[int] = None) -> bool:
        """Alias for publish_event."""
        return self.publish_event(topic=topic, payload=payload, qos=qos)

    def close(self) -> None:
        """Alias for stop."""
        self.stop()

    def publish_heartbeat(self, status: Dict[str, Any]) -> bool:
        """Publish periodic component liveness heartbeat."""
        return self.publish_event(
            topic=self.config.topics.heartbeat,
            payload=status,
            qos=self.config.qos.heartbeat,
        )

    def _drain_worker(self) -> None:
        """Background thread continuously draining spooled records upon broker reconnection."""
        logger.info("Started background MQTT spool drain worker.")
        while not self._stop_event.is_set():
            try:
                if self.is_connected and self.spooler.get_queue_depth() > 0:
                    batch = self.spooler.peek_batch(limit=50)
                    for rec_id, topic, payload, qos in batch:
                        if not self.is_connected or self._stop_event.is_set():
                            break
                        try:
                            info = self._client.publish(topic, payload, qos=qos)
                            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                                self.spooler.delete_acknowledged([rec_id])
                            else:
                                break
                        except Exception as exc:
                            logger.error(f"Error publishing spooled record {rec_id}: {exc}")
                            break
            except Exception as exc:
                logger.error(f"Unexpected exception in spool drain worker: {exc}")

            self._stop_event.wait(0.3)
        logger.info("Spool drain worker terminated.")