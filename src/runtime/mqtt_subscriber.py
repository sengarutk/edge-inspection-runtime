"""MQTT Event Subscriber & Ingestion Subsystem.

Listens to edge inspection MQTT topics and ingests messages into the SQLite audit database.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, Optional
from loguru import logger
import paho.mqtt.client as mqtt

from src.config import MQTTConfig, load_mqtt_config


class MQTTEventSubscriber:
    """Subscriber client receiving edge events, telemetry, and health heartbeats."""

    def __init__(
        self,
        config: Optional[MQTTConfig] = None,
        audit_db: Optional[Any] = None,
        on_event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize subscriber with MQTT config and optional persistence handler.

        Args:
            config: Optional MQTTConfig instance.
            audit_db: Optional AuditLogDB instance for auto-ingestion.
            on_event_callback: Optional custom callback invoked for every parsed event.
        """
        self.config = config or load_mqtt_config()
        self.audit_db = audit_db
        self.on_event_callback = on_event_callback

        client_id = f"{self.config.broker.client_id_prefix}_sub_{int(time.time())}"
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        self._is_connected = False
        self._running = False
        self._state_lock = threading.RLock()

        self._setup_callbacks()
        logger.info(f"Initialized MQTTEventSubscriber (client_id={client_id})")

    @property
    def is_connected(self) -> bool:
        """True if MQTT subscriber is actively connected."""
        with self._state_lock:
            return self._is_connected

    def _setup_callbacks(self) -> None:
        """Attach Paho MQTT event and message callbacks."""
        def on_connect(client: mqtt.Client, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
            code = getattr(rc, "value", rc)
            if code == 0:
                with self._state_lock:
                    self._is_connected = True
                # Subscribe to all line1 topics
                sub_pattern = "inspection/line1/#"
                client.subscribe(sub_pattern, qos=1)
                logger.info(f"MQTT Subscriber connected and subscribed to {sub_pattern}")
            else:
                with self._state_lock:
                    self._is_connected = False
                logger.warning(f"MQTT subscriber connection failed with code: {code}")

        def on_disconnect(client: mqtt.Client, userdata: Any, disconnect_flags: Any, rc: Any, properties: Any = None) -> None:
            with self._state_lock:
                self._is_connected = False
            logger.warning(f"MQTT subscriber disconnected (rc={rc})")

        def on_message(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
            try:
                payload_str = msg.payload.decode("utf-8")
                data = json.loads(payload_str)
                self._handle_message(msg.topic, data)
            except Exception as exc:
                logger.error(f"Failed to process incoming MQTT message on {msg.topic}: {exc}")

        self._client.on_connect = on_connect
        self._client.on_disconnect = on_disconnect
        self._client.on_message = on_message

    def _handle_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Dispatch deserialized message to audit database and custom callbacks."""
        if self.audit_db is not None:
            try:
                if topic == self.config.topics.risk_events:
                    self.audit_db.insert_risk_event(data)
                elif topic == self.config.topics.telemetry:
                    # Ingest telemetry dict if provided
                    pass
                elif topic == self.config.topics.health:
                    self.audit_db.insert_system_health(
                        component=data.get("component", "system"),
                        status=data.get("status", "HEALTHY"),
                        details=json.dumps(data),
                    )
            except Exception as exc:
                logger.error(f"Error persisting event to audit DB: {exc}")

        if self.on_event_callback is not None:
            try:
                self.on_event_callback(topic, data)
            except Exception as exc:
                logger.error(f"Error in custom on_event_callback: {exc}")

    def start(self) -> None:
        """Connect to MQTT broker and start network listener."""
        with self._state_lock:
            if self._running:
                return
            self._running = True

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
            logger.info("MQTTEventSubscriber started.")
        except Exception as exc:
            logger.warning(f"MQTT subscriber connection error: {exc}")

    def stop(self) -> None:
        """Stop network listener and disconnect cleanly."""
        with self._state_lock:
            if not self._running:
                return
            self._running = False

        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception as exc:
            logger.debug(f"Subscriber disconnect error: {exc}")

        logger.info("MQTTEventSubscriber stopped.")