"""
app/utils/cache.py
──────────────────
Redis-backed response cache.

• Uses a deterministic hash of (endpoint, payload) as the cache key.
• Gracefully degrades to a no-op if Redis is unavailable.
• TTL is configurable via settings.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ResponseCache:
    def __init__(self, settings: Settings) -> None:
        self._ttl = settings.redis_ttl_seconds
        try:
            self._client: redis.Redis | None = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            logger.info("Redis cache connected", url=settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable; cache disabled", error=str(exc))
            self._client = None

    @staticmethod
    def _make_key(namespace: str, payload: dict[str, Any]) -> str:
        content = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"support_triage:{namespace}:{digest}"

    def get(self, namespace: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self._client is None:
            return None
        key = self._make_key(namespace, payload)
        try:
            raw = self._client.get(key)
            if raw:
                logger.debug("Cache hit", key=key)
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Cache get failed", error=str(exc))
        return None

    def set(self, namespace: str, payload: dict[str, Any], value: dict[str, Any]) -> None:
        if self._client is None:
            return
        key = self._make_key(namespace, payload)
        try:
            self._client.setex(key, self._ttl, json.dumps(value))
            logger.debug("Cache set", key=key, ttl=self._ttl)
        except Exception as exc:
            logger.warning("Cache set failed", error=str(exc))

    def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False
