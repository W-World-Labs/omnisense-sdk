"""User-configured independent engines; no implementation or model supplier."""
from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit


def algorithm_suffix(algorithm_id: str) -> str:
    return "".join(c.upper() if c.isascii() and c.isalnum() else "_" for c in algorithm_id)


def engine_setting(name: str, algorithm_id: str) -> str | None:
    return (os.environ.get(f"{name}__{algorithm_suffix(algorithm_id)}", "").strip()
            or os.environ.get(name, "").strip() or None)


def engine_url(algorithm_id: str) -> str | None:
    value = engine_setting("OMNISENSE_ENGINE_URL", algorithm_id)
    if value is None:
        return None
    url = urlsplit(value)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("engine URL must be an HTTP(S) origin/path without credentials, query or fragment")
    if url.scheme == "http" and url.hostname != "localhost":
        try:
            address = ipaddress.ip_address(url.hostname)
        except ValueError as exc:
            raise ValueError("remote engine hostnames require HTTPS") from exc
        private = any(address in ipaddress.ip_network(network) for network in
                      (("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16") if address.version == 4 else ("fc00::/7",)))
        if not (address.is_loopback or private):
            raise ValueError("public engine endpoints require HTTPS")
    return value.rstrip("/")


def external_supplier(algorithm_id: str) -> dict[str, str | None] | None:
    # Configured is not a health/availability assertion. Never return the URL or token.
    if not engine_setting("OMNISENSE_ENGINE_URL", algorithm_id):
        return None
    return {"engine": "external-http", "model_id": engine_setting("OMNISENSE_ENGINE_MODEL", algorithm_id)}
