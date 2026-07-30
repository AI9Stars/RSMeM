"""HTTP endpoints for Perception tool microservices (env-configurable)."""

from __future__ import annotations

import os


def _host() -> str:
    return os.getenv("RSMEM_SERVICE_HOST", "127.0.0.1")


def service_url(port: int, path: str = "") -> str:
    base = f"http://{_host()}:{port}"
    if not path:
        return base
    return f"{base}/{path.lstrip('/')}"


# Default ports used by the RSMeM perception service interfaces.
SAM2_API_BASE = os.getenv("SAM2_API_URL", service_url(5554))
REMOTECLIP_CLASSIFY_URL = os.getenv("REMOTECLIP_API_URL", service_url(5556, "classify"))
REMOTECLIP_HEALTH_URL = os.getenv("REMOTECLIP_HEALTH_URL", service_url(5556, "health"))
STRIP_RCNN_DETECT_URL = os.getenv("STRIP_RCNN_API_URL", service_url(2533, "detect"))
SM3DET_API_URL = os.getenv("SM3DET_API_URL", service_url(8020, "detect"))
REMOTESAM_API_URL = os.getenv("REMOTESAM_API_URL", service_url(8018, "remotesam"))
CHANGEOS_API_URL = os.getenv("CHANGEOS_API_URL", service_url(8019, "changeos"))
BRIGHT_API_URL = os.getenv("BRIGHT_API_URL", service_url(8120, "bright_damage_assessment"))
