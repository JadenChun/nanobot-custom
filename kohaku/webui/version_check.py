"""On-demand version checker for Kohaku releases.

Checks the project's GitHub releases when explicitly requested.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from kohaku import __version__

logger = logging.getLogger(__name__)

_REPOSITORY = "JadenChun/nanobot-custom"
_RELEASE_API_URL = f"https://api.github.com/repos/{_REPOSITORY}/releases/latest"
_RELEASES_URL = f"https://github.com/{_REPOSITORY}/releases"
_CACHE_TTL_S = 300

_cache: tuple[float, str | None] = (0.0, None)


def check_for_update() -> dict[str, Any] | None:
    """Return update information if the project has a newer release.

    Uses a short cache to avoid repeated requests within the TTL window.
    This is a blocking call — invoke from a thread or background task.
    """
    global _cache
    now = time.monotonic()
    cached_at, cached_val = _cache
    if now - cached_at < _CACHE_TTL_S and cached_val is not None:
        latest = cached_val
    else:
        try:
            resp = httpx.get(_RELEASE_API_URL, timeout=5.0, follow_redirects=True)
            resp.raise_for_status()
            tag = resp.json().get("tag_name")
            latest = tag.removeprefix("v") if isinstance(tag, str) else None
        except Exception:
            logger.debug("Kohaku release check failed", exc_info=True)
            return None
        _cache = (now, latest)

    if not isinstance(latest, str) or not latest:
        return None
    try:
        if Version(latest) <= Version(__version__):
            return None
    except InvalidVersion:
        logger.debug("GitHub returned an invalid Kohaku version: %r", latest)
        return None
    return {
        "currentVersion": __version__,
        "latestVersion": latest,
        "releaseUrl": _RELEASES_URL,
    }
