"""Discover and normalize the model catalog for a configured provider."""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict, cast

import httpx

from kohaku.config.loader import resolve_env_refs
from kohaku.config.schema import ProviderConfig
from kohaku.providers.factory import provider_extra_headers
from kohaku.providers.oauth_credentials import CredentialOwner
from kohaku.providers.oauth_model_catalog import CatalogSource, get_oauth_model_catalog
from kohaku.providers.registry import ProviderSpec

HttpGet = Callable[..., httpx.Response]

_DEFAULT_API_BASES = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
}
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class DiscoveredModelPayload(TypedDict):
    id: str
    label: str | None
    owned_by: str | None
    context_window: int | None
    description: NotRequired[str | None]
    reasoning_efforts: NotRequired[list[str]]
    supports_backend_search: NotRequired[bool]


class ProviderModelCatalogPayload(TypedDict):
    provider: str
    label: str
    catalog_kind: str
    status: Literal["available", "unsupported", "not_configured", "missing_api_base", "error"]
    models: list[DiscoveredModelPayload]
    model_count: int
    message: str | None
    fetched_at: float
    source: NotRequired[CatalogSource]


def _resolve_env_placeholders(value: str | None) -> str | None:
    if not value:
        return None
    missing = False

    def replace(match: re.Match[str]) -> str:
        nonlocal missing
        env_value = os.environ.get(match.group(1))
        if env_value is None:
            missing = True
            return ""
        return env_value

    resolved = _ENV_REF_RE.sub(replace, value).strip()
    if missing and not resolved:
        return None
    return resolved or None


def provider_requires_api_key(spec: ProviderSpec) -> bool:
    if spec.name == "azure_openai":
        return False
    if spec.is_oauth:
        return False
    if spec.is_local or spec.is_direct:
        return False
    return True


def model_catalog_kind(spec: ProviderSpec) -> str:
    catalog = spec.model_catalog
    if catalog != "auto":
        return catalog
    if spec.is_transcription_only or spec.is_oauth:
        return "unsupported"
    if spec.backend == "anthropic":
        return "official"
    if spec.backend != "openai_compat":
        return "unsupported"
    if spec.is_local:
        return "local"
    if spec.is_direct:
        return "custom"
    if spec.is_gateway:
        return "catalog"
    return "official"


def _model_id_from_row(row: dict[object, object]) -> str | None:
    for key in ("id", "name", "model"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _model_context_window(row: dict[object, object]) -> int | None:
    for key in (
        "context_window",
        "context_length",
        "max_context_length",
        "max_model_len",
        "max_input_tokens",
    ):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if isinstance(value, float) and math.isfinite(value) and value >= 1:
            return int(value)
    return None


def _model_row_payload(row: object) -> DiscoveredModelPayload | None:
    if isinstance(row, str) and row.strip():
        return {"id": row.strip(), "label": None, "owned_by": None, "context_window": None}
    if not isinstance(row, dict):
        return None
    row_mapping = cast(dict[object, object], row)
    model_id = _model_id_from_row(row_mapping)
    if not model_id:
        return None

    raw_label = row_mapping.get("display_name") or row_mapping.get("label") or row_mapping.get("name")
    label = raw_label.strip() if isinstance(raw_label, str) else ""
    raw_owner = row_mapping.get("owned_by") or row_mapping.get("owner") or row_mapping.get("organization")
    payload: DiscoveredModelPayload = {
        "id": model_id,
        "label": label if label and label != model_id else None,
        "owned_by": (raw_owner.strip() or None) if isinstance(raw_owner, str) else None,
        "context_window": _model_context_window(row_mapping),
    }
    description = row_mapping.get("description")
    if isinstance(description, str) and description.strip():
        payload["description"] = description.strip()
    reasoning = row_mapping.get("reasoning_efforts")
    if isinstance(reasoning, list):
        payload["reasoning_efforts"] = [
            value.strip()
            for value in cast(list[object], reasoning)
            if isinstance(value, str) and value.strip()
        ]
    supports_search = row_mapping.get("supports_backend_search")
    if isinstance(supports_search, bool):
        payload["supports_backend_search"] = supports_search
    return payload


def _extract_model_rows(body: object) -> list[DiscoveredModelPayload]:
    raw_rows = cast(dict[object, object], body).get("data") if isinstance(body, dict) else body
    if not isinstance(raw_rows, list):
        return []
    rows: list[DiscoveredModelPayload] = []
    seen: set[str] = set()
    for raw_row in cast(list[object], raw_rows):
        row = _model_row_payload(raw_row)
        if row is None or row["id"] in seen:
            continue
        seen.add(row["id"])
        rows.append(row)
    return rows


def discover_provider_models(
    spec: ProviderSpec,
    provider_config: ProviderConfig,
    *,
    provider_key: str | None = None,
    http_get: HttpGet = httpx.get,
    credential_owner: CredentialOwner | None = None,
) -> ProviderModelCatalogPayload:
    """Fetch an advisory model list for one provider."""
    catalog_kind = model_catalog_kind(spec)
    base_payload: ProviderModelCatalogPayload = {
        "provider": provider_key or spec.name,
        "label": provider_config.display_name or spec.label,
        "catalog_kind": catalog_kind,
        "status": "available",
        "models": [],
        "model_count": 0,
        "message": None,
        "fetched_at": time.time(),
    }
    if catalog_kind == "unsupported":
        return {
            **base_payload,
            "status": "unsupported",
            "message": "Model list is not available for this provider. Type a model ID manually.",
        }
    if catalog_kind == "builtin":
        rows: list[DiscoveredModelPayload] = [
            {
                "id": model.id,
                "label": model.label or None,
                "description": model.description or None,
                "owned_by": spec.label,
                "context_window": model.context_window,
            }
            for model in spec.builtin_models
        ]
        return {
            **base_payload,
            "models": rows,
            "model_count": len(rows),
        }
    if catalog_kind == "hybrid":
        proxy = _resolve_env_placeholders(provider_config.proxy)
        catalog = get_oauth_model_catalog(spec.name, proxy=proxy, owner=credential_owner)
        oauth_rows: list[DiscoveredModelPayload] = [
            {
                "id": model.id,
                "label": model.label or None,
                "description": model.description or None,
                "owned_by": model.owned_by or spec.label,
                "context_window": model.context_window,
                "reasoning_efforts": list(model.reasoning_efforts),
                "supports_backend_search": model.supports_backend_search,
            }
            for model in catalog.models
        ]
        return {
            **base_payload,
            "source": catalog.source,
            "models": oauth_rows,
            "model_count": len(oauth_rows),
            "message": catalog.message,
            "fetched_at": catalog.fetched_at,
        }

    if provider_config.api_base:
        # Never send credentials for an explicitly configured endpoint to a
        # different origin if the endpoint's environment reference is unset.
        api_base = resolve_env_refs(provider_config.api_base).strip()
    else:
        api_base = spec.default_api_base or _DEFAULT_API_BASES.get(spec.name, "")
    if not api_base:
        return {
            **base_payload,
            "status": "missing_api_base",
            "message": "Configure an API base URL to load models.",
        }

    api_key = _resolve_env_placeholders(provider_config.api_key)
    if provider_requires_api_key(spec) and not api_key:
        return {
            **base_payload,
            "status": "not_configured",
            "message": "Configure this provider before loading models.",
        }

    headers = {"Accept": "application/json"}
    if spec.backend == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if api_key:
            headers["X-Api-Key"] = api_key
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    extra_headers: dict[str, str] = {}
    for name, value in (provider_extra_headers(spec, provider_config) or {}).items():
        resolved = resolve_env_refs(value)
        if value and not resolved:
            return {
                **base_payload,
                "status": "not_configured",
                "message": "A configured provider header is missing from the environment.",
            }
        extra_headers[name] = resolved

    api_base = api_base.rstrip("/")
    if spec.backend == "anthropic" and not api_base.endswith("/v1"):
        api_base += "/v1"
    models_url = f"{api_base}/models"

    try:
        request_headers = httpx.Headers(headers)
        request_headers.update(extra_headers)
        response = http_get(
            models_url,
            headers=request_headers,
            timeout=10.0,
            follow_redirects=False,
        )
        response.raise_for_status()
        models = _extract_model_rows(response.json())
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in {401, 403}:
            return {
                **base_payload,
                "status": "not_configured",
                "message": "The provider rejected the configured credential.",
            }
        return {
            **base_payload,
            "status": "error",
            "message": f"Model list request failed with HTTP {status}.",
        }
    except (httpx.HTTPError, ValueError):
        return {
            **base_payload,
            "status": "error",
            "message": "Could not load models from the configured provider.",
        }

    return {
        **base_payload,
        "models": models,
        "model_count": len(models),
    }
