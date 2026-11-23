"""Base class for agent tools."""
from __future__ import annotations

import inspect
import math
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from enum import StrEnum
from typing import Any, TypeVar, cast

if typing.TYPE_CHECKING:
    from pydantic import BaseModel

    from nanobot.agent.tools.context import ToolContext
    from nanobot.runtime_context import RuntimeContextProvider

_ToolT = TypeVar("_ToolT", bound="Tool")

# Matches :meth:`Tool._cast_value` / :meth:`Schema.validate_json_schema_value` behavior
_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

# Keywords that :meth:`Schema.validate_json_schema_value` enforces. Any other
# keyword would be advertised to the model but silently ignored before dispatch.
_ENFORCED_JSON_SCHEMA_KEYS = frozenset({
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "minItems", "maxItems",
    "nullable",
})
_JSON_SCHEMA_ANNOTATION_KEYS = frozenset({"description", "title", "default", "$schema"})
_SUPPORTED_JSON_SCHEMA_TYPES = frozenset({*_JSON_TYPE_MAP, "null"})
_NUMERIC_LIMIT_KEYS = ("multipleOf", "exclusiveMinimum", "exclusiveMaximum")


def numeric_limit_value(value: Any) -> int | float | None:
    """Return ``value`` as a numeric JSON Schema limit, or ``None`` when it is not one.

    Booleans are excluded because ``bool`` is an ``int`` subclass in Python.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        return value
    return value if math.isfinite(value) else None


def invalid_numeric_limit(
    mapping: dict[str, Any],
    key: str,
    *,
    path: str = "",
) -> str | None:
    """Return an error when a numeric constraint keyword carries an unusable limit."""
    if key not in mapping:
        return None
    if numeric_limit_value(mapping[key]) is None:
        return f"{path or 'schema'} requires a numeric {key!r}"
    if key == "multipleOf" and cast(int | float, mapping[key]) <= 0:
        return f"{path or 'schema'} requires a positive {key!r}"
    return None


def unsupported_json_schema_constraint(schema: Any, *, path: str = "") -> str | None:
    """Return the first advertised keyword or type the local validator cannot enforce.

    External or raw schemas may declare constraints such as ``pattern`` or ``$ref``
    that :meth:`Schema.validate_json_schema_value` would ignore. Publishers call this
    before the schema becomes model-visible. Annotation-only keywords (``description``,
    ``title``, ``default``) are accepted because they promise no enforcement.
    """
    if not isinstance(schema, dict):
        return f"{path or 'schema'} must be an object"
    mapping = cast(dict[str, Any], schema)
    for key in mapping:
        if key not in _ENFORCED_JSON_SCHEMA_KEYS and key not in _JSON_SCHEMA_ANNOTATION_KEYS:
            return f"{path or 'schema'} uses unsupported keyword {key!r}"
    if "$schema" in mapping:
        dialect = mapping["$schema"]
        if not isinstance(dialect, str) or not dialect.strip():
            return f"{path or 'schema'} requires a non-empty string '$schema'"
    for key in _NUMERIC_LIMIT_KEYS:
        error = invalid_numeric_limit(mapping, key, path=path)
        if error:
            return error
    raw_type = mapping.get("type")
    if raw_type is not None:
        types: list[Any] = (
            list(cast(list[Any], raw_type)) if isinstance(raw_type, list) else [raw_type]
        )
        if any(item not in _SUPPORTED_JSON_SCHEMA_TYPES for item in types):
            return f"{path or 'schema'} uses an unsupported type"
        if len([item for item in types if item != "null"]) > 1:
            return f"{path or 'schema'} uses an unsupported type union"
    properties = mapping.get("properties")
    if isinstance(properties, dict):
        for name, child in cast(dict[str, Any], properties).items():
            error = unsupported_json_schema_constraint(
                child, path=f"{path}.{name}" if path else name
            )
            if error:
                return error
    items = mapping.get("items")
    if isinstance(items, dict):
        error = unsupported_json_schema_constraint(items, path=f"{path}[]" if path else "[]")
        if error:
            return error
    additional = mapping.get("additionalProperties")
    if isinstance(additional, dict):
        error = unsupported_json_schema_constraint(
            additional, path=f"{path}.*" if path else "*"
        )
        if error:
            return error
    return None


class Schema(ABC):
    """Abstract base for JSON Schema fragments describing tool parameters.

    Concrete types live in :mod:`nanobot.agent.tools.schema`; all implement
    :meth:`to_json_schema` and :meth:`validate_value`. Class methods
    :meth:`validate_json_schema_value` and :meth:`fragment` are the shared validation and normalization entry points.
    """

    @staticmethod
    def resolve_json_schema_type(t: Any) -> str | None:
        """Resolve the non-null type name from JSON Schema ``type`` (e.g. ``['string','null']`` -> ``'string'``)."""
        if isinstance(t, list):
            types = cast(list[Any], t)
            return cast(str | None, next((x for x in types if x != "null"), None))
        return cast(str | None, t)

    @staticmethod
    def subpath(path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    @staticmethod
    def validate_json_schema_value(val: Any, schema: dict[str, Any], path: str = "") -> list[str]:
        """Validate ``val`` against a JSON Schema fragment; returns error messages (empty means valid).

        Used by :class:`Tool` and each concrete Schema's :meth:`validate_value`.
        """
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get("nullable", False)
        t = Schema.resolve_json_schema_type(raw_type)
        label = path or "parameter"

        if nullable and val is None:
            return []
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (
            not isinstance(val, _JSON_TYPE_MAP["number"]) or isinstance(val, bool)
        ):
            return [f"{label} should be number"]
        if t in _JSON_TYPE_MAP and t not in ("integer", "number") and not isinstance(val, _JSON_TYPE_MAP[t]):
            return [f"{label} should be {t}"]
        if t == "number" and isinstance(val, float) and not math.isfinite(val):
            return [f"{label} must be finite"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
            number_value = cast(int | float, val)
            multiple_of = numeric_limit_value(schema.get("multipleOf"))
            if multiple_of is not None and multiple_of > 0:
                if isinstance(number_value, int) and isinstance(multiple_of, int):
                    is_multiple = number_value % multiple_of == 0
                else:
                    try:
                        quotient = number_value / multiple_of
                    except OverflowError:
                        is_multiple = False
                    else:
                        is_multiple = math.isclose(
                            quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9
                        )
                if not is_multiple:
                    errors.append(f"{label} must be a multiple of {multiple_of}")
            exclusive_minimum = numeric_limit_value(schema.get("exclusiveMinimum"))
            if exclusive_minimum is not None and number_value <= exclusive_minimum:
                errors.append(f"{label} must be > {exclusive_minimum}")
            exclusive_maximum = numeric_limit_value(schema.get("exclusiveMaximum"))
            if exclusive_maximum is not None and number_value >= exclusive_maximum:
                errors.append(f"{label} must be < {exclusive_maximum}")
        if t == "string":
            string_value = cast(str, val)
            if "minLength" in schema and len(string_value) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(string_value) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if t == "object":
            object_value = cast(dict[str, Any], val)
            props = cast(dict[str, Any], schema.get("properties", {}))
            required = cast(list[Any], schema.get("required", []))
            for k in required:
                if k not in object_value:
                    errors.append(f"missing required {Schema.subpath(path, k)}")
            additional = schema.get("additionalProperties", True)
            for k, v in object_value.items():
                if k in props:
                    errors.extend(Schema.validate_json_schema_value(v, props[k], Schema.subpath(path, k)))
                elif additional is False:
                    errors.append(f"unexpected parameter {Schema.subpath(path, k)}")
                elif isinstance(additional, dict):
                    errors.extend(
                        Schema.validate_json_schema_value(
                            v,
                            cast(dict[str, Any], additional),
                            Schema.subpath(path, k),
                        )
                    )
        if t == "array":
            array_value = cast(list[Any], val)
            if "minItems" in schema and len(array_value) < schema["minItems"]:
                errors.append(f"{label} must have at least {schema['minItems']} items")
            if "maxItems" in schema and len(array_value) > schema["maxItems"]:
                errors.append(f"{label} must be at most {schema['maxItems']} items")
            if "items" in schema:
                prefix = f"{path}[{{}}]" if path else "[{}]"
                for i, item in enumerate(array_value):
                    errors.extend(
                        Schema.validate_json_schema_value(item, schema["items"], prefix.format(i))
                    )
        return errors

    @staticmethod
    def fragment(value: Any) -> dict[str, Any]:
        """Normalize a Schema instance or an existing JSON Schema dict to a fragment dict."""
        # Try to_json_schema first: Schema instances must be distinguished from dicts that are already JSON Schema
        to_js = getattr(value, "to_json_schema", None)
        if callable(to_js):
            return cast(dict[str, Any], to_js())
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
        raise TypeError(f"Expected schema object or dict, got {type(value).__name__}")

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        """Return a fragment dict compatible with :meth:`validate_json_schema_value`."""
        ...

    def validate_value(self, value: Any, path: str = "") -> list[str]:
        """Validate a single value; returns error messages (empty means pass). Subclasses may override for extra rules."""
        return Schema.validate_json_schema_value(value, self.to_json_schema(), path)


class ToolResult(str):
    """String-compatible tool output with structured status and media refs."""

    is_error: bool
    media_artifacts: tuple[Mapping[str, Any], ...]

    def __new__(
        cls,
        content: str,
        *,
        is_error: bool = False,
        media_artifacts: tuple[Mapping[str, Any], ...] = (),
    ) -> ToolResult:
        obj = str.__new__(cls, content)
        obj.is_error = is_error
        obj.media_artifacts = media_artifacts
        return obj

    @classmethod
    def error(cls, content: str) -> ToolResult:
        return cls(content, is_error=True)

    @classmethod
    def with_media(
        cls,
        content: str,
        artifacts: list[Mapping[str, Any]],
    ) -> ToolResult:
        """Return string-compatible output with durable media references."""
        return cls(content, media_artifacts=tuple(dict(item) for item in artifacts))


class ToolDenialKind(StrEnum):
    """Typed pre-dispatch denial that the owning tool can prove."""

    PRIVATE_NETWORK = "private_network"
    MANAGED_UNAVAILABLE = "managed_unavailable"
    WRITE_PERMISSION = "write_permission"


PRIVATE_NETWORK_DENIAL_NOTE = (
    "This is a non-bypassable security boundary for the denied target. "
    "Do not retry that access with curl, wget, encoded IPs, alternate DNS, "
    "redirects, proxies, or another tool. Unrelated permitted work may "
    "continue without touching the denied target. If the user explicitly "
    "trusts this private service, ask them to whitelist its exact IP/CIDR "
    "via tools.ssrfWhitelist."
)


class ToolDenialResult(ToolResult):
    """Structured denial for a tool call rejected before its side effect.

    Only first-party guard code may construct this subtype.  Message text is
    never sufficient evidence for recovery.
    """

    kind: ToolDenialKind

    def __new__(cls, content: str, *, kind: ToolDenialKind) -> ToolDenialResult:
        obj = cast(ToolDenialResult, super().__new__(cls, content, is_error=True))
        obj.kind = kind
        return obj

    def __reduce__(self) -> tuple[Callable[..., ToolDenialResult], tuple[str, ToolDenialKind]]:
        return _restore_tool_denial_result, (str(self), self.kind)


def _restore_tool_denial_result(
    content: str,
    kind: ToolDenialKind,
) -> ToolDenialResult:
    return ToolDenialResult(content, kind=kind)


class ToolPreparationErrorKind(StrEnum):
    """Typed reason why the registry rejected a tool call before dispatch."""

    UNKNOWN_TOOL = "unknown_tool"
    WITHHELD_TOOL = "withheld_tool"
    MANAGED_UNAVAILABLE = "managed_unavailable"


class ToolPreparationError(ToolResult):
    """Tool error that preserves a registry-owned pre-dispatch classification."""

    kind: ToolPreparationErrorKind

    def __new__(
        cls,
        content: str,
        *,
        kind: ToolPreparationErrorKind,
    ) -> ToolPreparationError:
        result = cast(ToolPreparationError, super().__new__(cls, content, is_error=True))
        result.kind = kind
        return result

    def __reduce__(self) -> tuple[Callable[..., ToolPreparationError], tuple[str, ToolPreparationErrorKind]]:
        return _restore_tool_preparation_error, (str(self), self.kind)


def _restore_tool_preparation_error(
    content: str,
    kind: ToolPreparationErrorKind,
) -> ToolPreparationError:
    return ToolPreparationError(content, kind=kind)


class Tool(ABC):
    """Agent capability: read files, run commands, etc."""

    _TYPE_MAP = _JSON_TYPE_MAP
    _BOOL_TRUE = frozenset(("true", "1", "yes"))
    _BOOL_FALSE = frozenset(("false", "0", "no"))

    @staticmethod
    def _resolve_type(t: Any) -> str | None:
        """Pick first non-null type from JSON Schema unions like ``['string','null']``."""
        return Schema.resolve_json_schema_type(t)

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @property
    def read_only(self) -> bool:
        """Whether this tool is side-effect free and safe to parallelize."""
        return False

    @property
    def concurrency_safe(self) -> bool:
        """Whether this tool can run alongside other concurrency-safe tools."""
        return self.read_only and not self.exclusive

    @property
    def exclusive(self) -> bool:
        """Whether this tool should run alone even if concurrency is enabled."""
        return False

    # --- Plugin metadata ---

    config_key: str = ""
    _plugin_discoverable: bool = True
    _scopes: set[str] = {"core"}

    @classmethod
    def config_cls(cls) -> type[BaseModel] | None:
        return None

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return True

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        return cls()

    def dispose(self) -> None:
        """Release resources acquired during ``create()``.

        The loader calls this when a constructed plugin is never published, for
        example after a duplicate-name, name, or schema rejection. Tools that own
        a resource override it; the default is a no-op. Publication keeps the
        resource, so the loader does not call this after a successful load.
        """
        return None

    def runtime_context_provider(self) -> RuntimeContextProvider | None:
        """Return optional per-turn prompt context owned by this tool."""
        return None

    # The schema-validated dynamic call edge erases heterogeneous parameters;
    # concrete signatures still govern callers holding a concrete tool type.
    if typing.TYPE_CHECKING:
        @abstractmethod
        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            ...
    else:
        @abstractmethod
        async def execute(self, **kwargs: Any) -> Any:
            """Run the tool; return content, or ``ToolResult.error(...)`` for failures."""
            ...

    @staticmethod
    def error(content: str) -> ToolResult:
        return ToolResult.error(content)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        props = cast(dict[str, Any], schema.get("properties", {}))
        additional = schema.get("additionalProperties")
        casted: dict[str, Any] = {}
        object_value = cast(dict[str, Any], obj)
        for k, v in object_value.items():
            if k in props:
                casted[k] = self._cast_value(v, props[k])
            elif isinstance(additional, dict):
                casted[k] = self._cast_value(v, cast(dict[str, Any], additional))
            else:
                casted[k] = v
        return casted

    if typing.TYPE_CHECKING:
        async def execute_with_context(self, context: Any, /, *args: Any, **kwargs: Any) -> Any:
            ...
    else:
        async def execute_with_context(self, context: Any, /, **kwargs: Any) -> Any:
            """Execute with an explicit turn context when the tool opts in.

        The legacy ``execute(**kwargs)`` contract remains unchanged.  A tool
        that needs per-turn state may either override this method or declare a
        ``context``/``execution_context`` parameter on ``execute``.
        Schema-declared parameters are user data, not injection slots. The
        positional-only argument keeps that data separate from turn context.
        We do not inject context into arbitrary ``**kwargs`` methods because several
        existing adapters forward unknown kwargs to remote tools.
        """
            execute = self.execute
            try:
                parameters = inspect.signature(execute).parameters
            except (TypeError, ValueError):
                return await execute(**kwargs)

            properties = self.parameters.get("properties", {})
            if "context" in parameters and "context" not in properties:
                kwargs["context"] = context
            elif "execution_context" in parameters and "execution_context" not in properties:
                kwargs["execution_context"] = context
            return await execute(**kwargs)

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply safe schema-driven casts before validation."""
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        t = self._resolve_type(schema.get("type"))

        if t == "boolean" and isinstance(val, bool):
            return val
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if t in self._TYPE_MAP and t not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[t]
            if isinstance(val, expected):
                return val

        if isinstance(val, str) and t in ("integer", "number"):
            try:
                return int(val) if t == "integer" else float(val)
            except ValueError:
                return val

        if t == "string":
            return val if val is None else str(val)

        if t == "boolean" and isinstance(val, str):
            low = val.lower()
            if low in self._BOOL_TRUE:
                return True
            if low in self._BOOL_FALSE:
                return False
            return val

        if t == "array" and isinstance(val, list):
            items = schema.get("items")
            array_value = cast(list[Any], val)
            return [self._cast_value(x, items) for x in array_value] if items else array_value

        if t == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate against JSON schema; empty list means valid."""
        if not isinstance(cast(object, params), dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return Schema.validate_json_schema_value(params, {**schema, "type": "object"}, "")

    def to_schema(self) -> dict[str, Any]:
        """OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool_parameters(schema: dict[str, Any]) -> Callable[[type[_ToolT]], type[_ToolT]]:
    """Class decorator: attach JSON Schema and inject a concrete ``parameters`` property.

    Use on ``Tool`` subclasses instead of writing ``@property def parameters``. The
    schema is stored on the class and returned as a fresh copy on each access.

    Example::

        @tool_parameters({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        class ReadFileTool(Tool):
            ...
    """

    def decorator(cls: type[_ToolT]) -> type[_ToolT]:
        frozen = deepcopy(schema)

        @property
        def parameters(self: Any) -> dict[str, Any]:
            return deepcopy(frozen)

        cls.parameters = parameters  # type: ignore[assignment]

        abstract = getattr(cls, "__abstractmethods__", None)
        if abstract is not None and "parameters" in abstract:
            cls.__abstractmethods__ = frozenset(abstract - {"parameters"})  # type: ignore[misc]

        return cls

    return decorator
