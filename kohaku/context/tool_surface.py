"""Select which tool schemas reach the model now and which wait for discovery.

Direct tools keep a stable prefix so provider prompt caches survive ordinary
turns. Deferred tools are withheld until ``tool_search`` activates them; the
activation appends to the tail of the schema list and never reorders that prefix.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from kohaku.agent.tools.base import Tool
    from kohaku.agent.tools.registry import ToolRegistry
    from kohaku.providers.base import LLMProvider

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_BM25_K1 = 1.5
_BM25_B = 0.75
MAX_SEARCH_RESULTS = 16
DEFAULT_SEARCH_RESULTS = 8
MAX_NAMESPACE_PREVIEW = 6
ACTIVATED_TOOLS_METADATA = "_activated_tools"


def _tokens(text: str) -> list[str]:
    """Split text into lowercase terms for keyword ranking."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ToolSurfaceState:
    """Per-session record of deferred tools already made visible."""

    activated: tuple[str, ...] = ()

    def with_activated(self, names: list[str]) -> ToolSurfaceState:
        """Append newly discovered names in insertion order, ignoring repeats."""
        if not names:
            return self
        seen = set(self.activated)
        added = tuple(name for name in dict.fromkeys(names) if name not in seen)
        return replace(self, activated=(*self.activated, *added)) if added else self

    @classmethod
    def from_metadata(cls, value: object) -> ToolSurfaceState:
        if not isinstance(value, list):
            return cls()
        return cls().with_activated([
            name for name in cast(list[object], value) if isinstance(name, str)
        ])


def _document_for(tool: Tool) -> str:
    """Collect every searchable field the tool declares."""
    try:
        properties: object = (tool.parameters or {}).get("properties", {})
        parameter_names = (
            " ".join(sorted(key for key in cast(dict[object, object], properties) if isinstance(key, str)))
            if isinstance(properties, dict) else ""
        )
    except Exception:
        parameter_names = ""
    return " ".join(
        part
        for part in (
            tool.name,
            tool.namespace,
            tool.description,
            parameter_names,
            " ".join(tool.search_keywords),
        )
        if part
    )


@dataclass
class ToolSurface:
    """Compute the model-visible tool list for one turn."""

    registry: ToolRegistry
    state: ToolSurfaceState = field(default_factory=ToolSurfaceState)
    requested_namespaces: set[str] = field(default_factory=set)

    def activate(self, names: list[str]) -> None:
        available = {tool.name for tool in self.deferred_tools()}
        self.state = self.state.with_activated([name for name in names if name in available])

    def activate_requested_namespaces(self) -> list[str]:
        """Activate small explicitly requested namespaces and return the rest."""
        namespaces: dict[str, list[str]] = {name: [] for name in self.requested_namespaces}
        for tool in self.deferred_tools():
            if tool.namespace in namespaces:
                namespaces[tool.namespace].append(tool.name)
        for names in namespaces.values():
            if 0 < len(names) <= MAX_SEARCH_RESULTS:
                self.state = self.state.with_activated(sorted(names))
        return sorted(name for name, tools in namespaces.items()
                      if not tools or len(tools) > MAX_SEARCH_RESULTS)

    def _tools(self) -> list[Tool]:
        return list(self.registry.iter_tools())

    def direct_names(self) -> tuple[str, ...]:
        """Names always present in the schema list, in stable registry order."""
        return tuple(tool.name for tool in self._tools() if tool.exposure == "direct")

    def deferred_tools(self) -> list[Tool]:
        """Tools withheld until a search activates them."""
        return [tool for tool in self._tools() if tool.exposure != "direct"]

    def visible_names(self) -> tuple[str, ...]:
        """Direct tools followed by the append-only activated tail."""
        allowed = {tool.name for tool in self.deferred_tools()}
        deferred = [name for name in self.state.activated if name in allowed]
        return (*self.direct_names(), *deferred)

    def definitions_for_provider(
        self, provider: LLMProvider | None = None, model: str | None = None,
        *, registry: ToolRegistry | None = None,
    ) -> list[dict[str, Any]]:
        """Keep the direct prefix stable and append deferred schemas in activation order."""
        from kohaku.agent.tools.registry import ToolRegistry

        source = registry if registry is not None else self.registry
        deferred = {tool.name for tool in source.iter_tools() if tool.exposure != "direct"}
        direct: list[dict[str, Any]] = []
        tail: dict[str, dict[str, Any]] = {}
        native = (
            bool(deferred) and provider is not None
            and provider.supports_native_tool_search is True
            and provider.native_tool_search_for_model(model)
        )
        for definition in source.get_definitions():
            name = ToolRegistry.schema_name(definition)
            if name in deferred:
                tail[name] = definition
            elif native and name == "tool_search":
                direct.append({"type": "tool_search"})
            else:
                direct.append(definition)
        for name in self.state.activated:
            if name in tail:
                direct.append(tail.pop(name))
        if native:
            direct.extend({**definition, "defer_loading": True} for definition in tail.values())
        return direct

    def namespace_index(self) -> list[str]:
        """Summarize withheld namespaces so the model knows what can be loaded."""
        deferred = self.deferred_tools()
        if not deferred:
            return []
        grouped: dict[str, list[str]] = {}
        for tool in deferred:
            grouped.setdefault(tool.namespace, []).append(tool.name)
        lines: list[str] = []
        for namespace in sorted(grouped):
            names = sorted(grouped[namespace])
            if namespace.startswith("mcp:"):
                lines.append(f"- {namespace}: {len(names)} tools")
                continue
            preview = ", ".join(names[:MAX_NAMESPACE_PREVIEW])
            extra = len(names) - MAX_NAMESPACE_PREVIEW
            more = "" if extra <= 0 else f", +{extra} more"
            lines.append(
                f"- {namespace}: {preview}{more} ({len(names)} tools) "
                f'— use tool_search(namespace="{namespace}", query="...")'
            )
        return lines

    def search(
        self, query: str, *, namespace: str | None = None,
        limit: int = DEFAULT_SEARCH_RESULTS,
    ) -> list[Tool]:
        """Rank deferred tools against a query with a small BM25 index."""
        if query.lower().startswith("namespace:") and " " not in query.strip():
            namespace, query = namespace or query[len("namespace:"):].strip(), ""
        candidates = [
            tool for tool in self.deferred_tools()
            if self.registry.dispatch_error(tool.name) is None
        ]
        if namespace:
            candidates = [tool for tool in candidates if tool.namespace == namespace]
        if not candidates:
            return []
        size = max(1, min(MAX_SEARCH_RESULTS, limit))
        query_terms = _tokens(query)
        if not query_terms:
            return sorted(candidates, key=lambda tool: tool.name)[:size]

        documents = [_tokens(_document_for(tool)) for tool in candidates]
        average = sum(len(document) for document in documents) / len(documents)
        frequencies = [Counter(document) for document in documents]
        document_frequency = Counter(term for document in documents for term in set(document))
        total = len(documents)
        lowered = query.strip().lower()
        scored: list[tuple[float, str]] = []
        for tool, frequency, document in zip(candidates, frequencies, documents):
            score = 0.0
            for term in query_terms:
                occurrences = frequency.get(term, 0)
                if not occurrences:
                    continue
                inverse = math.log(
                    1 + (total - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = occurrences + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * (len(document) / average if average else 1)
                )
                score += inverse * (occurrences * (_BM25_K1 + 1)) / denominator
            # An exact name or namespace match outranks incidental body matches.
            if tool.name.lower() == lowered:
                score += 10.0
            elif lowered and lowered in tool.name.lower():
                score += 4.0
            if score > 0:
                scored.append((score, tool.name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        by_name = {tool.name: tool for tool in candidates}
        return [by_name[name] for _, name in scored[:size]]
