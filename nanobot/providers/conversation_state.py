"""Provider-owned conversation-state lifecycle coordination."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from math import isfinite
from typing import Any, cast

from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
    ProviderConversationTransportReceipt,
)
from nanobot.providers.request_budget import derive_provider_request_limits
from nanobot.utils.helpers import estimate_prompt_tokens_chain

_PROVIDER_STATE_OUTPUT_META = "provider_state_output"
_PROVIDER_STATE_BOUNDARY_META = "provider_state_boundary"
_BINDING_VERSION = 1


def allows_conversation_message_merge(message: dict[str, Any]) -> bool:
    """Return whether new same-role input may merge into *message*."""
    internal_meta = cast(object, message.get("_meta"))
    return not (
        isinstance(internal_meta, dict)
        and cast(dict[str, Any], internal_meta).get(
            _PROVIDER_STATE_BOUNDARY_META
        ) is True
    )


def _canonical_value(value: object) -> object:
    """Recursively normalize only deterministic JSON-compatible input."""
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("binding mapping keys must be strings")
            normalized[key] = _canonical_value(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("binding floats must be finite")
        return value
    raise ValueError("binding contains an opaque value")


def _canonical_json(value: object) -> bytes | None:
    """Return a stable private representation without retaining its contents."""
    try:
        return json.dumps(
            _canonical_value(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        return None


def _digest(value: object) -> str | None:
    canonical = _canonical_json(value)
    return sha256(canonical).hexdigest() if canonical is not None else None


@dataclass(frozen=True, slots=True)
class ConversationRequestBinding:
    """Private authority and policy inputs that scope a continuation state.

    The portable prefix is supplied for each request so this object can be
    reused by callers that own structured session or projection identities.
    Only digests and a prefix length are persisted in provider state.
    """

    session_namespace: str | None = None
    authority_namespace: str | None = None
    execution_owner: str | None = None
    session_generation: str | int | None = None
    projection_epoch: str | int | None = None
    runtime_prefix: object | None = None
    policy_prefix: object | None = None
    tools: object | None = None
    tool_choice: object | None = None
    generation_settings: object | None = None

    @property
    def is_complete(self) -> bool:
        """Whether every state-scoping dimension was explicitly supplied."""
        return all(
            value is not None
            for value in (
                self.session_namespace,
                self.authority_namespace,
                self.execution_owner,
                self.session_generation,
                self.projection_epoch,
                self.runtime_prefix,
                self.policy_prefix,
                self.tools,
                self.tool_choice,
                self.generation_settings,
            )
        )

    def private_record(
        self,
        portable_prefix: list[dict[str, Any]],
        receipt: ProviderConversationTransportReceipt,
    ) -> dict[str, Any] | None:
        """Create the content-free sidecar record for one accepted request."""
        if not self.is_complete:
            return None
        prefix_digest = _digest(portable_prefix)
        config_digest = _digest({
            "session_namespace": self.session_namespace,
            "authority_namespace": self.authority_namespace,
            "execution_owner": self.execution_owner,
            "session_generation": self.session_generation,
            "projection_epoch": self.projection_epoch,
            "runtime_prefix": self.runtime_prefix,
            "policy_prefix": self.policy_prefix,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "generation_settings": self.generation_settings,
            "transport": receipt.to_private_record(),
        })
        if prefix_digest is None or config_digest is None:
            return None
        return {
            "version": _BINDING_VERSION,
            "config_digest": config_digest,
            "prefix_digest": prefix_digest,
            "prefix_length": len(portable_prefix),
        }


def _valid_binding_record(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    record = cast(dict[str, Any], value)
    if set(record) != {"version", "config_digest", "prefix_digest", "prefix_length"}:
        return None
    version = record.get("version")
    prefix_length = record.get("prefix_length")
    config_digest = record.get("config_digest")
    prefix_digest = record.get("prefix_digest")
    if (
        version != _BINDING_VERSION
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or prefix_length < 0
        or not isinstance(config_digest, str)
        or not isinstance(prefix_digest, str)
        or len(config_digest) != 64
        or len(prefix_digest) != 64
    ):
        return None
    try:
        int(config_digest, 16)
        int(prefix_digest, 16)
    except ValueError:
        return None
    return deepcopy(record)


class ProviderConversationStateController:
    """Keep provider conversation-state semantics outside the agent runner.

    A response state is deliberately provisional. Its owner must call
    :meth:`settle_response` only after the response and any required source or
    session record have committed. Until then a subsequent request rebuilds
    from the portable transcript.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str | None,
        messages: list[dict[str, Any]],
        state: ProviderConversationState | None = None,
        session_id: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._session_id = session_id
        self._state = (
            state
            if state is not None
            and provider.can_resume_conversation_state(state, model)
            else None
        )
        self._boundary = len(messages)
        self._request_messages: list[dict[str, Any]] = []
        self._active_portable_prefix: list[dict[str, Any]] | None = None
        self._active_binding: ConversationRequestBinding | None = None
        self._active_receipt: ProviderConversationTransportReceipt | None = None
        self._accepted_prefix: list[dict[str, Any]] | None = None
        self._provisional_state: ProviderConversationState | None = None
        self._provisional_prefix: list[dict[str, Any]] | None = None
        self._provisional_invocation = 0

    def independent_request_context(
        self,
        *,
        context_window_tokens: int | None,
        completion_reserve_tokens: int | None = None,
    ) -> ProviderCallContext | None:
        """Return typed provider context for a request that does not resume state."""
        request_limits = derive_provider_request_limits(
            context_window_tokens,
            completion_reserve_tokens,
        )
        if context_window_tokens is None and self._session_id is None and request_limits is None:
            return None
        return ProviderCallContext(
            request_limits=request_limits,
            context_window_tokens=context_window_tokens,
            session_id=self._session_id,
        )

    def estimate_request_context_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        model_messages: list[dict[str, Any]] | None = None,
        supplemental_messages: list[dict[str, Any]] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        binding: ConversationRequestBinding | None = None,
    ) -> int | None:
        """Estimate resumed state plus the pending delta for the next request."""
        if (
            binding is None
            or not binding.is_complete
            or not self._supports_bound_conversation_state()
        ):
            self.invalidate()
            return None
        receipt = self._transport_receipt()
        if self._state is None or receipt is None:
            self.invalidate()
            return None
        if not self._provider.can_resume_conversation_state(self._state, self._model):
            self.invalidate()
            return None
        if not self._state_has_known_context(self._state):
            self.invalidate()
            return None
        portable_prefix = self._portable_messages(
            model_messages if model_messages is not None else messages,
        )
        portable_prefix.extend(self._portable_messages(supplemental_messages or []))
        if not self._binding_matches(portable_prefix, binding, receipt):
            self.invalidate()
            return None
        state = self.checkpoint(messages, model_messages=model_messages)
        if state is None:
            return None
        context_tokens = state.payload.get("context_tokens")
        if (
            isinstance(context_tokens, bool)
            or not isinstance(context_tokens, int)
            or context_tokens < 0
        ):
            return None
        pending_messages = [
            *state.pending_messages,
            *(supplemental_messages or []),
        ]
        delta_tokens, _ = estimate_prompt_tokens_chain(
            self._provider,
            self._model,
            pending_messages,
            tool_definitions,
        )
        return context_tokens + max(0, delta_tokens)

    def replace_transcript(self, messages: list[dict[str, Any]]) -> None:
        """Discard append-only provider state after a transcript rewrite."""
        self.invalidate()
        self._boundary = len(messages)

    def prepare_request(
        self,
        messages: list[dict[str, Any]],
        *,
        context_window_tokens: int | None,
        completion_reserve_tokens: int | None = None,
        model_messages: list[dict[str, Any]] | None = None,
        supplemental_messages: list[dict[str, Any]] | None = None,
        resume_state: bool = True,
        binding: ConversationRequestBinding | None = None,
    ) -> ProviderCallContext | None:
        """Build a request context, reusing only an exact bound state."""
        # Starting another transport attempt abandons any prior response whose
        # caller did not durably settle it. This makes late callbacks unable to
        # revive an ambiguous provider handle.
        abandoned_provisional = self._provisional_state is not None
        if abandoned_provisional:
            # The prior state only covered the request before the unrecorded
            # response. Reusing it would omit that response's assistant tool
            # call while replaying its tool result.
            self._state = None
            self._accepted_prefix = None
            self._provisional_state = None
            self._provisional_prefix = None
            self._provisional_invocation += 1
        independent_context = self.independent_request_context(
            context_window_tokens=context_window_tokens,
            completion_reserve_tokens=completion_reserve_tokens,
        )
        portable_prefix = self._portable_messages(
            model_messages if model_messages is not None else messages,
        )
        portable_prefix.extend(self._portable_messages(supplemental_messages or []))
        receipt = self._transport_receipt()
        supports_bound_state = self._supports_bound_conversation_state()
        self._active_portable_prefix = portable_prefix
        self._active_binding = (
            binding
            if binding is not None and binding.is_complete and supports_bound_state
            else None
        )
        self._active_receipt = receipt if self._active_binding is not None else None
        self._request_messages = []

        if abandoned_provisional:
            return independent_context

        if not resume_state:
            # This request deliberately rebuilds from portable history, but a
            # valid response may establish a fresh state at that new boundary.
            # Keep the active request binding so ``observe_response`` can
            # stage that candidate without reviving the discarded handle.
            self._state = None
            self._accepted_prefix = None
            self._provisional_state = None
            self._provisional_prefix = None
            self._provisional_invocation += 1
            return independent_context
        if (
            self._active_binding is None
            or self._active_receipt is None
            or self._state is None
        ):
            self._state = None
            self._accepted_prefix = None
            return independent_context
        if not self._provider.can_resume_conversation_state(self._state, self._model):
            self.invalidate()
            return independent_context
        if not self._state_has_known_context(self._state):
            self.invalidate()
            return independent_context
        if not self._binding_matches(portable_prefix, binding, receipt):
            self.invalidate()
            return independent_context

        request_messages = self._request_delta(portable_prefix)
        self._request_messages = deepcopy(request_messages)
        request_state = self._state.with_pending_messages([
            *self._state.pending_messages,
            *request_messages,
        ])
        return ProviderCallContext(
            conversation_state=request_state,
            request_limits=(
                independent_context.request_limits
                if independent_context is not None
                else None
            ),
            context_window_tokens=(
                independent_context.context_window_tokens
                if independent_context is not None
                else None
            ),
            session_id=self._session_id,
        )

    def observe_response(
        self,
        response: LLMResponse,
        messages: list[dict[str, Any]],
        *,
        adopt_candidate_state: bool = True,
    ) -> int | None:
        """Stage a response candidate without making it reusable.

        The returned invocation token is single-use. A late or duplicate
        settlement cannot revive a candidate that was invalidated or replaced.
        """
        self._provisional_state = None
        self._provisional_prefix = None
        candidate = response.provider_state if adopt_candidate_state else None
        candidate_is_replayable = response.finish_reason in {
            "stop",
            "tool_calls",
            "function_call",
        }
        binding = self._active_binding
        receipt = self._active_receipt
        prefix = self._active_portable_prefix
        record = (
            binding.private_record(prefix, receipt)
            if binding is not None and receipt is not None and prefix is not None
            else None
        )
        if (
            candidate is not None
            and candidate_is_replayable
            and record is not None
            and receipt is not None
            and self._supports_bound_conversation_state()
            and self._provider.can_resume_conversation_state(candidate, self._model)
        ):
            self._provisional_invocation += 1
            self._provisional_state = self._state_with_metadata(candidate, record, receipt)
            self._provisional_prefix = deepcopy(prefix)
            self._request_messages = []
            return self._provisional_invocation
        self.invalidate()
        self._boundary = len(messages)
        return None

    def settle_response(
        self,
        invocation: int,
        messages: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Adopt the current candidate after its mandatory record commits."""
        if (
            invocation != self._provisional_invocation
            or self._provisional_state is None
            or self._provisional_prefix is None
        ):
            return False
        self._state = self._provisional_state
        self._accepted_prefix = deepcopy(self._provisional_prefix)
        self._boundary = len(self._accepted_prefix)
        self._provisional_state = None
        self._provisional_prefix = None
        self._request_messages = []
        if messages is not None:
            self._seal_boundary(messages)
        return True

    def discard_response(self, invocation: int | None = None) -> bool:
        """Discard an uncertain candidate and invalidate its prior handle."""
        if invocation is not None and invocation != self._provisional_invocation:
            return False
        self.invalidate()
        return True

    def invalidate(self) -> None:
        """Drop live, staged, and cached state after an unsafe transition."""
        self._state = None
        self._accepted_prefix = None
        self._request_messages = []
        self._active_portable_prefix = None
        self._active_binding = None
        self._active_receipt = None
        self._provisional_state = None
        self._provisional_prefix = None
        self._provisional_invocation += 1

    @staticmethod
    def project_response_message(
        message: dict[str, Any],
        response: LLMResponse,
    ) -> dict[str, Any]:
        """Mark a Chat projection already represented by provider output."""
        if response.provider_state is None:
            return message
        internal_meta = dict(message.get("_meta") or {})
        internal_meta[_PROVIDER_STATE_OUTPUT_META] = True
        message["_meta"] = internal_meta
        return message

    def checkpoint(
        self,
        messages: list[dict[str, Any]],
        *,
        model_messages: list[dict[str, Any]] | None = None,
    ) -> ProviderConversationState | None:
        """Return a durable settled-state snapshot without changing live state."""
        if self._state is None:
            return None
        durable_messages = self._messages_after_boundary(messages)
        governed_messages = (
            self._model_messages_after_boundary(model_messages)
            if model_messages is not None and durable_messages
            else None
        )
        pending_messages = governed_messages if governed_messages is not None else durable_messages
        return self._state.with_pending_messages([
            *self._state.pending_messages,
            *pending_messages,
        ])

    def finish(
        self,
        messages: list[dict[str, Any]],
    ) -> ProviderConversationState | None:
        """Return a final persistence candidate without reusing it in-process.

        The runner hands this private value to the session owner, which is the
        durability boundary for an ordinary non-source run.  Keeping it staged
        here ensures a tool-loop follow-up cannot consume a response that has
        not yet been recorded.
        """
        if self._provisional_state is not None:
            return self._provisional_state.with_pending_messages([
                *self._provisional_state.pending_messages,
                *self._messages_after_boundary(messages),
            ])
        self._state = self.checkpoint(messages)
        return self._state

    def _supports_bound_conversation_state(self) -> bool:
        try:
            return self._provider.supports_bound_conversation_state() is True
        except Exception:
            return False

    def _transport_receipt(self) -> ProviderConversationTransportReceipt | None:
        try:
            value = self._provider.conversation_state_transport_receipt(self._model)
        except Exception:
            value = None
        if isinstance(value, ProviderConversationTransportReceipt):
            return value
        # A third-party legacy provider can opt into state before it implements
        # the receipt method. Bind that state to this process object only: a
        # restored session gets a different opaque account value and therefore
        # rebuilds from portable history. First-party providers use the typed
        # base implementation above instead.
        provider_type = type(self._provider)
        runtime_identity = f"{provider_type.__module__}.{provider_type.__qualname__}:{id(self._provider)}"
        model = self._model if isinstance(self._model, str) and self._model else "<unknown>"
        return ProviderConversationTransportReceipt(
            provider=runtime_identity,
            account=sha256(runtime_identity.encode("utf-8")).hexdigest(),
            endpoint="<legacy-runtime>",
            protocol=runtime_identity,
            model=model,
        )

    @staticmethod
    def _state_has_known_context(state: ProviderConversationState) -> bool:
        context_tokens = state.payload.get("context_tokens")
        return (
            not isinstance(context_tokens, bool)
            and isinstance(context_tokens, int)
            and context_tokens >= 0
        )

    def _binding_matches(
        self,
        portable_prefix: list[dict[str, Any]],
        binding: ConversationRequestBinding,
        receipt: ProviderConversationTransportReceipt,
    ) -> bool:
        assert self._state is not None
        record = _valid_binding_record(self._state.conversation_binding)
        stored_receipt = self._state.transport_receipt
        if record is None or stored_receipt is None:
            return False
        if stored_receipt != receipt:
            return False
        current = binding.private_record(portable_prefix, receipt)
        if current is None or current["config_digest"] != record["config_digest"]:
            return False
        length = cast(int, record["prefix_length"])
        if len(portable_prefix) < length:
            return False
        accepted = portable_prefix[:length]
        if self._accepted_prefix is not None:
            return accepted == self._accepted_prefix
        return _digest(accepted) == record["prefix_digest"]

    def _request_delta(self, portable_prefix: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert self._state is not None
        record = _valid_binding_record(self._state.conversation_binding)
        if record is None:
            return self._messages_after_boundary(portable_prefix)
        start = cast(int, record["prefix_length"])
        return deepcopy(portable_prefix[start:])

    @staticmethod
    def _portable_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Detach model-visible messages and omit controller-only markers."""
        portable: list[dict[str, Any]] = []
        for message in messages:
            internal_meta = cast(object, message.get("_meta"))
            if (
                isinstance(internal_meta, dict)
                and cast(dict[str, Any], internal_meta).get(_PROVIDER_STATE_OUTPUT_META) is True
            ):
                continue
            detached = deepcopy(message)
            detached.pop("_meta", None)
            portable.append(detached)
        return portable

    @staticmethod
    def _without_provider_outputs(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        for message in messages:
            internal_meta = cast(object, message.get("_meta"))
            if (
                isinstance(internal_meta, dict)
                and cast(dict[str, Any], internal_meta).get(_PROVIDER_STATE_OUTPUT_META) is True
            ):
                continue
            pending.append(deepcopy(message))
        return pending

    @staticmethod
    def _state_with_metadata(
        state: ProviderConversationState,
        binding: dict[str, Any],
        receipt: ProviderConversationTransportReceipt,
    ) -> ProviderConversationState:
        return ProviderConversationState(
            kind=state.kind,
            provider=state.provider,
            model=state.model,
            version=state.version,
            payload=deepcopy(state.payload),
            pending_messages=deepcopy(state.pending_messages),
            conversation_binding=deepcopy(binding),
            transport_receipt=receipt,
            continuation_metadata_present=True,
        )

    def _messages_after_boundary(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._without_provider_outputs(messages[self._boundary:])

    @staticmethod
    def _model_messages_after_boundary(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Return the governed delta after the latest provider-owned boundary."""
        boundary = None
        for idx in range(len(messages) - 1, -1, -1):
            internal_meta = cast(object, messages[idx].get("_meta"))
            if (
                isinstance(internal_meta, dict)
                and cast(dict[str, Any], internal_meta).get(
                    _PROVIDER_STATE_BOUNDARY_META
                ) is True
            ):
                boundary = idx
                break
        if boundary is None:
            return None
        return ProviderConversationStateController._without_provider_outputs(messages[boundary + 1:])

    @staticmethod
    def _seal_boundary(messages: list[dict[str, Any]]) -> None:
        """Prevent later same-role injection merging across a state boundary."""
        if not messages:
            return
        internal_meta = dict(messages[-1].get("_meta") or {})
        internal_meta[_PROVIDER_STATE_BOUNDARY_META] = True
        messages[-1]["_meta"] = internal_meta
