"""Message bus module for decoupled channel-agent communication."""

from kohaku.bus.events import InboundMessage, OutboundMessage
from kohaku.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
