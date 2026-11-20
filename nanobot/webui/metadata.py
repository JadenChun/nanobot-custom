"""Shared WebUI metadata keys."""

WEBUI_TURN_METADATA_KEY = "webui_turn_id"
WEBUI_SYSTEM_COMMAND_TURN_PREFIX = "webui-system:"
WEBSOCKET_TURN_OWNER_METADATA_KEY = "_websocket_turn_owner"
WEBUI_MESSAGE_SOURCE_METADATA_KEY = "_webui_message_source"

WEBUI_SOURCE_TOKEN_METADATA_KEY = "webui_source_token"
WEBUI_REQUEST_INTENT_METADATA_KEY = "webui_request_intent"
WEBUI_TARGET_TURN_METADATA_KEY = "webui_target_turn_id"
WEBUI_INSTALLATION_METADATA_KEY = "webui_installation_id"

# Server-only display state. The ingress flag requests archival after admission;
# the source is set by the admitted turn's delivery, never by the request.
WEBUI_DISPLAY_ON_COMMIT_KEY = "_webui_display_on_commit"
WEBUI_DISPLAY_SOURCE_KEY = "_webui_display_source"
