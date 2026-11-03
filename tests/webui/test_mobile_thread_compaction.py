from __future__ import annotations

from nanobot.webui.ws_http import _compact_mobile_thread_payload


def test_mobile_thread_payload_drops_raw_tool_event_arrays() -> None:
    body = {
        "messages": [
            {
                "role": "tool",
                "kind": "trace",
                "content": "search projection helpers",
                "traces": ["search projection helpers"],
                "toolEvents": [{"result": "x" * 2_000_000}],
            },
            {"role": "assistant", "content": "Final answer."},
        ]
    }

    compacted = _compact_mobile_thread_payload(body)

    assert "toolEvents" not in compacted["messages"][0]
    assert compacted["messages"][0]["traces"] == ["search projection helpers"]
    assert compacted["messages"][1]["content"] == "Final answer."
