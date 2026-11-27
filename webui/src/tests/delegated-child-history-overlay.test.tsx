import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DelegatedChildHistoryOverlay } from "@/components/thread/DelegatedChildHistoryOverlay";
import type {
  DelegatedChildPayload,
  DelegatedChildSummaryPayload,
  SourceOrigin,
} from "@/lib/types";

const root: SourceOrigin = {
  store_identity: "root-store",
  session_key: "websocket:parent",
  incarnation: "root-incarnation",
  generation: 4,
};

const summaryChild: DelegatedChildSummaryPayload = {
  history_kind: "summary",
  parent_session_key: root.session_key,
  child_session_key: "delegated:legacy-child",
  child_id: "legacy-child",
  profile: "worker",
  task_label: null,
  task_label_truncated: false,
  state: "completed",
  outcome: "completed",
  runtime_generation: null,
  turn_id: null,
  parent_run_id: null,
  child_run_id: null,
  parent_store_identity: null,
  parent_incarnation: null,
  parent_generation: null,
  root_key: root.session_key,
  root_store_identity: null,
  root_incarnation: null,
  root_generation: null,
  child_store_identity: null,
  child_incarnation: null,
  child_generation: null,
  run_sequence: null,
  state_sequence: null,
  control_epoch: null,
  can_stop: false,
  result_summary: "The old result remains available as a summary.",
  result_summary_truncated: false,
};

const canonicalChild: DelegatedChildPayload = {
  parent_session_key: root.session_key,
  child_session_key: "delegated:canonical-child",
  child_id: "canonical-child",
  profile: "worker",
  task_label: "Review current task",
  task_label_truncated: false,
  state: "completed",
  outcome: "completed",
  runtime_generation: 1,
  turn_id: null,
  parent_run_id: "parent-run",
  child_run_id: "child-run",
  parent_store_identity: root.store_identity,
  parent_incarnation: root.incarnation,
  parent_generation: root.generation,
  root_key: root.session_key,
  root_store_identity: root.store_identity,
  root_incarnation: root.incarnation,
  root_generation: root.generation,
  child_store_identity: "canonical-child-store",
  child_incarnation: "canonical-child-incarnation",
  child_generation: 1,
  run_sequence: 1,
  state_sequence: 1,
  control_epoch: 1,
  holds_capacity: false,
  can_stop: false,
  owned_work: false,
  pending_approval_id: null,
  blocked_run_id: null,
  result_disposition: "settled",
  result_summary: null,
  result_summary_truncated: false,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DelegatedChildHistoryOverlay", () => {
  it("shows authorized legacy summary history with an explicit summary label", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        schemaVersion: 3,
        read_only: true,
        history_kind: "summary",
        parent_session_key: root.session_key,
        root_key: root.session_key,
        source: null,
        page: {
          before_cursor: null,
          has_more_before: false,
          loaded_message_count: 2,
        },
        messages: [{
          id: "summary-message",
          role: "assistant",
          content: "A summary of the old delegated work.",
          createdAt: 0,
          timestampAvailable: false,
        }, {
          id: "epoch-zero-message",
          role: "assistant",
          content: "A message with a real epoch-zero timestamp.",
          createdAt: 0,
          timestampAvailable: true,
        }],
      }),
    }));

    const onClose = vi.fn();
    render(
      <DelegatedChildHistoryOverlay
        child={summaryChild}
        rootKey={root.session_key}
        rootRows={[canonicalChild, summaryChild]}
        expectedRoot={root}
        provenanceComplete
        token="tok"
        onClose={onClose}
      />,
    );

    expect(await screen.findByText("Summary-only history · read only")).toBeInTheDocument();
    expect(screen.getByText(/A summary of the old delegated work\./)).toBeInTheDocument();
    expect(screen.getByText(/A message with a real epoch-zero timestamp\./)).toBeInTheDocument();
    expect(screen.getByText(/Full canonical activity is unavailable/)).toBeInTheDocument();
    const back = screen.getByRole("button", { name: "Back to parent chat" });
    expect(back.textContent).toBe("");
    expect(back.querySelector("svg")).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Fork" })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog").querySelectorAll("[data-message-timestamp]")).toHaveLength(1);
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/sessions/delegated%3Alegacy-child/webui-thread"),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    ));
  });

  it("refuses canonical history that lacks the exact child source", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        schemaVersion: 3,
        read_only: true,
        history_kind: "canonical",
        parent_session_key: root.session_key,
        root_key: root.session_key,
        source: null,
        page: { before_cursor: null, has_more_before: false, loaded_message_count: 0 },
        messages: [],
      }),
    }));

    render(
      <DelegatedChildHistoryOverlay
        child={canonicalChild}
        rootKey={root.session_key}
        rootRows={[canonicalChild]}
        expectedRoot={root}
        provenanceComplete
        token="tok"
        onClose={() => {}}
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This delegated history could not be verified.",
    );
    expect(screen.queryByText("No canonical activity is available.")).not.toBeInTheDocument();
  });
});

const ASSIGNMENT = "Investigate the token contract.\nRules: do not change anything.";

/** Rows shaped like the child history of a claude_sdk child: no event timestamps. */
function toolRow(index: number, name: string, args: Record<string, unknown>, withTraces = true) {
  const line = `${name}(${JSON.stringify(args)})`;
  const output = `raw output ${index}`;
  return {
    id: `tool-${index}`,
    role: "tool",
    kind: "trace",
    content: withTraces ? line : output,
    ...(withTraces ? { traces: [line] } : {}),
    toolEvents: [{
      phase: "end",
      call_id: `toolu_${index}|fc_${index}`,
      name,
      ...(withTraces ? { arguments: args } : {}),
      result: output,
    }],
    createdAt: 0,
    timestampAvailable: false,
    turnId: "child-run",
  };
}

function textRow(id: string, role: "user" | "assistant", content: string) {
  return { id, role, content, createdAt: 0, timestampAvailable: false, turnId: "child-run" };
}

function historyPayload(
  messages: unknown[],
  extra: Record<string, unknown> = {},
  beforeCursor: string | null = null,
) {
  return {
    schemaVersion: 3,
    read_only: true,
    history_kind: "canonical",
    parent_session_key: root.session_key,
    root_key: root.session_key,
    source: {
      store_identity: canonicalChild.child_store_identity,
      session_key: canonicalChild.child_session_key,
      incarnation: canonicalChild.child_incarnation,
      generation: canonicalChild.child_generation,
    },
    page: {
      before_cursor: beforeCursor,
      has_more_before: beforeCursor !== null,
      loaded_message_count: messages.length,
    },
    messages,
    ...extra,
  };
}

function stubPages(...pages: unknown[]) {
  const fetchMock = vi.fn();
  for (const page of pages) {
    fetchMock.mockResolvedValueOnce({ ok: true, status: 200, json: async () => page });
  }
  vi.stubGlobal("fetch", fetchMock);
}

function renderCanonical() {
  render(
    <DelegatedChildHistoryOverlay
      child={canonicalChild}
      rootKey={root.session_key}
      rootRows={[canonicalChild]}
      expectedRoot={root}
      provenanceComplete
      token="tok"
      onClose={() => {}}
    />,
  );
}

/** Open every collapsed activity group so its steps are in the document. */
function expandActivity() {
  for (const toggle of screen.getAllByRole("button", { expanded: false })) {
    if (/^(Worked|Activity)/.test(toggle.textContent ?? "")) fireEvent.click(toggle);
  }
}

const assignmentToggle = () => screen.getByRole("button", { name: /^Assignment/ });

describe("DelegatedChildHistoryOverlay child history", () => {
  it("pins the assignment, collapsed to its first line, while older pages are unloaded", async () => {
    stubPages(historyPayload(
      [toolRow(1, "read_file", { path: "a.ts" })],
      { assignment: ASSIGNMENT },
      "older",
    ));
    renderCanonical();

    await screen.findByRole("button", { name: "Load earlier activity" });
    const toggle = assignmentToggle();
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveTextContent("Investigate the token contract.");
    expect(screen.queryByText(/Rules: do not change anything/)).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Rules: do not change anything/)).toBeInTheDocument();
  });

  it("shows the assignment once after the first page of history is loaded", async () => {
    stubPages(
      historyPayload([toolRow(2, "grep", { pattern: "x" })], { assignment: ASSIGNMENT }, "older"),
      historyPayload(
        [textRow("assignment", "user", ASSIGNMENT), toolRow(1, "read_file", { path: "a.ts" })],
        { assignment: ASSIGNMENT },
      ),
    );
    renderCanonical();

    fireEvent.click(await screen.findByRole("button", { name: "Load earlier activity" }));
    await waitFor(() => expect(
      screen.queryByRole("button", { name: "Load earlier activity" }),
    ).not.toBeInTheDocument());
    expect(screen.getAllByText(/Investigate the token contract\./)).toHaveLength(1);
    expect(assignmentToggle()).toBeInTheDocument();
  });

  it("shows the duration of the child", async () => {
    stubPages(historyPayload(
      [toolRow(1, "read_file", { path: "a.ts" }), textRow("report", "assistant", "Report")],
      { assignment: ASSIGNMENT, duration_ms: 372_000 },
    ));
    renderCanonical();

    expect(await screen.findByText("Worked for 6m 12s")).toBeInTheDocument();
    expect(screen.getByRole("dialog").querySelectorAll("[data-message-timestamp]")).toHaveLength(0);
  });

  it.each([true, false])(
    "describes tool steps like the main stream (server traces: %s)",
    async (withTraces) => {
      stubPages(historyPayload([
        toolRow(1, "grep", { pattern: "token" }, withTraces),
        toolRow(2, "read_file", { path: "src/a.ts" }, withTraces),
        toolRow(3, "find_files", { pattern: "*.md" }, withTraces),
        toolRow(4, "exec", { command: "rg token src" }, withTraces),
      ]));
      renderCanonical();

      await screen.findByText(/Worked|Activity/);
      expandActivity();
      const text = screen.getByRole("dialog").textContent ?? "";
      expect(text).toContain("Searched files");
      expect(text).toContain("Read file");
      expect(text).toContain("Found files");
      expect(text).toMatch(/Ran |Searched/);
      expect(text).not.toContain("Completed step");
      expect(text).not.toContain("raw output");
    },
  );
});
