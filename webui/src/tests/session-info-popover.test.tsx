import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SessionInfoPopover, SessionSummaryCard } from "@/components/thread/SessionInfoPopover";
import { setAppLanguage } from "@/i18n";
import type { DelegatedChildPayload } from "@/lib/types";

function automationJob(
  nextRunAt = Date.now() + 3_600_000,
  state: Record<string, unknown> = {},
) {
  return {
    id: "job-1",
    name: "Morning check",
    enabled: true,
    schedule: { kind: "every", every_ms: 3_600_000 },
    payload: { message: "Check the project status" },
    state: { next_run_at_ms: nextRunAt, ...state },
  };
}

function automationsResponse(jobs: unknown[]) {
  return {
    ok: true,
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => ({
      jobs,
    }),
  } as Response;
}

describe("SessionInfoPopover", () => {
  beforeEach(async () => {
    await setAppLanguage("en");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(automationsResponse([automationJob()])),
    );
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("lists session subagents newest first, with their own status, metrics and actions", async () => {
    const user = userEvent.setup();
    const older: DelegatedChildPayload = {
      child_id: "old", child_session_key: "delegated:old", profile: "worker",
      state: "completed", runtime_generation: 0, task_label: "Review setup",
    };
    const newer: DelegatedChildPayload = {
      ...older, child_id: "new", child_session_key: "delegated:new", harness: "claude_sdk",
      task_label: "Trace recent work", outcome: "partial", duration_ms: 62_000, step_count: 0,
    };
    const onJump = vi.fn();
    const onOpenHistory = vi.fn();
    render(
      <SessionSummaryCard sessionKey="websocket:chat-1" token="tok" title="Release work" active
        subagents={{
          entries: [{ child: older, createdAt: 1000, canJump: false }, { child: newer, createdAt: 2000, canJump: true }],
          statusAvailable: true, onJump, onOpenHistory,
        }}
      />,
    );
    const section = within(screen.getByRole("region", { name: "Subagents (2)" }));
    const [first, second] = section.getAllByRole("listitem");
    expect(first).toHaveTextContent("Trace recent work");
    expect(first).toHaveTextContent("Claude");
    expect(first).toHaveTextContent("Partial");
    expect(first).toHaveTextContent("1m 2s");
    expect(first).toHaveTextContent("0 steps");
    expect(second).toHaveTextContent("Review setup");
    expect(second).toHaveTextContent("Kohaku");
    expect(second).not.toHaveTextContent(/\d steps/);
    expect(within(second).getByRole("button", { name: /Jump to.*Review setup/ })).toBeDisabled();
    await user.click(within(first).getByRole("button", { name: /Jump to.*Trace recent work/ }));
    expect(onJump).toHaveBeenCalledWith(newer);
    const openHistory = within(first).getByRole("button", { name: /Open history.*Trace recent work/ });
    await user.click(openHistory);
    expect(onOpenHistory).toHaveBeenCalledWith(newer, openHistory);
    expect(await screen.findByText("Morning check")).toBeInTheDocument();
  });

  it("loads and displays session automations when opened", async () => {
    const user = userEvent.setup();

    render(
      <SessionInfoPopover
        sessionKey="websocket:chat-1"
        token="tok"
        title="Release work"
      />,
    );

    await user.click(screen.getByRole("button", { name: "Session details" }));

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "/api/sessions/websocket%3Achat-1/automations",
        expect.objectContaining({
          headers: { Authorization: "Bearer tok" },
        }),
      );
    });
    expect(await screen.findByText("Morning check")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: /Subagents/ })).not.toBeInTheDocument();
    expect(screen.getByText("Check the project status")).toBeInTheDocument();
  });

  it("localizes the panel chrome in Simplified Chinese", async () => {
    await setAppLanguage("zh-CN");
    const user = userEvent.setup();

    render(
      <SessionInfoPopover
        sessionKey="websocket:chat-1"
        token="tok"
        title="@hyperframes 使用指南"
      />,
    );

    await user.click(screen.getByRole("button", { name: "会话详情" }));

    expect(await screen.findByText("会话")).toBeInTheDocument();
    expect(screen.getByText("自动任务")).toBeInTheDocument();
    expect(screen.getByText("Morning check")).toBeInTheDocument();
    expect(screen.getByText(/下次/)).toBeInTheDocument();
    expect(screen.queryByText("Session")).not.toBeInTheDocument();
    expect(screen.queryByText("Automations")).not.toBeInTheDocument();
  });

  it("shows a short pending label for deferred automations", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        automationsResponse([automationJob(Date.now() - 1000, { pending: true })]),
      ),
    );
    const user = userEvent.setup();

    render(
      <SessionInfoPopover
        sessionKey="websocket:chat-1"
        token="tok"
        title="Release work"
      />,
    );

    await user.click(screen.getByRole("button", { name: "Session details" }));

    expect(await screen.findByText("Runs shortly")).toBeInTheDocument();
    expect(screen.queryByText(/ago/i)).not.toBeInTheDocument();
  });

  it("shows the actual message received by a local trigger", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        automationsResponse([
          {
            id: "trg_123",
            name: "PR monitor",
            enabled: true,
            kind: "local_trigger",
            schedule: { kind: "local" },
            payload: {
              kind: "local_trigger",
              message: "Review PR #4591",
              command: 'kohaku trigger trg_123 "message"',
            },
            state: { pending: false },
          },
        ]),
      ),
    );
    const user = userEvent.setup();

    render(
      <SessionInfoPopover
        sessionKey="websocket:chat-1"
        token="tok"
        title="Release work"
      />,
    );

    await user.click(screen.getByRole("button", { name: "Session details" }));

    expect(await screen.findByText("Review PR #4591")).toBeInTheDocument();
    expect(screen.queryByText('kohaku trigger trg_123 "message"')).not.toBeInTheDocument();
  });

  it("refreshes while open so completed one-shot automations disappear", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce(automationsResponse([automationJob(Date.now() + 1000)]))
        .mockResolvedValue(automationsResponse([])),
    );
    const user = userEvent.setup();

    render(
      <SessionInfoPopover
        sessionKey="websocket:chat-1"
        token="tok"
        title="Release work"
      />,
    );

    await user.click(screen.getByRole("button", { name: "Session details" }));
    expect(await screen.findByText("Morning check")).toBeInTheDocument();

    await waitFor(
      () => {
        expect(screen.queryByText("Morning check")).not.toBeInTheDocument();
      },
      { timeout: 4500 },
    );
    expect(screen.getByText("No automations in this session yet.")).toBeInTheDocument();
  }, 8000);

  it("coalesces focus refreshes while a session automation request is in flight", async () => {
    let resolveRequest!: (response: Response) => void;
    const pendingRequest = new Promise<Response>((resolve) => {
      resolveRequest = resolve;
    });
    const fetchMock = vi.fn(() => pendingRequest);
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();

    render(
      <SessionInfoPopover
        sessionKey="websocket:chat-1"
        token="tok"
        title="Release work"
      />,
    );

    await user.click(screen.getByRole("button", { name: "Session details" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    window.dispatchEvent(new Event("focus"));
    window.dispatchEvent(new Event("focus"));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => {
      resolveRequest(automationsResponse([]));
      await pendingRequest;
    });
  });

  it("stays open on outside click and toggles closed when clicking the trigger again", async () => {
    const user = userEvent.setup();

    render(
      <div>
        <button type="button">Outside target</button>
        <SessionInfoPopover
          sessionKey="websocket:chat-1"
          token="tok"
          title="Release work"
        />
      </div>,
    );

    const toggle = screen.getByRole("button", { name: "Session details" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");

    await user.click(toggle);
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByRole("button", { name: "Outside target" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    await user.click(toggle);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(toggle).toHaveAttribute("aria-pressed", "false");
  });
});
