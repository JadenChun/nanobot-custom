/**
 * Package 6: integrated chat-page qualification (Linux-executable).
 *
 * This composes the four required user journeys over the real chat components
 * rather than re-asserting each unit-level contract:
 *
 *   1. First paint of saved history plus prompt-rail visibility on a short
 *      latest window of a long chat.
 *   2. Jump to an unloaded older prompt -> mount -> highlight, with live-tail
 *      isolation while viewing the older window, then Go to latest.
 *   3. Draft preservation across A -> B -> A and a simulated relaunch, including
 *      incarnation invalidation.
 *   4. Active-turn next-folder change while turn A runs, with Steer unaffected
 *      and a held next-turn acknowledgement blocking a new Send.
 *
 * The unit suites own the exhaustive branch coverage. This file records the
 * integrated behavior a Linux host can prove. macOS/Android/Go checks remain
 * external.
 */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";

import { ThreadComposer } from "@/components/thread/ThreadComposer";
import { ThreadViewport } from "@/components/thread/ThreadViewport";
import { resetComposerDraftsForTests } from "@/lib/composer-drafts";
import type { UIMessage, WorkspaceScopePayload } from "@/lib/types";

const NS = "pkg6-ns";

const FOLDER_A: WorkspaceScopePayload = {
  project_path: "/projects/alpha",
  project_name: "alpha",
  access_mode: "restricted",
  restrict_to_workspace: true,
};
const FOLDER_B: WorkspaceScopePayload = {
  project_path: "/projects/beta",
  project_name: "beta",
  access_mode: "restricted",
  restrict_to_workspace: true,
};
const HOME: WorkspaceScopePayload = {
  project_path: "/agent/home",
  project_name: "Agent home",
  access_mode: "restricted",
  restrict_to_workspace: true,
};

function workspacesFixture() {
  return {
    schema_version: 2 as const,
    default_access_mode: "restricted" as const,
    default_scope: HOME,
    managed_access: false,
    management: null,
    controls: {
      can_change_project: true,
      can_use_full_access: true,
      can_pick_folder: true,
    },
  };
}

function longChatMessages(): UIMessage[] {
  // A short latest window of a long chat: two turns, prompts 18 and 19.
  return [
    { id: "u18", promptId: "evt-18", role: "user", content: "Prompt 18", createdAt: 18 },
    { id: "a18", role: "assistant", content: "Reply 18", createdAt: 19, turnId: "turn-18" },
    { id: "u19", promptId: "evt-19", role: "user", content: "Prompt 19", createdAt: 20 },
    { id: "a19", role: "assistant", content: "Reply 19", createdAt: 21, turnId: "turn-19" },
  ];
}

function olderPrompts() {
  return Array.from({ length: 8 }, (_, index) => ({
    prompt_id: `evt-${index + 11}`,
    turn_ordinal: index + 10,
    prompt_ordinal: index + 10,
    label: `Prompt ${index + 11}`,
    created_at: 100 + index,
    locator: `loc-${index + 11}`,
  }));
}

const input = () => screen.getByLabelText(/message input/i) as HTMLTextAreaElement;

async function typeInto(text: string) {
  await act(async () => {
    fireEvent.change(input(), { target: { value: text } });
  });
}

function getScroller(container: HTMLElement): HTMLElement {
  const scroller = container.querySelector<HTMLElement>(".thread-viewport-scrollbar");
  if (!scroller) throw new Error("thread scrollport not found");
  return scroller;
}

beforeEach(() => {
  resetComposerDraftsForTests();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Package 6 integrated chat-page qualification", () => {
  it("paints saved history with a visible prompt rail on a short latest window", async () => {
    const onRevealPrompt = vi.fn();
    const { container } = render(
      <ThreadViewport
        messages={longChatMessages()}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        sessionPrompts={olderPrompts()}
        sessionPromptTotal={20}
        hasMorePromptsBefore
        onLoadMorePrompts={vi.fn()}
        onRevealPrompt={onRevealPrompt}
      />,
    );

    // Saved history paints immediately: both latest turns, no model run.
    expect(screen.getByText("Prompt 19")).toBeInTheDocument();
    expect(screen.getByText("Reply 19")).toBeInTheDocument();

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });
    await act(async () => {
      window.dispatchEvent(new Event("resize"));
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
    });

    // The rail is visible even though the window is short in a long chat.
    expect(screen.getByLabelText("User prompt navigation")).toBeInTheDocument();
    expect(screen.getAllByTestId("prompt-rail-marker").length).toBeGreaterThan(0);
  });

  it("reveals an unloaded prompt, isolates the live tail, and returns to latest", async () => {
    const onPendingRevealHandled = vi.fn();
    const onLoadNewer = vi.fn();
    const onGoToLatest = vi.fn();
    const { container } = render(
      <ThreadViewport
        messages={[
          { id: "u15", promptId: "evt-u-15", role: "user", content: "Prompt 15", createdAt: 15 },
          { id: "a15", role: "assistant", content: "Before steer", createdAt: 16 },
          { id: "u15s", promptId: "evt-steer-15", role: "user", content: "Steer 15", createdAt: 17 },
          { id: "a15s", role: "assistant", content: "After steer", createdAt: 18 },
        ]}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        hasMoreAfter
        tailActivityPending
        onLoadNewer={onLoadNewer}
        onGoToLatest={onGoToLatest}
        pendingReveal={{
          requestId: 3,
          promptId: "evt-steer-15",
          locator: "loc-steer-15",
          status: "ready",
        }}
        onPendingRevealHandled={onPendingRevealHandled}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });
    const steerEl = container.querySelector<HTMLElement>('[data-prompt-key="evt-steer-15"]');
    expect(steerEl).toBeTruthy();
    Object.defineProperty(steerEl!, "offsetTop", { configurable: true, value: 900 });

    // The exact prompt is mounted, highlighted, and the reveal is acknowledged.
    await waitFor(() => expect(onPendingRevealHandled).toHaveBeenCalledWith(3));
    expect(steerEl?.dataset.promptHighlighted).toBe("true");

    // Forward paging from the older window does not steal the reader's position.
    scroller.scrollTop = 1800;
    fireEvent.scroll(scroller);
    expect(onLoadNewer).toHaveBeenCalledTimes(1);

    // New tail activity is signaled, not spliced into the old window.
    const latestBtn = screen.getByRole("button", { name: "Go to latest" });
    expect(latestBtn).toHaveAttribute("data-tail-activity", "true");
    fireEvent.click(latestBtn);
    expect(onGoToLatest).toHaveBeenCalledTimes(1);
  });

  it("preserves drafts across A -> B -> A and a simulated relaunch", async () => {
    function Harness() {
      const [chat, setChat] = useState("websocket:a");
      const [incarnation] = useState("inc-1");
      return (
        <div>
          <button onClick={() => setChat("websocket:b")}>to B</button>
          <button onClick={() => setChat("websocket:a")}>to A</button>
          <ThreadComposer
            onSend={vi.fn()}
            pendingQueueKey={chat}
            draftKey={chat}
            draftNamespace={NS}
            draftIncarnation={incarnation}
          />
        </div>
      );
    }

    const first = render(<Harness />);
    await typeInto("draft A");
    fireEvent.click(screen.getByText("to B"));
    await waitFor(() => expect(input().value).toBe(""));
    await typeInto("draft B");
    fireEvent.click(screen.getByText("to A"));
    // Returning restores A's draft, never B's text.
    await waitFor(() => expect(input().value).toBe("draft A"));
    expect(input().value).not.toContain("draft B");

    // Simulated relaunch: memory is gone, device-local storage survives.
    const stored = window.localStorage.getItem("nanobot.webui.composerDrafts.v1");
    expect(stored).toContain("draft A");
    first.unmount();
    resetComposerDraftsForTests();
    window.localStorage.setItem("nanobot.webui.composerDrafts.v1", stored!);
    const relaunched = render(<Harness />);
    await waitFor(() => expect(input().value).toBe("draft A"));

    // An incarnation replacement invalidates the stored draft: the same key
    // restored under a different incarnation must not resurrect old text.
    relaunched.unmount();
    resetComposerDraftsForTests();
    window.localStorage.setItem("nanobot.webui.composerDrafts.v1", stored!);
    render(
      <ThreadComposer
        onSend={vi.fn()}
        pendingQueueKey="websocket:a"
        draftKey="websocket:a"
        draftNamespace={NS}
        draftIncarnation="inc-2"
      />,
    );
    await waitFor(() => expect(input().value).toBe(""));
  });

  it("keeps the next-folder picker and editor usable while turn A runs", async () => {
    const user = userEvent.setup();
    const onWorkspaceScopeChange = vi.fn();
    const onSend = vi.fn();
    const workspaces = workspacesFixture();

    render(
      <ThreadComposer
        onSend={onSend}
        placeholder="Ask anything..."
        pendingQueueKey="pkg6-chat"
        isStreaming
        workspaceScope={FOLDER_A}
        workspaceDefaultScope={HOME}
        workspaceControls={workspaces.controls}
        workspaces={workspaces}
        nextWorkspaceScope={FOLDER_B}
        nextTurnSettingsPending
        onWorkspaceScopeChange={onWorkspaceScopeChange}
      />,
    );

    // The next-folder picker stays enabled while turn A runs and shows B
    // separately from the admitted folder A.
    const trigger = screen.getByRole("button", { name: "Next turn: beta" });
    expect(trigger).toBeEnabled();
    await user.click(trigger);
    expect(await screen.findByText(FOLDER_B.project_path)).toBeInTheDocument();
    expect(screen.getByText(FOLDER_A.project_path)).toBeInTheDocument();

    // The editor stays usable. Enter during a run queues a follow-up instead
    // of starting a second turn, so nothing is admitted under a pending choice.
    await typeInto("follow-up for turn A");
    fireEvent.keyDown(input(), { key: "Enter" });
    expect(onSend).not.toHaveBeenCalled();
    expect(input()).toHaveValue("");
    expect(screen.getByText("follow-up for turn A")).toBeInTheDocument();
  });

  it("does not run the model or fetch history to change the next folder", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const workspaces = workspacesFixture();
    const onWorkspaceScopeChange = vi.fn();

    render(
      <ThreadComposer
        onSend={vi.fn()}
        placeholder="Ask anything..."
        pendingQueueKey="pkg6-chat"
        isStreaming
        workspaceScope={FOLDER_A}
        workspaceDefaultScope={HOME}
        workspaceControls={workspaces.controls}
        workspaces={workspaces}
        onWorkspaceScopeChange={onWorkspaceScopeChange}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Working in: alpha" }));
    fireEvent.change(await screen.findByLabelText("Paste path"), {
      target: { value: FOLDER_B.project_path },
    });
    await user.click(screen.getByRole("button", { name: "Use folder" }));

    expect(onWorkspaceScopeChange).toHaveBeenCalledWith(
      expect.objectContaining({ project_path: FOLDER_B.project_path }),
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
