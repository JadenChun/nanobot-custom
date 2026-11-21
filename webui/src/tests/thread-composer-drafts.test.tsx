import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { encodeImage } from "@/lib/imageEncode";
import { ThreadComposer } from "@/components/thread/ThreadComposer";
import {
  clearComposerDraftsForSession,
  draftIncarnationReplaced,
  handoffComposerDraft,
  loadComposerDraft,
  NEW_CHAT_DRAFT_KEY,
  resetComposerDraftsForTests,
} from "@/lib/composer-drafts";

vi.mock("@/lib/imageEncode", () => ({
  encodeImage: vi.fn(async (file: File) => ({
    ok: true,
    dataUrl: `data:image/png;base64,${btoa(file.name)}`,
    bytes: Math.max(1, file.size),
    normalized: false,
  })),
}));

const NS = "ns1";

function Composer({
  id,
  incarnation = null,
  revoked = false,
  persist = true,
  quote = null,
  onQuote,
}: {
  id: string;
  incarnation?: string | null;
  revoked?: boolean;
  persist?: boolean;
  quote?: string | null;
  onQuote?: (text: string | null) => void;
}) {
  return (
    <ThreadComposer
      onSend={vi.fn()}
      pendingQueueKey={persist ? id : null}
      draftKey={id}
      draftNamespace={NS}
      draftIncarnation={incarnation}
      draftRevoked={revoked}
      quotedContext={quote}
      onQuotedContextChange={onQuote}
    />
  );
}

const input = () => screen.getByLabelText(/message input/i) as HTMLTextAreaElement;

async function type(text: string) {
  await act(async () => {
    fireEvent.change(input(), { target: { value: text } });
  });
}

async function attach(name: string) {
  const file = new File([new Uint8Array(4)], name, { type: "image/png" });
  const fileInput = input().closest("form")!.querySelector('input[type="file"]') as HTMLInputElement;
  await act(async () => {
    fireEvent.change(fileInput, { target: { files: [file] } });
  });
  await waitFor(() => expect(screen.getByTestId("composer-chip")).toHaveTextContent(name));
}

describe("chat-scoped composer drafts", () => {
  beforeEach(() => {
    resetComposerDraftsForTests();
  });

  it("restores text and attachments for A -> B -> A without flashing B", async () => {
    const { rerender } = render(<Composer id="websocket:a" />);
    await type("draft A");
    await attach("a.png");
    rerender(<Composer id="websocket:b" />);
    expect(input().value).toBe("");
    expect(screen.queryByTestId("composer-chip")).toBeNull();
    await type("draft B");
    rerender(<Composer id="websocket:a" />);
    expect(input().value).toBe("draft A");
    expect(screen.getByTestId("composer-chip")).toHaveTextContent("a.png");
    rerender(<Composer id="websocket:b" />);
    expect(input().value).toBe("draft B");
    expect(screen.queryByTestId("composer-chip")).toBeNull();
  });

  it("keeps the quote per chat", async () => {
    const onQuote = vi.fn();
    const { rerender } = render(<Composer id="websocket:a" quote="quoted" onQuote={onQuote} />);
    rerender(<Composer id="websocket:b" quote="quoted" onQuote={onQuote} />);
    expect(onQuote).toHaveBeenLastCalledWith(null);
    rerender(<Composer id="websocket:a" quote={null} onQuote={onQuote} />);
    expect(onQuote).toHaveBeenLastCalledWith("quoted");
  });

  it("persists saved-chat text across relaunch and asks to reattach attachments", async () => {
    const first = render(<Composer id="websocket:a" />);
    await type("after relaunch");
    await attach("photo.png");
    await waitFor(() =>
      expect(window.localStorage.getItem("nanobot.webui.composerDrafts.v1")).toContain("photo.png"),
    );
    expect(window.localStorage.getItem("nanobot.webui.composerDrafts.v1")).not.toContain("base64");
    first.unmount();
    // Simulate relaunch: memory gone, storage kept.
    const stored = window.localStorage.getItem("nanobot.webui.composerDrafts.v1")!;
    resetComposerDraftsForTests();
    window.localStorage.setItem("nanobot.webui.composerDrafts.v1", stored);
    render(<Composer id="websocket:a" />);
    expect(input().value).toBe("after relaunch");
    expect(screen.queryByTestId("composer-chip")).toBeNull();
    expect(screen.getByTestId("composer-reattach-notice")).toHaveTextContent("photo.png");
  });

  it("does not persist temporary or new-chat drafts across relaunch", async () => {
    const view = render(<Composer id="temp-1" persist={false} />);
    await type("ephemeral");
    view.unmount();
    expect(window.localStorage.getItem("nanobot.webui.composerDrafts.v1")).toBeNull();
  });

  it("scopes persisted drafts to the namespace", async () => {
    const first = render(<Composer id="websocket:a" />);
    await type("mine");
    first.unmount();
    const stored = window.localStorage.getItem("nanobot.webui.composerDrafts.v1")!;
    resetComposerDraftsForTests();
    window.localStorage.setItem("nanobot.webui.composerDrafts.v1", stored.replaceAll(NS, "other"));
    render(<Composer id="websocket:a" />);
    expect(input().value).toBe("");
  });

  it("invalidates the draft when the session is replaced (new incarnation)", async () => {
    const { rerender } = render(<Composer id="websocket:a" incarnation="inc-1" />);
    await type("old session text");
    rerender(<Composer id="websocket:a" incarnation="inc-2" />);
    await waitFor(() => expect(input().value).toBe(""));
    expect(window.localStorage.getItem("nanobot.webui.composerDrafts.v1")).toBeNull();
  });

  it("learning the incarnation after history loads is not a switch", async () => {
    const { rerender } = render(<Composer id="websocket:a" />);
    await type("keep me");
    rerender(<Composer id="websocket:a" incarnation="inc-1" />);
    expect(input().value).toBe("keep me");
  });

  it("clears drafts on revoked access and never re-saves them on switch", async () => {
    const { rerender } = render(<Composer id="websocket:a" />);
    await type("secret");
    rerender(<Composer id="websocket:a" revoked />);
    await waitFor(() => expect(input().value).toBe(""));
    rerender(<Composer id="websocket:b" revoked={false} />);
    rerender(<Composer id="websocket:a" />);
    expect(input().value).toBe("");
    expect(window.localStorage.getItem("nanobot.webui.composerDrafts.v1")).toBeNull();
  });

  it("deleted sessions drop drafts and refuse a late switch-away save", async () => {
    const { rerender } = render(<Composer id="websocket:a" />);
    await type("doomed");
    clearComposerDraftsForSession(null, "websocket:a", { deleted: true });
    rerender(<Composer id="websocket:b" />);
    rerender(<Composer id="websocket:a" />);
    expect(input().value).toBe("");
    expect(window.localStorage.getItem("nanobot.webui.composerDrafts.v1")).toBeNull();
  });

  it("restores caret and focus when returning, without stealing focus for background chats", async () => {
    const { rerender } = render(<Composer id="websocket:a" />);
    await type("hello world");
    act(() => {
      input().focus();
      input().setSelectionRange(2, 5);
    });
    rerender(<Composer id="websocket:b" />);
    await type("other");
    act(() => input().blur());
    rerender(<Composer id="websocket:a" />);
    await waitFor(() => expect(document.activeElement).toBe(input()));
    expect(input().selectionStart).toBe(2);
    expect(input().selectionEnd).toBe(5);
    // B was not focused when we left it: returning must not focus.
    rerender(<Composer id="websocket:b" />);
    act(() => input().blur());
    rerender(<Composer id="websocket:a" />);
    rerender(<Composer id="websocket:b" />);
    await act(async () => { await new Promise((r) => requestAnimationFrame(() => r(null))); });
    expect(input().value).toBe("other");
    expect(document.activeElement).not.toBe(input());
  });

  it("rerendering the same chat keeps the editor mounted and focused", async () => {
    const { rerender } = render(<Composer id="websocket:a" />);
    await type("typing");
    const el = input();
    act(() => el.focus());
    rerender(<Composer id="websocket:a" incarnation="inc-1" />);
    expect(input()).toBe(el);
    expect(document.activeElement).toBe(el);
    expect(el.value).toBe("typing");
  });

  it("rejects a stale attachment completion after switching away and back", async () => {
    let finish!: (value: unknown) => void;
    vi.mocked(encodeImage).mockImplementationOnce(
      () => new Promise((resolve) => { finish = resolve; }) as never,
    );
    const { rerender } = render(<Composer id="websocket:a" />);
    await type("A text");
    const file = new File([new Uint8Array(4)], "late.png", { type: "image/png" });
    const fileInput = input().closest("form")!.querySelector('input[type="file"]') as HTMLInputElement;
    await act(async () => { fireEvent.change(fileInput, { target: { files: [file] } }); });
    rerender(<Composer id="websocket:b" />);
    await type("B newer");
    rerender(<Composer id="websocket:a" />);
    await act(async () => {
      finish({ ok: true, dataUrl: "data:image/png;base64,bGF0ZQ==", bytes: 4, normalized: false });
    });
    expect(screen.queryByTestId("composer-chip")).toBeNull();
    expect(input().value).toBe("A text");
    rerender(<Composer id="websocket:b" />);
    expect(input().value).toBe("B newer");
    expect(screen.queryByTestId("composer-chip")).toBeNull();
  });

  it("does not steal focus from another element during a background refresh", async () => {
    const { rerender } = render(
      <div>
        <button type="button">Other control</button>
        <Composer id="websocket:a" />
      </div>,
    );
    await type("keep typing");
    const other = screen.getByRole("button", { name: "Other control" });
    act(() => other.focus());
    expect(document.activeElement).toBe(other);
    rerender(
      <div>
        <button type="button">Other control</button>
        <Composer id="websocket:a" incarnation="inc-1" />
      </div>,
    );
    await act(async () => { await new Promise((r) => requestAnimationFrame(() => r(null))); });
    expect(document.activeElement).toBe(other);
  });

  it("clears the new-chat draft scope when an async first send resolves after switching away, without touching the active chat", async () => {
    let resolveSend!: (accepted: boolean) => void;
    const onSend = vi.fn(() => new Promise<boolean>((resolve) => {
      resolveSend = resolve;
    }));
    const { rerender } = render(
      <ThreadComposer
        onSend={onSend}
        pendingQueueKey={null}
        draftKey={null}
        draftNamespace={NS}
      />,
    );
    await type("new chat message");
    await attach("first.png");
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    });
    expect(onSend).toHaveBeenCalledTimes(1);

    // User switches away to chat B while new_chat is still in flight.
    rerender(
      <ThreadComposer
        onSend={vi.fn()}
        pendingQueueKey="chat-b"
        draftKey="websocket:b"
        draftNamespace={NS}
      />,
    );
    expect(input().value).toBe("");
    expect(screen.queryByTestId("composer-chip")).toBeNull();
    await type("draft in B");

    // First send completes for the new chat.
    await act(async () => {
      resolveSend(true);
    });
    // Chat B's draft is untouched.
    expect(input().value).toBe("draft in B");

    // Returning to New Chat shows an empty composer (the handed-off draft was consumed).
    rerender(
      <ThreadComposer
        onSend={onSend}
        pendingQueueKey={null}
        draftKey={null}
        draftNamespace={NS}
      />,
    );
    expect(input().value).toBe("");
    expect(screen.queryByTestId("composer-chip")).toBeNull();
  });

  it("transfers the full new-chat draft payload (text, quote, attachments) strictly to the created chat", async () => {
    const onQuote = vi.fn();
    const { rerender } = render(
      <ThreadComposer
        onSend={vi.fn()}
        pendingQueueKey={null}
        draftKey={null}
        draftNamespace={NS}
        quotedContext="quoted excerpt"
        onQuotedContextChange={onQuote}
      />,
    );
    await type("full handoff draft");
    await attach("handoff.png");

    const snapshot = loadComposerDraft({
      namespace: NS,
      key: NEW_CHAT_DRAFT_KEY,
      persist: false,
    });
    // Switch away to existing chat B before the new chat finishes creation.
    rerender(
      <ThreadComposer
        onSend={vi.fn()}
        pendingQueueKey="chat-b"
        draftKey="websocket:b"
        draftNamespace={NS}
        quotedContext={null}
        onQuotedContextChange={onQuote}
      />,
    );
    expect(input().value).toBe("");
    expect(screen.queryByTestId("composer-chip")).toBeNull();

    // New chat creation finishes and hands off the snapshot strictly to websocket:created.
    handoffComposerDraft(NS, "websocket:created", snapshot, "full handoff draft");

    // Chat B still sees nothing.
    expect(input().value).toBe("");
    expect(screen.queryByTestId("composer-chip")).toBeNull();

    // Switching to the created chat restores the full payload.
    rerender(
      <ThreadComposer
        onSend={vi.fn()}
        pendingQueueKey="created"
        draftKey="websocket:created"
        draftNamespace={NS}
        quotedContext={null}
        onQuotedContextChange={onQuote}
      />,
    );
    expect(input().value).toBe("full handoff draft");
    expect(screen.getByTestId("composer-chip")).toHaveTextContent("handoff.png");
    expect(onQuote).toHaveBeenLastCalledWith("quoted excerpt");

    // And the new-chat slot is clean.
    rerender(
      <ThreadComposer
        onSend={vi.fn()}
        pendingQueueKey={null}
        draftKey={null}
        draftNamespace={NS}
        quotedContext={null}
        onQuotedContextChange={onQuote}
      />,
    );
    expect(input().value).toBe("");
    expect(screen.queryByTestId("composer-chip")).toBeNull();
  });

  it("invalidates drafts on a /new reset generation bump (inc-1:0 -> inc-1:1), including post-relaunch", async () => {
    const first = render(<Composer id="websocket:a" incarnation="inc-1:0" />);
    await type("before /new reset");
    first.unmount();
    const stored = window.localStorage.getItem("nanobot.webui.composerDrafts.v1")!;
    resetComposerDraftsForTests();
    window.localStorage.setItem("nanobot.webui.composerDrafts.v1", stored);
    expect(draftIncarnationReplaced({ namespace: NS, key: "websocket:a" }, "inc-1:1")).toBe(true);
    const { rerender } = render(<Composer id="websocket:a" incarnation="inc-1:0" />);
    expect(input().value).toBe("before /new reset");
    rerender(<Composer id="websocket:a" incarnation="inc-1:1" />);
    await waitFor(() => expect(input().value).toBe(""));
    expect(window.localStorage.getItem("nanobot.webui.composerDrafts.v1")).toBeNull();
  });

  it("allows drafting again when a deleted session key is recreated with a new incarnation", async () => {
    const { rerender } = render(<Composer id="websocket:a" incarnation="inc-1:0" />);
    await type("doomed before delete");
    clearComposerDraftsForSession(null, "websocket:a", { deleted: true });
    // Late switch-away save from the deleted incarnation is blocked.
    rerender(<Composer id="websocket:b" incarnation="inc-b:0" />);
    rerender(<Composer id="websocket:a" incarnation="inc-1:0" />);
    expect(input().value).toBe("");

    // Recreated session under the same key with a new incarnation can draft again.
    rerender(<Composer id="websocket:a" incarnation="inc-2:0" />);
    await type("fresh draft in recreated session");
    rerender(<Composer id="websocket:b" incarnation="inc-b:0" />);
    rerender(<Composer id="websocket:a" incarnation="inc-2:0" />);
    expect(input().value).toBe("fresh draft in recreated session");
  });
});
