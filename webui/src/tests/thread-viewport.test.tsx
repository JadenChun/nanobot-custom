import { createRef, useRef } from "react";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PromptNavigator } from "@/components/thread/PromptNavigator";
import { ThreadComposer } from "@/components/thread/ThreadComposer";
import {
  HISTORY_WINDOW_INCREMENT,
  INITIAL_HISTORY_WINDOW,
  ThreadViewport,
  type ThreadViewportHandle,
  windowMessages,
} from "@/components/thread/ThreadViewport";
import { ThreadCameraController } from "@/components/thread/thread-camera";
import { ThreadMotionCoordinator } from "@/components/thread/thread-motion";
import type { UIMessage } from "@/lib/types";

const messages: UIMessage[] = [
  {
    id: "u1",
    role: "user",
    content: "hello",
    createdAt: Date.now(),
  },
];

const emptyMessages: UIMessage[] = [];

afterEach(() => {
  vi.restoreAllMocks();
});

async function flushAnimationFrame(): Promise<void> {
  await act(async () => {
    await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
  });
}

function dispatchUserScroll(scroller: HTMLElement): void {
  fireEvent.pointerDown(scroller, { button: 0 });
  scroller.dispatchEvent(new Event("scroll"));
}

interface ResizeObserverInstance {
  elements: Element[];
  callback: ResizeObserverCallback;
  disconnect: ReturnType<typeof vi.fn>;
}

function stubVisualViewport({
  height,
  innerHeight,
  offsetTop = 0,
}: {
  height: number;
  innerHeight: number;
  offsetTop?: number;
}) {
  const originalInnerHeight = window.innerHeight;
  const originalVisualViewport = window.visualViewport;
  const target = new EventTarget();
  const viewport = {
    width: 390,
    height,
    offsetTop,
    offsetLeft: 0,
    pageTop: offsetTop,
    pageLeft: 0,
    scale: 1,
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatchEvent: target.dispatchEvent.bind(target),
  } as unknown as VisualViewport;

  Object.defineProperty(window, "innerHeight", {
    configurable: true,
    value: innerHeight,
  });
  Object.defineProperty(window, "visualViewport", {
    configurable: true,
    value: viewport,
  });

  return {
    viewport,
    restore: () => {
      Object.defineProperty(window, "innerHeight", {
        configurable: true,
        value: originalInnerHeight,
      });
      if (originalVisualViewport) {
        Object.defineProperty(window, "visualViewport", {
          configurable: true,
          value: originalVisualViewport,
        });
      } else {
        Reflect.deleteProperty(window, "visualViewport");
      }
    },
  };
}

function stubResizeObserver() {
  const original = globalThis.ResizeObserver;
  const observers: ResizeObserverInstance[] = [];
  class MockResizeObserver {
    elements: Element[] = [];
    callback: ResizeObserverCallback;
    disconnect = vi.fn();

    constructor(callback: ResizeObserverCallback) {
      this.callback = callback;
      observers.push(this);
    }

    observe(element: Element) {
      this.elements.push(element);
    }
  }
  vi.stubGlobal("ResizeObserver", MockResizeObserver);
  return {
    observers,
    restore: () => vi.stubGlobal("ResizeObserver", original),
  };
}

function stubElementsFromPoint(resolve: () => Element[]) {
  const descriptor = Object.getOwnPropertyDescriptor(document, "elementsFromPoint");
  const mock = vi.fn(resolve);
  Object.defineProperty(document, "elementsFromPoint", {
    configurable: true,
    value: mock,
  });
  return {
    mock,
    restore: () => {
      if (descriptor) {
        Object.defineProperty(document, "elementsFromPoint", descriptor);
      } else {
        Reflect.deleteProperty(document, "elementsFromPoint");
      }
    },
  };
}

function makeLongMessages(count: number): UIMessage[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `m${index}`,
    role: "user" as const,
    content: `message ${index}`,
    createdAt: index,
  }));
}

function makePromptExchangeMessages(count: number): UIMessage[] {
  return Array.from({ length: count }, (_, index) => ([
    {
      id: `m${index}`,
      role: "user" as const,
      content: `message ${index}`,
      createdAt: index * 2,
    },
    {
      id: `a${index}`,
      role: "assistant" as const,
      content: `answer ${index}`,
      createdAt: index * 2 + 1,
    },
  ])).flat();
}

function getScroller(container: HTMLElement): HTMLElement {
  const scroller = container.querySelector<HTMLElement>(".thread-viewport-scrollbar");
  if (!scroller) throw new Error("thread scrollport not found");
  return scroller;
}

async function renderPromptRailViewport({
  scrollTo,
  messages: promptMessages = makePromptExchangeMessages(5),
}: {
  scrollTo?: (options?: ScrollToOptions) => void;
  messages?: UIMessage[];
} = {}) {
  const { container } = render(
    <ThreadViewport
      messages={promptMessages}
      isStreaming={false}
      composer={<div />}
    />,
  );

  const scroller = getScroller(container);
  Object.defineProperties(scroller, {
    scrollHeight: { configurable: true, value: 1800 },
    clientHeight: { configurable: true, value: 600 },
    scrollTop: { configurable: true, writable: true, value: 0 },
    ...(scrollTo ? { scrollTo: { configurable: true, value: scrollTo } } : {}),
  });

  const promptEls = Array.from(
    container.querySelectorAll<HTMLElement>("[data-user-prompt-id]"),
  );
  promptEls.forEach((el, index) => {
    Object.defineProperty(el, "offsetTop", {
      configurable: true,
      value: index * 360,
    });
  });

  await act(async () => {
    window.dispatchEvent(new Event("resize"));
    await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
  });

  return { promptEls, scroller };
}

function ViewportWithPromptNavigator({ messages }: { messages: UIMessage[] }) {
  const viewportRef = useRef<ThreadViewportHandle | null>(null);
  return (
    <div>
      <PromptNavigator
        messages={messages}
        onJumpToPrompt={(promptId) => viewportRef.current?.jumpToUserPrompt(promptId)}
      />
      <ThreadViewport
        ref={viewportRef}
        messages={messages}
        isStreaming={false}
        composer={<div />}
      />
    </div>
  );
}

describe("ThreadViewport", () => {
  it("keeps reasoning disclosure anchored for pointer and keyboard toggles", () => {
    const takeUserControl = vi.spyOn(
      ThreadMotionCoordinator.prototype,
      "takeUserControl",
    );
    render(
      <ThreadViewport
        messages={[{
          id: "reasoning-1",
          role: "assistant",
          content: "",
          reasoning: "A completed thought",
          createdAt: 1,
        }]}
        isStreaming={false}
        composer={<div>composer</div>}
      />,
    );

    const disclosure = screen.getByRole("button", { name: "Worked" });
    fireEvent.pointerDown(disclosure, { button: 0 });
    expect(takeUserControl).toHaveBeenCalledTimes(1);

    takeUserControl.mockClear();
    fireEvent.keyDown(disclosure, { key: "Enter" });
    expect(takeUserControl).toHaveBeenCalledTimes(1);

    takeUserControl.mockClear();
    fireEvent.keyDown(disclosure, { key: " " });
    expect(takeUserControl).toHaveBeenCalledTimes(1);
  });

  it("top-aligns short threads in the message rendering area", () => {
    render(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div>composer</div>}
      />,
    );

    const messageRegion = screen.getByTestId("thread-message-region");
    expect(messageRegion).toHaveClass("justify-start");
    expect(messageRegion).not.toHaveClass("justify-end");
    expect(messageRegion).toHaveClass("thread-message-viewport");
    expect(messageRegion).toHaveClass("pt-3");
    expect(messageRegion).toHaveClass("pb-0");
    expect(messageRegion.className).not.toContain("5rem");
  });

  it("allows long messages to shrink within the shared mobile grid column", () => {
    render(
      <ThreadViewport
        messages={[
          ...messages,
          {
            id: "a-long-link",
            role: "assistant",
            content:
              "https://github.com/HKUDS/nanobot/discussions/17788077"
              + "/a-very-long-unbroken-segment-that-must-not-widen-the-thread",
            createdAt: Date.now(),
          },
        ]}
        isStreaming={false}
        composer={<div>composer</div>}
      />,
    );

    expect(screen.getByTestId("thread-message-region")).toHaveClass("min-w-0");
  });

  it("top-aligns a short active turn while the agent is responding", () => {
    render(
      <ThreadViewport
        messages={messages}
        isStreaming
        composer={<div>composer</div>}
      />,
    );

    const messageRegion = screen.getByTestId("thread-message-region");
    expect(messageRegion).toHaveClass("justify-start");
    expect(messageRegion).not.toHaveClass("justify-end");
    expect(screen.getByTestId("thread-composer-dock")).not.toHaveClass("mt-auto");
  });

  it("does not render a working bar above the composer", () => {
    render(
      <ThreadViewport
        messages={messages}
        isStreaming
        runStartedAt={Date.now() / 1000 - 12}
        composer={<div>composer</div>}
      />,
    );

    // Live progress belongs to the in-chat activity fold only.
    expect(screen.queryByTestId("thread-working-bar")).toBeNull();
    expect(screen.queryByTestId("thread-working-bar-slot")).toBeNull();
  });

  it("counts the live in-chat duration from the server run start, not the send time", () => {
    vi.useFakeTimers();
    try {
      const sentAtMs = 1_800_000_000_000;
      const serverRunStart = (sentAtMs + 48_000) / 1000;
      vi.setSystemTime(sentAtMs + 60_000);

      render(
        <ThreadViewport
          messages={[
            {
              id: "u-sent",
              role: "user",
              content: "do the task",
              createdAt: sentAtMs,
              turnId: "t-timer",
              turnPhase: "user",
              promptId: "p-timer",
            },
            {
              id: "trace-timer",
              role: "tool",
              kind: "trace",
              content: "search()",
              traces: ["search()"],
              createdAt: sentAtMs + 2_000,
              turnId: "t-timer",
              turnPhase: "activity",
            },
          ]}
          isStreaming
          activeTurnId="t-timer"
          runStartedAt={serverRunStart}
          composer={<div />}
        />,
      );

      // 12s of model work elapsed; the user sent 60s ago.
      expect(screen.getByText("Working · 12s")).toBeInTheDocument();
      expect(screen.queryByText("Working · 60s")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("names the running MCP step and its live progress in the working bar", () => {
    const streaming: UIMessage[] = [
      ...messages,
      {
        id: "tool-row",
        role: "tool",
        kind: "trace",
        content: 'mcp_comfy_generate_video({"workflow":"wan"})',
        traces: ['mcp_comfy_generate_video({"workflow":"wan"})'],
        createdAt: Date.now(),
        toolEvents: [{
          phase: "start",
          call_id: "call-1",
          name: "mcp_comfy_generate_video",
          arguments: { workflow: "wan" },
        }, {
          phase: "progress",
          call_id: "call-1",
          name: "mcp_comfy_generate_video",
          arguments: { workflow: "wan" },
          progress: 62,
          total: 100,
          message: "ComfyUI is generating prompt...",
        }],
      },
    ];
    // MCP progress now lives in the in-chat activity fold, not a composer bar.
    const { container } = render(
      <ThreadViewport
        messages={streaming}
        isStreaming
        runStartedAt={Date.now() / 1000}
        mcpPresets={[{ name: "comfy", display_name: "ComfyUI" }]}
        composer={<div>composer</div>}
      />,
    );

    expect(container).toHaveTextContent("ComfyUI is generating prompt...");
    expect(container).toHaveTextContent("62%");
  });

  it("keeps the docked composer outside the message scrollport", () => {
    const { container } = render(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div>composer</div>}
      />,
    );

    const scroller = getScroller(container);
    const messageRegion = screen.getByTestId("thread-message-region");
    const composerDock = screen.getByTestId("thread-composer-dock");
    expect(scroller).toBe(messageRegion);
    expect(scroller).not.toContainElement(composerDock);
    expect(scroller.parentElement).toContainElement(composerDock);
    expect(composerDock).toHaveClass("relative");
    expect(composerDock).toHaveClass("thread-composer-dock");
    expect(composerDock).not.toHaveClass("sticky");
    expect(scroller.querySelector(".thread-message-end-gap")).toBeInTheDocument();
    expect(scroller.lastElementChild).toHaveClass("h-px", "shrink-0");
  });

  it("collapses mobile composer controls while scrolling up and restores them before the tail", () => {
    const matchMedia = window.matchMedia.bind(window);
    vi.spyOn(window, "matchMedia").mockImplementation((query) => {
      const media = matchMedia(query);
      if (query === "(max-width: 639px)") {
        Object.defineProperty(media, "matches", { value: true });
      }
      return media;
    });
    const { container } = render(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<ThreadComposer
          onSend={vi.fn()}
          hasConfiguredModel
          contextUsage={{ contextTokens: 1_000, contextWindowTokens: 10_000 }}
        />}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2_400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 1_800 },
    });
    const row = () => container.querySelector(".thread-composer-context-row");
    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "Keep my draft" } });
    expect(row()).toBeInTheDocument();

    fireEvent.touchStart(scroller, { touches: [{ clientY: 250 }] });
    act(() => {
      fireEvent.touchMove(scroller, { touches: [{ clientY: 400 }] });
      scroller.scrollTop = 1_350;
      scroller.dispatchEvent(new Event("scroll"));
    });
    expect(row()).not.toBeInTheDocument();

    // A layout/anchor adjustment is not a downward scroll gesture.
    act(() => {
      scroller.scrollTop = 1_400;
      scroller.dispatchEvent(new Event("scroll"));
    });
    expect(row()).not.toBeInTheDocument();

    fireEvent.touchMove(scroller, { touches: [{ clientY: 350 }] });
    fireEvent.touchEnd(scroller);
    expect(row()).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toBe(input);
    expect(input).toHaveValue("Keep my draft");
    expect(screen.getByRole("button", { name: "Scroll to bottom" })).toBeInTheDocument();
  });

  it("pins a waiting prompt to the exact lower scroll boundary", async () => {
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const threaded: UIMessage[] = [
      { id: "u1", role: "user", content: "old question", turnId: "turn-1", createdAt: 1 },
      { id: "a1", role: "assistant", content: "old answer", turnId: "turn-1", createdAt: 2 },
      { id: "u2", role: "user", content: "new question", turnId: "turn-2", createdAt: 3 },
    ];
    const scrollTo = vi.fn();
    const { container, rerender } = render(
      <ThreadViewport
        messages={threaded}
        isStreaming
        composer={<div>composer</div>}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1200 },
      clientHeight: { configurable: true, value: 500 },
      scrollTop: { configurable: true, writable: true, value: 700 },
      scrollTo: { configurable: true, value: scrollTo },
    });
    const prompt = container.querySelector<HTMLElement>('[data-user-prompt-id="u2"]');
    expect(prompt).not.toBeNull();
    Object.defineProperty(prompt, "offsetTop", {
      configurable: true,
      value: 420,
    });
    scrollTo.mockClear();
    jumpTo.mockClear();

    await act(async () => {
      rerender(
        <ThreadViewport
          messages={threaded}
          isStreaming
          composer={<div>composer</div>}
          activeTurnId="turn-2"
          activeTurnStartedHere
        />,
      );
    });
    await flushAnimationFrame();

    expect(jumpTo).toHaveBeenCalledWith(700);
    expect(screen.getByTestId("thread-message-region")).toHaveClass("justify-start");
  });

  it("keeps automatic ownership when layout emits a scroll before its measurement frame", async () => {
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const threaded: UIMessage[] = [
      { id: "u1", role: "user", content: "old question", turnId: "turn-1", createdAt: 1 },
      { id: "a1", role: "assistant", content: "old answer", turnId: "turn-1", createdAt: 2 },
      { id: "u2", role: "user", content: "new question", turnId: "turn-2", createdAt: 3 },
    ];
    const { container, rerender } = render(
      <ThreadViewport
        messages={threaded}
        isStreaming
        composer={<div>composer</div>}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1_200 },
      clientHeight: { configurable: true, value: 500 },
      scrollTop: { configurable: true, writable: true, value: 700 },
    });
    const prompt = container.querySelector<HTMLElement>('[data-user-prompt-id="u2"]');
    expect(prompt).not.toBeNull();
    Object.defineProperty(prompt, "offsetTop", {
      configurable: true,
      value: 420,
    });
    jumpTo.mockClear();

    await act(async () => {
      rerender(
        <ThreadViewport
          messages={threaded}
          isStreaming
          composer={<div>composer</div>}
          activeTurnId="turn-2"
          activeTurnStartedHere
        />,
      );
    });

    Object.defineProperty(scroller, "scrollHeight", {
      configurable: true,
      value: 1_280,
    });
    act(() => {
      scroller.dispatchEvent(new Event("scroll"));
    });

    expect(screen.queryByRole("button", { name: "Scroll to bottom" }))
      .not.toBeInTheDocument();
    await flushAnimationFrame();
    expect(jumpTo).toHaveBeenCalledWith(780);
    expect(scroller.scrollTop).toBe(780);
  });

  it("lets the first prompt supersede a pending empty-conversation camera command", async () => {
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const scrollTo = vi.fn();
    const firstPrompt: UIMessage = {
      id: "u-first",
      role: "user",
      content: "first question",
      turnId: "turn-first",
      createdAt: 1,
    };
    const { container, rerender } = render(
      <ThreadViewport
        messages={emptyMessages}
        isStreaming={false}
        composer={<div>composer</div>}
        conversationKey={null}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1200 },
      clientHeight: { configurable: true, value: 500 },
      scrollTop: { configurable: true, writable: true, value: 700 },
      scrollTo: { configurable: true, value: scrollTo },
    });
    scrollTo.mockClear();
    jumpTo.mockClear();

    await act(async () => {
      rerender(
        <ThreadViewport
          messages={[firstPrompt]}
          isStreaming
          composer={<div>composer</div>}
          conversationKey="chat-a"
          conversationReady={false}
          activeTurnId="turn-first"
          activeTurnStartedHere
        />,
      );
    });
    const threadScroller = getScroller(container);
    Object.defineProperties(threadScroller, {
      scrollHeight: { configurable: true, value: 1200 },
      clientHeight: { configurable: true, value: 500 },
      scrollTop: { configurable: true, writable: true, value: 0 },
      scrollTo: { configurable: true, value: scrollTo },
    });
    jumpTo.mockClear();
    await act(async () => {
      rerender(
        <ThreadViewport
          messages={[firstPrompt]}
          isStreaming
          composer={<div>composer</div>}
          conversationKey="chat-a"
          conversationReady
          activeTurnId="turn-first"
          activeTurnStartedHere
        />,
      );
    });
    await flushAnimationFrame();

    expect(jumpTo).toHaveBeenCalledTimes(1);
    expect(jumpTo).toHaveBeenCalledWith(700);
  });

  it("anchors the latest user prompt even when assistant output is already present", async () => {
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const threaded: UIMessage[] = [
      { id: "u1", role: "user", content: "old question", turnId: "turn-1", createdAt: 1 },
      { id: "a1", role: "assistant", content: "old answer", turnId: "turn-1", createdAt: 2 },
      { id: "u2", role: "user", content: "new question", turnId: "turn-2", createdAt: 3 },
      { id: "a2", role: "assistant", content: "fast answer", turnId: "turn-2", createdAt: 4 },
    ];
    const viewportRef = createRef<ThreadViewportHandle>();
    const scrollTo = vi.fn();
    const { container, rerender } = render(
      <ThreadViewport
        ref={viewportRef}
        messages={threaded}
        isStreaming
        composer={<div>composer</div>}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1904 },
      clientHeight: { configurable: true, value: 500 },
      scrollTop: { configurable: true, writable: true, value: 1404 },
      scrollTo: { configurable: true, value: scrollTo },
    });
    const prompt = container.querySelector<HTMLElement>('[data-user-prompt-id="u2"]');
    expect(prompt).not.toBeNull();
    Object.defineProperty(prompt, "offsetTop", {
      configurable: true,
      value: 1420,
    });
    viewportRef.current?.cancelAutoScroll();
    scrollTo.mockClear();
    jumpTo.mockClear();

    await act(async () => {
      rerender(
        <ThreadViewport
          ref={viewportRef}
          messages={threaded}
          isStreaming
          composer={<div>composer</div>}
          activeTurnId="turn-2"
          activeTurnStartedHere
        />,
      );
    });
    await flushAnimationFrame();

    expect(jumpTo).toHaveBeenCalledWith(1404);
  });

  it("drives the camera from a message commit when canonical replay replaces the prompt DOM id", async () => {
    const resizeObserver = stubResizeObserver();
    const followTo = vi.spyOn(ThreadCameraController.prototype, "followTo")
      .mockReturnValue("started");

    try {
      const optimistic: UIMessage[] = [
        {
          id: "optimistic-user",
          role: "user",
          content: "new question",
          turnId: "turn-stable",
          createdAt: 1,
        },
        {
          id: "optimistic-answer",
          role: "assistant",
          content: "answer",
          turnId: "turn-stable",
          isStreaming: true,
          createdAt: 2,
        },
      ];
      const { container, rerender } = render(
        <ThreadViewport
          messages={optimistic}
          isStreaming
          composer={<div>composer</div>}
        />,
      );
      const scroller = getScroller(container);
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 2_000 },
        clientHeight: { configurable: true, value: 500 },
        scrollTop: { configurable: true, writable: true, value: 1_500 },
        scrollTo: {
          configurable: true,
          value: ({ top }: ScrollToOptions) => {
            if (typeof top === "number") scroller.scrollTop = top;
          },
        },
      });
      const optimisticPrompt = container.querySelector<HTMLElement>(
        '[data-user-prompt-id="optimistic-user"]',
      );
      expect(optimisticPrompt).not.toBeNull();
      Object.defineProperty(optimisticPrompt, "offsetTop", {
        configurable: true,
        value: 1_420,
      });

      await act(async () => {
        rerender(
          <ThreadViewport
            messages={optimistic}
            isStreaming
            composer={<div>composer</div>}
            activeTurnId="turn-stable"
          />,
        );
      });
      followTo.mockClear();

      await act(async () => {
        rerender(
          <ThreadViewport
            messages={[
              {
                ...optimistic[0],
                id: "canonical-user",
              },
              {
                ...optimistic[1],
                id: "canonical-answer",
                content: "canonical answer",
              },
            ]}
            isStreaming
            composer={<div>composer</div>}
            activeTurnId="turn-stable"
          />,
        );
      });
      await flushAnimationFrame();
      // Streaming React commits only invalidate geometry. The next display
      // frame measures once and issues the follow command.
      expect(followTo).toHaveBeenCalledWith(1_500);

      followTo.mockClear();
      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 2_120,
      });
      const messageContent = screen.getByTestId("thread-message-region").firstElementChild;
      const contentObserver = resizeObserver.observers.find(
        (observer) => observer.elements.includes(messageContent!),
      );
      expect(contentObserver).toBeDefined();

      act(() => {
        contentObserver!.callback([], contentObserver as unknown as ResizeObserver);
      });
      await flushAnimationFrame();

      // ResizeObserver remains the asynchronous reflow fallback.
      expect(followTo).toHaveBeenCalledWith(1_620);
    } finally {
      followTo.mockRestore();
      resizeObserver.restore();
    }
  });

  it("settles observed streamed layout growth before paint", async () => {
    const resizeObserver = stubResizeObserver();
    const followTo = vi.spyOn(ThreadCameraController.prototype, "followTo")
      .mockReturnValue("started");

    try {
      const threaded: UIMessage[] = [
        { id: "u1", role: "user", content: "old question", turnId: "turn-1", createdAt: 1 },
        { id: "a1", role: "assistant", content: "old answer", turnId: "turn-1", createdAt: 2 },
        { id: "u2", role: "user", content: "new question", turnId: "turn-2", createdAt: 3 },
      ];
      const answer: UIMessage = {
        id: "a2",
        role: "assistant",
        content: "streaming answer",
        turnId: "turn-2",
        isStreaming: true,
        createdAt: 4,
      };
      const { container, rerender } = render(
        <ThreadViewport
          messages={threaded}
          isStreaming
          composer={<div>composer</div>}
        />,
      );

      const scroller = getScroller(container);
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 1904 },
        clientHeight: { configurable: true, value: 500 },
        scrollTop: { configurable: true, writable: true, value: 1404 },
      });
      const prompt = container.querySelector<HTMLElement>('[data-user-prompt-id="u2"]');
      expect(prompt).not.toBeNull();
      Object.defineProperty(prompt, "offsetTop", {
        configurable: true,
        value: 1420,
      });

      await act(async () => {
        rerender(
          <ThreadViewport
            messages={threaded}
            isStreaming
            composer={<div>composer</div>}
            activeTurnId="turn-2"
            activeTurnStartedHere
          />,
        );
      });
      await flushAnimationFrame();
      followTo.mockClear();

      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 1932,
      });
      await act(async () => {
        rerender(
          <ThreadViewport
            messages={[...threaded, answer]}
            isStreaming
            composer={<div>composer</div>}
            activeTurnId="turn-2"
          />,
        );
      });
      await flushAnimationFrame();
      expect(followTo).toHaveBeenCalledTimes(1);
      expect(followTo).toHaveBeenLastCalledWith(1432);
      followTo.mockClear();

      const messageRegion = screen.getByTestId("thread-message-region");
      const messageContent = messageRegion.firstElementChild;
      expect(messageContent).not.toBeNull();
      const contentObserver = resizeObserver.observers.find(
        (observer) => observer.elements.includes(messageContent!),
      );
      expect(contentObserver).toBeDefined();

      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 1948,
      });
      act(() => {
        contentObserver!.callback([], contentObserver as unknown as ResizeObserver);
      });
      expect(followTo).toHaveBeenCalledTimes(1);
      expect(followTo).toHaveBeenLastCalledWith(1448);
      followTo.mockClear();

      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 2548,
      });
      await act(async () => {
        rerender(
          <ThreadViewport
            messages={[
              ...threaded,
              {
                ...answer,
                content: "final answer received as one large batch",
                isStreaming: false,
              },
            ]}
            isStreaming={false}
            composer={<div>composer</div>}
          />,
        );
      });
      await flushAnimationFrame();
      expect(followTo).toHaveBeenLastCalledWith(2048);
      followTo.mockClear();

      scroller.scrollTop = 100;
      act(() => {
        dispatchUserScroll(scroller);
      });
      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 2560,
      });
      act(() => {
        contentObserver!.callback([], contentObserver as unknown as ResizeObserver);
      });
      await flushAnimationFrame();
      expect(followTo).not.toHaveBeenCalled();
      expect(scroller.scrollTop).toBe(100);
    } finally {
      followTo.mockRestore();
      resizeObserver.restore();
    }
  });

  it("keeps shallow wheel and touch scrolling user-owned until intent reverses", async () => {
    const followTo = vi.spyOn(ThreadCameraController.prototype, "followTo");
    const threaded: UIMessage[] = [
      { id: "u1", role: "user", content: "old question", turnId: "turn-1", createdAt: 1 },
      { id: "a1", role: "assistant", content: "old answer", turnId: "turn-1", createdAt: 2 },
      { id: "u2", role: "user", content: "new question", turnId: "turn-2", createdAt: 3 },
    ];
    const answer: UIMessage = {
      id: "a2",
      role: "assistant",
      content: "streaming answer",
      turnId: "turn-2",
      isStreaming: true,
      createdAt: 4,
    };
    const { container, rerender } = render(
      <ThreadViewport
        messages={threaded}
        isStreaming
        composer={<div>composer</div>}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1_904 },
      clientHeight: { configurable: true, value: 500 },
      scrollTop: { configurable: true, writable: true, value: 1_404 },
    });
    const prompt = container.querySelector<HTMLElement>('[data-user-prompt-id="u2"]');
    expect(prompt).not.toBeNull();
    Object.defineProperty(prompt, "offsetTop", {
      configurable: true,
      value: 1_420,
    });

    rerender(
      <ThreadViewport
        messages={[...threaded, answer]}
        isStreaming
        composer={<div>composer</div>}
        activeTurnId="turn-2"
        activeTurnStartedHere
      />,
    );
    await flushAnimationFrame();
    followTo.mockClear();

    act(() => {
      fireEvent.wheel(scroller, { deltaY: -24 });
      scroller.scrollTop = 1_380;
      scroller.dispatchEvent(new Event("scroll"));
    });
    await flushAnimationFrame();

    expect(followTo).not.toHaveBeenCalled();
    expect(scroller.scrollTop).toBe(1_380);
    expect(screen.getByRole("button", { name: "Scroll to bottom" })).toBeInTheDocument();

    act(() => {
      scroller.scrollTop = 1_404;
      scroller.dispatchEvent(new Event("scroll"));
      fireEvent.wheel(scroller, { deltaY: 24 });
    });
    await flushAnimationFrame();

    expect(followTo).toHaveBeenCalledWith(1_404);
    expect(scroller.scrollTop).toBe(1_404);
    expect(screen.queryByRole("button", { name: "Scroll to bottom" }))
      .not.toBeInTheDocument();

    followTo.mockClear();
    act(() => {
      fireEvent.touchStart(scroller, { touches: [{ clientY: 300 }] });
      fireEvent.touchMove(scroller, { touches: [{ clientY: 324 }] });
      scroller.scrollTop = 1_380;
      scroller.dispatchEvent(new Event("scroll"));
    });
    await flushAnimationFrame();

    expect(followTo).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Scroll to bottom" })).toBeInTheDocument();

    act(() => {
      fireEvent.touchMove(scroller, { touches: [{ clientY: 300 }] });
      scroller.scrollTop = 1_404;
      scroller.dispatchEvent(new Event("scroll"));
      fireEvent.touchEnd(scroller);
    });
    await flushAnimationFrame();

    expect(followTo).toHaveBeenCalledWith(1_404);
    expect(screen.queryByRole("button", { name: "Scroll to bottom" }))
      .not.toBeInTheDocument();
  });

  it("keeps the scroll-to-bottom button above a growing composer", async () => {
    const resizeObserver = stubResizeObserver();

    try {
      const { container } = render(
        <ThreadViewport
          messages={messages}
          isStreaming={false}
          composer={<div>composer</div>}
        />,
      );
      const scroller = getScroller(container);
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 2400 },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, value: 0 },
      });

      act(() => {
        dispatchUserScroll(scroller);
      });

      const button = screen.getByRole("button", { name: "Scroll to bottom" });
      const buttonPositioner = button.parentElement as HTMLElement;
      expect(button).not.toHaveClass("-translate-x-1/2");
      expect(buttonPositioner).toHaveStyle({ bottom: "192px" });

      const composerDock = screen.getByTestId("thread-composer-dock");
      composerDock.getBoundingClientRect = () =>
        ({
          height: 240,
          width: 800,
          top: 0,
          right: 800,
          bottom: 240,
          left: 0,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        }) as DOMRect;

      const composerObserver = resizeObserver.observers.find(
        (observer) => observer.elements.includes(composerDock),
      );
      expect(composerObserver).toBeDefined();

      act(() => {
        composerObserver!.callback([], composerObserver as unknown as ResizeObserver);
      });
      await flushAnimationFrame();

      expect(buttonPositioner).toHaveStyle({ bottom: "256px" });
    } finally {
      resizeObserver.restore();
    }
  });

  it("gives smooth scroll-to-bottom navigation ownership of the latest target", () => {
    const navigateLatestTo = vi.spyOn(
      ThreadMotionCoordinator.prototype,
      "navigateLatestTo",
    ).mockReturnValue("started");
    const { container } = render(
      <ThreadViewport
        messages={messages}
        isStreaming
        composer={<div>composer</div>}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2_400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });

    act(() => {
      dispatchUserScroll(scroller);
    });
    fireEvent.click(screen.getByRole("button", { name: "Scroll to bottom" }));

    expect(navigateLatestTo).toHaveBeenCalledWith(1_800);
  });

  it("pins the waiting boundary across composer and grid-track growth", async () => {
    const resizeObserver = stubResizeObserver();
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");

    try {
      const threaded: UIMessage[] = [
        { id: "u1", role: "user", content: "old question", turnId: "turn-1", createdAt: 1 },
        { id: "a1", role: "assistant", content: "old answer", turnId: "turn-1", createdAt: 2 },
        { id: "u2", role: "user", content: "new question", turnId: "turn-2", createdAt: 3 },
      ];
      const scrollTo = vi.fn();
      const { container, rerender } = render(
        <ThreadViewport
          messages={threaded}
          isStreaming
          composer={<div>composer</div>}
        />,
      );

      const scroller = getScroller(container);
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 1200 },
        clientHeight: { configurable: true, value: 500 },
        scrollTop: { configurable: true, writable: true, value: 700 },
        scrollTo: { configurable: true, value: scrollTo },
      });
      const prompt = container.querySelector<HTMLElement>('[data-user-prompt-id="u2"]');
      expect(prompt).not.toBeNull();
      Object.defineProperty(prompt, "offsetTop", {
        configurable: true,
        value: 420,
      });

      await act(async () => {
        rerender(
          <ThreadViewport
            messages={threaded}
            isStreaming
            composer={<div>composer</div>}
            activeTurnId="turn-2"
          />,
        );
      });
      await flushAnimationFrame();

      scrollTo.mockClear();
      jumpTo.mockClear();
      const composerDock = screen.getByTestId("thread-composer-dock");
      composerDock.getBoundingClientRect = () =>
        ({
          height: 240,
          width: 800,
          top: 0,
          right: 800,
          bottom: 240,
          left: 0,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        }) as DOMRect;
      const composerObserver = resizeObserver.observers.find(
        (observer) => observer.elements.includes(composerDock),
      );
      expect(composerObserver).toBeDefined();
      expect(composerObserver!.elements).toContain(
        screen.getByTestId("thread-message-region"),
      );

      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 1228,
      });
      act(() => {
        composerObserver!.callback([], composerObserver as unknown as ResizeObserver);
      });
      await flushAnimationFrame();

      expect(jumpTo).toHaveBeenCalledWith(728);

      jumpTo.mockClear();
      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 1268,
      });
      act(() => {
        composerObserver!.callback([], composerObserver as unknown as ResizeObserver);
      });
      await flushAnimationFrame();

      expect(jumpTo).toHaveBeenCalledWith(768);
    } finally {
      resizeObserver.restore();
    }
  });

  it("releases completion follow when single-line input starts before completion", async () => {
    const resizeObserver = stubResizeObserver();
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const followTo = vi.spyOn(ThreadCameraController.prototype, "followTo");

    try {
      const threaded: UIMessage[] = [
        { id: "u1", role: "user", content: "question", turnId: "turn-1", createdAt: 1 },
        { id: "a1", role: "assistant", content: "answer", turnId: "turn-1", createdAt: 2 },
      ];
      const viewport = (isStreaming: boolean, activeTurnId?: string) => (
        <ThreadViewport
          messages={threaded}
          isStreaming={isStreaming}
          composer={<textarea aria-label="Message input" />}
          activeTurnId={activeTurnId}
        />
      );
      const { container, rerender } = render(viewport(true));
      const scroller = getScroller(container);
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 1_200 },
        clientHeight: { configurable: true, value: 500 },
        scrollTop: { configurable: true, writable: true, value: 700 },
      });

      rerender(viewport(true, "turn-1"));
      await flushAnimationFrame();
      jumpTo.mockClear();
      followTo.mockClear();

      const composerDock = screen.getByTestId("thread-composer-dock");
      const composerObserver = resizeObserver.observers.find(
        (observer) => observer.elements.includes(composerDock),
      );
      expect(composerObserver).toBeDefined();
      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 1_240,
      });

      act(() => {
        composerObserver!.callback(
          [{ target: composerDock }] as ResizeObserverEntry[],
          composerObserver as unknown as ResizeObserver,
        );
      });
      await flushAnimationFrame();

      expect(followTo).toHaveBeenCalled();

      followTo.mockClear();
      fireEvent.input(screen.getByLabelText("Message input"), {
        target: { value: "x" },
      });
      const scrollTopAfterInput = scroller.scrollTop;
      rerender(viewport(false));
      await flushAnimationFrame();
      expect(followTo).not.toHaveBeenCalled();
      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 1_280,
      });
      act(() => {
        composerObserver!.callback(
          [{ target: composerDock }] as ResizeObserverEntry[],
          composerObserver as unknown as ResizeObserver,
        );
      });
      await flushAnimationFrame();

      expect(jumpTo).not.toHaveBeenCalled();
      expect(followTo).not.toHaveBeenCalled();
      expect(scroller.scrollTop).toBe(scrollTopAfterInput);
    } finally {
      resizeObserver.restore();
    }
  });

  it("restores thread scroll after the textarea autosize measurement collapses it", () => {
    let scroller: HTMLElement | null = null;
    const { container } = render(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={(
          <textarea
            aria-label="Message input"
            onInput={() => {
              if (scroller) scroller.scrollTop = 692;
            }}
          />
        )}
      />,
    );
    scroller = getScroller(container);
    Object.defineProperty(scroller, "scrollTop", {
      configurable: true,
      writable: true,
      value: 700,
    });

    fireEvent.input(screen.getByLabelText("Message input"), {
      target: { value: "中文" },
    });

    expect(scroller.scrollTop).toBe(700);
  });

  it.each([
    { name: "visual viewport", layoutHeight: 800, inset: "320px" },
    { name: "layout viewport", layoutHeight: 480, inset: "" },
  ])("keeps the chat position when the mobile keyboard shrinks the $name", async ({
    layoutHeight, inset,
  }) => {
    const visualViewport = stubVisualViewport({ innerHeight: 800, height: 800 });
    try {
      const { container } = render(
        <ThreadViewport
          messages={messages}
          isStreaming={false}
          composer={<textarea aria-label="Message input" />}
        />,
      );
      const scroller = getScroller(container);
      const viewportFrame = container.querySelector<HTMLElement>(".thread-viewport-frame");
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 2400 },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, writable: true, value: 700 },
      });

      act(() => {
        dispatchUserScroll(scroller);
      });
      const previousScrollTop = scroller.scrollTop;

      const input = screen.getByLabelText("Message input");
      act(() => {
        input.focus();
      });
      expect(scroller.scrollTop).toBe(previousScrollTop);

      act(() => {
        Object.defineProperty(window, "innerHeight", { configurable: true, value: layoutHeight });
        Object.defineProperty(visualViewport.viewport, "height", { value: 480 });
        Object.defineProperty(scroller, "clientHeight", { value: 280 });
        visualViewport.viewport.dispatchEvent(new Event("resize"));
        if (layoutHeight !== 800) window.dispatchEvent(new Event("resize"));
      });
      await flushAnimationFrame();

      expect(scroller.scrollTop).toBe(previousScrollTop);
      expect(input).toHaveFocus();
      expect(viewportFrame?.style.bottom).toBe(inset);
    } finally {
      visualViewport.restore();
    }
  });

  it("keeps the welcome composer above a mobile soft keyboard", async () => {
    const visualViewport = stubVisualViewport({ innerHeight: 800, height: 480 });
    try {
      const { container } = render(
        <ThreadViewport
          messages={emptyMessages}
          isStreaming={false}
          composer={<textarea aria-label="Message input" />}
          emptyState={<div>welcome</div>}
          showScrollToBottomButton={false}
        />,
      );
      const scroller = container.querySelector(".thread-viewport-scrollbar") as HTMLElement;
      const input = screen.getByLabelText("Message input");

      act(() => {
        input.focus();
        fireEvent.focusIn(input);
      });

      await waitFor(() => expect(scroller).toHaveStyle({ bottom: "320px" }));
    } finally {
      visualViewport.restore();
    }
  });

  it("keeps one composer instance while the landing layout docks into a thread", () => {
    const { rerender } = render(
      <ThreadViewport
        messages={emptyMessages}
        isStreaming={false}
        composer={(
          <input
            data-testid="persistent-composer"
            defaultValue="draft"
          />
        )}
        emptyState={<div>welcome</div>}
      />,
    );

    const initialComposer = screen.getByTestId("persistent-composer");
    fireEvent.change(initialComposer, { target: { value: "kept draft" } });

    rerender(
      <ThreadViewport
        messages={messages}
        isStreaming
        composer={(
          <input
            data-testid="persistent-composer"
            defaultValue="draft"
          />
        )}
        emptyState={<div>welcome</div>}
      />,
    );

    const dockedComposer = screen.getByTestId("persistent-composer");
    expect(dockedComposer).toBe(initialComposer);
    expect(dockedComposer).toHaveValue("kept draft");
    expect(dockedComposer.closest(".thread-layout")).toHaveAttribute(
      "data-layout",
      "thread",
    );
  });

  it("separates the mobile welcome copy and composer into responsive rows", () => {
    render(
      <ThreadViewport
        messages={emptyMessages}
        isStreaming={false}
        composer={<div>composer</div>}
        emptyState={<div>welcome</div>}
      />,
    );

    const layout = screen.getByTestId("thread-welcome-layout");
    expect(layout).toHaveClass("thread-layout", "grid");
    expect(layout).toHaveAttribute("data-layout", "hero");
    expect(screen.getByText("welcome").parentElement).toHaveClass(
      "min-h-0",
      "items-center",
      "sm:items-end",
      "sm:pb-8",
    );
    expect(screen.getByTestId("thread-composer-motion")).toContainElement(
      screen.getByText("composer"),
    );
  });

  it("allows the welcome view to scroll when a short viewport overflows", async () => {
    const { container } = render(
      <ThreadViewport
        messages={emptyMessages}
        isStreaming={false}
        composer={<div>composer</div>}
        emptyState={<div>welcome</div>}
        showScrollToBottomButton={false}
      />,
    );
    const scroller = container.querySelector(".thread-viewport-scrollbar") as HTMLElement;
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 620 },
      clientHeight: { configurable: true, value: 320 },
    });

    act(() => {
      window.dispatchEvent(new Event("resize"));
    });

    await waitFor(() => expect(scroller).toHaveClass("overflow-y-auto"));
    expect(scroller).toHaveClass(
      "[scrollbar-width:none]",
      "[&::-webkit-scrollbar]:hidden",
    );
  });

  it("keeps the chat position when the composer receives focus", async () => {
    const { container } = render(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<textarea aria-label="Message input" />}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 700 },
    });

    act(() => {
      dispatchUserScroll(scroller);
    });
    const previousScrollTop = scroller.scrollTop;

    const input = screen.getByLabelText("Message input");
    act(() => {
      input.focus();
    });
    await flushAnimationFrame();

    expect(scroller.scrollTop).toBe(previousScrollTop);
  });

  it("hides the scroll-to-bottom button when disabled for the welcome view", () => {
    const { container } = render(
      <ThreadViewport
        messages={emptyMessages}
        isStreaming={false}
        composer={<div>composer</div>}
        emptyState={<div>welcome</div>}
        showScrollToBottomButton={false}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });

    act(() => {
      dispatchUserScroll(scroller);
    });

    expect(screen.queryByRole("button", { name: "Scroll to bottom" })).not.toBeInTheDocument();
  });

  it("renders only the tail window for long history by default", () => {
    const longMessages = makeLongMessages(300);

    render(
      <ThreadViewport
        messages={longMessages}
        isStreaming={false}
        composer={<div />}
      />,
    );

    expect(screen.queryByText("message 139")).not.toBeInTheDocument();
    expect(screen.getByText("message 140")).toBeInTheDocument();
    expect(screen.getByText("message 299")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load earlier messages" })).not.toBeInTheDocument();
  });

  it("automatically expands earlier local history near the top", () => {
    const longMessages = makeLongMessages(300);

    const { container } = render(
      <ThreadViewport
        messages={longMessages}
        isStreaming={false}
        composer={<div />}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });

    act(() => {
      dispatchUserScroll(scroller);
    });

    const firstVisible =
      300 - INITIAL_HISTORY_WINDOW - HISTORY_WINDOW_INCREMENT;

    expect(
      screen.queryByText(`message ${firstVisible - 1}`),
    ).not.toBeInTheDocument();
    expect(screen.getByText(`message ${firstVisible}`)).toBeInTheDocument();
    expect(screen.getAllByText("message 299").length).toBeGreaterThan(0);
  });

  it("prefetches earlier history several viewports before the top", () => {
    const { container } = render(
      <ThreadViewport
        messages={makeLongMessages(300)}
        isStreaming={false}
        composer={<div />}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 6000 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 1501 },
    });

    act(() => {
      dispatchUserScroll(scroller);
    });
    expect(screen.queryByText("message 139")).not.toBeInTheDocument();

    scroller.scrollTop = 1400;
    act(() => {
      dispatchUserScroll(scroller);
    });
    expect(screen.getByText("message 20")).toBeInTheDocument();
    expect(screen.queryByText("message 19")).not.toBeInTheDocument();
  });

  it("requests the next older page after one lands while still near the top", async () => {
    const onLoadOlder = vi.fn();
    const props = {
      isStreaming: false,
      composer: <div />,
      hasMoreBefore: true,
      onLoadOlder,
    };
    const { container, rerender } = render(
      <ThreadViewport {...props} messages={makeLongMessages(40)} />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 6000 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 100 },
    });
    act(() => {
      dispatchUserScroll(scroller);
    });
    expect(onLoadOlder).toHaveBeenCalledTimes(1);

    // The page lands (messages grow) and the viewport is still in the
    // prefetch zone: the next page must be requested without another scroll.
    rerender(<ThreadViewport {...props} loadingOlder messages={makeLongMessages(60)} />);
    rerender(<ThreadViewport {...props} messages={makeLongMessages(80)} />);
    await waitFor(() => expect(onLoadOlder).toHaveBeenCalledTimes(2));
  });

  it("keeps the first visible history item fixed while deferred rows materialize", () => {
    const resizeObserver = stubResizeObserver();
    let hitTarget: Element | null = null;
    const hitTest = stubElementsFromPoint(() => hitTarget ? [hitTarget] : []);
    try {
      const { container } = render(
        <ThreadViewport
          messages={makeLongMessages(300)}
          isStreaming={false}
          composer={<div />}
        />,
      );

      const scroller = getScroller(container);
      let scrollHeight = 2_400;
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, get: () => scrollHeight },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, writable: true, value: 80 },
        getBoundingClientRect: {
          configurable: true,
          value: () => DOMRect.fromRect({ y: 0, width: 800, height: 600 }),
        },
      });

      const anchor = screen.getByText("message 140")
        .closest<HTMLElement>("[data-thread-display-unit]");
      expect(anchor).not.toBeNull();
      hitTarget = anchor;
      let anchorDocumentTop = 200;
      Object.defineProperty(anchor, "getBoundingClientRect", {
        configurable: true,
        value: () => DOMRect.fromRect({
          y: anchorDocumentTop - scroller.scrollTop,
          width: 800,
          height: 40,
        }),
      });

      act(() => {
        dispatchUserScroll(scroller);
      });
      expect(hitTest.mock).toHaveBeenCalled();

      const replacement = anchor.cloneNode(true) as HTMLElement;
      anchor.replaceWith(replacement);
      anchorDocumentTop += 180;
      scrollHeight += 180;
      Object.defineProperty(replacement, "getBoundingClientRect", {
        configurable: true,
        value: () => DOMRect.fromRect({
          y: anchorDocumentTop - scroller.scrollTop,
          width: 800,
          height: 40,
        }),
      });
      const content = screen.getByTestId("thread-message-region").firstElementChild;
      const observer = resizeObserver.observers.find((candidate) =>
        content ? candidate.elements.includes(content) : false,
      );
      expect(observer).toBeDefined();
      act(() => {
        observer?.callback([], observer as unknown as ResizeObserver);
      });

      expect(scroller.scrollTop).toBe(260);
      expect(replacement.getBoundingClientRect().top).toBe(120);
    } finally {
      hitTest.restore();
      resizeObserver.restore();
    }
  });

  it("does not snap scrollTop backward when ResizeObserver fires mid-scroll", () => {
    const resizeObserver = stubResizeObserver();
    let hitTarget: Element | null = null;
    const hitTest = stubElementsFromPoint(() => hitTarget ? [hitTarget] : []);
    try {
      const { container } = render(
        <ThreadViewport
          messages={makeLongMessages(300)}
          isStreaming={false}
          composer={<div />}
        />,
      );

      const scroller = getScroller(container);
      let scrollHeight = 6_000;
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, get: () => scrollHeight },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, writable: true, value: 2_000 },
        getBoundingClientRect: {
          configurable: true,
          value: () => DOMRect.fromRect({ y: 0, width: 800, height: 600 }),
        },
      });

      const anchor = screen.getByText("message 200")
        .closest<HTMLElement>("[data-thread-display-unit]");
      expect(anchor).not.toBeNull();
      hitTarget = anchor;
      let anchorDocumentTop = 2_120;
      Object.defineProperty(anchor, "getBoundingClientRect", {
        configurable: true,
        value: () => DOMRect.fromRect({
          y: anchorDocumentTop - scroller.scrollTop,
          width: 800,
          height: 40,
        }),
      });

      act(() => {
        dispatchUserScroll(scroller);
      });

      // Compositor moves scrollTop upward before the next JS scroll event fires.
      scroller.scrollTop = 1_940;
      const content = screen.getByTestId("thread-message-region").firstElementChild;
      const observer = resizeObserver.observers.find((candidate) =>
        content ? candidate.elements.includes(content) : false,
      );
      expect(observer).toBeDefined();

      // ResizeObserver fires without layout shift above anchor: must NOT snap back to 2000.
      act(() => {
        observer?.callback([], observer as unknown as ResizeObserver);
      });
      expect(scroller.scrollTop).toBe(1_940);

      // Now older content above anchor grows by +180px while user scrolled another -20px.
      scroller.scrollTop = 1_920;
      anchorDocumentTop += 180;
      scrollHeight += 180;
      act(() => {
        observer?.callback([], observer as unknown as ResizeObserver);
      });
      // Compensates the +180px shift while preserving the user's -20px scroll.
      expect(scroller.scrollTop).toBe(2_100);
    } finally {
      hitTest.restore();
      resizeObserver.restore();
    }
  });

  it("does not chain another older-page request when loadingOlder finishes without new messages", async () => {
    const onLoadOlder = vi.fn();
    const initialMessages = makeLongMessages(40);
    const props = {
      isStreaming: false,
      composer: <div />,
      hasMoreBefore: true,
      onLoadOlder,
    };
    const { container, rerender } = render(
      <ThreadViewport {...props} messages={initialMessages} />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 6000 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 100 },
    });

    act(() => {
      dispatchUserScroll(scroller);
    });
    expect(onLoadOlder).toHaveBeenCalledTimes(1);

    // loadingOlder toggles true -> false without messages growing (e.g. error or
    // intermediate commit before ThreadShell adopts historical): must not re-fire.
    rerender(<ThreadViewport {...props} loadingOlder messages={initialMessages} />);
    rerender(<ThreadViewport {...props} loadingOlder={false} messages={initialMessages} />);
    await new Promise((resolve) => window.requestAnimationFrame(() => resolve(undefined)));
    expect(onLoadOlder).toHaveBeenCalledTimes(1);
  });

  it("renders an entire turn-aligned older server page without splitting it", () => {
    const initialTail = makeLongMessages(160).map((m, i) => ({
      ...m,
      id: `tail-${i}`,
      content: `tail ${i}`,
    }));
    const onLoadOlder = vi.fn();
    const { container, rerender } = render(
      <ThreadViewport
        messages={initialTail}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        onLoadOlder={onLoadOlder}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 6000 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 100 },
    });

    act(() => {
      dispatchUserScroll(scroller);
    });
    expect(onLoadOlder).toHaveBeenCalledTimes(1);

    // Backend returns a turn-aligned page larger than HISTORY_WINDOW_INCREMENT (145 messages).
    const olderPage = makeLongMessages(145).map((m, i) => ({
      ...m,
      id: `older-${i}`,
      content: `older ${i}`,
    }));
    rerender(
      <ThreadViewport
        messages={[...olderPage, ...initialTail]}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore={false}
        onLoadOlder={onLoadOlder}
      />,
    );

    expect(screen.getByText("older 0")).toBeInTheDocument();
    expect(screen.getByText("tail 159")).toBeInTheDocument();
  });

  it("keeps messages from the same turn together when windowing history", () => {
    const messages = makeLongMessages(INITIAL_HISTORY_WINDOW + 10);
    messages[8] = { ...messages[8], turnId: "shared-turn", role: "user" };
    messages[9] = { ...messages[9], turnId: "shared-turn", role: "assistant" };
    messages[10] = { ...messages[10], turnId: "shared-turn", role: "assistant" };

    const visible = windowMessages(messages, INITIAL_HISTORY_WINDOW);
    expect(visible[0].id).toBe(messages[8].id);
  });

  it("automatically requests older transcript pages near the top", () => {
    const onLoadOlder = vi.fn();

    const { container } = render(
      <ThreadViewport
        messages={makeLongMessages(20)}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        onLoadOlder={onLoadOlder}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1800 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });

    act(() => {
      dispatchUserScroll(scroller);
    });

    expect(onLoadOlder).toHaveBeenCalledTimes(1);
  });

  it("loads older transcript pages on upward wheel even when already at scrollTop 0", () => {
    const onLoadOlder = vi.fn();

    const { container } = render(
      <ThreadViewport
        messages={makeLongMessages(4)}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        onLoadOlder={onLoadOlder}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });

    act(() => {
      scroller.dispatchEvent(new WheelEvent("wheel", { deltaY: -120, bubbles: true }));
    });

    expect(onLoadOlder).toHaveBeenCalledTimes(1);
  });

  it("renders a prompt rail that jumps to user messages", async () => {
    const navigateTo = vi.spyOn(ThreadCameraController.prototype, "navigateTo")
      .mockReturnValue("started");
    const scrollTo = vi.fn();
    const { promptEls } = await renderPromptRailViewport({ scrollTo });

    expect(promptEls).toHaveLength(5);

    expect(screen.getByLabelText("User prompt navigation")).toBeInTheDocument();
    const promptMarkers = screen.getAllByRole("button", { name: /Jump to prompt:/ });
    const markerTops = promptMarkers.map((marker) => Number.parseFloat(marker.style.top));
    expect(markerTops[2]).toBeCloseTo(50);
    expect(markerTops[1] - markerTops[0]).toBeCloseTo(16 / 3);
    expect(markerTops[4] - markerTops[0]).toBeCloseTo(64 / 3);

    const railMarkers = screen.getAllByTestId("prompt-rail-marker");
    expect(railMarkers).toHaveLength(promptMarkers.length);
    expect(railMarkers.every((marker) => marker.style.width === "9px")).toBe(true);

    fireEvent.pointerEnter(promptMarkers[2]);
    expect(railMarkers.map((marker) => marker.style.width)).toEqual([
      "16px",
      "22px",
      "28px",
      "22px",
      "16px",
    ]);

    fireEvent.pointerLeave(promptMarkers[2]);
    expect(railMarkers.every((marker) => marker.style.width === "9px")).toBe(true);

    const targetPrompt = screen.getByRole("button", { name: "Jump to prompt: message 3" });
    fireEvent.pointerEnter(targetPrompt);
    const preview = screen.getByTestId("prompt-rail-preview");
    expect(within(preview).getByText("message 3")).toBeInTheDocument();
    expect(within(preview).getByText("answer 3")).toBeInTheDocument();

    fireEvent.click(targetPrompt);

    expect(navigateTo).toHaveBeenCalledWith(1064);
  });

  it("reserves top inset for load-older button and clamps preview vertically", async () => {
    const prompts = Array.from({ length: 25 }, (_, i) => ({
      prompt_id: `p-${i + 1}`,
      turn_ordinal: i + 1,
      label: `Prompt ${i + 1}`,
      created_at: 1000 + i,
      locator: `loc-${i + 1}`,
    }));
    const onLoadMorePrompts = vi.fn();
    render(
      <ThreadViewport
        messages={makePromptExchangeMessages(5)}
        isStreaming={false}
        composer={<div />}
        sessionPrompts={prompts}
        sessionPromptTotal={40}
        hasMoreBefore
        hasMorePromptsBefore
        onLoadMorePrompts={onLoadMorePrompts}
      />,
    );
    await act(async () => {
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
    });

    expect(screen.getByTestId("prompt-rail-load-older")).toBeInTheDocument();
    const markerButtons = screen.getAllByRole("button", { name: /Jump to prompt:/ });
    const firstTop = Number.parseFloat(markerButtons[0].style.top);
    // 28px inset on 300px fallback rail height is > 9%, keeping markers clear of the top `…` button.
    expect(firstTop).toBeGreaterThanOrEqual(9);

    fireEvent.pointerEnter(markerButtons[0]);
    const topPreview = screen.getByTestId("prompt-rail-preview");
    expect(Number.parseFloat(topPreview.style.top)).toBeGreaterThanOrEqual(14);

    fireEvent.pointerEnter(markerButtons[markerButtons.length - 1]);
    const bottomPreview = screen.getByTestId("prompt-rail-preview");
    expect(Number.parseFloat(bottomPreview.style.top)).toBeLessThanOrEqual(86);
  });

  it("shows one marker per turn in the whole session, not just the loaded window", async () => {
    // Five turns are rendered; the session has twenty. The rail must report the
    // real conversation length, because the left navigator is the only place a
    // reader can see how much history exists.
    const messages = makePromptExchangeMessages(5);
    const sessionTurns = Array.from({ length: 20 }, (_, ordinal) => ({
      ordinal,
      label: `Prompt ${ordinal + 1}`,
      created_at: null,
      cursor: `cursor-${ordinal}`,
    }));

    const onJumpToTurn = vi.fn();
    const { container } = render(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div />}
        sessionTurns={sessionTurns}
        sessionTurnTotal={20}
        onNeedSessionTurns={() => {}}
        onJumpToTurn={onJumpToTurn}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1800 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });
    Array.from(container.querySelectorAll<HTMLElement>("[data-user-prompt-id]"))
      .forEach((el, index) => {
        Object.defineProperty(el, "offsetTop", { configurable: true, value: index * 360 });
      });

    await act(async () => {
      window.dispatchEvent(new Event("resize"));
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
    });

    // One rail, and it carries every turn in the session.
    expect(screen.getAllByLabelText("User prompt navigation")).toHaveLength(1);
    const markers = screen.getAllByTestId("prompt-rail-marker");
    expect(markers).toHaveLength(20);

    // A turn outside the rendered window jumps by its cursor.
    fireEvent.click(screen.getByLabelText("Jump to prompt: Prompt 1"));
    expect(onJumpToTurn).toHaveBeenCalledWith("cursor-0");
  });

  it("does not render or measure the prompt rail below the desktop breakpoint", async () => {
    const matchMedia = vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }) as unknown as MediaQueryList);

    try {
      const { container } = render(
        <ThreadViewport
          messages={makePromptExchangeMessages(5)}
          isStreaming={false}
          composer={<div />}
        />,
      );

      const scroller = getScroller(container);
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 1800 },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, value: 0 },
      });

      await act(async () => {
        window.dispatchEvent(new Event("resize"));
        await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
      });

      expect(screen.queryByLabelText("User prompt navigation")).not.toBeInTheDocument();
      expect(screen.queryAllByTestId("prompt-rail-marker")).toHaveLength(0);
    } finally {
      matchMedia.mockRestore();
    }
  });

  it("renders markdown in prompt rail previews", async () => {
    const promptMessages = makePromptExchangeMessages(5);
    const answer = promptMessages.find((message) => message.id === "a3");
    if (!answer) throw new TypeError("prompt answer fixture missing");
    answer.content = "### Confirmed limit\n\nUse the **policy cap**.";

    await renderPromptRailViewport({ messages: promptMessages });

    const targetPrompt = screen.getByRole("button", { name: "Jump to prompt: message 3" });
    fireEvent.pointerEnter(targetPrompt);
    const preview = screen.getByTestId("prompt-rail-preview");

    await waitFor(() => {
      expect(preview.querySelector("h3")).toHaveTextContent("Confirmed limit");
    });
    expect(preview.querySelector("strong")).toHaveTextContent("policy cap");
    expect(preview).not.toHaveTextContent("###");
    expect(preview).not.toHaveTextContent("**");
  });

  it("lets direct paging input interrupt prompt rail navigation", async () => {
    vi.spyOn(ThreadCameraController.prototype, "navigateTo")
      .mockReturnValue("started");
    const cancel = vi.spyOn(ThreadCameraController.prototype, "cancel");
    const { scroller } = await renderPromptRailViewport();
    const targetPrompt = screen.getByRole("button", { name: "Jump to prompt: message 3" });
    const restartNavigation = () => {
      fireEvent.click(targetPrompt);
      cancel.mockClear();
    };

    scroller.scrollTop = 500;
    restartNavigation();
    fireEvent.wheel(scroller, { deltaY: 120 });
    expect(cancel).toHaveBeenCalledTimes(1);

    restartNavigation();
    fireEvent.wheel(scroller, { deltaY: -120 });
    expect(cancel).toHaveBeenCalledTimes(1);

    restartNavigation();
    fireEvent.keyDown(scroller, { key: "PageDown" });
    expect(cancel).toHaveBeenCalledTimes(1);

    restartNavigation();
    fireEvent.touchStart(scroller, { touches: [{ clientY: 300 }] });
    fireEvent.touchMove(scroller, { touches: [{ clientY: 200 }] });
    expect(cancel).toHaveBeenCalledTimes(1);

    restartNavigation();
    scroller.scrollTop = 1200;
    fireEvent.wheel(scroller, { deltaY: 120 });
    expect(cancel).toHaveBeenCalledTimes(1);

    cancel.mockClear();
    fireEvent.wheel(scroller, { deltaY: 120 });
    expect(cancel).not.toHaveBeenCalled();
  });

  it("opens a prompt navigator list and jumps to a selected prompt", async () => {
    const navigateTo = vi.spyOn(ThreadCameraController.prototype, "navigateTo")
      .mockReturnValue("started");
    const promptMessages = makeLongMessages(5);
    const { container } = render(<ViewportWithPromptNavigator messages={promptMessages} />);

    const scroller = container.querySelector(".thread-viewport-scrollbar") as HTMLElement;
    const scrollTo = vi.fn();
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1800 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, value: 0 },
      scrollTo: { configurable: true, value: scrollTo },
    });

    const promptEls = Array.from(
      container.querySelectorAll<HTMLElement>("[data-user-prompt-id]"),
    );
    promptEls.forEach((el, index) => {
      Object.defineProperty(el, "offsetTop", {
        configurable: true,
        value: index * 360,
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "Open prompt navigator" }));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Prompts")).toBeInTheDocument();
    expect(within(dialog).getByText("message 4")).toBeInTheDocument();

    fireEvent.change(within(dialog).getByRole("textbox", { name: "Search prompts" }), {
      target: { value: "message 4" },
    });
    expect(within(dialog).queryByText("message 1")).not.toBeInTheDocument();

    fireEvent.click(within(dialog).getByRole("button", { name: "Jump to prompt: message 4" }));

    expect(navigateTo).toHaveBeenCalledWith(1200);
  });

  it("expands the history window before jumping to an older prompt from the navigator", async () => {
    const longMessages = makeLongMessages(300);
    render(<ViewportWithPromptNavigator messages={longMessages} />);

    expect(screen.queryByText("message 20")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Open prompt navigator" }));
    const dialog = screen.getByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Jump to prompt: message 20" }));

    await waitFor(() => expect(screen.getByText("message 20")).toBeInTheDocument());
  });

  it("renders the prompt rail for compact scroll ranges", async () => {
    const promptMessages = makeLongMessages(3);
    const { container } = render(
      <ThreadViewport
        messages={promptMessages}
        isStreaming={false}
        composer={<div />}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 700 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, value: 0 },
    });

    const promptEls = Array.from(
      container.querySelectorAll<HTMLElement>("[data-user-prompt-id]"),
    );
    expect(promptEls).toHaveLength(3);
    promptEls.forEach((el, index) => {
      Object.defineProperty(el, "offsetTop", {
        configurable: true,
        value: index * 50,
      });
    });

    await act(async () => {
      window.dispatchEvent(new Event("resize"));
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
    });

    expect(screen.getByLabelText("User prompt navigation")).toBeInTheDocument();
  });

  it("buckets dense prompt rails without rendering every prompt as a marker", async () => {
    const navigateTo = vi.spyOn(ThreadCameraController.prototype, "navigateTo")
      .mockReturnValue("started");
    const promptMessages = makeLongMessages(100);
    const { container } = render(
      <ThreadViewport
        messages={promptMessages}
        isStreaming={false}
        composer={<div />}
      />,
    );

    const scroller = getScroller(container);
    const scrollTo = vi.fn();
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 10000 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, value: 0 },
      scrollTo: { configurable: true, value: scrollTo },
    });

    const promptEls = Array.from(
      container.querySelectorAll<HTMLElement>("[data-user-prompt-id]"),
    );
    expect(promptEls).toHaveLength(100);
    promptEls.forEach((el, index) => {
      Object.defineProperty(el, "offsetTop", {
        configurable: true,
        value: index * 90,
      });
    });

    await act(async () => {
      window.dispatchEvent(new Event("resize"));
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
    });

    const promptMarkers = screen.getAllByRole("button", { name: /Jump to prompt:/ });
    expect(promptMarkers.length).toBeGreaterThan(3);
    expect(promptMarkers.length).toBeLessThan(100);
    expect(
      promptMarkers.some((marker) =>
        marker.getAttribute("aria-label")?.includes("prompts, latest"),
      ),
    ).toBe(true);

    fireEvent.click(promptMarkers[promptMarkers.length - 1]);

    expect(navigateTo).toHaveBeenCalledWith(8894);
  });

  it("expands the window start to avoid cutting an agent activity cluster", () => {
    const clustered = makeLongMessages(200);
    clustered.splice(
      38,
      3,
      {
        id: "r0",
        role: "assistant",
        content: "",
        reasoning: "first reasoning",
        createdAt: 38,
      },
      {
        id: "t0",
        role: "tool",
        kind: "trace",
        content: "tool()",
        traces: ["tool()"],
        createdAt: 39,
      },
      {
        id: "r1",
        role: "assistant",
        content: "",
        reasoning: "second reasoning",
        createdAt: 40,
      },
    );

    const visible = windowMessages(clustered, INITIAL_HISTORY_WINDOW);

    expect(visible[0].id).toBe("r0");
    expect(visible).toHaveLength(INITIAL_HISTORY_WINDOW + 2);
  });

  it("resets to the bottom when opening a different conversation", async () => {
    const scrollTo = vi.fn();
    const { container, rerender } = render(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div />}
        conversationKey="chat-a"
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
      scrollTo: { configurable: true, value: scrollTo },
    });
    act(() => {
      dispatchUserScroll(scroller);
    });
    scrollTo.mockClear();

    rerender(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div />}
        conversationKey="chat-b"
      />,
    );

    await waitFor(() => expect(scroller.scrollTop).toBe(1800));
  });

  it("pins an opened conversation to late layout growth without animating", async () => {
    const resizeObserver = stubResizeObserver();
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const followTo = vi.spyOn(ThreadCameraController.prototype, "followTo");

    try {
      const { container, rerender } = render(
        <ThreadViewport
          messages={messages}
          isStreaming={false}
          composer={<div />}
          conversationKey="chat-a"
        />,
      );
      const scroller = getScroller(container);
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 2400 },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, writable: true, value: 300 },
      });
      act(() => {
        dispatchUserScroll(scroller);
      });

      rerender(
        <ThreadViewport
          messages={messages}
          isStreaming={false}
          composer={<div />}
          conversationKey="chat-b"
        />,
      );
      await waitFor(() => expect(scroller.scrollTop).toBe(1800));
      jumpTo.mockClear();
      followTo.mockClear();

      const messageRegion = screen.getByTestId("thread-message-region");
      const messageContent = messageRegion.firstElementChild;
      expect(messageContent).not.toBeNull();
      const contentObserver = resizeObserver.observers.find(
        (observer) => observer.elements.includes(messageContent!),
      );
      expect(contentObserver).toBeDefined();

      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 3000,
      });
      act(() => {
        contentObserver!.callback([], contentObserver as unknown as ResizeObserver);
      });
      await flushAnimationFrame();

      expect(jumpTo).toHaveBeenCalledWith(2400);
      expect(scroller.scrollTop).toBe(2400);
      expect(followTo).not.toHaveBeenCalled();
    } finally {
      jumpTo.mockRestore();
      followTo.mockRestore();
      resizeObserver.restore();
    }
  });

  it("animates the bottom button target and pins later layout growth", async () => {
    const resizeObserver = stubResizeObserver();
    const navigateLatestTo = vi.spyOn(
      ThreadMotionCoordinator.prototype,
      "navigateLatestTo",
    );
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const followTo = vi.spyOn(ThreadCameraController.prototype, "followTo");

    try {
      const { container } = render(
        <ThreadViewport
          messages={messages}
          isStreaming={false}
          composer={<div />}
          conversationKey="chat-a"
        />,
      );
      const scroller = getScroller(container);
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, value: 2400 },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, writable: true, value: 300 },
      });
      act(() => {
        dispatchUserScroll(scroller);
      });

      navigateLatestTo.mockClear();
      jumpTo.mockClear();
      followTo.mockClear();
      fireEvent.click(screen.getByRole("button", { name: "Scroll to bottom" }));
      expect(navigateLatestTo).toHaveBeenCalledWith(1800);
      expect(scroller.scrollTop).toBe(300);

      const messageRegion = screen.getByTestId("thread-message-region");
      const messageContent = messageRegion.firstElementChild;
      expect(messageContent).not.toBeNull();
      const contentObserver = resizeObserver.observers.find(
        (observer) => observer.elements.includes(messageContent!),
      );
      expect(contentObserver).toBeDefined();

      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 3000,
      });
      act(() => {
        contentObserver!.callback([], contentObserver as unknown as ResizeObserver);
      });
      await waitFor(() => expect(scroller.scrollTop).toBe(2400));

      expect(jumpTo).not.toHaveBeenCalled();
      expect(followTo).not.toHaveBeenCalled();

      act(() => {
        scroller.dispatchEvent(new Event("scroll"));
      });
      await flushAnimationFrame();
      jumpTo.mockClear();

      Object.defineProperty(scroller, "scrollHeight", {
        configurable: true,
        value: 3400,
      });
      act(() => {
        contentObserver!.callback([], contentObserver as unknown as ResizeObserver);
      });
      await flushAnimationFrame();

      expect(jumpTo).toHaveBeenCalledWith(2800);
      expect(scroller.scrollTop).toBe(2800);
    } finally {
      navigateLatestTo.mockRestore();
      jumpTo.mockRestore();
      followTo.mockRestore();
      resizeObserver.restore();
    }
  });

  it("waits for the next conversation's transcript before restoring its bottom", async () => {
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const followTo = vi.spyOn(ThreadCameraController.prototype, "followTo");
    const handoffAnimation = {
      cancel: vi.fn(),
      oncancel: null,
      onfinish: null,
    } as unknown as Animation;
    const animate = vi.fn(() => handoffAnimation);
    const oldMessages: UIMessage[] = [
      {
        id: "old-user",
        role: "user",
        content: "old question",
        turnId: "old-turn",
        createdAt: 1,
      },
    ];
    const nextMessages: UIMessage[] = [
      {
        id: "next-user",
        role: "user",
        content: "next question",
        turnId: "next-turn",
        createdAt: 2,
      },
      {
        id: "next-answer",
        role: "assistant",
        content: "next answer",
        turnId: "next-turn",
        createdAt: 3,
      },
    ];
    const { container, rerender } = render(
      <ThreadViewport
        messages={oldMessages}
        isStreaming
        composer={<div />}
        conversationKey="chat-a"
        activeTurnId="old-turn"
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 300 },
      animate: { configurable: true, value: animate },
    });
    jumpTo.mockClear();

    rerender(
      <ThreadViewport
        messages={oldMessages}
        isStreaming
        composer={<div />}
        conversationKey="chat-b"
        activeTurnId="old-turn"
        conversationReady={false}
      />,
    );
    expect(scroller.scrollTop).toBe(300);
    expect(jumpTo).not.toHaveBeenCalled();
    expect(animate).toHaveBeenCalledWith(
      [{ opacity: 1 }, { opacity: 0.82 }],
      {
        duration: 80,
        easing: "cubic-bezier(0.2, 0, 0, 1)",
        fill: "forwards",
      },
    );

    Object.defineProperty(scroller, "scrollHeight", {
      configurable: true,
      value: 3000,
    });
    rerender(
      <ThreadViewport
        messages={nextMessages}
        isStreaming
        composer={<div />}
        conversationKey="chat-b"
        activeTurnId="next-turn"
        conversationReady
      />,
    );

    await waitFor(() => expect(scroller.scrollTop).toBe(2400));
    await flushAnimationFrame();
    expect(jumpTo.mock.calls).toEqual([[2400]]);
    expect(followTo).toHaveBeenCalledWith(2400);
    expect(handoffAnimation.cancel).toHaveBeenCalled();
    expect(animate).toHaveBeenCalledTimes(1);
  });

  it("waits for hydrated messages before fulfilling open-chat bottom scroll", async () => {
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    const scrollTo = vi.fn();
    const { container, rerender } = render(
      <ThreadViewport
        messages={emptyMessages}
        isStreaming={false}
        composer={<div />}
        conversationKey={null}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 0 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
      scrollTo: { configurable: true, value: scrollTo },
    });
    scrollTo.mockClear();
    jumpTo.mockClear();

    rerender(
      <ThreadViewport
        messages={emptyMessages}
        isStreaming={false}
        composer={<div />}
        conversationKey="chat-a"
      />,
    );
    expect(jumpTo).toHaveBeenCalledWith(0);

    rerender(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div />}
        conversationKey="chat-a"
        conversationReady={false}
      />,
    );
    const hydratedScroller = getScroller(container);
    Object.defineProperties(hydratedScroller, {
      scrollHeight: { configurable: true, value: 2400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
      scrollTo: { configurable: true, value: scrollTo },
    });
    scrollTo.mockClear();
    jumpTo.mockClear();

    rerender(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div />}
        conversationKey="chat-a"
        conversationReady
      />,
    );

    await waitFor(() => expect(hydratedScroller.scrollTop).toBe(1800));
    expect(jumpTo).toHaveBeenCalledWith(1800);
  });

  it("scrolls to the bottom when explicitly signalled after send", async () => {
    const scrollTo = vi.fn();
    const { container, rerender } = render(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div />}
        scrollToBottomSignal={0}
      />,
    );
    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 2400 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
      scrollTo: { configurable: true, value: scrollTo },
    });
    scrollTo.mockClear();

    rerender(
      <ThreadViewport
        messages={messages}
        isStreaming={false}
        composer={<div />}
        scrollToBottomSignal={1}
      />,
    );

    await waitFor(() => expect(scroller.scrollTop).toBe(1800));
  });

  it("keeps the prompt rail visible for a short latest window in a long chat and reveals by prompt id", async () => {
    const shortLatestWindow: UIMessage[] = [
      { id: "u-latest", promptId: "evt-20", role: "user", content: "Latest prompt", createdAt: 20 },
      { id: "a-latest", role: "assistant", content: "Latest reply", createdAt: 21 },
    ];
    const sessionPrompts = Array.from({ length: 10 }, (_, idx) => ({
      prompt_id: `evt-${idx + 11}`,
      turn_ordinal: idx + 10,
      prompt_ordinal: idx + 10,
      label: `Prompt ${idx + 11}`,
      created_at: 100 + idx,
      locator: `loc-${idx + 11}`,
    }));
    const onRevealPrompt = vi.fn();
    const onLoadMorePrompts = vi.fn();
    const { container } = render(
      <ThreadViewport
        messages={shortLatestWindow}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        sessionPrompts={sessionPrompts}
        sessionPromptTotal={20}
        hasMorePromptsBefore
        onLoadMorePrompts={onLoadMorePrompts}
        onRevealPrompt={onRevealPrompt}
      />,
    );

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

    expect(screen.getByLabelText("User prompt navigation")).toBeInTheDocument();
    expect(screen.getAllByTestId("prompt-rail-marker")).toHaveLength(10);
    fireEvent.click(screen.getByTestId("prompt-rail-load-older"));
    expect(onLoadMorePrompts).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByLabelText("Jump to prompt: Prompt 11"));
    expect(onRevealPrompt).toHaveBeenCalledWith({
      promptId: "evt-11",
      locator: "loc-11",
      cursor: null,
    });
  });

  it("scrolls and highlights the exact Steer prompt on pendingReveal ready and triggers loadNewer / goToLatest", async () => {
    const navigateTo = vi.spyOn(ThreadCameraController.prototype, "navigateTo");
    const steerWindow: UIMessage[] = [
      { id: "u15a", promptId: "evt-u-15", role: "user", content: "Prompt 15", createdAt: 15 },
      { id: "a15a", role: "assistant", content: "Before steer", createdAt: 16 },
      { id: "u15b", promptId: "evt-steer-15", role: "user", content: "Steer 15", createdAt: 17 },
      { id: "a15b", role: "assistant", content: "After steer with turn_end text", createdAt: 18 },
    ];
    const onPendingRevealHandled = vi.fn();
    const onLoadNewer = vi.fn();
    const onGoToLatest = vi.fn();
    const { container } = render(
      <ThreadViewport
        messages={steerWindow}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        hasMoreAfter
        tailActivityPending
        onLoadNewer={onLoadNewer}
        onGoToLatest={onGoToLatest}
        pendingReveal={{
          requestId: 7,
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
    const firstEl = container.querySelector<HTMLElement>('[data-prompt-key="evt-u-15"]');
    expect(steerEl).toBeTruthy();
    expect(firstEl).toBeTruthy();
    Object.defineProperty(firstEl!, "offsetTop", { configurable: true, value: 120 });
    Object.defineProperty(steerEl!, "offsetTop", { configurable: true, value: 940 });

    await waitFor(() => expect(onPendingRevealHandled).toHaveBeenCalledWith(7));
    expect(steerEl?.dataset.promptHighlighted).toBe("true");
    expect(firstEl?.dataset.promptHighlighted).toBeUndefined();
    expect(navigateTo).toHaveBeenCalled();

    // Scrolling near the bottom of an older window triggers forward paging.
    scroller.scrollTop = 1750;
    fireEvent.scroll(scroller);
    expect(onLoadNewer).toHaveBeenCalledTimes(1);

    // Go to latest button shows tail activity and calls onGoToLatest.
    const latestBtn = screen.getByRole("button", { name: "Go to latest" });
    expect(latestBtn).toHaveAttribute("data-tail-activity", "true");
    fireEvent.click(latestBtn);
    expect(onGoToLatest).toHaveBeenCalledTimes(1);
  });

  it("shows loading feedback on an unloaded prompt rail marker while revealing and allows loading older history without a turn index", async () => {
    const onLoadOlder = vi.fn();
    const onRevealPrompt = vi.fn();
    const tailMessages: UIMessage[] = [
      { id: "u3", promptId: "evt-3", role: "user", content: "Prompt 3", createdAt: 3 },
      { id: "a3", role: "assistant", content: "Answer 3", createdAt: 4 },
      { id: "u4", promptId: "evt-4", role: "user", content: "Prompt 4", createdAt: 5 },
      { id: "a4", role: "assistant", content: "Answer 4", createdAt: 6 },
      { id: "u5", promptId: "evt-5", role: "user", content: "Prompt 5", createdAt: 7 },
      { id: "a5", role: "assistant", content: "Answer 5", createdAt: 8 },
    ];

    const { container, rerender } = render(
      <ThreadViewport
        messages={tailMessages}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        onLoadOlder={onLoadOlder}
        onRevealPrompt={onRevealPrompt}
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, value: 1800 },
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 1200 },
    });

    // Without a prompt/turn index, the rail's top "…" button loads older transcript pages directly.
    const loadOlderBtn = await screen.findByTestId("prompt-rail-load-older");
    fireEvent.click(loadOlderBtn);
    expect(onLoadOlder).toHaveBeenCalledTimes(1);

    // With a prompt index, clicking an unloaded marker triggers reveal and shows loading pulse.
    rerender(
      <ThreadViewport
        messages={tailMessages}
        isStreaming={false}
        composer={<div />}
        hasMoreBefore
        onLoadOlder={onLoadOlder}
        onRevealPrompt={onRevealPrompt}
        sessionPrompts={[
          { prompt_id: "evt-1", turn_ordinal: 0, prompt_ordinal: 0, label: "Older prompt 1", locator: "loc-1", created_at: 1 },
          { prompt_id: "evt-3", turn_ordinal: 2, prompt_ordinal: 2, label: "Prompt 3", locator: "loc-3", created_at: 3 },
          { prompt_id: "evt-4", turn_ordinal: 3, prompt_ordinal: 3, label: "Prompt 4", locator: "loc-4", created_at: 5 },
          { prompt_id: "evt-5", turn_ordinal: 4, prompt_ordinal: 4, label: "Prompt 5", locator: "loc-5", created_at: 7 },
        ]}
        sessionPromptTotal={4}
        pendingReveal={{
          requestId: 9,
          promptId: "evt-1",
          locator: "loc-1",
          status: "loading",
        }}
      />,
    );

    const olderMarkerBtn = await screen.findByRole("button", { name: /Older prompt 1/ });
    expect(olderMarkerBtn).toHaveAttribute("aria-busy", "true");
    expect(olderMarkerBtn.querySelector('[data-revealing="true"]')).not.toBeNull();
    fireEvent.click(olderMarkerBtn);
    expect(onRevealPrompt).toHaveBeenCalledWith({
      promptId: "evt-1",
      locator: "loc-1",
      cursor: null,
    });
  });

  it("virtualizes long transcripts with TanStack Virtual instead of mounting every message", () => {
    const longMessages = makeLongMessages(200);
    const { container } = render(
      <ThreadViewport
        messages={longMessages}
        isStreaming={false}
        composer={<div />}
        virtualize
      />,
    );

    const scroller = getScroller(container);
    Object.defineProperties(scroller, {
      clientHeight: { configurable: true, value: 600 },
      scrollTop: { configurable: true, writable: true, value: 0 },
    });

    const virtualList = screen.getByTestId("thread-virtual-list");
    expect(virtualList).toBeInTheDocument();
    const mountedUnits = virtualList.querySelectorAll("[data-thread-display-unit]");
    expect(mountedUnits.length).toBeLessThan(longMessages.length);
  });

  it("enables native CSS scroll anchoring when browsing history and skips redundant JS scrollTop writes when native anchoring already compensated", () => {
    const resizeObserver = stubResizeObserver();
    let hitTarget: Element | null = null;
    const hitTest = stubElementsFromPoint(() => (hitTarget ? [hitTarget] : []));
    const jumpTo = vi.spyOn(ThreadCameraController.prototype, "jumpTo");
    try {
      const { container } = render(
        <ThreadViewport
          messages={makeLongMessages(300)}
          isStreaming={false}
          composer={<div />}
        />,
      );

      const scroller = getScroller(container);
      expect(scroller).toHaveClass("[overflow-anchor:none]");

      let scrollHeight = 6_000;
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, get: () => scrollHeight },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, writable: true, value: 2_000 },
        getBoundingClientRect: {
          configurable: true,
          value: () => DOMRect.fromRect({ y: 0, width: 800, height: 600 }),
        },
      });

      const anchor = screen.getByText("message 200")
        .closest<HTMLElement>("[data-thread-display-unit]");
      expect(anchor).not.toBeNull();
      hitTarget = anchor;
      let anchorDocumentTop = 2_120;
      Object.defineProperty(anchor, "getBoundingClientRect", {
        configurable: true,
        value: () => DOMRect.fromRect({
          y: anchorDocumentTop - scroller.scrollTop,
          width: 800,
          height: 40,
        }),
      });

      act(() => {
        dispatchUserScroll(scroller);
      });
      expect(scroller).toHaveClass("[overflow-anchor:auto]");
      jumpTo.mockClear();

      // Simulate native CSS overflow-anchor:auto compensating a +180px growth
      // above the anchor during layout before ResizeObserver fires.
      anchorDocumentTop += 180;
      scrollHeight += 180;
      scroller.scrollTop += 180;
      const content = screen.getByTestId("thread-message-region").firstElementChild;
      const observer = resizeObserver.observers.find((candidate) =>
        content ? candidate.elements.includes(content) : false,
      );
      expect(observer).toBeDefined();
      act(() => {
        observer?.callback([], observer as unknown as ResizeObserver);
      });

      // Must NOT double-add +180 or emit a programmatic scrollTop write.
      expect(scroller.scrollTop).toBe(2_180);
      expect(jumpTo).not.toHaveBeenCalled();
    } finally {
      hitTest.restore();
      resizeObserver.restore();
    }
  });

  it("anchors on the clicked disclosure display unit and does not prematurely expand history on wheel when scrollTop > 0", () => {
    const resizeObserver = stubResizeObserver();
    try {
      const longMessages = makeLongMessages(300);
      longMessages[220] = {
        id: "t-220",
        role: "tool",
        kind: "trace",
        content: 'exec({"cmd":"ls"})',
        traces: ['exec({"cmd":"ls"})'],
        turnId: "turn-220",
        createdAt: 220,
      };
      const { container } = render(
        <ThreadViewport
          messages={longMessages}
          isStreaming={false}
          composer={<div />}
        />,
      );

      const scroller = getScroller(container);
      let scrollHeight = 6_000;
      Object.defineProperties(scroller, {
        scrollHeight: { configurable: true, get: () => scrollHeight },
        clientHeight: { configurable: true, value: 600 },
        scrollTop: { configurable: true, writable: true, value: 800 },
        getBoundingClientRect: {
          configurable: true,
          value: () => DOMRect.fromRect({ y: 0, width: 800, height: 600 }),
        },
      });

      // Backward wheel while scrollTop > 0 yields camera to user but waits for
      // the browser's resulting scroll event to expand the window.
      act(() => {
        scroller.dispatchEvent(new WheelEvent("wheel", { deltaY: -120, bubbles: true }));
      });
      expect(screen.queryByText("message 139")).not.toBeInTheDocument();

      // Clicking a disclosure pins its enclosing display unit as the scroll anchor.
      const disclosure = container.querySelector<HTMLElement>("[data-thread-disclosure]");
      expect(disclosure).not.toBeNull();
      const disclosureUnit = disclosure!.closest<HTMLElement>("[data-thread-display-unit]");
      expect(disclosureUnit).not.toBeNull();
      let unitDocumentTop = 960;
      Object.defineProperty(disclosureUnit, "getBoundingClientRect", {
        configurable: true,
        value: () => DOMRect.fromRect({
          y: unitDocumentTop - scroller.scrollTop,
          width: 800,
          height: 40,
        }),
      });

      act(() => {
        fireEvent.pointerDown(disclosure!, { button: 0 });
      });

      // Layout above the clicked disclosure unit shifts by +140px without native anchor:
      unitDocumentTop += 140;
      scrollHeight += 140;
      const content = screen.getByTestId("thread-message-region").firstElementChild;
      const observer = resizeObserver.observers.find((candidate) =>
        content ? candidate.elements.includes(content) : false,
      );
      act(() => {
        observer?.callback([], observer as unknown as ResizeObserver);
      });
      expect(scroller.scrollTop).toBe(940);
    } finally {
      resizeObserver.restore();
    }
  });
});
