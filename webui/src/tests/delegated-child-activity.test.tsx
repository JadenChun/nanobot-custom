import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ThreadMessages } from "@/components/thread/ThreadMessages";
import { formatToolCallTrace } from "@/lib/tool-traces";
import type { DelegatedChildPayload, UIMessage } from "@/lib/types";

function childRow(overrides: Partial<DelegatedChildPayload> = {}): UIMessage {
  return {
    id: "delegated-child-child-a",
    role: "assistant",
    kind: "delegated_child",
    content: "",
    createdAt: 3,
    turnId: "t1",
    delegatedChild: {
      child_session_key: "delegated:child-a",
      child_id: "child-a",
      profile: "worker",
      task_label: "Inspect the API contract",
      state: "active",
      runtime_generation: 0,
      turn_id: "t1",
      spawn_call_id: "call-spawn",
      ...overrides,
    },
  };
}

function thread(childMessage: UIMessage): UIMessage[] {
  const trace = (id: string, callId: string, command: string): UIMessage => {
    const event = { phase: "end", call_id: callId, name: "exec", arguments: { command } };
    const line = formatToolCallTrace(event)!;
    return {
      id, role: "tool", kind: "trace", content: line, traces: [line], createdAt: 2, turnId: "t1",
      toolEvents: [event],
    };
  };
  return [
    { id: "user", role: "user", content: "go", createdAt: 1, turnId: "t1" },
    trace("spawn", "call-spawn", "exec before-child"),
    childMessage,
    trace("later", "call-later", "exec after-child"),
    { id: "answer", role: "assistant", content: "Final answer", createdAt: 4, turnId: "t1", turnPhase: "answer" },
  ];
}

const childElement = (container: HTMLElement) => (
  container.querySelector<HTMLElement>('[data-delegated-child="child-a"]')
);
const disclosure = (container: HTMLElement) => (
  container.querySelector<HTMLButtonElement>("button[data-thread-disclosure]")!
);

describe("inline delegated child activity", () => {
  it("renders a live child inside the single activity group and opens its history", () => {
    const onOpen = vi.fn();
    const { container } = render(
      <ThreadMessages messages={thread(childRow())} onOpenChildHistory={onOpen} />,
    );

    const units = container.querySelectorAll("[data-thread-display-unit]");
    expect(units).toHaveLength(3);
    const element = childElement(container);
    expect(element).not.toBeNull();
    expect(units[1]!.contains(element)).toBe(true);
    expect(disclosure(container)).toHaveAttribute("aria-expanded", "true");
    const text = units[1]!.textContent!;
    expect(text.indexOf("before-child")).toBeLessThan(text.indexOf("Inspect the API contract"));
    expect(text.indexOf("Inspect the API contract")).toBeLessThan(text.indexOf("after-child"));
    fireEvent.click(within(element!).getByRole("button"));
    expect(onOpen).toHaveBeenCalledWith(
      expect.objectContaining({ child_id: "child-a" }),
      expect.any(HTMLButtonElement),
    );
  });

  it("shows one step for the spawning call when its end arrives after the child", () => {
    const [prompt, spawn, delegated, , answer] = thread(childRow());
    const start: UIMessage = {
      ...spawn!,
      toolEvents: spawn!.toolEvents!.map((event) => ({ ...event, phase: "start" })),
    };
    const { container } = render(
      <ThreadMessages
        messages={[prompt!, start, delegated!, { ...spawn!, id: "spawn-end" }, answer!]}
      />,
    );

    const text = container.querySelectorAll("[data-thread-display-unit]")[1]!.textContent!;
    expect(text.match(/before-child/g)).toHaveLength(1);
    expect(text.indexOf("before-child")).toBeLessThan(text.indexOf("Inspect the API contract"));
  });

  it("collapses a settled child with the other steps", () => {
    const { container } = render(
      <ThreadMessages messages={thread(childRow({ state: "completed" }))} />,
    );
    expect(disclosure(container)).toHaveAttribute("aria-expanded", "false");
    expect(childElement(container)!.closest('[aria-hidden="true"]')).not.toBeNull();

    fireEvent.click(disclosure(container));
    expect(childElement(container)!.closest('[aria-hidden="true"]')).toBeNull();
  });

  it("does not hold the group open for a stale live state", () => {
    const { container } = render(
      <ThreadMessages messages={thread(childRow())} delegatedStatusAvailable={false} />,
    );
    expect(disclosure(container)).toHaveAttribute("aria-expanded", "false");
  });

  it("shows an unanchored child directly, without a Worked fold", () => {
    const { container } = render(
      <ThreadMessages messages={[childRow({ state: "completed", spawn_call_id: null })]} />,
    );

    expect(childElement(container)).not.toBeNull();
    expect(screen.queryByText(/worked/i)).toBeNull();
  });
});
