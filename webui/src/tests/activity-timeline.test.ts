import { describe, expect, it } from "vitest";

import { normalizeActivityTimeline } from "@/lib/activity-timeline";
import type { UIMessage } from "@/lib/types";

const reply = (id: string, turnSeq: number, overrides: Partial<UIMessage> = {}): UIMessage => ({
  id, role: "assistant", content: id, createdAt: turnSeq * 1000, turnId: "turn", turnSeq,
  turnPhase: "answer", ...overrides,
});
const tool = (id: string, turnSeq: number, overrides: Partial<UIMessage> = {}): UIMessage => ({
  id, role: "tool", kind: "trace", content: "read_file()", createdAt: turnSeq * 1000,
  turnId: "turn", turnSeq, turnPhase: "activity", ...overrides,
});

describe("interim assistant text", () => {
  it("uses the tool's causal order and preserves one final answer and the source count", () => {
    const note = reply("Checking the result", 1, { reasoning: "Find supporting records" });
    const work = tool("tool", 2);
    const answer = reply("Final answer", 3);
    expect(normalizeActivityTimeline([note, answer, work])).toEqual([
      expect.objectContaining({
        type: "activity",
        sourceMessageCount: 2,
        messages: [
          expect.objectContaining({ reasoning: "Find supporting records", content: "" }),
          expect.objectContaining({ id: note.id, content: note.content, activityKind: "model" }),
          work,
        ],
      }),
      expect.objectContaining({
        type: "message", sourceMessageCount: 1, message: expect.objectContaining({ id: answer.id }),
      }),
    ]);
    expect(note.activityKind).toBeUndefined();
  });

  it("reclassifies the live answer when work starts and keeps it in activity on completion", () => {
    const streaming = reply("I'll read it now", 1, { isStreaming: true });
    const call = { call_id: "read-1", name: "read_file" };
    expect(normalizeActivityTimeline([streaming])[0]).toMatchObject({
      type: "message", message: { id: streaming.id, isStreaming: true },
    });
    for (const phase of ["start", "progress", "end"]) {
      expect(normalizeActivityTimeline([
        streaming, tool("tool", 2, { toolEvents: [{ ...call, phase }] }),
      ])).toMatchObject([{
        type: "activity",
        messages: [{ id: streaming.id, activityKind: "model", isStreaming: false }, { id: "tool" }],
      }]);
    }
  });

  it("does not demote final prose for a different turn, a late result, or child status", () => {
    const finished = reply("Final answer", 2, { completedAt: 2_500 });
    const started = tool("started", 1, { toolEvents: [{ phase: "start", call_id: "read-1" }] });
    const ended = tool("ended", 3, { toolEvents: [{ phase: "end", call_id: "read-1" }] });
    const distinct = tool("distinct", 4, { turnId: "other" });
    const late = tool("late", 5);
    const child = reply("child", 3, { kind: "delegated_child", content: "" });
    const expectAnswer = (rows: UIMessage[], answer = finished) => expect(normalizeActivityTimeline(rows))
      .toContainEqual(expect.objectContaining({ type: "message", message: expect.objectContaining({ id: answer.id }) }));
    expectAnswer([finished, distinct]);
    expectAnswer([finished, late]);
    expectAnswer([started, reply(finished.id, 2), ended]);
    expectAnswer([started, reply(finished.id, 2), {
      ...ended, toolEvents: [{ phase: "progress", call_id: "read-1" }],
    }]);
    expectAnswer([finished, child]);
  });

  it("keeps delivered media in the answer surface", () => {
    const media = reply("Preview", 1, { media: [{ kind: "file", url: "/result.csv", name: "result.csv" }] });
    expect(normalizeActivityTimeline([media, tool("tool", 2), reply("Final answer", 3)])[0]).toMatchObject({
      type: "message", message: { id: media.id, media: media.media },
    });
  });

  it("uses chronological evidence for legacy turns without IDs", () => {
    const legacy = { turnId: undefined, turnSeq: undefined };
    const note = reply("Checking", 1, legacy);
    const trace = tool("trace", 2, legacy);
    const answer = reply("Done", 3, legacy);
    expect(normalizeActivityTimeline([note, trace, answer])).toMatchObject([
      { type: "activity", messages: [{ id: note.id, activityKind: "model" }, { id: trace.id }] },
      { type: "message", message: { id: answer.id } },
    ]);
    expect(normalizeActivityTimeline([answer, trace])[0]).toMatchObject({
      type: "message", message: { id: answer.id },
    });
  });
});
