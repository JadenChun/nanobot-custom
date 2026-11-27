import { describe, expect, it } from "vitest";

import {
  describeTraceLine,
  traceLinesForMessage,
} from "@/components/thread/activity/trace-activity-model";
import type { GenericToolStatus } from "@/components/thread/activity/generic-tool-model";
import type { UIMessage } from "@/lib/types";

function describeTrace(line: string, status: GenericToolStatus = "done") {
  return describeTraceLine(line, status);
}

describe("trace activity semantics", () => {
  it("uses readable copy when a hosted search query is unavailable", () => {
    expect(describeTrace("web_search()").label).toBe("Searched the web");
  });

  it.each([
    ['web_search({"query":"nanobot latest release"})', "done", "Searched nanobot latest release", ""],
    ['web_fetch({"url":"https://example.com/docs?token=private"})', "done", "Read", "example.com/docs"],
    ['read_file({"path":"/Users/alice/project/README.md"})', "done", "Read", "~/project/README.md"],
    ['exec({"command":"date +%Y-%m-%d"})', "done", "Checked current time", ""],
    ['exec_command({"cmd":"API_KEY=secret npm test"})', "running", "Running command", "API_KEY=•••• npm test"],
    ['write_file({"path":"/home/alice/project/output.txt"})', "done", "Wrote file", "~/project/output.txt"],
    ['apply_patch({"file_path":"src/app.tsx","patch":"private"})', "error", "Could not edit file", "src/app.tsx"],
    ['third_party_sync({"token":"secret","payload":"private"})', "done", "Completed Third party sync", ""],
    ["Finished collecting results", "done", "Completed step", "Finished collecting results"],
  ] as const)("describes %s as one safe activity line", (line, status, label, detail) => {
    const result = describeTrace(line, status);
    expect(result).toMatchObject({ label, detail });
    expect(`${result.label} ${result.detail}`).not.toMatch(/[{}]|private|\/Users\/alice|\/home\/alice/);
  });

  it.each([
    ["running", "Searching status test"],
    ["done", "Searched status test"],
    ["error", "Could not search status test"],
  ] as const)("uses status-aware search copy for %s", (status, label) => {
    expect(describeTrace('web_search({"query":"status test"})', status).label).toBe(label);
  });

  it.each([
    ["running", "Searching X · status test"],
    ["done", "Searched X · status test"],
    ["error", "Could not search X · status test"],
  ] as const)("identifies hosted X search activity for %s", (status, label) => {
    expect(describeTrace('x_search({"query":"status test"})', status).label).toBe(label);
  });

  it("never exposes URL credentials, query secrets, or private-network links", () => {
    const publicResult = describeTrace(
      'web_fetch({"url":"https://user:password@example.com/docs?api_key=secret#section"})',
    );
    expect(publicResult).toMatchObject({ detail: "example.com/docs", host: "example.com" });
    expect(JSON.stringify(publicResult)).not.toMatch(/password|api_key|secret/);

    const privateResult = describeTrace('web_fetch({"url":"http://127.0.0.1:8765/private"})');
    expect(privateResult.url).toBeUndefined();
    expect(privateResult.detail).not.toContain("127.0.0.1");
  });

  it("summarizes multi-line commands without exposing every script line", () => {
    const result = describeTrace(
      'exec({"command":"npm test\\necho second-secret-line\\necho third-line"})',
    );
    expect(result).toMatchObject({
      label: "Ran command",
      detail: "npm test · script, 3 lines",
    });
    expect(result.detail).not.toContain("second-secret-line");
  });

  it("displays explicit shell command description as the action intention", () => {
    expect(
      describeTraceLine(
        'exec({"command":"python3 check.py","description":"Check image luminance"})',
        "done",
      ),
    ).toMatchObject({ label: "Ran", detail: "check image luminance" });

    expect(
      describeTraceLine(
        'exec("python3 check.py")',
        "running",
        undefined,
        { command: "python3 check.py", description: "WebUI build" },
      ),
    ).toMatchObject({ label: "Running", detail: "WebUI build" });
  });

  it("infers clean intentions for common shell commands without explicit descriptions", () => {
    expect(
      describeTrace(
        'exec({"command":"cd ~/.nanobot-staging/workspace/tiktok-test && python3 - <<\'EOF\'\\nfrom PIL import Image\\nim = Image.open(\'a.png\')\\nEOF"})',
        "done",
      ),
    ).toMatchObject({ label: "Analyzed images", detail: "Python · 4 lines" });

    expect(
      describeTrace('exec({"command":"pytest tests/test_shell.py -v"})', "running"),
    ).toMatchObject({ label: "Running tests", detail: "" });

    expect(
      describeTrace('exec({"command":"ruff check nanobot/"})', "done"),
    ).toMatchObject({ label: "Ran linter", detail: "" });

    expect(
      describeTrace('exec({"command":"git status --short"})', "done"),
    ).toMatchObject({ label: "Checked git status", detail: "" });
  });
});

describe("trace lines of an activity row", () => {
  const row = (extra: Partial<UIMessage>): UIMessage => ({
    id: "row",
    role: "tool",
    kind: "trace",
    content: "src/a.ts:1: raw tool output",
    createdAt: 0,
    ...extra,
  });

  it("prefers the traces the server sent", () => {
    expect(traceLinesForMessage(row({
      traces: ['grep({"pattern":"token"})'],
      toolEvents: [{ phase: "start", call_id: "a", name: "read_file" }],
    }))).toEqual(['grep({"pattern":"token"})']);
  });

  it("builds lines from the start events instead of the output", () => {
    expect(traceLinesForMessage(row({
      toolEvents: [
        { phase: "end", call_id: "a", name: "read_file", result: "raw" },
        { phase: "start", call_id: "a", name: "read_file", arguments: { path: "a.ts" } },
        { phase: "start", call_id: "b", name: "grep", arguments: { pattern: "x" } },
        { phase: "progress", call_id: "b", name: "grep", message: "half" },
      ],
    }))).toEqual(['read_file({"path":"a.ts"})', 'grep({"pattern":"x"})']);
  });

  it("names the call when only its result event is on the row", () => {
    const lines = traceLinesForMessage(row({
      toolEvents: [{ phase: "end", call_id: "a", name: "grep", result: "raw" }],
    }));
    expect(lines).toEqual(["grep()"]);
  });

  it("never falls back to content when the row has tool events", () => {
    expect(traceLinesForMessage(row({
      toolEvents: [{ phase: "end", call_id: "a" }],
    }))).toEqual([]);
  });

  it("keeps the content line of a legacy row without tool events", () => {
    expect(traceLinesForMessage(row({ content: "Reading files" }))).toEqual(["Reading files"]);
    expect(traceLinesForMessage(row({ content: "  " }))).toEqual([]);
  });
});
