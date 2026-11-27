import { toolTraceLinesFromEvents } from "@/lib/tool-traces";
import type { UIMessage } from "@/lib/types";

import type { GenericToolStatus } from "./generic-tool-model";
import { safeActivityDetail, summarizeShellCommand } from "./activity-text";
import { presentWebSearchAction } from "./web-search-model";
import { displayWebHost, formatCompactWebUrl, parseSafeActivityHttpUrl } from "./web-url";

export interface TraceDescription {
  kind: "search" | "tool" | "done" | "trace";
  label: string;
  detail: string;
  icon?: "clock";
  url?: string;
  host?: string;
}

/**
 * The lines an activity row describes. A row with tool events names its calls
 * (start events first); its content may be raw tool output and is never a line.
 */
export function traceLinesForMessage(message: UIMessage): string[] {
  if (message.traces?.length) return message.traces;
  const events = message.toolEvents;
  if (!events?.length) return message.content.trim() ? [message.content] : [];
  // One line per call, taken from its start event when the row has one.
  const startByCallId = new Map(
    events
      .filter((event) => event.phase === "start" && event.call_id)
      .map((event) => [event.call_id, event]),
  );
  return toolTraceLinesFromEvents(
    events.map((event) => startByCallId.get(event.call_id) ?? event),
  );
}

export function describeTraceLine(
  line: string,
  status: GenericToolStatus,
  result?: unknown,
  toolArgs?: unknown,
): TraceDescription {
  const trimmed = line.trim();
  const functionMatch = /^([a-zA-Z0-9_.-]+)\((.*)\)$/.exec(trimmed);
  const name = (functionMatch?.[1] ?? "").toLowerCase().split(".").pop() || "";
  const args = functionMatch?.[2] ?? "";
  const parsedUrl = traceUrlFromArgs(args, trimmed);
  const webDetail = parsedUrl ? formatCompactWebUrl(parsedUrl) : "";
  const plainWebReadTrace =
    !!parsedUrl && /\b(fetch(?:ing|ed)?|read(?:ing)?|opened?|opening)\b/i.test(trimmed);

  if (/search/i.test(name)) {
    const query = traceFieldFromArgs(args, ["query", "q", "text"]) || args;
    return {
      kind: "search",
      label: presentWebSearchAction(query, status, name === "x_search" ? "x" : "web"),
      detail: "",
    };
  }
  if (/fetch|read|open/i.test(name) || plainWebReadTrace) {
    const rawTarget = traceFieldFromArgs(args, ["path", "file_path", "url"]) || args || trimmed;
    const pageTitle = parsedUrl ? webPageTitle(result) : "";
    return {
      kind: "tool",
      label: pageTitle || statusCopy(status, "Reading", "Read", "Could not read"),
      detail: webDetail || (/^https?:\/\//i.test(rawTarget.trim())
        ? "Private address"
        : safeActivityDetail(rawTarget)),
      url: parsedUrl?.href,
      host: parsedUrl ? displayWebHost(parsedUrl.hostname) : undefined,
    };
  }
  if (isShellTraceName(name)) return describeShellTrace(args, trimmed, status, toolArgs);
  if (name === "write_file") {
    return describeFileMutationTrace(args, status, "Writing file", "Wrote file", "Could not write file");
  }
  if (name === "edit_file" || name === "apply_patch") {
    return describeFileMutationTrace(args, status, "Editing file", "Edited file", "Could not edit file");
  }
  if (name) {
    const action = humanizeTraceToolName(name);
    return {
      kind: "tool",
      label: statusCopy(
        status,
        `Running ${action}`,
        `Completed ${action}`,
        `Could not complete ${action}`,
      ),
      detail: "",
    };
  }
  if (/done|complete|success/i.test(trimmed)) {
    return { kind: "done", label: "Completed step", detail: safeActivityDetail(trimmed) };
  }
  return {
    kind: status === "done" ? "done" : "trace",
    label: statusCopy(status, "Working", "Completed step", "Step failed"),
    detail: safeActivityDetail(trimmed),
  };
}

function webPageTitle(result: unknown): string {
  if (result && typeof result === "object" && !Array.isArray(result)) {
    const title = (result as Record<string, unknown>).title;
    if (typeof title === "string") return safeActivityDetail(title);
  }
  if (typeof result !== "string") return "";
  const heading = result.match(/^#\s+(.+)$/m)?.[1]?.trim();
  return heading ? safeActivityDetail(heading) : "";
}

function describeShellTrace(
  args: string,
  fallback: string,
  status: GenericToolStatus,
  toolArgs?: unknown,
): TraceDescription {
  const description =
    shellFieldFromToolArgs(toolArgs, ["description", "summary"])
    || traceFieldFromArgs(args, ["description", "summary"]);
  if (description) {
    const cleaned = formatShellIntentPhrase(safeActivityDetail(description, 84));
    if (cleaned) {
      return {
        kind: "tool",
        label: statusCopy(status, "Running", "Ran", "Failed"),
        detail: cleaned,
      };
    }
  }
  const command =
    shellFieldFromToolArgs(toolArgs, ["command", "cmd", "script"])
    || shellCommandFromArgs(args)
    || fallback;
  if (/^(?:\/(?:usr\/)?bin\/)?date(?:\s|$)/i.test(command.trim())) {
    return {
      kind: "tool",
      label: statusCopy(
        status,
        "Checking current time",
        "Checked current time",
        "Could not check current time",
      ),
      detail: "",
      icon: "clock",
    };
  }
  const inferred = inferShellIntent(command, status);
  if (inferred) return inferred;
  return {
    kind: "tool",
    label: statusCopy(status, "Running command", "Ran command", "Command failed"),
    detail: summarizeShellCommand(command),
  };
}

function shellFieldFromToolArgs(toolArgs: unknown, keys: string[]): string {
  if (!toolArgs || typeof toolArgs !== "object" || Array.isArray(toolArgs)) return "";
  const record = toolArgs as Record<string, unknown>;
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function formatShellIntentPhrase(phrase: string): string {
  const trimmed = phrase.trim().replace(/\.+$/, "");
  if (!trimmed) return "";
  const firstWord = trimmed.split(/\s+/, 1)[0] ?? "";
  // Lowercase leading sentence-case word ("Check image size" -> "check image size")
  // while preserving acronyms ("MCP", "API") and CamelCase proper nouns ("WebUI", "ComfyUI").
  if (/^[A-Z][a-z]+$/.test(firstWord)) {
    return trimmed[0].toLowerCase() + trimmed.slice(1);
  }
  return trimmed;
}

function inferShellIntent(command: string, status: GenericToolStatus): TraceDescription | null {
  const normalized = command.replace(/\r\n/g, "\n").trim();
  const lines = normalized.split("\n").map((l) => l.trim()).filter(Boolean);
  const firstLine = lines[0] ?? "";
  // Strip leading `cd <dir> &&` chains so we inspect the actual command.
  const coreFirst = firstLine
    .replace(/^(?:cd\s+(?:"[^"]*"|'[^']*'|[^\s;&|]+)\s*&&\s*)+/i, "")
    .trim();
  if (!coreFirst) return null;

  // Python inline heredoc / -c script (e.g. `cd ... && python3 - <<'EOF'`)
  if (/\bpython3?\s+(?:-\s*<<|-c\b)/i.test(coreFirst)) {
    const body = normalized;
    if (/\b(?:PIL|Image\.open|cv2|luminance|contact.?sheet|montage|crop|resize|review_)\b/i.test(body)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Analyzing images", "Analyzed images", "Image analysis failed"),
        detail: lines.length > 1 ? `Python · ${lines.length} lines` : "",
      };
    }
    if (/\b(?:ffmpeg|ffprobe|moviepy|video)\b/i.test(body)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Processing media", "Processed media", "Media processing failed"),
        detail: lines.length > 1 ? `Python · ${lines.length} lines` : "",
      };
    }
    return {
      kind: "tool",
      label: statusCopy(status, "Running Python script", "Ran Python script", "Python script failed"),
      detail: lines.length > 1 ? `${lines.length} lines` : "",
    };
  }

  // Only infer single-line CLI intents when there are no leading env-var assignments.
  if (lines.length === 1 && !/^[A-Z0-9_]+=/i.test(coreFirst)) {
    if (/^(?:\.?\.?\/?[\w./-]*\/)?(?:pytest|vitest|jest)\b|^(?:bun|npm|pnpm|yarn)\s+(?:run\s+)?test\b|^(?:cargo|go)\s+test\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Running tests", "Ran tests", "Tests failed"),
        detail: "",
      };
    }
    if (/^(?:\.?\.?\/?[\w./-]*\/)?(?:ruff|eslint|flake8)\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Running linter", "Ran linter", "Linter failed"),
        detail: "",
      };
    }
    if (/^(?:\.?\.?\/?[\w./-]*\/)?(?:tsc|basedpyright|pyright|mypy)\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Checking types", "Checked types", "Type check failed"),
        detail: "",
      };
    }
    if (/^(?:bun|npm|pnpm|yarn)\s+(?:run\s+)?build\b|^(?:vite|hatchling|cargo)\s+build\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Building project", "Built project", "Build failed"),
        detail: "",
      };
    }
    if (/^git\s+status\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Checking git status", "Checked git status", "Git status failed"),
        detail: "",
      };
    }
    if (/^git\s+diff\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Inspecting git diff", "Inspected git diff", "Git diff failed"),
        detail: "",
      };
    }
    if (/^git\s+log\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Checking git history", "Checked git history", "Git log failed"),
        detail: "",
      };
    }
    if (/^ffprobe\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Inspecting media", "Inspected media", "Media inspection failed"),
        detail: "",
      };
    }
    if (/^ffmpeg\b/i.test(coreFirst)) {
      return {
        kind: "tool",
        label: statusCopy(status, "Processing media", "Processed media", "Media processing failed"),
        detail: "",
      };
    }
  }

  return null;
}

function describeFileMutationTrace(
  args: string,
  status: GenericToolStatus,
  running: string,
  done: string,
  failed: string,
): TraceDescription {
  const path = traceFieldFromArgs(args, ["path", "file_path"]);
  return {
    kind: "tool",
    label: statusCopy(status, running, done, failed),
    detail: path ? safeActivityDetail(path) : "",
  };
}

function statusCopy(
  status: GenericToolStatus,
  running: string,
  done: string,
  failed: string,
): string {
  return status === "running" ? running : status === "error" ? failed : done;
}

function traceFieldFromArgs(args: string, keys: string[]): string {
  const compactArgs = args.trim();
  if (!compactArgs) return "";
  try {
    const parsed = JSON.parse(compactArgs) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return "";
    const record = parsed as Record<string, unknown>;
    for (const key of keys) {
      const value = record[key];
      if (typeof value === "string" && value.trim()) return value.trim();
    }
  } catch {
    return "";
  }
  return "";
}

function isShellTraceName(name: string): boolean {
  return [
    "exec",
    "exec_command",
    "execute_command",
    "run_command",
    "run_shell",
    "shell",
    "terminal",
    "bash",
    "sh",
  ].includes(name.toLowerCase().split(".").pop() || name.toLowerCase());
}

function shellCommandFromArgs(args: string): string {
  const compactArgs = args.trim();
  if (!compactArgs) return "";
  try {
    const parsed = JSON.parse(compactArgs) as unknown;
    if (typeof parsed === "string") return parsed;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return "";
    const record = parsed as Record<string, unknown>;
    for (const key of ["command", "cmd", "script", "input"]) {
      const value = record[key];
      if (typeof value === "string" && value.trim()) return value;
    }
  } catch {
    return compactArgs.replace(/^["']|["']$/g, "");
  }
  return "";
}

function humanizeTraceToolName(name: string): string {
  const words = name
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
  return words ? `${words[0].toUpperCase()}${words.slice(1)}` : "tool action";
}

function traceUrlFromArgs(args: string, fallback: string): URL | null {
  const candidates: string[] = [];
  const compactArgs = args.trim();
  if (compactArgs) {
    try {
      collectUrlCandidates(JSON.parse(compactArgs), candidates);
    } catch {
      candidates.push(compactArgs.replace(/^["']|["']$/g, ""));
    }
  }
  candidates.push(fallback);
  for (const candidate of candidates) {
    const url = parseSafeActivityHttpUrl(candidate);
    if (url) return url;
    const embedded = candidate.match(/https?:\/\/[^\s"'<>),]+/i)?.[0];
    if (embedded) {
      const embeddedUrl = parseSafeActivityHttpUrl(embedded);
      if (embeddedUrl) return embeddedUrl;
    }
  }
  return null;
}

function collectUrlCandidates(value: unknown, candidates: string[]) {
  if (typeof value === "string") {
    candidates.push(value);
    return;
  }
  if (!value || typeof value !== "object") return;
  if (Array.isArray(value)) {
    for (const item of value.slice(0, 6)) collectUrlCandidates(item, candidates);
    return;
  }
  const record = value as Record<string, unknown>;
  for (const key of ["url", "uri", "href", "link"]) {
    if (typeof record[key] === "string") candidates.push(record[key]);
  }
}
