import { describe, expect, it } from "vitest";

import { describeMcpActivity } from "@/components/thread/activity/mcp-activity-model";

describe("describeMcpActivity", () => {
  it.each([
    ["browser_navigate", { url: "https://example.com/docs" }, "done", "Opened", "example.com/docs"],
    ["browser_click", { element: "Submit" }, "running", "Clicking", "Submit"],
    ["browser_snapshot", {}, "done", "Inspected page", undefined],
    ["browser_screenshot", {}, "done", "Captured screenshot", undefined],
    ["browser_press_key", { key: "Enter" }, "error", "Could not press", "Enter"],
  ] as const)("turns %s into user-facing activity copy", (tool, args, status, action, target) => {
    expect(describeMcpActivity(tool, args, status)).toEqual({ action, target });
  });

  it("does not expose entered text in the activity timeline", () => {
    expect(describeMcpActivity(
      "browser_fill",
      { element: "Password", text: "not-for-the-timeline" },
      "done",
    )).toEqual({ action: "Entered text", target: "in Password" });
  });

  it("drops URL credentials and query parameters from browser activity", () => {
    expect(describeMcpActivity(
      "browser_navigate",
      { url: "https://user:password@example.com/docs?token=private#section" },
      "done",
    )).toEqual({ action: "Opened", target: "example.com/docs" });
  });

  it("humanizes unknown tool names instead of exposing function syntax", () => {
    expect(describeMcpActivity("browser_export_report", {}, "done")).toEqual({
      action: "Export report completed",
    });
  });

  it.each([
    ["generate_video", { workflow: "wan" }, "running", "Generating video", "wan"],
    ["generate_video", { workflow: "wan" }, "done", "Generated video", "wan"],
    ["generate_video", { workflow: "wan" }, "error", "Video generation failed", "wan"],
    ["generate_image", { mode: "t2i" }, "running", "Generating image", "t2i"],
    ["edit_image", {}, "done", "Edited image", undefined],
    ["comfy_status", {}, "running", "Checking ComfyUI status", undefined],
    ["media_workflow_catalog", {}, "done", "Checked media workflows", undefined],
    ["image_lora_catalog", {}, "done", "Checked LoRA catalog", undefined],
  ] as const)("describes the %s media tool as %s", (tool, args, status, action, target) => {
    expect(describeMcpActivity(tool, args, status)).toEqual({ action, target });
  });
});
