import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GeneratedAssetsSheet } from "@/components/thread/GeneratedAssetsSheet";
import { setAppLanguage } from "@/i18n";
import type { SessionGeneratedAssetsPayload } from "@/lib/types";

function response(payload: SessionGeneratedAssetsPayload): Response {
  return {
    ok: true,
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => payload,
  } as Response;
}

function asset(id: string, name: string) {
  return {
    id,
    kind: "image" as const,
    mime: "image/png",
    name,
    url: `/api/media/sig/${id}`,
    path: `/home/test/media/generated/${name}`,
    bytes: 123,
    createdAt: 1_793_438_339_000,
    prompt: `${name} prompt`,
  };
}

const payload: SessionGeneratedAssetsPayload = {
  schemaVersion: 1,
  sessionKey: "websocket:chat-1",
  total: 3,
  groups: [
    {
      id: "turn-1|call-1",
      turnId: "turn-1",
      toolCallId: "call-1",
      toolName: "mcp_comfy_generate_image",
      sequence: 1,
      startedAt: 1_793_438_338_000,
      arguments: { prompt: "first batch prompt", workflow: "qwen_image_2_1" },
      artifacts: [asset("a1", "a1.png"), asset("a2", "a2.png")],
    },
    {
      id: "turn-1|call-2",
      turnId: "turn-1",
      toolCallId: "call-2",
      toolName: "mcp_comfy_edit_image",
      sequence: 2,
      startedAt: 1_793_438_339_000,
      arguments: { prompt: "second batch prompt" },
      artifacts: [asset("b1", "b1.png")],
    },
  ],
};

describe("GeneratedAssetsSheet", () => {
  beforeEach(async () => {
    await setAppLanguage("en");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(payload)));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows session assets by default and groups by tool call on request", async () => {
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining("/api/sessions/websocket%3Achat-1/generated-assets?limit=50"),
        expect.objectContaining({
          headers: { Authorization: "Bearer tok" },
        }),
      );
    });

    expect(screen.getByTestId("generated-assets-flat")).toBeInTheDocument();
    expect(await screen.findAllByTestId("generated-asset-thumbnail")).toHaveLength(3);

    await user.click(screen.getByRole("button", { name: "By tool call" }));

    expect(await screen.findByText("first batch prompt")).toBeInTheDocument();
    expect(screen.getByText("second batch prompt")).toBeInTheDocument();
    expect(screen.getAllByTestId("generated-assets-group")).toHaveLength(2);
    expect(screen.getByText("Candidates: 2")).toBeInTheDocument();
    expect(screen.getByText("Candidates: 1")).toBeInTheDocument();
    expect(screen.getByText("qwen_image_2_1")).toBeInTheDocument();
  });

  it("switches back to the session-assets view", async () => {
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));
    await user.click(screen.getByRole("button", { name: "By tool call" }));
    await user.click(screen.getByRole("button", { name: "Session assets" }));

    expect(screen.getByTestId("generated-assets-flat")).toBeInTheDocument();
    expect(screen.getAllByTestId("generated-asset-thumbnail")).toHaveLength(3);
    expect(screen.queryByTestId("generated-assets-groups")).not.toBeInTheDocument();
  });

  it("loads the next page of groups", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce(response({
          ...payload,
          total: 3,
          groups: [payload.groups[0]],
          next: "1",
        }))
        .mockResolvedValueOnce(response({
          ...payload,
          total: 3,
          groups: [payload.groups[1]],
          next: null,
        })),
    );
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));
    await user.click(screen.getByRole("button", { name: "By tool call" }));
    expect(await screen.findByText("first batch prompt")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Load more" }));

    expect(await screen.findByText("second batch prompt")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
    expect(fetch).toHaveBeenLastCalledWith(
      expect.stringContaining("cursor=1"),
      expect.any(Object),
    );
  });

  it("opens the lightbox from a thumbnail", async () => {
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));
    await user.click(await screen.findByRole("button", { name: "Open a1.png" }));

    expect(await screen.findAllByAltText("a1.png")).toHaveLength(2);
  });

  it("copies the media link and path from the viewer", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    try {
      render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);
      await user.click(screen.getByRole("button", { name: "Open generated assets" }));
      await user.click(await screen.findByRole("button", { name: "Open a1.png" }));

      await user.click(await screen.findByRole("button", { name: "Copy media link" }));
      expect(writeText).toHaveBeenLastCalledWith(
        expect.stringMatching(/\/api\/media\/sig\/a1$/),
      );

      await user.click(screen.getByRole("button", { name: "Copy file path for agent" }));
      expect(writeText).toHaveBeenLastCalledWith("/home/test/media/generated/a1.png");
    } finally {
      Reflect.deleteProperty(navigator, "clipboard");
    }
  });

  it("loads the cross-session asset library", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/generated-assets?")) {
          return Promise.resolve(response({
            ...payload,
            groups: [{ ...payload.groups[0], sessionKey: "websocket:lib" }],
          }));
        }
        return Promise.resolve(response(payload));
      }),
    );
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));
    await screen.findByTestId("generated-assets-flat");
    await user.click(screen.getByRole("button", { name: "All sessions" }));

    expect(await screen.findByTestId("generated-assets-library")).toBeInTheDocument();
    expect(screen.getByTitle("websocket:lib")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/generated-assets?limit=50"),
      expect.any(Object),
    );
  });

  it("renders and opens generated video assets", async () => {
    const videoGroup = {
      ...payload.groups[1],
      artifacts: [{
        id: "v1",
        kind: "video" as const,
        mime: "video/mp4",
        name: "clip.mp4",
        url: "/api/media/sig/v1",
        bytes: 100,
      }],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(response({
        ...payload,
        total: 1,
        groups: [videoGroup],
      })),
    );
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));
    const thumbnail = await screen.findByTestId("generated-asset-thumbnail");
    expect(thumbnail.querySelector("video")).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "Open clip.mp4" }));

    expect(await screen.findAllByLabelText("clip.mp4")).toHaveLength(3);
  });

  it("copies the media link and agent-usable path", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    try {
      render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);
      await user.click(screen.getByRole("button", { name: "Open generated assets" }));

      await user.click(await screen.findByRole("button", { name: "Asset actions: a1.png" }));
      await user.click(await screen.findByRole("menuitem", { name: "Copy media link" }));
      expect(writeText).toHaveBeenLastCalledWith(
        expect.stringMatching(/\/api\/media\/sig\/a1$/),
      );

      await user.click(screen.getByRole("button", { name: "Asset actions: a1.png" }));
      await user.click(await screen.findByRole("menuitem", { name: "Copy file path for agent" }));
      expect(writeText).toHaveBeenLastCalledWith("/home/test/media/generated/a1.png");
      expect(await screen.findByText("Path copied")).toBeInTheDocument();
    } finally {
      Reflect.deleteProperty(navigator, "clipboard");
    }
  });

  it("shows an empty state", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(response({ ...payload, total: 0, groups: [] })),
    );
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));

    expect(await screen.findByText("No generated assets in this session yet.")).toBeInTheDocument();
  });

  it("retries after a failed load", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockRejectedValueOnce(new Error("offline"))
        .mockResolvedValueOnce(response(payload)),
    );
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));
    expect(await screen.findByText("Could not load generated assets.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByTestId("generated-assets-flat")).toBeInTheDocument();
  });
});
