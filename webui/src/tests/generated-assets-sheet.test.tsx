import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GeneratedAssetsPanel, GeneratedAssetsSheet } from "@/components/thread/GeneratedAssetsSheet";
import { setAppLanguage } from "@/i18n";
import {
  createMobileRuntime, installMobileRuntime, resetMobileRuntime, type MobileTransport,
} from "@/lib/mobile-runtime";
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

const libraryClips = ["one", "two", "three"].map((clip, index) => ({
  ...asset(clip, `clip-${clip}.mp4`),
  kind: "video" as const,
  mime: "video/mp4",
  createdAt: 1_793_438_339_000 - index,
}));

const videoLibrary: SessionGeneratedAssetsPayload = {
  ...payload,
  groups: [{ ...payload.groups[0], sessionKey: payload.sessionKey, artifacts: libraryClips }],
};

describe("GeneratedAssetsSheet", () => {
  beforeEach(async () => {
    await setAppLanguage("en");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(payload)));
  });

  afterEach(() => {
    resetMobileRuntime();
    vi.restoreAllMocks();
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
    expect(screen.getByText("2 loaded")).toBeInTheDocument();
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
            sessionKey: "*",
            total: 2,
            totalScope: "page",
            next: '["websocket:next",0]',
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
    expect(screen.getByText("2 loaded")).toBeInTheDocument();
    expect(screen.getByTitle("websocket:lib")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/generated-assets?limit=50"),
      expect.any(Object),
    );

    vi.mocked(fetch).mockResolvedValueOnce(response({
      ...payload,
      sessionKey: "*",
      total: 1,
      totalScope: "page",
      next: null,
      groups: [{
        ...payload.groups[1],
        sessionKey: "websocket:lib",
        artifacts: [{
          ...asset("v1", "clip.mp4"),
          kind: "video",
          mime: "video/mp4",
        }],
      }],
    }));
    await user.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("Assets: 3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open clip.mp4" })).toBeInTheDocument();
  });

  it("renders newest-first session assets first in flat and call views", async () => {
    const newestFirst = {
      ...payload,
      groups: [...payload.groups].reverse().map((group, index) => ({
        ...group,
        sequence: payload.groups.length - index,
      })),
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(newestFirst)));
    const user = userEvent.setup();
    render(<GeneratedAssetsSheet sessionKey="websocket:chat-1" token="tok" />);

    await user.click(screen.getByRole("button", { name: "Open generated assets" }));

    const flatThumbs = await screen.findAllByTestId("generated-asset-thumbnail");
    expect(
      flatThumbs.map((thumb) => thumb.querySelector("button")?.getAttribute("aria-label")),
    ).toEqual(["Open b1.png", "Open a1.png", "Open a2.png"]);

    await user.click(screen.getByRole("button", { name: "By tool call" }));

    const groups = await screen.findAllByTestId("generated-assets-group");
    expect(groups[0]).toHaveTextContent("second batch prompt");
    expect(groups[1]).toHaveTextContent("first batch prompt");
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
    expect(screen.getByText("Assets: 1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Open clip.mp4" }));

    expect(await screen.findAllByLabelText("clip.mp4")).toHaveLength(3);
  });

  it("defers desktop WebKit library playback and releases a clip when leaving it", async () => {
    vi.stubGlobal("__TAURI_INTERNALS__", {});
    vi.spyOn(navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 Safari/605.1.15",
    );
    vi.mocked(fetch).mockResolvedValue(response(videoLibrary));
    vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<GeneratedAssetsPanel sessionKey={payload.sessionKey} token="tok" modal={false} />);

    await user.click(screen.getByRole("button", { name: "All sessions" }));
    const tiles = await Promise.all(libraryClips.map(({ name }) =>
      screen.findByRole("button", { name: `Open ${name}` })));
    expect(document.querySelectorAll("video[src]")).toHaveLength(0);

    await user.click(tiles[0]);
    expect(screen.getByRole("group", { name: "All images" }).querySelectorAll("button")).toHaveLength(3);
    expect(document.querySelectorAll("video[src]")).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Play video" }));
    const first = document.querySelector("video[src]");
    expect(first).toHaveAttribute("src", libraryClips[0].url);
    expect(document.querySelectorAll("video[src]")).toHaveLength(1);

    for (const clip of libraryClips.slice(1)) {
      await user.click(screen.getByRole("button", { name: "Next image" }));
      await screen.findByRole("dialog", { name: clip.name });
      expect(first).not.toHaveAttribute("src");
      expect(document.querySelectorAll("video[src]")).toHaveLength(0);
    }

    await user.click(screen.getByRole("button", { name: "Play video" }));
    const last = document.querySelector("video[src]");
    expect(last).toHaveAttribute("src", libraryClips[2].url);
    expect(document.querySelectorAll("video[src]")).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: "Close image preview" }));
    expect(last).not.toHaveAttribute("src");
    expect(document.querySelectorAll("video[src]")).toHaveLength(0);
  });

  it("shows mobile library and filmstrip frames and a poster before playback", async () => {
    vi.stubGlobal("IntersectionObserver", undefined);
    const runtime = createMobileRuntime({} as MobileTransport);
    vi.spyOn(runtime, "request").mockResolvedValue({
      requestId: 1,
      status: 200,
      headers: {},
      bodyBase64: btoa(JSON.stringify(videoLibrary)),
    });
    installMobileRuntime(runtime);
    const poster = "data:image/jpeg;base64,cG9zdGVy";
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      { drawImage: vi.fn() } as unknown as CanvasRenderingContext2D,
    );
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue(poster);
    const user = userEvent.setup();
    render(<GeneratedAssetsPanel sessionKey={payload.sessionKey} token="tok" modal={false} />);

    function showFrame(video: Element) {
      Object.defineProperties(video, { videoWidth: { value: 320 }, videoHeight: { value: 180 } });
      fireEvent.loadedData(video);
    }

    async function showFrames(container: HTMLElement) {
      for (const clip of libraryClips) {
        const video = await waitFor(() => {
          const element = container.querySelector(`video[src="${clip.url}"]`);
          if (!element) throw new Error(`Waiting for ${clip.name}`);
          return element;
        });
        showFrame(video);
      }
      expect(container.querySelectorAll("canvas[width='320'][height='180']")).toHaveLength(3);
    }

    await user.click(screen.getByRole("button", { name: "All sessions" }));
    await showFrames(await screen.findByTestId("generated-assets-library"));
    await user.click(screen.getByRole("button", { name: "Open clip-one.mp4" }));
    await showFrames(screen.getByRole("group", { name: "All images" }));

    const dialog = screen.getByRole("dialog", { name: "clip-one.mp4" });
    const player = dialog.querySelector("video[controls]");
    const posterFrame = dialog.querySelector("video[aria-hidden][preload='auto']");
    expect(posterFrame).not.toBeNull();
    showFrame(posterFrame!);
    expect(player).toHaveAttribute("poster", poster);
    expect(player).toHaveAttribute("preload", "none");
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

describe("GeneratedAssetsPanel", () => {
  beforeEach(async () => {
    await setAppLanguage("en");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(payload)));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads assets without a sheet wrapper and opens the lightbox from it", async () => {
    const user = userEvent.setup();
    render(<GeneratedAssetsPanel sessionKey="websocket:chat-1" token="tok" modal={false} />);

    expect(await screen.findByTestId("generated-assets-flat")).toBeInTheDocument();
    expect(await screen.findAllByTestId("generated-asset-thumbnail")).toHaveLength(3);
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/sessions/websocket%3Achat-1/generated-assets?limit=50"),
      expect.objectContaining({ headers: { Authorization: "Bearer tok" } }),
    );

    await user.click(screen.getByRole("button", { name: "Open a1.png" }));
    expect(await screen.findAllByAltText("a1.png")).toHaveLength(2);
  });

  it("stays idle until the tab becomes active", async () => {
    const { rerender } = render(
      <GeneratedAssetsPanel sessionKey="websocket:chat-1" token="tok" modal={false} active={false} />,
    );
    expect(screen.queryByTestId("generated-asset-thumbnail")).not.toBeInTheDocument();
    await waitFor(() => expect(fetch).not.toHaveBeenCalled());

    rerender(
      <GeneratedAssetsPanel sessionKey="websocket:chat-1" token="tok" modal={false} active />,
    );
    expect(await screen.findByTestId("generated-assets-flat")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("renders a dialog with a close control in modal mode", async () => {
    const onClose = vi.fn();
    render(
      <GeneratedAssetsPanel
        sessionKey="websocket:chat-1"
        token="tok"
        modal
        visible
        onClose={onClose}
      />,
    );

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close generated assets" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("fills the docked inspector tabpanel width with flex-1 and min-w-0", async () => {
    render(
      <GeneratedAssetsPanel sessionKey="websocket:chat-1" token="tok" modal={false} active />,
    );
    const panel = await screen.findByTestId("generated-assets-panel");
    expect(panel.className).toContain("flex-1");
    expect(panel.className).toContain("min-w-0");
  });
});
