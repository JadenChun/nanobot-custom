import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { FilePreviewPanel } from "@/components/FilePreviewPanel";
import { setAppLanguage } from "@/i18n";
import { fetchFilePreview } from "@/lib/api";
import { isLinuxDesktopWebKit } from "@/lib/runtime";

vi.mock("@/components/CodeBlock", () => ({
  CodeBlock: ({
    code,
    language,
    highlight,
  }: {
    code: string;
    language?: string;
    highlight?: boolean;
  }) => (
    <pre
      data-testid="mock-code-block"
      data-language={language}
      data-highlight={String(highlight)}
    >
      {code}
    </pre>
  ),
}));

vi.mock("@/components/PdfPreview", () => ({
  default: ({ src, fileName }: { src: string; fileName: string }) => (
    <div data-testid="mock-pdf-preview" data-src={src} data-filename={fileName} />
  ),
}));

vi.mock("@/lib/runtime", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/runtime")>();
  return {
    ...actual,
    isLinuxDesktopWebKit: vi.fn(() => false),
  };
});

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchFilePreview: vi.fn(),
  };
});

describe("FilePreviewPanel", () => {
  beforeEach(async () => {
    await setAppLanguage("en");
    vi.mocked(fetchFilePreview).mockReset();
    vi.mocked(isLinuxDesktopWebKit).mockReturnValue(false);
  });

  it("shows a compact breadcrumb with one file name and a visible close action", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    vi.mocked(fetchFilePreview).mockResolvedValue({
      path: "/Users/hr/workspace/quicksort.py",
      display_path: "quicksort.py",
      language: "python",
      content: "print('ok')",
      truncated: false,
    });

    render(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="quicksort.py"
        token="tok"
        onClose={onClose}
      />,
    );

    const codeBlock = await screen.findByTestId("mock-code-block");
    expect(codeBlock).toHaveTextContent("print('ok')");
    expect(codeBlock).toHaveAttribute("data-language", "python");
    expect(codeBlock).toHaveAttribute("data-highlight", "true");
    expect(screen.getByTestId("file-preview-breadcrumb")).toHaveTextContent("...");
    expect(screen.getByTestId("file-preview-breadcrumb")).toHaveTextContent("workspace");
    expect(screen.getByTestId("file-preview-title")).toHaveTextContent("quicksort.py");
    expect(screen.getAllByText("quicksort.py")).toHaveLength(1);

    const closeButton = screen.getByRole("button", { name: "Close file preview" });
    expect(closeButton).toBeVisible();

    await user.click(closeButton);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("updates translated chrome without refetching the open file", async () => {
    vi.mocked(fetchFilePreview).mockResolvedValue({
      path: "/workspace/notes.txt",
      display_path: "notes.txt",
      language: "text",
      content: "plain notes",
      truncated: false,
    });

    render(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="notes.txt"
        token="tok"
        onClose={() => {}}
      />,
    );

    await screen.findByTestId("mock-code-block");
    expect(fetchFilePreview).toHaveBeenCalledTimes(1);

    await act(async () => {
      await setAppLanguage("zh-CN");
    });

    expect(fetchFilePreview).toHaveBeenCalledTimes(1);
  });

  it("refetches only when the refresh key changes for the same preview", async () => {
    vi.mocked(fetchFilePreview)
      .mockResolvedValueOnce({
        path: "/workspace/notes.txt",
        display_path: "notes.txt",
        language: "text",
        content: "notes v1",
        truncated: false,
      })
      .mockResolvedValueOnce({
        path: "/workspace/notes.txt",
        display_path: "notes.txt",
        language: "text",
        content: "notes v2",
        truncated: false,
      });

    const { rerender } = render(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="notes.txt"
        token="tok"
        refreshKey={0}
        onClose={() => {}}
      />,
    );

    const codeBlock = await screen.findByTestId("mock-code-block");
    expect(codeBlock).toHaveTextContent("notes v1");
    expect(fetchFilePreview).toHaveBeenCalledTimes(1);

    rerender(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="notes.txt"
        token="tok"
        refreshKey={1}
        onClose={() => {}}
      />,
    );

    await screen.findByText("notes v2");
    expect(screen.getByTestId("mock-code-block")).toHaveTextContent("notes v2");
    expect(fetchFilePreview).toHaveBeenCalledTimes(2);

    rerender(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="notes.txt"
        token="tok"
        refreshKey={1}
        onClose={() => {}}
      />,
    );

    expect(fetchFilePreview).toHaveBeenCalledTimes(2);
    expect(screen.queryByTestId("file-preview-download")).not.toBeInTheDocument();
  });

  it("renders image preview with header download link", async () => {
    vi.mocked(fetchFilePreview).mockResolvedValue({
      kind: "image",
      path: "/workspace/chart.png",
      display_path: "chart.png",
      project_path: "/workspace",
      language: "",
      content: "",
      size: 2048,
      truncated: false,
      mime: "image/png",
      media_url: "/api/media/sig/chart",
    });

    render(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="chart.png"
        token="tok"
        onClose={() => {}}
      />,
    );

    const img = await screen.findByRole("img", { name: "chart.png" });
    expect(img).toHaveAttribute("src", "/api/media/sig/chart");
    const download = screen.getByTestId("file-preview-download");
    expect(download).toHaveAttribute("href", "/api/media/sig/chart");
    expect(download).toHaveAttribute("download", "chart.png");
  });

  it("renders standard video preview and defers video loading on Linux desktop WebKit", async () => {
    const user = userEvent.setup();
    vi.mocked(fetchFilePreview).mockResolvedValue({
      kind: "video",
      path: "/workspace/demo.mp4",
      display_path: "demo.mp4",
      project_path: "/workspace",
      language: "",
      content: "",
      size: 4096,
      truncated: false,
      mime: "video/mp4",
      media_url: "/api/media/sig/demo",
    });

    const { rerender } = render(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="demo.mp4"
        token="tok"
        onClose={() => {}}
      />,
    );

    const container = await screen.findByTestId("file-preview-video");
    const video = container.querySelector("video");
    expect(video).not.toBeNull();
    expect(video).toHaveAttribute("src", "/api/media/sig/demo");
    expect(video).toHaveAttribute("controls");
    expect(screen.getByTestId("file-preview-download")).toHaveAttribute("href", "/api/media/sig/demo");

    vi.mocked(isLinuxDesktopWebKit).mockReturnValue(true);
    rerender(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="demo-webkit.mp4"
        token="tok"
        onClose={() => {}}
      />,
    );

    const playButton = await screen.findByRole("button", { name: "Play video" });
    const deferredContainer = screen.getByTestId("file-preview-video");
    const deferredVideo = deferredContainer.querySelector("video");
    expect(deferredVideo).not.toHaveAttribute("src");

    await user.click(playButton);
    expect(deferredVideo).toHaveAttribute("src", "/api/media/sig/demo");
  });

  it("renders audio preview card with metadata and controls", async () => {
    vi.mocked(fetchFilePreview).mockResolvedValue({
      kind: "audio",
      path: "/workspace/voice.m4a",
      display_path: "voice.m4a",
      project_path: "/workspace",
      language: "",
      content: "",
      size: 10240,
      truncated: false,
      mime: "audio/mp4",
      media_url: "/api/media/sig/voice",
    });

    render(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="voice.m4a"
        token="tok"
        onClose={() => {}}
      />,
    );

    const container = await screen.findByTestId("file-preview-audio");
    const audio = container.querySelector("audio");
    expect(audio).not.toBeNull();
    expect(audio).toHaveAttribute("src", "/api/media/sig/voice");
    expect(audio).toHaveAttribute("controls");
    expect(container).toHaveTextContent("10 KB");
    expect(container).toHaveTextContent("audio/mp4");
    expect(screen.getByTestId("file-preview-download")).toHaveAttribute("href", "/api/media/sig/voice");
  });

  it("lazy-loads PdfPreview for pdf payloads with header download link", async () => {
    vi.mocked(fetchFilePreview).mockResolvedValue({
      kind: "pdf",
      path: "/home/jaden/Documents/Freelance/Records/T&C/company website/docs/INV-000009.pdf",
      display_path: "INV-000009.pdf",
      project_path: "/home/jaden/Documents/Freelance",
      language: "",
      content: "",
      size: 878937,
      truncated: false,
      mime: "application/pdf",
      media_url: "/api/media/sig/inv9",
    });

    render(
      <FilePreviewPanel
        sessionKey="websocket:chat-1"
        path="INV-000009.pdf"
        token="tok"
        onClose={() => {}}
      />,
    );

    const pdf = await screen.findByTestId("mock-pdf-preview");
    expect(pdf).toHaveAttribute("data-src", "/api/media/sig/inv9");
    expect(pdf).toHaveAttribute("data-filename", "INV-000009.pdf");
    const download = screen.getByTestId("file-preview-download");
    expect(download).toHaveAttribute("href", "/api/media/sig/inv9");
    expect(download).toHaveAttribute("download", "INV-000009.pdf");
  });
});
