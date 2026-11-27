import { useEffect, useRef, useState } from "react";
import { AlertCircle, Download, Loader2, ZoomIn, ZoomOut } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  getDocument,
  GlobalWorkerOptions,
  type PDFDocumentLoadingTask,
  type PDFDocumentProxy,
  type PDFPageProxy,
  type RenderTask,
} from "pdfjs-dist/legacy/build/pdf.mjs";
import pdfWorkerUrl from "pdfjs-dist/legacy/build/pdf.worker.min.mjs?url";

import { isLinuxDesktopWebKit } from "@/lib/runtime";
import { cn } from "@/lib/utils";

GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

interface PdfPreviewProps {
  src: string;
  fileName: string;
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 2.5;
const ZOOM_STEP = 0.25;
const HORIZONTAL_PADDING_PX = 48;
const RESIZE_DEBOUNCE_MS = 150;
const RESIZE_DEADBAND_PX = 4;
// Cap each rendered canvas to ~3 MP (~12 MB RGBA) so multi-page PDFs never
// exhaust GPU/VRAM or WebKitGTK Skia backing stores.
const MAX_CANVAS_PIXELS = 3_145_728;

function releaseCanvas(canvas: HTMLCanvasElement | null): void {
  if (!canvas) return;
  if (canvas.width !== 0) canvas.width = 0;
  if (canvas.height !== 0) canvas.height = 0;
}

export default function PdfPreview({ src, fileName }: PdfPreviewProps) {
  const { t } = useTranslation();
  // Measure width from an outer non-scrolling container so scrollbar visibility
  // can never trigger a ResizeObserver oscillation loop.
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollRoot, setScrollRoot] = useState<HTMLDivElement | null>(null);
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [numPages, setNumPages] = useState(0);
  const [basePageSize, setBasePageSize] = useState<{ width: number; height: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [zoom, setZoom] = useState<number | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    setZoom(null);
  }, [src]);

  const hasError = status === "error";

  useEffect(() => {
    if (hasError) return;
    const node = containerRef.current;
    if (!node) return;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const applyWidth = (next: number) => {
      if (next <= 0) return;
      setContainerWidth((prev) => (
        prev === 0 || Math.abs(next - prev) >= RESIZE_DEADBAND_PX ? next : prev
      ));
    };
    applyWidth(node.clientWidth);
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      const next = node.clientWidth;
      if (next <= 0) return;
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        applyWidth(node.clientWidth);
      }, RESIZE_DEBOUNCE_MS);
    });
    observer.observe(node);
    return () => {
      if (timer !== null) clearTimeout(timer);
      observer.disconnect();
    };
  }, [hasError]);

  useEffect(() => {
    let cancelled = false;
    const abortController = new AbortController();
    let loadingTask: PDFDocumentLoadingTask | null = null;

    setStatus("loading");
    setDoc(null);
    setNumPages(0);
    setBasePageSize(null);

    async function loadPdf() {
      const response = await fetch(src, {
        signal: abortController.signal,
        credentials: "same-origin",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      if (cancelled) return;
      loadingTask = getDocument({ data: new Uint8Array(buffer) });
      const loaded = await loadingTask.promise;
      if (cancelled) return;
      const firstPage = await loaded.getPage(1);
      if (cancelled) return;
      const viewport = firstPage.getViewport({ scale: 1 });
      firstPage.cleanup();
      setBasePageSize({ width: viewport.width, height: viewport.height });
      setNumPages(loaded.numPages);
      setDoc(loaded);
      setStatus("ready");
    }

    void loadPdf().catch(() => {
      if (!cancelled) setStatus("error");
    });

    return () => {
      cancelled = true;
      abortController.abort();
      if (loadingTask) {
        // Destroying the loading task cleanly destroys both the PDFDocumentProxy
        // and its underlying worker without double-destroy races.
        void loadingTask.destroy().catch(() => {});
      }
    };
  }, [src]);

  const rawFitScale = containerWidth > 0 && basePageSize && basePageSize.width > 0
    ? Math.max(
        MIN_ZOOM,
        Math.min(MAX_ZOOM, (containerWidth - HORIZONTAL_PADDING_PX) / basePageSize.width),
      )
    : 1;
  const fitScale = Math.round(rawFitScale * 100) / 100;
  const effectiveScale = zoom ?? fitScale;

  const stepZoom = (delta: number) => {
    const base = zoom ?? fitScale;
    const next = Math.round(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base + delta)) * 100) / 100;
    setZoom(next);
  };

  if (status === "error") {
    return (
      <div
        className="flex h-full items-center justify-center px-8 text-center text-sm text-muted-foreground"
        data-testid="pdf-preview-error"
      >
        <div className="max-w-sm space-y-3">
          <AlertCircle className="mx-auto h-5 w-5 text-muted-foreground/70" aria-hidden />
          <p>{t("filePreview.pdfFailed", { defaultValue: "Could not display PDF inline." })}</p>
          <a
            href={src}
            download={fileName}
            className={cn(
              "inline-flex items-center gap-2 rounded-control border border-border/60",
              "bg-muted/40 px-3 py-1.5 text-xs font-medium text-foreground",
              "transition-colors hover:bg-muted",
            )}
          >
            <Download className="h-3.5 w-3.5" aria-hidden />
            {t("filePreview.download", { defaultValue: "Download file" })}
          </a>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex h-full min-h-0 flex-col overflow-hidden bg-muted/20"
      data-testid="pdf-preview"
    >
      <div
        className={cn(
          "flex shrink-0 items-center justify-between gap-2 border-b border-border/50",
          "bg-background px-3 py-1.5 text-xs text-muted-foreground",
        )}
      >
        <span>
          {numPages > 0
            ? t("filePreview.pdfPages", {
                count: numPages,
                defaultValue: numPages === 1 ? "1 page" : "{{count}} pages",
              })
            : t("filePreview.loading", { defaultValue: "Loading preview..." })}
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => stepZoom(-ZOOM_STEP)}
            disabled={status !== "ready" || effectiveScale <= MIN_ZOOM}
            className={cn(
              "inline-flex h-7 w-7 items-center justify-center rounded-md",
              "transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40",
            )}
            aria-label={t("filePreview.zoomOut", { defaultValue: "Zoom out" })}
            title={t("filePreview.zoomOut", { defaultValue: "Zoom out" })}
          >
            <ZoomOut className="h-3.5 w-3.5" aria-hidden />
          </button>
          <button
            type="button"
            onClick={() => setZoom(null)}
            disabled={status !== "ready"}
            className={cn(
              "min-w-[3.25rem] rounded-md px-1.5 py-1 text-center font-mono text-[11px]",
              "transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40",
            )}
            aria-label={t("filePreview.fitWidth", { defaultValue: "Fit to width" })}
            title={t("filePreview.fitWidth", { defaultValue: "Fit to width" })}
          >
            {Math.round(effectiveScale * 100)}%
          </button>
          <button
            type="button"
            onClick={() => stepZoom(ZOOM_STEP)}
            disabled={status !== "ready" || effectiveScale >= MAX_ZOOM}
            className={cn(
              "inline-flex h-7 w-7 items-center justify-center rounded-md",
              "transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40",
            )}
            aria-label={t("filePreview.zoomIn", { defaultValue: "Zoom in" })}
            title={t("filePreview.zoomIn", { defaultValue: "Zoom in" })}
          >
            <ZoomIn className="h-3.5 w-3.5" aria-hidden />
          </button>
        </div>
      </div>

      <div
        ref={setScrollRoot}
        className="min-h-0 flex-1 overflow-y-scroll overflow-x-auto [scrollbar-gutter:stable]"
      >
        {status === "loading" || !doc ? (
          <div className="flex h-full items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            {t("filePreview.loading", { defaultValue: "Loading preview..." })}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 p-4">
            {Array.from({ length: numPages }, (_, index) => (
              <PdfPageCanvas
                key={`${src}-${index + 1}`}
                doc={doc}
                pageNumber={index + 1}
                scale={effectiveScale}
                scrollRoot={scrollRoot}
                fallbackWidth={basePageSize ? Math.floor(basePageSize.width * effectiveScale) : 600}
                fallbackHeight={basePageSize ? Math.floor(basePageSize.height * effectiveScale) : 800}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function PdfPageCanvas({
  doc,
  pageNumber,
  scale,
  scrollRoot,
  fallbackWidth,
  fallbackHeight,
}: {
  doc: PDFDocumentProxy;
  pageNumber: number;
  scale: number;
  scrollRoot: HTMLDivElement | null;
  fallbackWidth: number;
  fallbackHeight: number;
}) {
  const holderRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const activeTaskRef = useRef<RenderTask | null>(null);
  const renderChainRef = useRef<Promise<void>>(Promise.resolve());
  const [inView, setInView] = useState(
    () => pageNumber <= 2 || typeof IntersectionObserver === "undefined",
  );
  const [renderedSize, setRenderedSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const node = holderRef.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry) return;
        setInView(entry.isIntersecting);
      },
      { root: scrollRoot, rootMargin: "300px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [scrollRoot]);

  useEffect(() => {
    let cancelled = false;
    const linuxWebKit = isLinuxDesktopWebKit();

    if (!inView) {
      activeTaskRef.current?.cancel();
      renderChainRef.current = renderChainRef.current
        .catch(() => {})
        .then(() => {
          if (!cancelled) {
            releaseCanvas(canvasRef.current);
          }
        });
      return () => {
        cancelled = true;
      };
    }

    const previousTask = activeTaskRef.current;
    if (previousTask) {
      previousTask.cancel();
    }

    const nextRender = renderChainRef.current
      .catch(() => {})
      .then(async () => {
        if (cancelled) return;
        let page: PDFPageProxy | null = null;
        try {
          page = await doc.getPage(pageNumber);
          if (cancelled) return;
          const canvas = canvasRef.current;
          if (!canvas) return;
          // Use software 2D canvas backing store on Linux WebKitGTK to avoid
          // AMDGPU/Mesa DMA-BUF surface exhaustion during PDF rendering.
          const context = canvas.getContext("2d", {
            alpha: false,
            willReadFrequently: linuxWebKit,
          });
          if (!context) return;

          const viewport = page.getViewport({ scale });
          const cssWidth = Math.max(1, Math.floor(viewport.width));
          const cssHeight = Math.max(1, Math.floor(viewport.height));
          const maxDpr = linuxWebKit ? 1.5 : 2;
          const baseDpr = Math.min(window.devicePixelRatio || 1, maxDpr);
          const pixelCapDpr = Math.sqrt(
            MAX_CANVAS_PIXELS / Math.max(1, viewport.width * viewport.height),
          );
          const dpr = Math.max(0.5, Math.min(baseDpr, pixelCapDpr));
          const backingWidth = Math.max(1, Math.floor(viewport.width * dpr));
          const backingHeight = Math.max(1, Math.floor(viewport.height * dpr));

          canvas.width = backingWidth;
          canvas.height = backingHeight;
          canvas.style.width = `${cssWidth}px`;
          canvas.style.height = `${cssHeight}px`;
          context.fillStyle = "#ffffff";
          context.fillRect(0, 0, backingWidth, backingHeight);
          setRenderedSize((prev) => (
            prev && prev.width === cssWidth && prev.height === cssHeight
              ? prev
              : { width: cssWidth, height: cssHeight }
          ));

          const task = page.render({
            canvasContext: context,
            viewport,
            transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined,
          });
          activeTaskRef.current = task;
          try {
            await task.promise;
          } finally {
            if (activeTaskRef.current === task) {
              activeTaskRef.current = null;
            }
          }
        } catch {
          // Cancellation during zoom, scroll-out, or unmount is expected.
        } finally {
          if (page && typeof page.cleanup === "function") {
            page.cleanup();
          }
        }
      });

    renderChainRef.current = nextRender;

    return () => {
      cancelled = true;
      activeTaskRef.current?.cancel();
    };
  }, [doc, pageNumber, scale, inView]);

  useEffect(() => {
    return () => {
      activeTaskRef.current?.cancel();
      const canvas = canvasRef.current;
      void renderChainRef.current.catch(() => {}).finally(() => {
        releaseCanvas(canvas);
      });
    };
  }, []);

  const width = renderedSize?.width ?? fallbackWidth;
  const height = renderedSize?.height ?? fallbackHeight;

  return (
    <div
      ref={holderRef}
      style={{ width, height }}
      className="max-w-full shrink-0 overflow-hidden rounded-sm bg-white shadow-sm ring-1 ring-border/40"
    >
      <canvas ref={canvasRef} className="block max-w-full" />
    </div>
  );
}
