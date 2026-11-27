import { useEffect, useRef, useState } from "react";
import { AlertCircle, Download, Loader2, ZoomIn, ZoomOut } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  getDocument,
  GlobalWorkerOptions,
  type PDFDocumentLoadingTask,
  type PDFDocumentProxy,
  type RenderTask,
} from "pdfjs-dist/legacy/build/pdf.mjs";
import pdfWorkerUrl from "pdfjs-dist/legacy/build/pdf.worker.min.mjs?url";

import { cn } from "@/lib/utils";

GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

interface PdfPreviewProps {
  src: string;
  fileName: string;
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 2.5;
const ZOOM_STEP = 0.25;
const HORIZONTAL_PADDING_PX = 32;

export default function PdfPreview({ src, fileName }: PdfPreviewProps) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [numPages, setNumPages] = useState(0);
  const [basePageSize, setBasePageSize] = useState<{ width: number; height: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [zoom, setZoom] = useState<number | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    setZoom(null);
  }, [src]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const updateWidth = () => {
      const next = node.clientWidth;
      if (next > 0) setContainerWidth(next);
    };
    updateWidth();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => updateWidth());
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    const abortController = new AbortController();
    let loadingTask: PDFDocumentLoadingTask | null = null;
    let activeDoc: PDFDocumentProxy | null = null;

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
      if (cancelled) {
        await loaded.destroy();
        return;
      }
      activeDoc = loaded;
      const firstPage = await loaded.getPage(1);
      if (cancelled) {
        await loaded.destroy();
        return;
      }
      const viewport = firstPage.getViewport({ scale: 1 });
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
        void loadingTask.destroy().catch(() => {});
      }
      if (activeDoc) {
        void activeDoc.destroy().catch(() => {});
      }
    };
  }, [src]);

  const fitScale = containerWidth > 0 && basePageSize && basePageSize.width > 0
    ? Math.max(
        MIN_ZOOM,
        Math.min(MAX_ZOOM, (containerWidth - HORIZONTAL_PADDING_PX) / basePageSize.width),
      )
    : 1;
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
    <div ref={containerRef} className="flex min-h-full flex-col bg-muted/20" data-testid="pdf-preview">
      <div
        className={cn(
          "sticky top-0 z-10 flex items-center justify-between gap-2 border-b border-border/50",
          "bg-background/90 px-3 py-1.5 text-xs text-muted-foreground backdrop-blur-sm",
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

      {status === "loading" || !doc ? (
        <div className="flex flex-1 items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
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
              fallbackWidth={basePageSize ? Math.floor(basePageSize.width * effectiveScale) : 600}
              fallbackHeight={basePageSize ? Math.floor(basePageSize.height * effectiveScale) : 800}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function PdfPageCanvas({
  doc,
  pageNumber,
  scale,
  fallbackWidth,
  fallbackHeight,
}: {
  doc: PDFDocumentProxy;
  pageNumber: number;
  scale: number;
  fallbackWidth: number;
  fallbackHeight: number;
}) {
  const holderRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [visible, setVisible] = useState(typeof IntersectionObserver === "undefined");
  const [renderedSize, setRenderedSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    if (visible) return;
    const node = holderRef.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry?.isIntersecting) return;
        observer.disconnect();
        setVisible(true);
      },
      { rootMargin: "200px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    let activeTask: RenderTask | null = null;

    async function renderPage() {
      const page = await doc.getPage(pageNumber);
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const context = canvas.getContext("2d");
      if (!context) return;

      const viewport = page.getViewport({ scale });
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssWidth = Math.max(1, Math.floor(viewport.width));
      const cssHeight = Math.max(1, Math.floor(viewport.height));
      canvas.width = Math.max(1, Math.floor(viewport.width * dpr));
      canvas.height = Math.max(1, Math.floor(viewport.height * dpr));
      canvas.style.width = `${cssWidth}px`;
      canvas.style.height = `${cssHeight}px`;
      setRenderedSize({ width: cssWidth, height: cssHeight });

      activeTask = page.render({
        canvasContext: context,
        viewport,
        transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined,
      });
      await activeTask.promise;
    }

    void renderPage().catch(() => {
      // Cancellation during zoom or unmount is expected.
    });

    return () => {
      cancelled = true;
      if (activeTask) {
        activeTask.cancel();
      }
    };
  }, [doc, pageNumber, scale, visible]);

  const width = renderedSize?.width ?? fallbackWidth;
  const height = renderedSize?.height ?? fallbackHeight;

  return (
    <div
      ref={holderRef}
      style={{ width, height }}
      className="max-w-full overflow-hidden rounded-sm bg-white shadow-sm ring-1 ring-border/40"
    >
      <canvas ref={canvasRef} className="block max-w-full" />
    </div>
  );
}
