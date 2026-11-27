import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Download,
  ExternalLink,
  FolderOpen,
  Loader2,
  Music,
  Play,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import { CodeBlock } from "@/components/CodeBlock";
import { MarkdownText } from "@/components/MarkdownText";
import { splitFilePath } from "@/components/FileReferenceChip";
import { MobileVideoPlayer } from "@/components/MobileVideoPlayer";
import { ThreadInspectorDialog } from "@/components/thread/ThreadInspector";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useMobileBack } from "@/hooks/useMobileBack";
import {
  ApiError,
  fetchFilePreview,
  openLocalFilePreview,
  type LocalFilePreviewAction,
  type WebUIMutationTransport,
} from "@/lib/api";
import { useMediaSource } from "@/lib/mobile-media";
import { isMobileRuntimeActive } from "@/lib/mobile-runtime";
import { isLinuxDesktopWebKit } from "@/lib/runtime";
import type { FilePreviewPayload } from "@/lib/types";
import { cn } from "@/lib/utils";

const LazyPdfPreview = lazy(() => import("@/components/PdfPreview"));

interface FilePreviewPanelProps {
  sessionKey: string;
  path: string;
  token: string;
  transport?: WebUIMutationTransport | null;
  refreshKey?: number;
  modal?: boolean;
  active?: boolean;
  visible?: boolean;
  onClose: () => void;
}

type PreviewState =
  | { status: "loading" }
  | { status: "error"; error: unknown }
  | { status: "ready"; payload: FilePreviewPayload };

function formatEntrySize(size: number): string {
  if (!Number.isFinite(size) || size <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

export function FilePreviewPanel({
  sessionKey,
  path,
  token,
  transport = null,
  refreshKey = 0,
  modal = true,
  active = true,
  visible = true,
  onClose,
}: FilePreviewPanelProps) {
  const { t } = useTranslation();
  const [state, setState] = useState<PreviewState>({ status: "loading" });
  const [actionBusy, setActionBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const [activePath, setActivePath] = useState(path);
  const [viewMode, setViewMode] = useState<"formatted" | "source">("formatted");
  const pathHistoryRef = useRef<string[]>([]);
  const tokenRef = useRef(token);
  tokenRef.current = token;

  useEffect(() => {
    pathHistoryRef.current = [];
    setActivePath(path);
  }, [path]);

  useEffect(() => {
    setViewMode("formatted");
    setActionError(null);
  }, [activePath]);

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    fetchFilePreview(tokenRef.current, sessionKey, activePath)
      .then((payload) => {
        if (!cancelled) setState({ status: "ready", payload });
      })
      .catch((error: unknown) => {
        if (!cancelled) setState({ status: "error", error });
      });
    return () => {
      cancelled = true;
    };
  }, [activePath, sessionKey, refreshKey]);

  const navigateToPath = (next: string) => {
    if (next === activePath) return;
    pathHistoryRef.current.push(activePath);
    setActivePath(next);
  };

  // Android back unwinds folder/file navigation before closing the preview.
  useMobileBack(() => {
    const previous = pathHistoryRef.current.pop();
    if (previous !== undefined) {
      setActivePath(previous);
      return true;
    }
    onClose();
    return true;
  }, active);

  const displayPath = state.status === "ready" ? state.payload.display_path : activePath;
  const previewPath = state.status === "ready" ? state.payload.path : displayPath;
  const normalizedPreviewPath = previewPath.replace(/\\/g, "/");
  const hasRootPrefix = normalizedPreviewPath.startsWith("/");
  const { name } = splitFilePath(displayPath);
  const fileName = name || displayPath;
  const pathParts = useMemo(
    () => normalizedPreviewPath.split("/").filter(Boolean),
    [normalizedPreviewPath],
  );
  const directoryParts = useMemo(
    () => (pathParts.length > 1 ? pathParts.slice(0, -1) : []),
    [pathParts],
  );
  const breadcrumbTitle = `${hasRootPrefix ? "/" : ""}${[
    ...directoryParts,
    fileName,
  ].join("/")}`;
  const payloadKind = state.status === "ready" ? state.payload.kind ?? "text" : "text";
  const rawMediaUrl = state.status === "ready" ? state.payload.media_url : undefined;
  const resolvedMediaUrl = useMediaSource(rawMediaUrl).src;
  const isDirectory = payloadKind === "directory";
  const isMarkdown = payloadKind === "text" && (
    (state.status === "ready" && state.payload.language === "markdown")
    || /\.(md|markdown)$/i.test(activePath)
  );
  const navTrail = useMemo(() => {
    const absolute = state.status === "ready" ? state.payload.path : activePath;
    const normalized = absolute.replace(/\\/g, "/");
    const parts = normalized.split("/").filter(Boolean);
    const prefixes: string[] = [];
    let accumulated = normalized.startsWith("/") ? "" : "";
    for (const part of parts) {
      accumulated = `${accumulated}/${part}`;
      prefixes.push(accumulated);
    }
    return { parts, prefixes };
  }, [activePath, state]);
  const visibleTrail = useMemo(() => {
    if (navTrail.parts.length <= 3) return navTrail.parts.map((part, index) => ({ part, index }));
    return navTrail.parts
      .slice(-3)
      .map((part, index) => ({ part, index: navTrail.parts.length - 3 + index }));
  }, [navTrail]);
  const canNavigateBack = isDirectory || activePath !== path;
  const canOpenLocally = (
    state.status === "ready"
    && state.payload.local_actions === true
    && transport !== null
    && !isMobileRuntimeActive()
  );
  const openWithApps = state.status === "ready" ? (state.payload.open_with_apps ?? []) : [];
  const defaultOpenApp = openWithApps.find((app) => app.default) ?? null;

  const triggerLocalOpen = async (action: LocalFilePreviewAction, appId?: string) => {
    if (!transport || actionBusy) return;
    setActionBusy(true);
    setActionError(null);
    try {
      await openLocalFilePreview(transport, {
        sessionKey,
        path: previewPath,
        action,
        appId,
      });
    } catch (error: unknown) {
      setActionError(
        error instanceof Error && error.message
          ? error.message
          : t("filePreview.openFailed", { defaultValue: "Could not open file on this device." }),
      );
    } finally {
      setActionBusy(false);
    }
  };

  const errorMessage = state.status === "error"
    ? (state.error instanceof ApiError
      ? (state.error.status === 404 && /API route not found/i.test(state.error.message)
        ? t("filePreview.routeMissing", {
          defaultValue: "File preview needs the latest gateway. Restart kohaku gateway and try again.",
        })
        : state.error.message)
      : t("filePreview.failed", { defaultValue: "Could not preview this file." }))
    : null;

  const content = (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col" data-testid="file-preview-panel">
      <div
        className="flex h-11 shrink-0 items-center gap-2 border-b border-border/60 px-3"
        title={previewPath}
      >
        {canNavigateBack ? (
          <button
            type="button"
            onClick={() => {
              const parent = activePath.replace(/\/[^/]+\/?$/, "") || "/";
              navigateToPath(parent);
            }}
            className={cn(
              "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
              "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            )}
            title={t("filePreview.back", { defaultValue: "Back to parent folder" })}
            aria-label={t("filePreview.back", { defaultValue: "Back to parent folder" })}
          >
            <ChevronLeft className="h-4 w-4" aria-hidden />
          </button>
        ) : null}
        <nav
          aria-label={t("filePreview.breadcrumb", { defaultValue: "File path" })}
          className="flex min-w-0 flex-1 items-center overflow-hidden text-sm leading-5"
          title={breadcrumbTitle}
          data-testid="file-preview-breadcrumb"
        >
          {navTrail.parts.length > visibleTrail.length ? (
            <>
              <span className="shrink-0 text-muted-foreground/55">...</span>
              <ChevronRight
                className="mx-1 h-3.5 w-3.5 shrink-0 text-muted-foreground/35"
                aria-hidden
              />
            </>
          ) : normalizedPreviewPath.startsWith("/") ? (
            <>
              <span className="shrink-0 text-muted-foreground/55">/</span>
              <ChevronRight
                className="mx-1 h-3.5 w-3.5 shrink-0 text-muted-foreground/35"
                aria-hidden
              />
            </>
          ) : null}
          {visibleTrail.map(({ part, index }, position) => {
            const isLast = position === visibleTrail.length - 1;
            const label = (
              <span
                className={cn(
                  "min-w-0 truncate rounded-mark px-1 py-0.5",
                  isLast
                    ? "font-medium text-foreground"
                    : "max-w-[26vw] shrink text-muted-foreground/78",
                )}
                data-testid={isLast ? "file-preview-title" : undefined}
              >
                {part}
              </span>
            );
            return (
              <span
                key={`${part}-${index}`}
                className="flex min-w-0 items-center overflow-hidden"
              >
                {position > 0 ? (
                  <ChevronRight
                    className="mx-1 h-3.5 w-3.5 shrink-0 text-muted-foreground/35"
                    aria-hidden
                  />
                ) : null}
                {isDirectory && !isLast ? (
                  <button
                    type="button"
                    onClick={() => navigateToPath(navTrail.prefixes[index])}
                    className="min-w-0 rounded-mark px-1 py-0.5 hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    title={navTrail.prefixes[index]}
                  >
                    {label}
                  </button>
                ) : (
                  label
                )}
              </span>
            );
          })}
        </nav>
        {isMarkdown ? (
          <div
            role="group"
            aria-label={t("filePreview.viewMode", { defaultValue: "Preview mode" })}
            className="flex shrink-0 items-center rounded-full border border-border/60 bg-muted/30 p-0.5 text-[11px] font-medium"
          >
            {(["formatted", "source"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setViewMode(mode)}
                aria-pressed={viewMode === mode}
                className={cn(
                  "rounded-full px-2 py-0.5 transition-colors",
                  viewMode === mode
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {mode === "formatted"
                  ? t("filePreview.formatted", { defaultValue: "Formatted" })
                  : t("filePreview.source", { defaultValue: "Source" })}
              </button>
            ))}
          </div>
        ) : null}
        {canOpenLocally ? (
          <button
            type="button"
            disabled={actionBusy}
            onClick={() => void triggerLocalOpen("reveal")}
            className={cn(
              "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
              "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            )}
            title={t("filePreview.openInFolder", { defaultValue: "Open in folder" })}
            aria-label={t("filePreview.openInFolder", { defaultValue: "Open in folder" })}
            data-testid="file-preview-open-folder"
          >
            <FolderOpen className="h-4 w-4" aria-hidden />
          </button>
        ) : null}
        {canOpenLocally && !isDirectory ? (
          <DropdownMenu modal={false}>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                disabled={actionBusy}
                className={cn(
                  "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
                  "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                )}
                title={t("filePreview.openWith", { defaultValue: "Open with" })}
                aria-label={t("filePreview.openWith", { defaultValue: "Open with" })}
                data-testid="file-preview-open-with"
              >
                <ExternalLink className="h-4 w-4" aria-hidden />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="min-w-48">
              <DropdownMenuLabel className="text-[11px] font-medium text-muted-foreground">
                {t("filePreview.openWith", { defaultValue: "Open with" })}
              </DropdownMenuLabel>
              <DropdownMenuItem onSelect={() => void triggerLocalOpen("open")}>
                {defaultOpenApp
                  ? t("filePreview.openInDefaultNamed", {
                      app: defaultOpenApp.name,
                      defaultValue: "{{app}} (Default)",
                    })
                  : t("filePreview.openInDefault", { defaultValue: "Open in default app" })}
              </DropdownMenuItem>
              {openWithApps.length > 0 ? (
                <>
                  {openWithApps.some((app) => !app.default) ? <DropdownMenuSeparator /> : null}
                  {openWithApps
                    .filter((app) => !app.default)
                    .map((app) => (
                      <DropdownMenuItem
                        key={app.id}
                        onSelect={() => void triggerLocalOpen("open_with", app.id)}
                      >
                        {app.name}
                      </DropdownMenuItem>
                    ))}
                </>
              ) : (
                <DropdownMenuItem onSelect={() => void triggerLocalOpen("open_with")}>
                  {t("filePreview.chooseApp", { defaultValue: "Choose application..." })}
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
        {state.status === "ready" && resolvedMediaUrl ? (
          <a
            href={resolvedMediaUrl}
            download={fileName}
            className={cn(
              "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
              "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            )}
            title={t("filePreview.download", { defaultValue: "Download file" })}
            aria-label={t("filePreview.download", { defaultValue: "Download file" })}
            data-testid="file-preview-download"
          >
            <Download className="h-4 w-4" aria-hidden />
          </a>
        ) : null}
        {modal ? (
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            className={cn(
              "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
              "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            )}
            title={t("filePreview.close", { defaultValue: "Close file preview" })}
            aria-label={t("filePreview.close", { defaultValue: "Close file preview" })}
            data-testid="file-preview-close"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        ) : null}
      </div>

      {actionError ? (
        <div
          role="alert"
          className="flex shrink-0 items-center justify-between gap-2 border-b border-destructive/25 bg-destructive/10 px-3 py-1.5 text-xs text-destructive"
        >
          <span className="min-w-0 truncate">{actionError}</span>
          <button
            type="button"
            onClick={() => setActionError(null)}
            className="shrink-0 rounded px-1 py-0.5 hover:bg-destructive/10"
            aria-label={t("filePreview.dismissError", { defaultValue: "Dismiss error" })}
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </button>
        </div>
      ) : null}

      <div
        className={cn(
          "min-h-0 flex-1",
          state.status === "ready" && state.payload.kind === "pdf"
            ? "overflow-hidden"
            : "overflow-auto",
        )}
      >
        {state.status === "loading" ? (
          <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            {t("filePreview.loading", { defaultValue: "Loading preview..." })}
          </div>
        ) : state.status === "error" ? (
          <div className="flex h-full items-center justify-center px-8 text-center text-sm text-muted-foreground">
            <div className="max-w-sm">
              <AlertCircle
                className="mx-auto mb-3 h-5 w-5 text-muted-foreground/70"
                aria-hidden
              />
              <p>{errorMessage}</p>
            </div>
          </div>
        ) : state.payload.kind === "image" && resolvedMediaUrl ? (
          <div className="flex h-full items-center justify-center bg-muted/20 p-3">
            <img
              src={resolvedMediaUrl}
              alt={fileName}
              className="max-h-full max-w-full rounded-control object-contain"
              loading="eager"
              decoding="async"
              draggable={false}
            />
          </div>
        ) : state.payload.kind === "video" && resolvedMediaUrl ? (
          <div
            className="flex h-full items-center justify-center bg-muted/20 p-3"
            data-testid="file-preview-video"
          >
            {isMobileRuntimeActive() ? (
              <MobileVideoPlayer
                key={resolvedMediaUrl}
                src={resolvedMediaUrl}
                label={fileName}
                containerClassName="flex h-full w-full items-center justify-center"
                className="max-h-full max-w-full rounded-control"
              />
            ) : isLinuxDesktopWebKit() ? (
              <DeferredPreviewVideo src={resolvedMediaUrl} label={fileName} />
            ) : (
              <StandardPreviewVideo src={resolvedMediaUrl} label={fileName} />
            )}
          </div>
        ) : state.payload.kind === "audio" && resolvedMediaUrl ? (
          <div
            className="flex h-full items-center justify-center bg-muted/20 p-6"
            data-testid="file-preview-audio"
          >
            <div className="w-full max-w-md space-y-4 rounded-xl border border-border/60 bg-background p-5 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-sky-500/10 text-sky-600 dark:text-sky-300">
                  <Music className="h-5 w-5" aria-hidden />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-foreground">{fileName}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatEntrySize(state.payload.size)}
                    {state.payload.mime ? ` · ${state.payload.mime}` : ""}
                  </p>
                </div>
              </div>
              <audio
                src={resolvedMediaUrl}
                controls
                preload="metadata"
                aria-label={fileName}
                className="w-full"
              />
            </div>
          </div>
        ) : state.payload.kind === "pdf" && resolvedMediaUrl ? (
          visible ? (
            <Suspense
              fallback={(
                <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                  {t("filePreview.loading", { defaultValue: "Loading preview..." })}
                </div>
              )}
            >
              <LazyPdfPreview src={resolvedMediaUrl} fileName={fileName} />
            </Suspense>
          ) : null
        ) : state.payload.kind === "image"
          || state.payload.kind === "video"
          || state.payload.kind === "audio"
          || state.payload.kind === "pdf" ? (
          <div className="flex h-full items-center justify-center px-8 text-center text-sm text-muted-foreground">
            <div className="max-w-sm">
              <AlertCircle
                className="mx-auto mb-3 h-5 w-5 text-muted-foreground/70"
                aria-hidden
              />
              <p>{t("filePreview.failed", { defaultValue: "Could not preview this file." })}</p>
            </div>
          </div>
        ) : state.payload.kind === "directory" ? (
          <ul className="min-h-full py-1" aria-label={t("filePreview.directory", { defaultValue: "Folder contents" })}>
            {(state.payload.entries ?? []).map((entry) => (
              <li key={entry.path}>
                <button
                  type="button"
                  onClick={() => navigateToPath(entry.path)}
                  className={cn(
                    "flex w-full items-center gap-2 px-4 py-2 text-left text-[13px]",
                    "transition-colors hover:bg-muted/55 focus-visible:bg-muted/55 focus-visible:outline-none",
                  )}
                >
                  <span className={cn("shrink-0", entry.kind === "directory" ? "text-sky-500" : "text-muted-foreground")}>
                    {entry.kind === "directory" ? "▸" : "·"}
                  </span>
                  <span className={cn("min-w-0 flex-1 truncate", entry.kind === "directory" && "font-medium")}>
                    {entry.name}
                  </span>
                  {entry.kind === "file" ? (
                    <span className="shrink-0 text-[11px] text-muted-foreground/70">
                      {formatEntrySize(entry.size)}
                    </span>
                  ) : null}
                </button>
              </li>
            ))}
            {(state.payload.entries ?? []).length === 0 ? (
              <li className="px-4 py-6 text-center text-sm text-muted-foreground">
                {t("filePreview.emptyFolder", { defaultValue: "This folder is empty." })}
              </li>
            ) : null}
          </ul>
        ) : isMarkdown && viewMode === "formatted" ? (
          <div className="min-h-full px-4 py-3 text-[13.5px] leading-6">
            {state.payload.truncated ? (
              <div className="mb-3 rounded-md border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-200">
                {t("filePreview.truncated", {
                  defaultValue: "Preview is truncated because this file is large.",
                })}
              </div>
            ) : null}
            <MarkdownText
              className={cn(
                "max-w-none break-words text-[13.5px] leading-6",
                "[&_pre]:max-w-full [&_pre]:overflow-x-auto",
                "[&_table]:block [&_table]:max-w-full [&_table]:overflow-x-auto",
              )}
            >
              {state.payload.content}
            </MarkdownText>
          </div>
        ) : (
          <div className="min-h-full">
            {state.payload.truncated ? (
              <div className="mx-4 mt-3 rounded-md border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-200">
                {t("filePreview.truncated", {
                  defaultValue: "Preview is truncated because this file is large.",
                })}
              </div>
            ) : null}
            <CodeBlock
              language={state.payload.language}
              code={state.payload.content}
              chrome="none"
              highlight
              showLineNumbers
              wrapLongLines={false}
              className="min-h-full"
            />
          </div>
        )}
      </div>
    </div>
  );
  if (!modal) return content;
  if (!visible) return null;
  return (
    <ThreadInspectorDialog focusRef={closeButtonRef} onClose={onClose}>
      <Dialog.Title className="sr-only">{t("filePreview.aria", { defaultValue: "File preview" })}</Dialog.Title>
      <Dialog.Description className="sr-only">{breadcrumbTitle}</Dialog.Description>
      {content}
    </ThreadInspectorDialog>
  );
}

function releaseVideoElement(video: HTMLVideoElement | null): void {
  if (!video?.hasAttribute("src")) return;
  video.removeAttribute("src");
  if (typeof video.load === "function") {
    video.load();
  }
}

function StandardPreviewVideo({ src, label }: { src: string; label: string }) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    return () => releaseVideoElement(video);
  }, [src]);

  return (
    <video
      ref={ref}
      src={src}
      controls
      playsInline
      preload="metadata"
      aria-label={label}
      className="max-h-full max-w-full rounded-control bg-black"
    />
  );
}

function DeferredPreviewVideo({ src, label }: { src: string; label: string }) {
  const { t } = useTranslation();
  const ref = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const showPlayer = status === "loading" || status === "ready";

  useEffect(() => {
    setStatus("idle");
    const video = ref.current;
    return () => releaseVideoElement(video);
  }, [src]);

  function play() {
    const video = ref.current;
    if (!video) return;
    // Attach the source and start playback within the user's click so Linux
    // desktop WebKitGTK does not allocate GStreamer pipelines until requested.
    video.src = src;
    setStatus("loading");
    video.focus();
    const pending = typeof video.play === "function" ? video.play() : undefined;
    void pending?.catch(() => {
      if (video.hasAttribute("src")) {
        setStatus((current) => (current === "loading" ? "ready" : current));
      }
    });
  }

  return (
    <div className="relative flex h-full w-full items-center justify-center">
      <video
        ref={ref}
        controls={showPlayer}
        playsInline
        preload="none"
        tabIndex={showPlayer ? 0 : -1}
        aria-label={label}
        aria-hidden={!showPlayer}
        onLoadedData={() => setStatus("ready")}
        onPlaying={() => setStatus("ready")}
        onWaiting={() => setStatus("loading")}
        onPause={() => setStatus((current) => (current === "loading" ? "ready" : current))}
        onError={() => setStatus("error")}
        className={cn(
          "max-h-full max-w-full rounded-control bg-black",
          !showPlayer && "pointer-events-none absolute opacity-0",
        )}
      />
      {status === "idle" ? (
        <button
          type="button"
          className={cn(
            "z-10 inline-flex items-center gap-2 rounded-control border border-border/60",
            "bg-background px-4 py-2 text-sm font-medium text-foreground shadow-sm",
            "transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          )}
          onClick={play}
        >
          <Play className="h-4 w-4" aria-hidden />
          {t("lightbox.playVideo", { defaultValue: "Play video" })}
        </button>
      ) : null}
      {status === "loading" ? (
        <Loader2
          className="pointer-events-none absolute h-5 w-5 animate-spin text-white/80"
          aria-hidden
        />
      ) : null}
      {status === "error" ? (
        <p role="status" className="text-center text-sm text-muted-foreground">
          {t("message.attachmentUnavailable", { defaultValue: "Attachment unavailable" })}
        </p>
      ) : null}
    </div>
  );
}
