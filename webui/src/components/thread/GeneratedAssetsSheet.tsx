import { useMemo, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import {
  Check,
  CircleAlert,
  FileText,
  Images,
  Link2,
  Loader2,
  MoreHorizontal,
  Play,
  RefreshCcw,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import { ImageLightbox } from "@/components/ImageLightbox";
import { ThreadInspectorDialog } from "@/components/thread/ThreadInspector";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { SegmentedControl } from "@/components/ui/segmented-control";
import {
  Sheet,
  SheetContent,
} from "@/components/ui/sheet";
import { useGeneratedAssetLibrary } from "@/hooks/useGeneratedAssetLibrary";
import { useMobileBack } from "@/hooks/useMobileBack";
import { useSessionGeneratedAssets } from "@/hooks/useSessionGeneratedAssets";
import { copyTextToClipboard } from "@/lib/clipboard";
import { fmtDateTime } from "@/lib/format";
import { mediaLink } from "@/lib/media-links";
import { useMobileMediaUrl } from "@/lib/mobile-media";
import type { GeneratedAsset, GeneratedAssetGroup, UIImage } from "@/lib/types";
import { cn } from "@/lib/utils";

type ViewMode = "call" | "flat" | "library";

interface GeneratedAssetsSheetProps {
  sessionKey: string;
  token: string;
  refreshKey?: string | number;
}

export interface GeneratedAssetsPanelProps extends GeneratedAssetsSheetProps {
  /** Fetch and show assets while true; keeps idle state when false. */
  active?: boolean;
  /** True when the panel fills a modal/sheet surface instead of a docked pane. */
  modal?: boolean;
  /** True when the modal surface is the selected inspector tab. */
  visible?: boolean;
  /** Render the title with the Radix dialog primitive (inside a Sheet or dialog). */
  dialogTitle?: boolean;
  /** Renders a close control; required for modal surfaces. */
  onClose?: () => void;
}

interface MediaGroup extends GeneratedAssetGroup {
  assets: GeneratedAsset[];
}

interface FlatAsset {
  asset: GeneratedAsset;
  group: MediaGroup;
  candidate: number;
}

interface LightboxState {
  images: UIImage[];
  index: number;
}

interface DatedAsset {
  asset: GeneratedAsset;
  createdAt: number;
  startedAt: number;
}

export function GeneratedAssetsPanel({
  sessionKey,
  token,
  refreshKey = 0,
  active = true,
  modal = true,
  visible = true,
  dialogTitle = false,
  onClose,
}: GeneratedAssetsPanelProps) {
  const { i18n, t } = useTranslation();
  const [view, setView] = useState<ViewMode>("flat");
  const [lightbox, setLightbox] = useState<LightboxState | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const {
    groups,
    next,
    loading,
    loadingMore,
    loadFailed,
    refresh,
    loadMore,
  } = useSessionGeneratedAssets(active, token, sessionKey, refreshKey);
  const {
    groups: libraryGroups,
    next: libraryNext,
    loading: libraryLoading,
    loadingMore: libraryLoadingMore,
    loadFailed: libraryLoadFailed,
    refresh: refreshLibrary,
    loadMore: loadMoreLibrary,
  } = useGeneratedAssetLibrary(active && view === "library", token, refreshKey);
  const locale = i18n.resolvedLanguage || i18n.language;

  // Android back closes the full-screen panel before the chat behind it.
  useMobileBack(() => {
    onClose?.();
    return true;
  }, modal && visible && Boolean(onClose));

  const mediaGroups = useMemo<MediaGroup[]>(
    () => groups
      .map((group) => ({
        ...group,
        assets: group.artifacts.filter(
          (artifact) => artifact.kind === "image" || artifact.kind === "video",
        ),
      }))
      .filter((group) => group.assets.length > 0),
    [groups],
  );
  const flatAssets = useMemo<FlatAsset[]>(
    () => mediaGroups.flatMap((group) => group.assets.map((asset, index) => ({
      asset,
      group,
      candidate: index + 1,
    }))),
    [mediaGroups],
  );
  const totalAssets = flatAssets.length;
  const librarySessions = useMemo(() => {
    const bySession = new Map<string, DatedAsset[]>();
    for (const group of libraryGroups) {
      const sessionKey = group.sessionKey || "unknown";
      const startedAt = group.startedAt ?? 0;
      const assets = bySession.get(sessionKey) ?? [];
      for (const asset of group.artifacts) {
        if (asset.kind !== "image" && asset.kind !== "video") continue;
        assets.push({ asset, createdAt: asset.createdAt ?? startedAt, startedAt });
      }
      if (assets.length > 0) bySession.set(sessionKey, assets);
    }
    const sessions = Array.from(bySession, ([sessionKey, assets]) => {
      assets.sort((a, b) => b.createdAt - a.createdAt || b.startedAt - a.startedAt);
      return {
        sessionKey,
        latestAt: assets[0]?.createdAt ?? 0,
        assets: assets.map((item) => item.asset),
      };
    });
    return sessions.sort((a, b) => b.latestAt - a.latestAt);
  }, [libraryGroups]);
  const libraryTotal = librarySessions.reduce((total, session) => total + session.assets.length, 0);
  const libraryMode = view === "library";
  const activeTotal = libraryMode ? libraryTotal : totalAssets;
  const activeLoading = libraryMode ? libraryLoading : loading;
  const activeLoadingMore = libraryMode ? libraryLoadingMore : loadingMore;
  const activeFailed = libraryMode ? libraryLoadFailed : loadFailed;
  const activeNext = libraryMode ? libraryNext : next;
  const activeRefresh = libraryMode ? refreshLibrary : refresh;
  const activeLoadMore = libraryMode ? loadMoreLibrary : loadMore;

  const openGroup = (group: MediaGroup, index: number) => {
    setLightbox({ images: group.assets, index });
  };
  const openFlat = (index: number) => {
    setLightbox({ images: flatAssets.map((item) => item.asset), index });
  };

  const Title = modal || dialogTitle ? Dialog.Title : "h2";
  const content = (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col" data-testid="generated-assets-panel">
      {modal ? (
        <Dialog.Description className="sr-only">
          {t("thread.generatedAssets.title")}
        </Dialog.Description>
      ) : null}
      <div className="shrink-0 border-b border-border/60 px-4 pb-3 pt-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <Title className="truncate text-sm font-medium">
              {t("thread.generatedAssets.title")}
            </Title>
            {activeTotal > 0 ? (
              <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                {activeNext
                  ? t("thread.generatedAssets.loaded", {
                      count: activeTotal,
                      defaultValue: "{{count}} loaded",
                    })
                  : t("thread.generatedAssets.total", { count: activeTotal })}
              </span>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label={t("thread.generatedAssets.refresh")}
              disabled={activeLoading}
              onClick={activeRefresh}
              className="h-8 w-8 rounded-full text-muted-foreground hover:bg-accent/40 hover:text-foreground"
            >
              <RefreshCcw className={cn("h-4 w-4", activeLoading && "animate-spin")} />
            </Button>
            {modal && onClose ? (
              <Button
                ref={closeButtonRef}
                type="button"
                variant="ghost"
                size="icon"
                aria-label={t("thread.generatedAssets.close")}
                onClick={onClose}
                className="h-8 w-8 rounded-full text-muted-foreground hover:bg-accent/40 hover:text-foreground"
              >
                <X className="h-4 w-4" aria-hidden />
              </Button>
            ) : null}
          </div>
        </div>
        <SegmentedControl
          className="mt-2.5"
          value={view}
          onChange={setView}
          ariaLabel={t("thread.generatedAssets.view")}
          options={[
            { value: "flat", label: t("thread.generatedAssets.all") },
            { value: "call", label: t("thread.generatedAssets.byCall") },
            { value: "library", label: t("thread.generatedAssets.allSessions") },
          ]}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {activeLoading && activeTotal === 0 ? (
          <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t("thread.generatedAssets.loading")}
          </div>
        ) : activeFailed ? (
          <div className="mx-auto flex max-w-sm flex-col items-center gap-3 py-16 text-center">
            <CircleAlert className="h-5 w-5 text-destructive" />
            <p className="text-sm text-muted-foreground">
              {t("thread.generatedAssets.loadFailed")}
            </p>
            <Button type="button" variant="outline" size="sm" onClick={activeRefresh}>
              {t("thread.generatedAssets.retry")}
            </Button>
          </div>
        ) : activeTotal === 0 ? (
          <div className="mx-auto max-w-sm py-16 text-center text-sm text-muted-foreground">
            {libraryMode && activeNext
              ? t("thread.generatedAssets.moreSessions", {
                  defaultValue: "No media found yet. Load more to check the remaining chats.",
                })
              : t(libraryMode
                ? "thread.generatedAssets.emptyLibrary"
                : "thread.generatedAssets.empty")}
          </div>
        ) : view === "flat" ? (
          <div
            data-testid="generated-assets-flat"
            className="grid grid-cols-2 gap-2 sm:grid-cols-3"
          >
            {flatAssets.map((item, index) => (
              <AssetThumbnail
                key={`${item.group.id}:${item.asset.id}`}
                asset={item.asset}
                badge={`#${item.group.sequence}.${item.candidate}`}
                onClick={() => openFlat(index)}
              />
            ))}
          </div>
        ) : view === "library" ? (
          <div data-testid="generated-assets-library" className="space-y-7">
            {librarySessions.map((session) => (
              <section key={session.sessionKey} className="space-y-2">
                <div
                  className="truncate text-[12px] font-medium text-muted-foreground"
                  title={session.sessionKey}
                >
                  {shortSessionLabel(session.sessionKey)}
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {session.assets.map((asset, index) => (
                    <AssetThumbnail
                      key={`${session.sessionKey}:${asset.id}`}
                      asset={asset}
                      badge={`#${index + 1}`}
                      onClick={() => setLightbox({ images: session.assets, index })}
                    />
                  ))}
                </div>
              </section>
            ))}
          </div>
        ) : (
          <div data-testid="generated-assets-groups" className="space-y-7">
            {mediaGroups.map((group) => (
              <AssetGroupSection
                key={group.id}
                group={group}
                locale={locale}
                onOpen={(index) => openGroup(group, index)}
              />
            ))}
          </div>
        )}
        {activeNext ? (
          <div className="flex justify-center pt-5">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={activeLoadingMore}
              onClick={activeLoadMore}
            >
              {activeLoadingMore
                ? t("thread.generatedAssets.loadingMore")
                : t("thread.generatedAssets.loadMore")}
            </Button>
          </div>
        ) : null}
      </div>

      {lightbox ? (
        <ImageLightbox
          images={lightbox.images}
          index={lightbox.index}
          onIndexChange={(index) => setLightbox((current) => (
            current ? { ...current, index } : current
          ))}
          onOpenChange={(nextOpen) => {
            if (!nextOpen) setLightbox(null);
          }}
        />
      ) : null}
    </div>
  );
  if (!modal) return content;
  if (!visible) return null;
  return (
    <ThreadInspectorDialog focusRef={closeButtonRef} onClose={onClose ?? (() => {})}>
      {content}
    </ThreadInspectorDialog>
  );
}

/** Standalone trigger + sheet kept for callers that are not inside the inspector. */
export function GeneratedAssetsSheet({
  sessionKey,
  token,
  refreshKey = 0,
}: GeneratedAssetsSheetProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={cn(
          "host-no-drag relative h-8 w-8 rounded-full text-muted-foreground/80",
          "hover:bg-accent/40 hover:text-foreground",
        )}
        aria-label={t("thread.generatedAssets.open")}
        onClick={() => setOpen(true)}
      >
        <Images className="h-4 w-4" />
      </Button>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent
          side="right"
          aria-describedby={undefined}
          className="w-[min(96vw,56rem)] gap-0 p-0 sm:max-w-[56rem]"
        >
          {open ? (
            <GeneratedAssetsPanel
              sessionKey={sessionKey}
              token={token}
              refreshKey={refreshKey}
              active={open}
              modal={false}
              dialogTitle
            />
          ) : null}
        </SheetContent>
      </Sheet>
    </>
  );
}


function AssetGroupSection({
  group,
  locale,
  onOpen,
}: {
  group: MediaGroup;
  locale: string;
  onOpen: (index: number) => void;
}) {
  const { t } = useTranslation();
  const prompt = groupPrompt(group);
  const metadata = groupMetadata(group);
  const startedAt = group.startedAt ? fmtDateTime(group.startedAt, locale) : "";
  return (
    <section data-testid="generated-assets-group" aria-label={formatToolName(group.toolName)}>
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 grid h-6 min-w-6 place-items-center rounded-full bg-muted px-1.5 text-[11px] font-semibold text-muted-foreground">
          {group.sequence}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[12px]">
            <span className="font-medium text-foreground">
              {formatToolName(group.toolName)}
            </span>
            <span className="text-muted-foreground">
              {t("thread.generatedAssets.candidates", { count: group.assets.length })}
            </span>
            {startedAt ? (
              <>
                <span aria-hidden className="text-muted-foreground/50">·</span>
                <span className="text-muted-foreground/80">{startedAt}</span>
              </>
            ) : null}
          </div>
          {prompt ? (
            <p className="mt-1 line-clamp-2 text-[12px] leading-snug text-muted-foreground">
              {prompt}
            </p>
          ) : null}
          {metadata ? (
            <div className="mt-1 text-[10.5px] text-muted-foreground/70">{metadata}</div>
          ) : null}
        </div>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
        {group.assets.map((asset, index) => (
          <AssetThumbnail
            key={asset.id}
            asset={asset}
            badge={`#${index + 1}`}
            onClick={() => onOpen(index)}
          />
        ))}
      </div>
    </section>
  );
}

function AssetThumbnail({
  asset,
  badge,
  onClick,
}: {
  asset: GeneratedAsset;
  badge: string;
  onClick: () => void;
}) {
  const { t } = useTranslation();
  const [failed, setFailed] = useState(false);
  const [copied, setCopied] = useState<"link" | "path" | null>(null);
  const url = useMobileMediaUrl(asset.url);

  const copy = (kind: "link" | "path") => {
    const value = kind === "link" ? mediaLink(asset.url) : (asset.path ?? "");
    if (!value) return;
    void copyTextToClipboard(value).then((ok) => {
      if (!ok) return;
      setCopied(kind);
      window.setTimeout(() => setCopied(null), 1200);
    });
  };

  return (
    <div
      data-testid="generated-asset-thumbnail"
      className={cn(
        "group relative aspect-square overflow-hidden rounded-xl border border-border/50 bg-muted/25",
        "transition hover:border-border hover:bg-muted/45",
      )}
    >
      <button
        type="button"
        className="absolute inset-0 h-full w-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/40"
        aria-label={t("thread.generatedAssets.openAsset", { name: asset.name })}
        onClick={onClick}
      >
        {url && !failed ? (
          asset.kind === "video" ? (
            <>
              <video
                src={url}
                muted
                playsInline
                preload="metadata"
                aria-label={asset.name}
                onError={() => setFailed(true)}
                className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]"
              />
              <span className="absolute inset-0 grid place-items-center bg-black/20">
                <Play className="h-6 w-6 fill-white text-white/90" aria-hidden />
              </span>
            </>
          ) : (
            <img
              src={url}
              alt={asset.name}
              loading="lazy"
              decoding="async"
              draggable={false}
              onError={() => setFailed(true)}
              className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]"
            />
          )
        ) : (
          <span className="grid h-full place-items-center px-2 text-center text-[10px] text-muted-foreground">
            {asset.name}
          </span>
        )}
      </button>
      <span className="pointer-events-none absolute left-1.5 top-1.5 rounded-full bg-black/65 px-1.5 py-0.5 text-[10px] font-medium text-white">
        {badge}
      </span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            aria-label={t("thread.generatedAssets.actions", { name: asset.name })}
            className={cn(
              "absolute right-1 top-1 z-10 grid h-6 w-6 place-items-center rounded-full",
              "bg-black/60 text-white/90 opacity-90 transition hover:bg-black/80 hover:text-white",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80",
            )}
            onClick={(event) => event.stopPropagation()}
          >
            <MoreHorizontal className="h-3.5 w-3.5" aria-hidden />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuItem onSelect={() => copy("link")}>
            <Link2 className="mr-2 h-3.5 w-3.5" aria-hidden />
            {t("thread.generatedAssets.copyLink")}
          </DropdownMenuItem>
          {asset.path ? (
            <DropdownMenuItem onSelect={() => copy("path")}>
              <FileText className="mr-2 h-3.5 w-3.5" aria-hidden />
              {t("thread.generatedAssets.copyPath")}
            </DropdownMenuItem>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>
      {copied ? (
        <span className="pointer-events-none absolute bottom-1.5 left-1.5 z-10 flex items-center gap-1 rounded-full bg-black/75 px-2 py-0.5 text-[10px] font-medium text-white">
          <Check className="h-3 w-3" aria-hidden />
          {copied === "link"
            ? t("thread.generatedAssets.linkCopied")
            : t("thread.generatedAssets.pathCopied")}
        </span>
      ) : null}
    </div>
  );
}


function shortSessionLabel(sessionKey: string): string {
  const raw = sessionKey.replace(/^websocket:/, "");
  return raw.length > 12 ? `${raw.slice(0, 8)}…` : raw;
}

function groupPrompt(group: GeneratedAssetGroup): string {
  const prompt = group.arguments.prompt;
  if (typeof prompt === "string" && prompt.trim()) return prompt;
  return group.artifacts.find((artifact) => artifact.prompt)?.prompt ?? "";
}

function groupMetadata(group: GeneratedAssetGroup): string {
  const workflow = group.arguments.workflow;
  const model = group.artifacts.find((artifact) => artifact.model)?.model;
  return [workflow, model]
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .join(" · ");
}

function formatToolName(name: string): string {
  const stripped = name.replace(/^mcp_[a-zA-Z0-9]+_/, "").replace(/^mcp_/, "");
  return stripped.replaceAll("_", " ");
}
