import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactElement,
  type Ref,
} from "react";

import { MarkdownText, preloadMarkdownText } from "@/components/MarkdownText";
import {
  CapabilityMentionToken,
  cliAppInitials,
  mcpPresetInitials,
  splitCapabilityMentionSegments,
  type CapabilityMentionSegment,
} from "@/components/CliAppMentionText";
import { INLINE_TOKEN_HIGHLIGHT_COLOR } from "@/components/InlineTokenHighlight";
import {
  Activity,
  Archive,
  ArrowUp,
  BookOpen,
  Brain,
  ChevronDown,
  ChevronUp,
  CircleHelp,
  CornerDownRight,
  FileText,
  GripVertical,
  History,
  ImageIcon,
  Loader2,
  MessageCircle,
  Mic,
  Plus,
  Quote,
  RotateCw,
  Shield,
  Sparkles,
  Square,
  SquarePen,
  Target,
  Trash2,
  Undo2,
  X,
  type LucideIcon,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { isMobileRuntimeActive } from "@/lib/mobile-runtime";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  floatingItemClassName,
  floatingSurfaceElevationClassName,
  floatingSurfaceVisualClassName,
} from "@/components/ui/floating-surface";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  WorkspacePermissionV2Control,
  WorkspaceProjectPicker,
} from "@/components/thread/WorkspaceControls";
import { useThreadComposerChromeHidden } from "@/components/thread/composerChrome";
import {
  ModelPresetBadge,
  type ModelPresetOption,
  type ModelReasoningControl,
} from "@/components/thread/ModelPresetBadge";
import {
  ComposerUsagePopover,
  type ComposerContextUsage,
  type ComposerRoundUsage,
} from "@/components/thread/ComposerUsagePopover";
import {
  ACCEPT_ATTR,
  MAX_ATTACHMENTS_PER_MESSAGE,
  useAttachedImages,
  type AttachedImage,
  type AttachmentError,
  type AttachmentKind,
  type RestoredReadyImage,
} from "@/hooks/useAttachedImages";
import {
  clearComposerDraftsForSession,
  draftIncarnationReplaced,
  draftScopeId,
  loadComposerDraft,
  NEW_CHAT_DRAFT_KEY,
  saveComposerDraft,
  updateComposerDraftSelection,
  type DraftAttachment,
  type DraftScope,
} from "@/lib/composer-drafts";
import { useClipboardAndDrop } from "@/hooks/useClipboardAndDrop";
import { useLogoFallback } from "@/hooks/useLogoFallback";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import type {
  SendAttachment,
  SendOptions,
  SubmittedTurn,
} from "@/hooks/useNanobotStream";
import type {
  QueuedAttemptBinding,
  QueuedAttemptOutcome,
} from "@/lib/nanobot-client";
import { useVoiceRecorder, type VoiceRecorderErrorKey } from "@/hooks/useVoiceRecorder";
import { getMobileRuntime } from "@/lib/mobile-runtime";
import type {
  CliAppInfo,
  ChatSummary,
  GoalStateWsPayload,
  McpPresetInfo,
  OutboundCliAppMention,
  OutboundMcpPresetMention,
  SessionMention,
  SlashCommand,
  SkillSummary,
  WebUIIngressLimits,
  WorkspaceScopePayload,
  WorkspaceAccessMode,
  WorkspacePermissionProjection,
  WorkspacesPayload,
} from "@/lib/types";
import {
  logoFallbackUrls,
} from "@/lib/provider-brand";
import { sessionHandleColor } from "@/lib/session-handle";
import {
  isSideChannelLifecycle,
  slashCommandLifecycle,
} from "@/lib/slash-command";
import {
  clearDraggedSession,
  hasDraggedSession,
  readDraggedSession,
} from "@/lib/session-drag";
import { formatQuotedUserMessage } from "@/lib/user-message-quote";
import { cn } from "@/lib/utils";

const VOICE_SHORTCUT_CODE = "KeyD";
const VOICE_SHORTCUT_ARIA = "Control+Shift+D";
const VOICE_ERROR_VISIBLE_MS = 3_500;
const VOICE_ERROR_FADE_MS = 500;
type VoiceShortcutPlatform = "apple" | "chromeos" | "linux" | "other" | "windows";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function utf8Bytes(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function isVoiceShortcutDown(event: KeyboardEvent): boolean {
  return (
    event.code === VOICE_SHORTCUT_CODE
    && event.ctrlKey
    && event.shiftKey
    && !event.altKey
    && !event.metaKey
  );
}

function isVoiceShortcutRelease(event: KeyboardEvent): boolean {
  return (
    event.code === VOICE_SHORTCUT_CODE
    || event.key === "Control"
    || event.key === "Shift"
  );
}

function getVoiceShortcutPlatform(): VoiceShortcutPlatform {
  if (typeof navigator === "undefined") return "other";
  const userAgentData = (navigator as Navigator & { userAgentData?: { platform?: string } })
    .userAgentData;
  const platform = [
    userAgentData?.platform,
    navigator.platform,
    navigator.userAgent,
  ].filter(Boolean).join(" ").toLowerCase();
  const isIpadPretendingToBeMac =
    navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  if (isIpadPretendingToBeMac || /mac|iphone|ipad|ipod/.test(platform)) return "apple";
  if (/win/.test(platform)) return "windows";
  if (/cros/.test(platform)) return "chromeos";
  if (/linux|x11|android/.test(platform)) return "linux";
  return "other";
}

function getVoiceShortcutLabel(): string {
  switch (getVoiceShortcutPlatform()) {
    case "apple":
      return "⌃⇧D";
    case "chromeos":
    case "linux":
    case "windows":
    case "other":
      return "Ctrl ⇧ D";
  }
}

interface ThreadComposerProps {
  onSend: (
    content: string,
    images?: SendAttachment[],
    options?: SendOptions,
  ) => SubmittedTurn | null | boolean | void | Promise<boolean | void>;
  onJoinQueuedAttempt?: (
    binding: QueuedAttemptBinding,
  ) => Promise<QueuedAttemptOutcome> | null;
  onConsumeQueuedAttempt?: (binding: QueuedAttemptBinding) => boolean;
  /** A queued send lost its admission and should return to the waiting stack. */
  admissionRejectedTurn?: { turnId: string; reason?: string } | null;
  onAdmissionRejectedPromptRestored?: (turnId: string) => void;
  disabled?: boolean;
  /** The editor and live controls stay usable while a new turn cannot start. */
  newTurnBlocked?: boolean;
  steerDisabled?: boolean;
  placeholder?: string;
  inputAriaLabel?: string;
  isStreaming?: boolean;
  /** The stream owner just observed a natural completion for the selected chat. */
  queueCompletionTurnId?: string | null;
  /** Connectivity and admission uncertainty consume, rather than defer, a drain. */
  automaticQueueSendBlocked?: boolean;
  /** A server-owned root/child snapshot permits Stop after the root reply. */
  stopAvailable?: boolean;
  stopDisabled?: boolean;
  stopPending?: boolean;
  modelLabel?: string | null;
  modelDetail?: string | null;
  modelPreset?: string | null;
  modelPresets?: ModelPresetOption[];
  onModelPresetChange?: (name: string) => void;
  modelReasoning?: ModelReasoningControl;
  modelProvider?: string | null;
  modelProviderLabel?: string | null;
  modelNeedsSetup?: boolean;
  fallbackModelName?: string | null;
  onModelBadgeClick?: () => void;
  onManageModels?: () => void;
  contextUsage?: ComposerContextUsage | null;
  hasConfiguredModel?: boolean;
  recentRoundUsage?: readonly ComposerRoundUsage[];
  variant?: "thread" | "hero";
  slashCommands?: SlashCommand[];
  cliApps?: CliAppInfo[];
  mcpPresets?: McpPresetInfo[];
  sessions?: ChatSummary[];
  skills?: SkillSummary[];
  onStop?: () => void;
  surfaceRef?: Ref<HTMLDivElement>;
  onTranscribeAudio?: (dataUrl: string, options?: { durationMs?: number }) => Promise<string>;
  /** Sustained objective for this chat (WebSocket ``goal_state``). */
  goalState?: GoalStateWsPayload;
  workspaceScope?: WorkspaceScopePayload | null;
  workspaceControlsHidden?: boolean;
  workspaceDefaultScope?: WorkspaceScopePayload | null;
  workspaceControls?: WorkspacesPayload["controls"] | null;
  workspaces?: WorkspacesPayload | null;
  onRefreshWorkspaces?: () => Promise<void>;
  workspaceScopeDisabled?: boolean;
  /** Detached working-folder choice for the next turn while one is running. */
  nextWorkspaceScope?: WorkspaceScopePayload | null;
  /** A next-turn setting change is awaiting its acknowledgement. */
  nextTurnSettingsPending?: boolean;
  /** A next-turn setting change was refused; the previous choice still applies. */
  nextTurnSettingsError?: string | null;
  onDismissNextTurnSettingsError?: () => void;
  /**
   * Settle any in-flight next-turn setting before a queued row is admitted.
   * Rejects when the choice was refused so the row stays ready to retry.
   */
  onBeforeNewTurnSend?: () => Promise<void>;
  /** UNKNOWN input disposition blocks persistent-chat access mutations, not known live controls. */
  workspaceAccessModeUnknownBlocked?: boolean;
  workspaceError?: string | null;
  onDismissWorkspaceError?: () => void;
  onPickWorkspaceFolder?: () => Promise<string | null>;
  onWorkspaceScopeChange?: (scope: WorkspaceScopePayload) => void;
  workspacePermissions?: WorkspacePermissionProjection | null;
  workspacePermissionsLoading?: boolean;
  workspacePermissionsError?: boolean;
  persistentWorkspaceChat?: boolean;
  onWorkspaceAccessModeChange?: (mode: WorkspaceAccessMode) => void | Promise<void>;
  onRetryWorkspaceRestricted?: () => void | Promise<void>;
  onDraftFullAccess?: (draft: string) => void | Promise<void>;
  initialDraft?: string | null;
  onInitialDraftConsumed?: () => void;
  pendingQueueKey?: string | null;
  /** Can identify an editor on a validated route before it has a send owner. */
  draftKey?: string | null;
  /** Authenticated gateway/client namespace for device-local drafts. */
  draftNamespace?: string;
  /** Session incarnation; a replaced session never inherits an older draft. */
  draftIncarnation?: string | null;
  /** Access revoked: discard the draft and never re-save it. */
  draftRevoked?: boolean;
  transcriptionProvider?: string | null;
  ingressLimits?: WebUIIngressLimits | null;
  quotedContext?: string | null;
  focusRequest?: number;
  onQuotedContextChange?: (text: string | null) => void;
}

const COMMAND_ICONS: Record<string, LucideIcon> = {
  activity: Activity,
  archive: Archive,
  "book-open": BookOpen,
  brain: Brain,
  "circle-help": CircleHelp,
  history: History,
  "rotate-cw": RotateCw,
  shield: Shield,
  sparkles: Sparkles,
  square: Square,
  "square-pen": SquarePen,
  "undo-2": Undo2,
};

const SLASH_PALETTE_GAP_PX = 8;
const SLASH_PALETTE_MAX_HEIGHT_PX = 288;
const SLASH_PALETTE_MIN_HEIGHT_PX = 144;
const SLASH_PALETTE_CHROME_PX = 12;
const SLASH_RECENTS_STORAGE_KEY = "nanobot.webui.slashCommandRecents";
const SLASH_RECENTS_LIMIT = 5;
const QUEUED_PROMPTS_STORAGE_PREFIX = "nanobot.webui.composerQueuedGuidance.v1:";
const QUEUED_PROMPTS_LIMIT = 20;
const QUEUED_PROMPT_MAX_CHARS = 4000;
const SESSION_MENTIONS_LIMIT = 8;

function VoiceRecordingMeter({
  ariaLabel,
  className,
  elapsedLabel,
  isHero,
  levels,
}: {
  ariaLabel: string;
  className?: string;
  elapsedLabel: string;
  isHero: boolean;
  levels: number[];
}) {
  return (
    <div
      className={cn(
        "flex min-w-0 items-center gap-2 text-neutral-700 dark:text-white",
        isHero ? "h-8" : "h-9",
        className,
      )}
      aria-live="polite"
      aria-label={ariaLabel}
    >
      <span className="flex h-5 min-w-0 flex-1 items-center justify-between overflow-hidden" aria-hidden>
        {levels.map((height, index) => (
          <span
            key={index}
            className="w-[2px] rounded-full bg-current opacity-85 transition-[height] duration-75 ease-linear motion-reduce:transition-none"
            style={{ height }}
          />
        ))}
      </span>
      <span className="min-w-[2.1rem] text-right text-[12px] font-medium tabular-nums text-muted-foreground">
        {elapsedLabel}
      </span>
    </div>
  );
}

type SlashPalettePlacement = "above" | "below";

interface SlashPaletteLayout {
  placement: SlashPalettePlacement;
  maxHeight: number;
}

interface QueuedPrompt {
  id: string;
  text: string;
  submission: QueuedPromptSubmission;
  images?: QueuedPromptImage[];
  quotedContext?: string;
  sessionMentions?: SessionMention[];
}

type QueuedPromptSubmission =
  | { version: 1; state: "ready" }
  | { version: 1; state: "submitted"; binding: QueuedAttemptBinding }
  | {
      version: 1;
      state: "not_admitted";
      binding: QueuedAttemptBinding;
      reason: string;
    }
  | { version: 1; state: "unknown" };

interface QueuedPromptImage {
  dataUrl: string;
  name?: string;
  kind?: AttachmentKind;
}

interface CliAppMentionQuery {
  query: string;
  start: number;
  end: number;
}

type MentionCandidate = {
  name: string;
  displayName: string;
} & (
  | { kind: "session"; mention: SessionMention }
  | {
      kind: "cli" | "mcp";
      brandColor: string | null;
      logoUrl: string | null;
      initials: string;
    }
);

interface MentionInsertion {
  value: string;
  cursor: number;
  tokenStart: number;
  tokenEnd: number;
}

function mentionInsertion(
  value: string,
  name: string,
  start: number,
  end: number,
): MentionInsertion {
  const from = Math.min(Math.max(start, 0), value.length);
  const to = Math.min(Math.max(end, from), value.length);
  const prefix = value.slice(0, from);
  const suffix = value.slice(to);
  const leadingSpace = prefix && !/\s$/.test(prefix) ? " " : "";
  const trailingSpace = /^\s/.test(suffix) ? "" : " ";
  const tokenStart = prefix.length + leadingSpace.length;
  const tokenEnd = tokenStart + name.length + 1;
  return {
    value: `${prefix}${leadingSpace}@${name}${trailingSpace}${suffix}`,
    cursor: tokenEnd + trailingSpace.length,
    tokenStart,
    tokenEnd,
  };
}

function sessionMentionOptions(sessions: ChatSummary[]): SessionMention[] {
  return sessions.flatMap((session) => {
    if (!session.handle) return [];
    return [{
      id: session.handle.id,
      name: session.handle.name,
      session_key: session.key,
      title: session.title?.trim() || session.preview.trim(),
    }];
  });
}

interface SlashPaletteCommand {
  command: string;
  title: string;
  description: string;
  icon: string;
  kind?: "skill";
  argHint?: string;
  detail: string;
  badge?: string;
  recent: boolean;
}

function skillMatchRank(skill: SkillSummary, query: string): number | null {
  if (!query) return 0;
  const name = skill.name.toLowerCase();
  if (name === query) return 0;
  if (name.startsWith(query)) return 1;
  if (name.includes(query)) return 2;
  if (skill.description.toLowerCase().includes(query)) return 3;
  return null;
}

function slashCommandI18nKey(command: string): string {
  return command.replace(/^\//, "").replace(/-/g, "_");
}

function readSlashRecents(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(SLASH_RECENTS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string").slice(0, SLASH_RECENTS_LIMIT)
      : [];
  } catch {
    return [];
  }
}

function storeSlashRecents(commands: string[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      SLASH_RECENTS_STORAGE_KEY,
      JSON.stringify(commands.slice(0, SLASH_RECENTS_LIMIT)),
    );
  } catch {
    // localStorage may be unavailable in private contexts; command insertion still works.
  }
}

function queuedPromptsStorageKey(key?: string | null): string | null {
  const clean = key?.trim();
  return clean ? `${QUEUED_PROMPTS_STORAGE_PREFIX}${clean}` : null;
}

function normalizeQueuedSessionMentions(value: unknown): SessionMention[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const candidate = item as Partial<SessionMention>;
    const name = candidate.name?.trim().slice(0, 80);
    const sessionKey = candidate.session_key?.trim().slice(0, 512);
    if (
      !name
      || !sessionKey?.startsWith("websocket:")
      || !/^[\p{L}\p{N}_-]+$/u.test(name)
    ) return [];
    return [{
      ...(typeof candidate.id === "string" && /^handle_[a-f0-9]{32}$/i.test(candidate.id)
        ? { id: candidate.id }
        : {}),
      name,
      session_key: sessionKey,
      title: candidate.title?.trim().slice(0, 160) ?? "",
    }];
  }).slice(0, SESSION_MENTIONS_LIMIT);
}

function normalizeQueuedAttemptBinding(value: unknown, promptId: string): QueuedAttemptBinding | null {
  if (!value || typeof value !== "object") return null;
  const binding = value as Partial<QueuedAttemptBinding>;
  if (
    (binding.queueKey !== null && typeof binding.queueKey !== "string")
    || binding.promptId !== promptId
    || typeof binding.turnId !== "string"
    || !binding.turnId.trim()
  ) return null;
  return {
    queueKey: binding.queueKey,
    promptId,
    turnId: binding.turnId,
  };
}

function normalizeQueuedPromptSubmission(value: unknown, promptId: string): QueuedPromptSubmission {
  if (!value || typeof value !== "object") return { version: 1, state: "unknown" };
  const candidate = value as {
    version?: unknown;
    state?: unknown;
    binding?: unknown;
    reason?: unknown;
  };
  if (candidate.version !== 1) return { version: 1, state: "unknown" };
  if (candidate.state === "ready") return { version: 1, state: "ready" };
  if (candidate.state === "unknown") return { version: 1, state: "unknown" };
  if (candidate.state !== "submitted" && candidate.state !== "not_admitted") {
    return { version: 1, state: "unknown" };
  }
  const binding = normalizeQueuedAttemptBinding(candidate.binding, promptId);
  if (!binding) return { version: 1, state: "unknown" };
  if (candidate.state === "submitted") return { version: 1, state: "submitted", binding };
  const reason = typeof candidate.reason === "string" ? candidate.reason.trim().slice(0, 512) : "";
  return reason
    ? { version: 1, state: "not_admitted", binding, reason }
    : { version: 1, state: "unknown" };
}

function normalizeQueuedPrompt(item: unknown, index: number): QueuedPrompt | null {
  if (!item || typeof item !== "object") return null;
  const record = item as Partial<QueuedPrompt>;
  if (typeof record.text !== "string") return null;
  const text = record.text.trim();
  const images = Array.isArray(record.images)
    ? record.images.flatMap((image) => {
        if (!image || typeof image !== "object") return [];
        const candidate = image as Partial<QueuedPromptImage>;
        if (typeof candidate.dataUrl !== "string" || !candidate.dataUrl.startsWith("data:")) {
          return [];
        }
        const kind = candidate.kind === "file" || candidate.kind === "image"
          ? candidate.kind
          : candidate.dataUrl.startsWith("data:image/")
            ? "image"
            : "file";
        return [{
          dataUrl: candidate.dataUrl,
          kind,
          ...(typeof candidate.name === "string" && candidate.name.trim()
            ? { name: candidate.name.trim() }
            : {}),
        }];
      }).slice(0, MAX_ATTACHMENTS_PER_MESSAGE)
    : [];
  const quotedContext = typeof record.quotedContext === "string"
    ? record.quotedContext.trim().slice(0, QUEUED_PROMPT_MAX_CHARS)
    : "";
  const sessionMentions = normalizeQueuedSessionMentions(record.sessionMentions);
  if (!text && images.length === 0) return null;
  const id = typeof record.id === "string" && record.id.trim()
    ? record.id
    : `queued-prompt-restored-${index}`;
  return {
    id,
    text,
    submission: normalizeQueuedPromptSubmission(record.submission, id),
    ...(images.length > 0 ? { images } : {}),
    ...(quotedContext ? { quotedContext } : {}),
    ...(sessionMentions.length > 0 ? { sessionMentions } : {}),
  };
}

function readQueuedPrompts(storageKey: string): QueuedPrompt[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(storageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item, index) => normalizeQueuedPrompt(item, index))
      .filter((item): item is QueuedPrompt => item != null);
  } catch {
    return [];
  }
}

function storeQueuedPrompts(storageKey: string, prompts: QueuedPrompt[]): boolean {
  if (typeof window === "undefined" || prompts.length > QUEUED_PROMPTS_LIMIT) return false;
  try {
    if (prompts.length === 0) {
      window.localStorage.removeItem(storageKey);
      return window.localStorage.getItem(storageKey) === null;
    }
    const serialized = JSON.stringify(prompts.map((prompt) => ({
      id: prompt.id,
      text: prompt.text,
      ...(prompt.images?.length ? { images: prompt.images.slice(0, MAX_ATTACHMENTS_PER_MESSAGE) } : {}),
      ...(prompt.quotedContext ? { quotedContext: prompt.quotedContext } : {}),
      ...(prompt.sessionMentions?.length
        ? { sessionMentions: prompt.sessionMentions.slice(0, SESSION_MENTIONS_LIMIT) }
        : {}),
      submission: prompt.submission,
    })));
    window.localStorage.setItem(storageKey, serialized);
    const stored = window.localStorage.getItem(storageKey);
    if (stored !== serialized) return false;
    const parsed: unknown = JSON.parse(stored);
    if (!Array.isArray(parsed) || parsed.length !== prompts.length) return false;
    return parsed.every((item, index) => {
      if (!item || typeof item !== "object") return false;
      const candidate = item as { id?: unknown; submission?: unknown };
      const expected = prompts[index];
      return candidate.id === expected?.id
        && JSON.stringify(candidate.submission) === JSON.stringify(expected.submission);
    });
  } catch {
    return false;
  }
}

function readyImagesToQueuedImages(
  images: Array<AttachedImage & { dataUrl: string }>,
): QueuedPromptImage[] {
  return images.map((img) => ({
    dataUrl: img.dataUrl,
    kind: img.kind,
    name: img.file.name,
  }));
}

function queuedImagesToSendImages(images?: QueuedPromptImage[]): SendAttachment[] | undefined {
  if (!images?.length) return undefined;
  return images.map((img) => ({
    media: {
      data_url: img.dataUrl,
      ...(img.name ? { name: img.name } : {}),
    },
    preview: {
      kind: img.kind ?? (img.dataUrl.startsWith("data:image/") ? "image" : "file"),
      url: img.dataUrl,
      ...(img.name ? { name: img.name } : {}),
    },
  }));
}

function queuedPromptLabel(prompt: QueuedPrompt): string {
  const text = prompt.text.trim();
  if (text) return text;
  return prompt.images?.map((img) => img.name).filter(Boolean).join(", ") || "File attachment";
}

function suppressNativeDragPreview(dataTransfer: DataTransfer): void {
  if (typeof document === "undefined" || typeof dataTransfer.setDragImage !== "function") {
    return;
  }
  const ghost = document.createElement("div");
  ghost.style.position = "fixed";
  ghost.style.left = "-9999px";
  ghost.style.top = "-9999px";
  ghost.style.width = "1px";
  ghost.style.height = "1px";
  ghost.style.opacity = "0";
  document.body.appendChild(ghost);
  try {
    dataTransfer.setDragImage(ghost, 0, 0);
  } catch {
    ghost.remove();
    return;
  }
  window.setTimeout(() => ghost.remove(), 0);
}

function visualViewportBounds(): { top: number; bottom: number; height: number } {
  const viewport = window.visualViewport;
  if (!viewport) {
    return { top: 0, bottom: window.innerHeight, height: window.innerHeight };
  }
  const top = Math.max(0, viewport.offsetTop);
  const height = Math.max(0, viewport.height);
  return { top, bottom: top + height, height };
}

function getVisibleBounds(el: HTMLElement): { top: number; bottom: number } {
  const viewport = visualViewportBounds();
  let top = viewport.top;
  let bottom = viewport.bottom;
  let parent = el.parentElement;

  while (parent) {
    const style = window.getComputedStyle(parent);
    if (/(auto|scroll|hidden|clip)/.test(style.overflowY)) {
      const rect = parent.getBoundingClientRect();
      top = Math.max(top, rect.top);
      bottom = Math.min(bottom, rect.bottom);
    }
    parent = parent.parentElement;
  }

  return { top, bottom };
}

function goalStateStripPreview(
  goal: GoalStateWsPayload | undefined,
  t: (key: string) => string,
): string | null {
  if (!goal?.active) return null;
  const summary = goal.ui_summary?.trim();
  if (summary) return summary;
  const obj = goal.objective?.trim();
  if (obj) return obj.length > 72 ? `${obj.slice(0, 72)}…` : obj;
  return t("thread.composer.goalStateFallback");
}

const GOAL_PANEL_VIEWPORT_TOP_PAD = 20;
const GOAL_PANEL_GAP_ABOVE_STRIP_PX = 10;
const GOAL_PANEL_MIN_HEIGHT_PX = 112;
const GOAL_PANEL_MAX_VIEWPORT_RATIO = 0.62;

function measureGoalPanelMaxCssHeight(stripTopY: number): number {
  const viewport = visualViewportBounds();
  const spaceAboveStrip =
    stripTopY - viewport.top - GOAL_PANEL_VIEWPORT_TOP_PAD - GOAL_PANEL_GAP_ABOVE_STRIP_PX;
  return Math.min(
    Math.max(spaceAboveStrip, GOAL_PANEL_MIN_HEIGHT_PX),
    Math.floor(viewport.height * GOAL_PANEL_MAX_VIEWPORT_RATIO),
  );
}

function buildGoalMarkdownBody(summary: string, objective: string): string {
  const s = summary.trim();
  const o = objective.trim();
  if (s && o) return `${s}\n\n---\n\n${o}`;
  return o || s;
}

function cliAppMentionPayload(app: CliAppInfo): OutboundCliAppMention {
  return {
    name: app.name,
    display_name: app.display_name,
    category: app.category,
    entry_point: app.entry_point,
    logo_url: app.logo_url ?? null,
    brand_color: app.brand_color ?? null,
  };
}

function mcpPresetMentionPayload(preset: McpPresetInfo): OutboundMcpPresetMention {
  return {
    name: preset.name,
    display_name: preset.display_name,
    category: preset.category,
    transport: preset.transport,
    status: preset.status,
    configured: preset.configured,
    logo_url: preset.logo_url ?? null,
    brand_color: preset.brand_color ?? null,
  };
}

function GoalStateStrip({
  goalState,
}: {
  goalState?: GoalStateWsPayload;
}) {
  const { t } = useTranslation();
  const [goalPanelOpen, setGoalPanelOpen] = useState(false);
  const stripLabel = goalStateStripPreview(goalState, t);
  const active = !!stripLabel?.trim();
  const [, setTick] = useState(0);
  const stripWrapperRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const expandToggleRef = useRef<HTMLButtonElement>(null);
  const stripSnapshotRef = useRef<{
    goalState?: GoalStateWsPayload;
    stripLabel: string | null;
  } | null>(null);
  const [panelMaxPx, setPanelMaxPx] = useState(280);

  if (active) {
    stripSnapshotRef.current = { goalState, stripLabel };
  }

  useEffect(() => {
    if (!active) setGoalPanelOpen(false);
  }, [active]);

  const display = active
    ? { goalState, stripLabel }
    : stripSnapshotRef.current;
  const displayGoalState = display?.goalState;
  const displayStripLabel = display?.stripLabel ?? null;

  const objectiveFull = displayGoalState?.objective?.trim() ?? "";
  const summaryFull = displayGoalState?.ui_summary?.trim() ?? "";
  const canExpandGoal = !!(active && displayGoalState?.active && (objectiveFull || summaryFull));

  const markdownBody =
    objectiveFull || summaryFull
      ? buildGoalMarkdownBody(summaryFull, objectiveFull)
      : "";

  useLayoutEffect(() => {
    if (!goalPanelOpen) return;

    function relayout(): void {
      const el = stripWrapperRef.current;
      if (!el) return;
      const top = el.getBoundingClientRect().top;
      setPanelMaxPx(measureGoalPanelMaxCssHeight(top));
    }

    relayout();

    void preloadMarkdownText();
    const ro =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => relayout())
        : null;
    if (stripWrapperRef.current && ro) {
      ro.observe(stripWrapperRef.current);
    }
    const viewport = window.visualViewport;
    viewport?.addEventListener("resize", relayout);
    viewport?.addEventListener("scroll", relayout);
    window.addEventListener("resize", relayout);
    window.addEventListener("scroll", relayout, true);
    return () => {
      ro?.disconnect();
      viewport?.removeEventListener("resize", relayout);
      viewport?.removeEventListener("scroll", relayout);
      window.removeEventListener("resize", relayout);
      window.removeEventListener("scroll", relayout, true);
    };
  }, [goalPanelOpen]);

  useEffect(() => {
    if (!goalPanelOpen) return;

    function onPointerDown(ev: MouseEvent): void {
      const target = ev.target as Node | null;
      if (!target) return;
      if (panelRef.current?.contains(target)) return;
      if (expandToggleRef.current?.contains(target)) return;
      setGoalPanelOpen(false);
    }

    function onKey(ev: KeyboardEvent): void {
      if (ev.key === "Escape") setGoalPanelOpen(false);
    }

    window.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [goalPanelOpen]);

  if (!display) return null;

  const ariaLabel = displayStripLabel
    ? t("thread.composer.goalStateStrip", { label: displayStripLabel })
    : t("thread.composer.goalStateFallback");

  return (
    <div
      ref={stripWrapperRef}
      className="composer-status-drawer relative z-30"
      data-composer-status-drawer=""
      data-state={active ? "open" : "closed"}
      aria-hidden={active ? undefined : true}
      onTransitionEnd={(event) => {
        if (active || event.target !== event.currentTarget) return;
        stripSnapshotRef.current = null;
        setTick((n) => n + 1);
      }}
    >
      {goalPanelOpen && canExpandGoal && markdownBody ? (
        <div
          ref={panelRef}
          id="nanobot-goal-panel-root"
          role="dialog"
          aria-modal="false"
          aria-labelledby="nanobot-goal-panel-title"
          tabIndex={-1}
          className={cn(
            "absolute bottom-[calc(100%+8px)] left-3 right-3 z-[50] flex max-w-none flex-col overflow-hidden",
            "rounded-2xl",
            floatingSurfaceElevationClassName,
          )}
          style={{ maxHeight: `${Math.round(panelMaxPx)}px` }}
        >
          <div className="flex shrink-0 items-center justify-between gap-2 border-b border-black/[0.06] px-3 py-2 dark:border-white/[0.08]">
            <h2
              id="nanobot-goal-panel-title"
              className="min-w-0 truncate text-[13px] font-semibold tracking-tight text-foreground"
            >
              {t("thread.composer.goalStateSheetTitle")}
            </h2>
            <button
              type="button"
              className={cn(
                "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                "text-muted-foreground transition-colors hover:bg-muted/65 hover:text-foreground",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              )}
              aria-label={t("thread.composer.goalStateCloseAria")}
              onClick={() => setGoalPanelOpen(false)}
            >
              <X className="h-4 w-4" aria-hidden />
            </button>
          </div>
          <div
            id="nanobot-goal-panel-scroll"
            className="min-h-0 flex-1 overflow-y-auto scrollbar-thin px-3 pb-3 pt-2"
          >
            <MarkdownText className="max-w-none text-[13.5px] leading-relaxed text-foreground/90">
              {markdownBody}
            </MarkdownText>
          </div>
        </div>
      ) : null}
      <div className="composer-status-drawer-clip">
        {display ? (
          <div
            className="composer-status-drawer-content flex min-h-[36px] items-center gap-2 px-3 py-2"
            role="status"
            aria-label={ariaLabel}
          >
            <Target className="h-4 w-4 shrink-0 text-primary/75" aria-hidden />
            <span className="flex min-w-0 flex-1 items-center gap-1.5 text-[12px] font-medium text-foreground/75">
              {displayStripLabel ? (
                <span className="truncate">
                  {t("thread.composer.goalStateStrip", { label: displayStripLabel })}
                </span>
              ) : null}
            </span>
            {canExpandGoal ? (
              <button
                ref={expandToggleRef}
                type="button"
                className={cn(
                  "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                  "text-muted-foreground transition-colors hover:bg-muted/55 hover:text-foreground",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                )}
                aria-expanded={goalPanelOpen}
                aria-controls={goalPanelOpen ? "nanobot-goal-panel-root" : undefined}
                aria-label={t("thread.composer.goalStateExpandAria")}
                title={t("thread.composer.goalStateExpandAria")}
                onClick={() => setGoalPanelOpen((o) => !o)}
              >
                {goalPanelOpen ? (
                  <ChevronDown className="h-4 w-4" aria-hidden />
                ) : (
                  <ChevronUp className="h-4 w-4" aria-hidden />
                )}
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function ThreadComposer({
  onSend,
  onJoinQueuedAttempt,
  onConsumeQueuedAttempt,
  admissionRejectedTurn = null,
  onAdmissionRejectedPromptRestored,
  disabled,
  newTurnBlocked = false,
  steerDisabled = false,
  placeholder,
  inputAriaLabel,
  isStreaming = false,
  queueCompletionTurnId = null,
  automaticQueueSendBlocked = false,
  stopAvailable = false,
  stopDisabled = false,
  stopPending = false,
  modelLabel = null,
  modelDetail = null,
  modelPreset = null,
  modelPresets = [],
  onModelPresetChange,
  modelReasoning,
  modelProvider = null,
  modelProviderLabel = null,
  modelNeedsSetup = false,
  fallbackModelName = null,
  onModelBadgeClick,
  onManageModels,
  contextUsage = null,
  hasConfiguredModel = false,
  recentRoundUsage = [],
  variant = "thread",
  slashCommands = [],
  cliApps = [],
  mcpPresets = [],
  sessions = [],
  skills = [],
  onStop,
  surfaceRef,
  onTranscribeAudio,
  goalState,
  workspaceScope = null,
  workspaceControlsHidden = false,
  workspaceDefaultScope = null,
  workspaceControls = null,
  workspaces = null,
  onRefreshWorkspaces,
  workspaceScopeDisabled = false,
  nextWorkspaceScope = null,
  nextTurnSettingsPending = false,
  nextTurnSettingsError = null,
  onDismissNextTurnSettingsError,
  onBeforeNewTurnSend,
  workspaceAccessModeUnknownBlocked = false,
  workspaceError = null,
  onDismissWorkspaceError,
  onPickWorkspaceFolder,
  onWorkspaceScopeChange,
  workspacePermissions = null,
  workspacePermissionsLoading = false,
  workspacePermissionsError = false,
  persistentWorkspaceChat = false,
  onWorkspaceAccessModeChange,
  onRetryWorkspaceRestricted,
  onDraftFullAccess,
  initialDraft = null,
  onInitialDraftConsumed,
  pendingQueueKey = null,
  draftKey = pendingQueueKey,
  draftNamespace = "",
  draftIncarnation = null,
  draftRevoked = false,
  transcriptionProvider = null,
  ingressLimits = null,
  quotedContext = null,
  focusRequest = 0,
  onQuotedContextChange,
}: ThreadComposerProps) {
  const { t } = useTranslation();
  const hasStopAction = isStreaming || stopAvailable;
  const mobileRuntimeActive = isMobileRuntimeActive();
  const isNarrowViewport = useMediaQuery("(max-width: 639px)");
  const mobileComposer = mobileRuntimeActive || isNarrowViewport;
  const composerChromeHidden = useThreadComposerChromeHidden();
  const [value, setValue] = useState("");
  const [selectedSessionMentions, setSelectedSessionMentions] = useState<SessionMention[]>([]);
  const [sessionDragPreview, setSessionDragPreview] = useState<{
    mention: SessionMention;
    start: number;
    end: number;
  } | null>(null);
  const [inlineError, setInlineError] = useState<string | null>(null);
  const [dismissedNextTurnError, setDismissedNextTurnError] = useState<string | null>(null);
  useEffect(() => {
    if (!nextTurnSettingsError) setDismissedNextTurnError(null);
  }, [nextTurnSettingsError]);
  const visibleNextTurnSettingsError =
    nextTurnSettingsError && nextTurnSettingsError !== dismissedNextTurnError
      ? nextTurnSettingsError
      : null;
  const [sendPending, setSendPending] = useState(false);
  const pendingSendRef = useRef<symbol | null>(null);
  const draftViewVersionRef = useRef(0);
  const [modelSetupAttentionRequest, setModelSetupAttentionRequest] = useState(0);
  const interactionDisabled = !!disabled || sendPending;
  const [voiceErrorFading, setVoiceErrorFading] = useState(false);
  const [slashMenuDismissed, setSlashMenuDismissed] = useState(false);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [cliAppMenuDismissed, setCliAppMenuDismissed] = useState(false);
  const [selectedCliAppIndex, setSelectedCliAppIndex] = useState(0);
  const [cursorPosition, setCursorPosition] = useState(0);
  const [recentSlashCommands, setRecentSlashCommands] = useState<string[]>(() => readSlashRecents());
  const [queuedPrompts, setQueuedPrompts] = useState<QueuedPrompt[]>([]);
  const [queuedEdit, setQueuedEdit] = useState<{ id: string; text: string } | null>(null);
  const queuedPromptsRef = useRef<QueuedPrompt[]>([]);
  const [awaitingConfirmationPromptIds, setAwaitingConfirmationPromptIds] = useState<Set<string>>(
    () => new Set(),
  );
  const hasTouchPrimaryPointer = useMediaQuery("(hover: none) and (pointer: coarse)");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chipRefs = useRef(new Map<string, HTMLButtonElement>());
  const queuedPromptCounterRef = useRef(0);
  const draggedQueuedPromptIdRef = useRef<string | null>(null);
  const heldQueuedPromptIdsRef = useRef(new Set<string>());
  const awaitingConfirmationPromptIdsRef = useRef(new Set<string>());
  const queuedSubmissionInProgressRef = useRef(new Set<string>());
  const queuedAttemptJoinKeysRef = useRef(new Set<string>());
  const queueViewGenerationRef = useRef(0);
  const currentPendingQueueKeyRef = useRef(pendingQueueKey);
  currentPendingQueueKeyRef.current = pendingQueueKey;
  const previousDraftKeyRef = useRef<string | null | undefined>(undefined);
  const draftAppliedRef = useRef(false);
  const draftLiveRef = useRef<{
    value: string;
    mentions: SessionMention[];
    attachments: DraftAttachment[];
    quote: string | null;
  }>({ value: "", mentions: [], attachments: [], quote: null });
  const [reattachNames, setReattachNames] = useState<string[]>([]);
  const reattachNamesRef = useRef<string[]>([]);
  const handledQueueCompletionRef = useRef(queueCompletionTurnId);
  const voiceShortcutDownRef = useRef(false);
  const voiceErrorFadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isHero = variant === "hero";
  const voiceShortcutLabel = useMemo(getVoiceShortcutLabel, []);
  const queuedPromptStorageKey = useMemo(
    () => queuedPromptsStorageKey(pendingQueueKey),
    [pendingQueueKey],
  );
  const setPromptAwaitingConfirmation = useCallback((
    promptId: string,
    queueKey: string | null,
    awaiting: boolean,
  ) => {
    if (currentPendingQueueKeyRef.current !== queueKey) return;
    if (awaiting) awaitingConfirmationPromptIdsRef.current.add(promptId);
    else awaitingConfirmationPromptIdsRef.current.delete(promptId);
    setAwaitingConfirmationPromptIds((current) => {
      if (current.has(promptId) === awaiting) return current;
      const next = new Set(current);
      if (awaiting) next.add(promptId);
      else next.delete(promptId);
      return next;
    });
  }, []);

  const commitQueuedPrompts = useCallback((next: QueuedPrompt[]): boolean => {
    if (next === queuedPromptsRef.current) return true;
    if (queuedPromptStorageKey && !storeQueuedPrompts(queuedPromptStorageKey, next)) {
      setInlineError(t("thread.composer.queuePersistenceError", {
        defaultValue: "Queued messages could not be saved safely. Try again before sending.",
      }));
      return false;
    }
    queuedPromptsRef.current = next;
    setQueuedPrompts(next);
    return true;
  }, [queuedPromptStorageKey, t]);

  const applyQueuedAttemptOutcome = useCallback((
    binding: QueuedAttemptBinding,
    outcome: QueuedAttemptOutcome,
  ) => {
    if (currentPendingQueueKeyRef.current !== binding.queueKey) return;
    const current = queuedPromptsRef.current;
    const promptIndex = current.findIndex((prompt) => (
      prompt.id === binding.promptId
      && prompt.submission.state === "submitted"
      && prompt.submission.binding.queueKey === binding.queueKey
      && prompt.submission.binding.promptId === binding.promptId
      && prompt.submission.binding.turnId === binding.turnId
    ));
    if (promptIndex < 0) return;
    const prompt = current[promptIndex];
    if (!prompt) return;
    const next = outcome.kind === "admitted"
      ? current.filter((item) => item.id !== binding.promptId)
      : current.map((item, index) => index === promptIndex
          ? {
              ...item,
              submission: {
                version: 1 as const,
                state: "not_admitted" as const,
                binding,
                reason: outcome.reason,
              },
            }
          : item);
    if (!commitQueuedPrompts(next)) return;
    setPromptAwaitingConfirmation(prompt.id, binding.queueKey, false);
    heldQueuedPromptIdsRef.current.delete(prompt.id);
    onConsumeQueuedAttempt?.(binding);
    if (outcome.kind === "not_admitted") {
      onAdmissionRejectedPromptRestored?.(binding.turnId);
    }
  }, [
    commitQueuedPrompts,
    onAdmissionRejectedPromptRestored,
    onConsumeQueuedAttempt,
    setPromptAwaitingConfirmation,
  ]);

  const observeQueuedAttempt = useCallback((binding: QueuedAttemptBinding) => {
    if (!onJoinQueuedAttempt) return;
    const joinKey = JSON.stringify(binding);
    if (queuedAttemptJoinKeysRef.current.has(joinKey)) return;
    const joined = onJoinQueuedAttempt(binding);
    if (!joined) return;
    queuedAttemptJoinKeysRef.current.add(joinKey);
    const generation = queueViewGenerationRef.current;
    void joined.then((outcome) => {
      if (queueViewGenerationRef.current !== generation) return;
      applyQueuedAttemptOutcome(binding, outcome);
    }).catch(() => {
      // Lost observation remains held; only an exact owner outcome can unlock it.
    });
  }, [applyQueuedAttemptOutcome, onJoinQueuedAttempt]);

  useEffect(() => {
    const generation = ++queueViewGenerationRef.current;
    heldQueuedPromptIdsRef.current.clear();
    awaitingConfirmationPromptIdsRef.current.clear();
    queuedSubmissionInProgressRef.current.clear();
    queuedAttemptJoinKeysRef.current.clear();
    const restored = queuedPromptStorageKey ? readQueuedPrompts(queuedPromptStorageKey) : [];
    queuedPromptsRef.current = restored;
    setQueuedPrompts(restored);
    const awaiting = new Set<string>();
    for (const prompt of restored) {
      if (prompt.submission.state === "submitted" || prompt.submission.state === "unknown") {
        awaiting.add(prompt.id);
        heldQueuedPromptIdsRef.current.add(prompt.id);
      }
    }
    awaitingConfirmationPromptIdsRef.current = awaiting;
    setAwaitingConfirmationPromptIds(awaiting);
    return () => {
      if (queueViewGenerationRef.current === generation) queueViewGenerationRef.current += 1;
    };
  }, [pendingQueueKey, queuedPromptStorageKey]);

  useEffect(() => {
    for (const prompt of queuedPrompts) {
      if (prompt.submission.state === "submitted") {
        observeQueuedAttempt(prompt.submission.binding);
      }
    }
  }, [observeQueuedAttempt, queuedPrompts]);

  useEffect(() => {
    const rejectedTurnId = admissionRejectedTurn?.turnId;
    const reason = admissionRejectedTurn?.reason?.trim();
    if (!rejectedTurnId || !reason) return;
    const prompt = queuedPromptsRef.current.find((item) => (
      item.submission.state === "submitted"
      && item.submission.binding.turnId === rejectedTurnId
    ));
    if (!prompt || prompt.submission.state !== "submitted") return;
    applyQueuedAttemptOutcome(prompt.submission.binding, { kind: "not_admitted", reason });
  }, [admissionRejectedTurn?.reason, admissionRejectedTurn?.turnId, applyQueuedAttemptOutcome]);

  const projectPickerAvailable =
    !!workspaceDefaultScope
    && !!onWorkspaceScopeChange;
  const showWorkspacePicker = projectPickerAvailable && !workspaceControlsHidden;

  // The transcript's activity fold owns the "still working" signal. The
  // composer keeps its normal placeholder and shows the stop control instead,
  // so one state is never announced twice.
  const resolvedPlaceholder = placeholder ?? t("thread.composer.placeholderThread");

  const maxAttachments = ingressLimits?.attachments.max_count
    ?? MAX_ATTACHMENTS_PER_MESSAGE;
  const maxTextBytes = ingressLimits?.message.max_text_bytes ?? 64 * 1024;
  const { images, enqueue, remove, clear, restoreReadyImages, encoding, full } =
    useAttachedImages({ ingressLimits });

  const formatRejection = useCallback(
    (reason: AttachmentError): string => {
      const key = `thread.composer.imageRejected.${reason}`;
      const fallback = reason === "too_many_attachments"
        ? `Max ${maxAttachments} attachments per message`
        : reason === "empty_file"
          ? "Empty files cannot be attached"
          : reason === "total_too_large"
            ? "Attachments are too large together — remove some or use smaller files"
            : reason === "transport_too_large"
              ? "This attachment would exceed the gateway transport limit"
              : reason === "too_large"
                ? "File is too large"
                : "Unsupported file type";
      return t(key, { max: maxAttachments, defaultValue: fallback });
    },
    [maxAttachments, t],
  );

  const textTooLargeMessage = useCallback(
    () => t("thread.composer.textTooLarge", {
      max: formatBytes(maxTextBytes),
      defaultValue: `Message text is too large (max ${formatBytes(maxTextBytes)})`,
    }),
    [maxTextBytes, t],
  );

  const addFiles = useCallback(
    (files: File[]) => {
      if (interactionDisabled || files.length === 0) return;
      const { rejected } = enqueue(files);
      if (rejected.length > 0) {
        setInlineError(formatRejection(rejected[0].reason));
      } else {
        setInlineError(null);
      }
    },
    [enqueue, formatRejection, interactionDisabled],
  );

  const mobileTransport = getMobileRuntime()?.transport;
  const pickCamera = useCallback(async () => {
    if (!mobileTransport?.pickCamera) return;
    const draftViewVersion = draftViewVersionRef.current;
    try {
      const picked = await mobileTransport.pickCamera();
      if (!picked || draftViewVersionRef.current !== draftViewVersion) return;
      const binary = atob(picked.dataBase64);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      addFiles([new File([bytes], picked.name, { type: picked.mediaType })]);
    } catch (error) {
      if (draftViewVersionRef.current === draftViewVersion) {
        setInlineError(error instanceof Error ? error.message : "Camera capture failed");
      }
    }
  }, [addFiles, mobileTransport]);

  const {
    isDragging,
    onPaste,
    onDragEnter,
    onDragOver,
    onDragLeave,
    onDrop,
  } = useClipboardAndDrop(addFiles);

  const autoFocusScopeRef = useRef<string | null>(null);
  const autoFocusKey = `${draftNamespace}\u0000${draftKey ?? NEW_CHAT_DRAFT_KEY}`;
  useEffect(() => {
    if (interactionDisabled || hasTouchPrimaryPointer || (workspaceError && showWorkspacePicker)) {
      return;
    }
    if (autoFocusScopeRef.current === autoFocusKey) return;
    autoFocusScopeRef.current = autoFocusKey;
    const existingDraft = loadComposerDraft({
      namespace: draftNamespace,
      key: draftKey ?? NEW_CHAT_DRAFT_KEY,
      incarnation: draftIncarnation,
      persist: false,
    });
    if (existingDraft && !existingDraft.focused) return;
    const el = textareaRef.current;
    if (!el) return;
    const id = requestAnimationFrame(() => {
      const active = document.activeElement;
      if (active && active !== document.body && active !== document.documentElement && active !== el) {
        return;
      }
      el.focus();
    });
    return () => cancelAnimationFrame(id);
  }, [autoFocusKey, draftIncarnation, draftKey, draftNamespace, hasTouchPrimaryPointer, interactionDisabled, showWorkspacePicker, workspaceError]);

  useEffect(() => {
    if (!focusRequest || interactionDisabled) return;
    const id = requestAnimationFrame(() => textareaRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, [focusRequest, interactionDisabled]);

  const normalizedQuotedContext = quotedContext?.trim().slice(0, QUEUED_PROMPT_MAX_CHARS) || null;

  const readyImages = useMemo(
    () => images.filter((img): img is AttachedImage & { dataUrl: string } =>
      img.status === "ready" && typeof img.dataUrl === "string",
    ),
    [images],
  );
  const hasErrors = images.some((img) => img.status === "error");
  const workspaceFullRetiring = persistentWorkspaceChat
    && workspacePermissions?.mode === "full"
    && workspacePermissions.state === "retiring";

  const hasComposerContent = value.trim().length > 0 || readyImages.length > 0;
  const canSend =
    !interactionDisabled
    && !newTurnBlocked
    && !workspaceFullRetiring
    && !modelNeedsSetup
    && !encoding
    && !hasErrors
    && hasComposerContent;
  const canOpenModelSettings = Boolean(
    modelNeedsSetup && onModelBadgeClick && !interactionDisabled,
  );
  const canQueueGuidance =
    isStreaming
    && !interactionDisabled
    && !workspaceFullRetiring
    && !modelNeedsSetup
    && !encoding
    && !hasErrors
    && hasComposerContent
    && !value.trimStart().startsWith("/");

  const slashQuery = useMemo(() => {
    if (interactionDisabled || slashMenuDismissed || !value.startsWith("/")) return null;
    const commandToken = value.slice(1);
    if (/\s/.test(commandToken)) return null;
    return commandToken.toLowerCase();
  }, [interactionDisabled, slashMenuDismissed, value]);

  const skillQuery = useMemo(() => {
    if (interactionDisabled || slashMenuDismissed) return null;
    const caret = Math.min(Math.max(cursorPosition, 0), value.length);
    const beforeCaret = value.slice(0, caret);
    const match = /\$([A-Za-z0-9_-]*)$/i.exec(beforeCaret);
    if (!match) return null;
    return {
      end: caret,
      start: match.index,
      text: match[1].toLowerCase(),
    };
  }, [cursorPosition, interactionDisabled, slashMenuDismissed, value]);

  const visibleSlashCommands = useMemo(() => {
    if (!(hasStopAction && onStop)) return slashCommands;
    const stopCommand = slashCommands.find((command) => command.command === "/stop");
    if (!stopCommand) return slashCommands;
    return [
      stopCommand,
      ...slashCommands.filter((command) => command.command !== "/stop"),
    ];
  }, [hasStopAction, onStop, slashCommands]);

  const filteredSlashCommands = useMemo<SlashPaletteCommand[]>(() => {
    if (skillQuery !== null) {
      const query = skillQuery.text;
      return skills
        .filter((skill) => skill.enabled !== false && skill.available)
        .flatMap((skill) => {
          const matchRank = skillMatchRank(skill, query);
          return matchRank === null
            ? []
            : [{
                command: `$${skill.name}`,
                matchRank,
                skill,
              }];
        })
        .sort((a, b) => {
          if (a.matchRank !== b.matchRank) return a.matchRank - b.matchRank;
          if (query !== "") return 0;
          const aRecent = recentSlashCommands.indexOf(a.command);
          const bRecent = recentSlashCommands.indexOf(b.command);
          if (aRecent === -1 && bRecent === -1) return 0;
          if (aRecent === -1) return 1;
          if (bRecent === -1) return -1;
          return aRecent - bRecent;
        })
        .slice(0, 8)
        .map(({ command, skill }) => {
          const description = skill.description || skill.name;
          return {
            command,
            title: skill.name,
            description,
            detail: description,
            icon: "brain",
            kind: "skill" as const,
            recent: recentSlashCommands.includes(command),
          };
        });
    }
    if (slashQuery === null) return [];
    const withDetails = visibleSlashCommands
      .filter((command) => {
        if (
          slashQuery === ""
          && (
            command.command === "/restart"
            || (command.command === "/stop" && !(hasStopAction && onStop))
          )
        ) {
          return false;
        }
        const commandKey = slashCommandI18nKey(command.command);
        const title = t(`thread.composer.slash.commands.${commandKey}.title`, {
          defaultValue: command.title,
        });
        const description = t(`thread.composer.slash.commands.${commandKey}.description`, {
          defaultValue: command.description,
        });
        const haystack = [
          command.command,
          command.title,
          command.description,
          command.argHint ?? "",
          title,
          description,
        ].join(" ").toLowerCase();
        return haystack.includes(slashQuery);
      })
      .map((command) => {
        const commandKey = slashCommandI18nKey(command.command);
        const description = t(`thread.composer.slash.commands.${commandKey}.description`, {
          defaultValue: command.description,
        });
        let detail = description;
        let badge: string | undefined;
        if (command.command === "/model" && modelLabel) {
          detail = modelLabel;
          badge = t("thread.composer.slash.badges.current");
        } else if (command.command === "/goal") {
          detail = goalState?.active
            ? t("thread.composer.slash.details.goalActive")
            : t("thread.composer.slash.details.goalReady");
        } else if (command.command === "/stop" && hasStopAction) {
          detail = t("thread.composer.slash.details.stopRunning");
        } else if (command.command === "/history") {
          detail = t("thread.composer.slash.details.history");
        }
        return {
          ...command,
          detail,
          badge,
          recent: recentSlashCommands.includes(command.command),
        };
      })
      .sort((a, b) => {
        if (hasStopAction) {
          if (a.command === "/stop") return -1;
          if (b.command === "/stop") return 1;
        }
        if (slashQuery !== "") return 0;
        const aRecent = recentSlashCommands.indexOf(a.command);
        const bRecent = recentSlashCommands.indexOf(b.command);
        if (aRecent !== -1 || bRecent !== -1) {
          if (aRecent === -1) return 1;
          if (bRecent === -1) return -1;
          return aRecent - bRecent;
        }
        return 0;
      });

    return withDetails
      .slice(0, 8);
  }, [goalState?.active, hasStopAction, modelLabel, recentSlashCommands, skills, skillQuery, slashQuery, t, visibleSlashCommands]);

  const showSlashMenu = filteredSlashCommands.length > 0;
  const cliAppMention = useMemo<CliAppMentionQuery | null>(() => {
    if (interactionDisabled || cliAppMenuDismissed) return null;
    const caret = Math.min(Math.max(cursorPosition, 0), value.length);
    const beforeCaret = value.slice(0, caret);
    const match = /(?:^|\s)@([\p{L}\p{N}_-]*)$/iu.exec(beforeCaret);
    if (!match) return null;
    const query = match[1].toLowerCase();
    return {
      query,
      start: caret - query.length - 1,
      end: caret,
    };
  }, [cliAppMenuDismissed, cursorPosition, interactionDisabled, value]);

  const availableSessionMentions = useMemo(
    () => sessionMentionOptions(sessions),
    [sessions],
  );
  const mentionSegments = useMemo(
    () => splitCapabilityMentionSegments(value, cliApps, mcpPresets, selectedSessionMentions),
    [cliApps, mcpPresets, selectedSessionMentions, value],
  );
  const sessionDragInsertion = sessionDragPreview
    ? mentionInsertion(
        value,
        sessionDragPreview.mention.name,
        sessionDragPreview.start,
        sessionDragPreview.end,
      )
    : null;
  const displayMentionSegments = sessionDragInsertion && sessionDragPreview
    ? splitCapabilityMentionSegments(
        sessionDragInsertion.value,
        cliApps,
        mcpPresets,
        [...selectedSessionMentions, sessionDragPreview.mention],
      )
    : mentionSegments;
  const activeSessionMentions = useMemo(() => {
    const seen = new Set<string>();
    return mentionSegments.flatMap((segment) => {
      if (segment.kind !== "session" || seen.has(segment.mention.session_key)) return [];
      seen.add(segment.mention.session_key);
      return [segment.mention];
    }).slice(0, SESSION_MENTIONS_LIMIT);
  }, [mentionSegments]);
  const filteredMentionCandidates = useMemo<MentionCandidate[]>(() => {
    if (!cliAppMention) return [];
    const sessionCandidates: MentionCandidate[] = availableSessionMentions
      .filter((mention) => (
        activeSessionMentions.length < SESSION_MENTIONS_LIMIT
        || activeSessionMentions.some(
          (selected) => selected.session_key === mention.session_key,
        )
      ))
      .filter((mention) => [
        mention.name,
        mention.title,
      ].join(" ").toLowerCase().includes(cliAppMention.query))
      .map((mention) => ({
        kind: "session",
        name: mention.name,
        displayName: mention.title || mention.name,
        mention,
      }));
    const sessionNames = new Set(
      availableSessionMentions.map((mention) => mention.name.toLowerCase()),
    );
    const cliCandidates: MentionCandidate[] = cliApps
      .filter((app) => app.installed)
      .filter((app) => !sessionNames.has(app.name.toLowerCase()))
      .filter((app) => {
        const haystack = [
          app.name,
          app.display_name,
          app.category,
          app.description,
          app.entry_point,
        ].join(" ").toLowerCase();
        return haystack.includes(cliAppMention.query);
      })
      .map((app) => ({
        kind: "cli",
        name: app.name,
        displayName: app.display_name,
        brandColor: app.brand_color ?? null,
        logoUrl: app.logo_url ?? null,
        initials: cliAppInitials(app),
      }));
    const mcpCandidates: MentionCandidate[] = mcpPresets
      .filter((preset) => preset.installed && preset.configured)
      .filter((preset) => !sessionNames.has(preset.name.toLowerCase()))
      .filter((preset) => {
        const haystack = [
          preset.name,
          preset.display_name,
          preset.category,
          preset.description,
          preset.transport,
        ].join(" ").toLowerCase();
        return haystack.includes(cliAppMention.query);
      })
      .map((preset) => ({
        kind: "mcp",
        name: preset.name,
        displayName: preset.display_name,
        brandColor: preset.brand_color ?? null,
        logoUrl: preset.logo_url ?? null,
        initials: mcpPresetInitials(preset),
      }));
    const groups = [
      { candidates: sessionCandidates, reserved: 4 },
      { candidates: cliCandidates, reserved: 2 },
      { candidates: mcpCandidates, reserved: 2 },
    ];
    let remaining = 8;
    const counts = groups.map(({ candidates, reserved }) => {
      const count = Math.min(candidates.length, reserved);
      remaining -= count;
      return count;
    });
    for (const index of [0, 1, 2]) {
      const extra = Math.min(remaining, groups[index].candidates.length - counts[index]);
      counts[index] += extra;
      remaining -= extra;
    }
    return groups.flatMap(({ candidates }, index) => candidates.slice(0, counts[index]));
  }, [activeSessionMentions, availableSessionMentions, cliAppMention, cliApps, mcpPresets]);

  const showCliAppMenu = filteredMentionCandidates.length > 0;
  const showAnyPalette = showSlashMenu || showCliAppMenu;
  const hasMentionDecorations = displayMentionSegments.some(
    (segment) => segment.kind !== "text",
  );
  const activeCliMentionApps = useMemo(() => {
    const seen = new Set<string>();
    return mentionSegments.flatMap((segment) => {
      if (segment.kind !== "cli" || seen.has(segment.app.name)) return [];
      seen.add(segment.app.name);
      return [segment.app];
    });
  }, [mentionSegments]);
  const activeMcpPresetMentions = useMemo(() => {
    const seen = new Set<string>();
    return mentionSegments.flatMap((segment) => {
      if (segment.kind !== "mcp" || seen.has(segment.preset.name)) return [];
      seen.add(segment.preset.name);
      return [segment.preset];
    });
  }, [mentionSegments]);
  const [slashPaletteLayout, setSlashPaletteLayout] = useState<SlashPaletteLayout>({
    placement: "above",
    maxHeight: SLASH_PALETTE_MAX_HEIGHT_PX,
  });

  useEffect(() => {
    setSelectedCommandIndex(0);
  }, [slashQuery]);

  useEffect(() => {
    setSelectedCliAppIndex(0);
  }, [cliAppMention?.query]);

  useEffect(() => {
    if (selectedCommandIndex >= filteredSlashCommands.length) {
      setSelectedCommandIndex(0);
    }
  }, [filteredSlashCommands.length, selectedCommandIndex]);

  useEffect(() => {
    if (selectedCliAppIndex >= filteredMentionCandidates.length) {
      setSelectedCliAppIndex(0);
    }
  }, [filteredMentionCandidates.length, selectedCliAppIndex]);

  useEffect(() => {
    if (!showAnyPalette) return;

    const dismissOnPointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Node && formRef.current?.contains(target)) return;
      setSlashMenuDismissed(true);
      setCliAppMenuDismissed(true);
    };

    document.addEventListener("pointerdown", dismissOnPointerDown, true);
    return () => {
      document.removeEventListener("pointerdown", dismissOnPointerDown, true);
    };
  }, [showAnyPalette]);

  useLayoutEffect(() => {
    if (!showAnyPalette) return;

    const updateLayout = () => {
      const form = formRef.current;
      if (!form) return;
      const rect = form.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) return;

      const bounds = getVisibleBounds(form);
      const spaceAbove = Math.max(0, rect.top - bounds.top - SLASH_PALETTE_GAP_PX);
      const spaceBelow = Math.max(0, bounds.bottom - rect.bottom - SLASH_PALETTE_GAP_PX);
      const placement: SlashPalettePlacement =
        spaceAbove >= SLASH_PALETTE_MIN_HEIGHT_PX || spaceAbove >= spaceBelow
          ? "above"
          : "below";
      const available = placement === "above" ? spaceAbove : spaceBelow;
      const maxHeight = Math.min(SLASH_PALETTE_MAX_HEIGHT_PX, available);

      setSlashPaletteLayout((current) =>
        current.placement === placement && current.maxHeight === maxHeight
          ? current
          : { placement, maxHeight },
      );
    };

    updateLayout();
    const viewport = window.visualViewport;
    viewport?.addEventListener("resize", updateLayout);
    viewport?.addEventListener("scroll", updateLayout);
    window.addEventListener("resize", updateLayout);
    document.addEventListener("scroll", updateLayout, true);
    return () => {
      viewport?.removeEventListener("resize", updateLayout);
      viewport?.removeEventListener("scroll", updateLayout);
      window.removeEventListener("resize", updateLayout);
      document.removeEventListener("scroll", updateLayout, true);
    };
  }, [filteredMentionCandidates.length, filteredSlashCommands.length, showAnyPalette]);

  const resizeTextarea = useCallback((restoreFocus = true) => {
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 260)}px`;
      if (restoreFocus) el.focus();
    });
  }, []);

  const draftScope = useMemo<DraftScope>(() => ({
    namespace: draftNamespace,
    key: draftKey ?? NEW_CHAT_DRAFT_KEY,
    incarnation: draftIncarnation,
    persist: !!draftKey && !!pendingQueueKey,
  }), [draftIncarnation, draftKey, draftNamespace, pendingQueueKey]);
  const draftScopeRef = useRef(draftScope);
  draftScopeRef.current = draftScope;
  const knownIncarnationRef = useRef<string | null>(null);
  const draftRevokedRef = useRef(draftRevoked);
  draftRevokedRef.current = draftRevoked;
  const draftScopeId_ = draftScopeId(draftScope);
  const previousDraftScopeRef = useRef<DraftScope | null>(null);
  draftLiveRef.current = {
    value,
    mentions: selectedSessionMentions,
    attachments: readyImages.map((img) => ({
      dataUrl: img.dataUrl,
      name: img.file.name,
      kind: img.kind,
    })),
    quote: normalizedQuotedContext,
  };

  // Runs before paint so switching sessions never flashes stale draft text:
  // the leaving chat's draft (with caret/focus) is saved, the entering
  // chat's draft is restored.
  useLayoutEffect(() => {
    const previousScope = previousDraftScopeRef.current;
    const first = previousScope === null;
    if (!first && draftScopeId(previousScope) === draftScopeId_) {
      previousDraftScopeRef.current = draftScope;
      return;
    }
    const el = textareaRef.current;
    if (previousScope && !draftRevokedRef.current) {
      const live = draftLiveRef.current;
      saveComposerDraft(previousScope, {
        text: live.value,
        quotedContext: live.quote,
        mentions: live.mentions,
        attachments: live.attachments,
        reattachNames: reattachNamesRef.current,
        selection: el ? { start: el.selectionStart, end: el.selectionEnd } : null,
        focused: !!el && document.activeElement === el,
      });
    }
    previousDraftScopeRef.current = draftScope;
    previousDraftKeyRef.current = draftKey;
    draftAppliedRef.current = true;
    const draft = loadComposerDraft(draftScope);
    knownIncarnationRef.current = draftScope.incarnation ?? draft?.incarnation ?? null;
    if (first && !draft) return;
    const restored = draft?.text ?? "";
    setQueuedEdit(null);
    setValue(restored);
    setSelectedSessionMentions(normalizeQueuedSessionMentions(draft?.mentions ?? []));
    setInlineError(null);
    setSlashMenuDismissed(false);
    setCliAppMenuDismissed(false);
    const caret = draft?.selection
      ? {
        start: Math.min(draft.selection.start, restored.length),
        end: Math.min(draft.selection.end, restored.length),
      }
      : { start: restored.length, end: restored.length };
    setCursorPosition(caret.end);
    clear();
    if (draft?.attachments.length) restoreReadyImages(draft.attachments);
    reattachNamesRef.current = draft?.reattachNames ?? [];
    setReattachNames(reattachNamesRef.current);
    onQuotedContextChange?.(draft?.quotedContext ?? null);
    const scopeId = draftScopeId_;
    requestAnimationFrame(() => {
      const target = textareaRef.current;
      if (!target || draftScopeId(draftScopeRef.current) !== scopeId) return;
      target.style.height = "auto";
      target.style.height = `${Math.min(target.scrollHeight, 260)}px`;
      if (!draft) return;
      try { target.setSelectionRange(caret.start, caret.end); } catch { /* detached */ }
      // Restore focus only where the user left it, and never pop a soft keyboard.
      if (draft.focused && !hasTouchPrimaryPointer && document.activeElement !== target) {
        target.focus({ preventScroll: true });
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftScope, draftScopeId_]);

  // A different incarnation of the same session key is a replacement/reset:
  // the draft and attachment references belong to the old one.
  useEffect(() => {
    if (!draftIncarnation) return;
    const known = knownIncarnationRef.current;
    const replaced =
      (!!known && known !== draftIncarnation)
      || draftIncarnationReplaced(draftScopeRef.current, draftIncarnation);
    knownIncarnationRef.current = draftIncarnation;
    if (!replaced) return;
    clearComposerDraftsForSession(draftScopeRef.current.namespace, draftScopeRef.current.key);
    reattachNamesRef.current = [];
    setReattachNames([]);
    setValue("");
    setSelectedSessionMentions([]);
    clear();
    onQuotedContextChange?.(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftIncarnation]);

  useEffect(() => {
    if (!draftRevoked) return;
    clearComposerDraftsForSession(draftScopeRef.current.namespace, draftScopeRef.current.key);
    reattachNamesRef.current = [];
    setReattachNames([]);
    setValue("");
    setSelectedSessionMentions([]);
    clear();
    onQuotedContextChange?.(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftRevoked]);

  // Persist the live draft (memory immediately, localStorage when persistent)
  // for the state that belongs to the active scope. Skip the pass in which a
  // switch just applied but state still belongs to the previous chat.
  useEffect(() => {
    if (draftAppliedRef.current) {
      draftAppliedRef.current = false;
      return;
    }
    if (draftRevokedRef.current) return;
    const live = draftLiveRef.current;
    const el = textareaRef.current;
    saveComposerDraft(draftScopeRef.current, {
      text: live.value,
      quotedContext: live.quote,
      mentions: live.mentions,
      attachments: live.attachments,
      reattachNames,
      selection: el ? { start: el.selectionStart, end: el.selectionEnd } : null,
      focused: !!el && document.activeElement === el,
    });
  }, [
    draftIncarnation,
    draftScopeId_,
    normalizedQuotedContext,
    readyImages,
    reattachNames,
    selectedSessionMentions,
    value,
  ]);
  useLayoutEffect(() => {
    setSendPending(false);
    return () => {
      pendingSendRef.current = null;
      draftViewVersionRef.current += 1;
      voiceShortcutDownRef.current = false;
    };
  }, [draftKey]);

  useLayoutEffect(() => {
    if (!initialDraft) return;
    setValue(initialDraft);
    onInitialDraftConsumed?.();
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 260)}px`;
      el.focus();
    });
  }, [initialDraft, onInitialDraftConsumed]);

  const appendTranscription = useCallback((text: string) => {
    const transcript = text.trim();
    if (!transcript) return;
    setValue((current) => {
      if (!current.trim()) return transcript;
      const separator = /[\s\n]$/.test(current) ? "" : " ";
      return `${current}${separator}${transcript}`;
    });
    setSlashMenuDismissed(false);
    setCliAppMenuDismissed(false);
    setInlineError(null);
    resizeTextarea();
  }, [resizeTextarea]);

  const clearVoiceErrorTimers = useCallback(() => {
    if (voiceErrorFadeTimerRef.current !== null) clearTimeout(voiceErrorFadeTimerRef.current);
    voiceErrorFadeTimerRef.current = null;
  }, []);
  const clearInlineError = useCallback(() => {
    clearVoiceErrorTimers();
    setVoiceErrorFading(false);
    setInlineError(null);
  }, [clearVoiceErrorTimers]);
  const setVoiceError = useCallback((key: VoiceRecorderErrorKey) => {
    clearVoiceErrorTimers();
    setVoiceErrorFading(false);
    const isNative = mobileRuntimeActive || typeof window.__TAURI_INTERNALS__ !== "undefined";
    const messageKey = key === "permission" && isNative ? "nativePermission" : key;
    setInlineError(t(`thread.composer.voiceErrors.${messageKey}`));
    voiceErrorFadeTimerRef.current = setTimeout(() => {
      setVoiceErrorFading(true);
      voiceErrorFadeTimerRef.current = setTimeout(() => {
        setInlineError(null);
        setVoiceErrorFading(false);
        voiceErrorFadeTimerRef.current = null;
      }, VOICE_ERROR_FADE_MS);
    }, VOICE_ERROR_VISIBLE_MS);
  }, [clearVoiceErrorTimers, mobileRuntimeActive, t]);
  const voiceRecorder = useVoiceRecorder({
    disabled: interactionDisabled,
    draftKey,
    onClearError: clearInlineError,
    onError: setVoiceError,
    onTranscript: appendTranscription,
    onTranscribeAudio,
    wantsWav: transcriptionProvider === "xiaomi_mimo",
  });

  useEffect(() => () => clearVoiceErrorTimers(), [clearVoiceErrorTimers, draftKey]);

  useEffect(() => {
    if (!onTranscribeAudio) return;

    function onKeyDown(event: KeyboardEvent): void {
      if (!isVoiceShortcutDown(event) || event.repeat || voiceShortcutDownRef.current) return;
      event.preventDefault();
      voiceShortcutDownRef.current = true;
      voiceRecorder.beginShortcutHold();
    }

    function onKeyUp(event: KeyboardEvent): void {
      if (!voiceShortcutDownRef.current || !isVoiceShortcutRelease(event)) return;
      event.preventDefault();
      voiceShortcutDownRef.current = false;
      voiceRecorder.endShortcutHold();
    }

    function onWindowBlur(): void {
      if (!voiceShortcutDownRef.current) return;
      voiceShortcutDownRef.current = false;
      voiceRecorder.endShortcutHold();
    }

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onWindowBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onWindowBlur);
    };
  }, [onTranscribeAudio, voiceRecorder.beginShortcutHold, voiceRecorder.endShortcutHold]);

  const handleStop = useCallback(() => {
    if (disabled || stopDisabled || stopPending || !onStop) return false;
    handledQueueCompletionRef.current = queueCompletionTurnId;
    onStop();
    return true;
  }, [disabled, onStop, queueCompletionTurnId, stopDisabled, stopPending]);

  const chooseSlashCommand = useCallback(
    (command: SlashPaletteCommand) => {
      if (command.command === "/stop" && hasStopAction && onStop) {
        if (!handleStop()) return;
        setValue("");
        setSlashMenuDismissed(true);
        setCliAppMenuDismissed(false);
        setInlineError(null);
        resizeTextarea();
        return;
      }

      const nextRecents = [
        command.command,
        ...recentSlashCommands.filter((item) => item !== command.command),
      ].slice(0, SLASH_RECENTS_LIMIT);
      setRecentSlashCommands(nextRecents);
      storeSlashRecents(nextRecents);

      if (skillQuery !== null) {
        const suffix = value.slice(skillQuery.end);
        const inserted = `${command.command}${suffix.startsWith(" ") ? "" : " "}`;
        const next = `${value.slice(0, skillQuery.start)}${inserted}${suffix}`;
        const nextCursor = skillQuery.start + inserted.length;
        setValue(next);
        setCursorPosition(nextCursor);
        requestAnimationFrame(() => {
          const el = textareaRef.current;
          if (!el) return;
          el.focus();
          el.setSelectionRange(nextCursor, nextCursor);
        });
      } else {
        setValue(command.argHint ? `${command.command} ` : command.command);
      }
      setSlashMenuDismissed(true);
      setCliAppMenuDismissed(false);
      setInlineError(null);
      resizeTextarea();
    },
    [handleStop, hasStopAction, onStop, recentSlashCommands, resizeTextarea, skillQuery, value],
  );

  const insertMentionCandidate = useCallback(
    (candidate: MentionCandidate, start: number, end: number) => {
      if (candidate.kind === "session") {
        const alreadySelected = activeSessionMentions.some(
          (mention) => mention.session_key === candidate.mention.session_key,
        );
        if (!alreadySelected && activeSessionMentions.length >= SESSION_MENTIONS_LIMIT) return;
        const name = candidate.name.toLowerCase();
        setSelectedSessionMentions([
          ...activeSessionMentions.filter((mention) => (
            mention.name.toLowerCase() !== name
            && mention.session_key !== candidate.mention.session_key
          )),
          candidate.mention,
        ]);
      }
      const insertion = mentionInsertion(value, candidate.name, start, end);
      setValue(insertion.value);
      setCursorPosition(insertion.cursor);
      setCliAppMenuDismissed(true);
      setSlashMenuDismissed(false);
      setInlineError(null);
      resizeTextarea();
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.focus();
        el.setSelectionRange(insertion.cursor, insertion.cursor);
      });
    },
    [activeSessionMentions, resizeTextarea, value],
  );

  const chooseMentionCandidate = useCallback(
    (candidate: MentionCandidate) => {
      if (!cliAppMention) return;
      insertMentionCandidate(candidate, cliAppMention.start, cliAppMention.end);
    },
    [cliAppMention, insertMentionCandidate],
  );

  const handleSessionDrop = useCallback((event: React.DragEvent) => {
    if (!hasDraggedSession(event.dataTransfer)) return false;
    event.preventDefault();
    clearDraggedSession();
    const preview = sessionDragPreview;
    setSessionDragPreview(null);
    if (interactionDisabled) return true;
    const sessionKey = readDraggedSession(event.dataTransfer);
    const mention = availableSessionMentions.find(
      (candidate) => candidate.session_key === (sessionKey ?? preview?.mention.session_key),
    );
    if (!mention) return true;
    const caret = preview?.start ?? textareaRef.current?.selectionStart ?? value.length;
    insertMentionCandidate(
      {
        kind: "session",
        name: mention.name,
        displayName: mention.title || mention.name,
        mention,
      },
      caret,
      preview?.end ?? textareaRef.current?.selectionEnd ?? caret,
    );
    return true;
  }, [
    availableSessionMentions,
    insertMentionCandidate,
    interactionDisabled,
    sessionDragPreview,
    value.length,
  ]);

  const previewSessionDrop = useCallback((event: React.DragEvent) => {
    if (!hasDraggedSession(event.dataTransfer)) return false;
    if (interactionDisabled) {
      event.dataTransfer.dropEffect = "none";
      setSessionDragPreview(null);
      return true;
    }
    const sessionKey = readDraggedSession(event.dataTransfer);
    const mention = availableSessionMentions.find(
      (candidate) => candidate.session_key === sessionKey,
    );
    const alreadySelected = mention && activeSessionMentions.some(
      (candidate) => candidate.session_key === mention.session_key,
    );
    if (!mention || (!alreadySelected && activeSessionMentions.length >= SESSION_MENTIONS_LIMIT)) {
      event.dataTransfer.dropEffect = "none";
      setSessionDragPreview(null);
      return true;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    const start = textareaRef.current?.selectionStart ?? value.length;
    const end = textareaRef.current?.selectionEnd ?? start;
    setSessionDragPreview((current) => (
      current?.mention.session_key === mention.session_key
      && current.start === start
      && current.end === end
        ? current
        : { mention, start, end }
    ));
    return true;
  }, [activeSessionMentions, availableSessionMentions, interactionDisabled, value.length]);

  useEffect(() => {
    if (!sessionDragPreview) return;
    const clearPreview = () => {
      clearDraggedSession();
      setSessionDragPreview(null);
    };
    document.addEventListener("dragend", clearPreview);
    return () => document.removeEventListener("dragend", clearPreview);
  }, [sessionDragPreview]);

  const clearComposerText = useCallback((restoreFocus = true) => {
    setValue("");
    setSelectedSessionMentions([]);
    setInlineError(null);
    setSlashMenuDismissed(false);
    setCliAppMenuDismissed(false);
    setCursorPosition(0);
    resizeTextarea(restoreFocus);
  }, [resizeTextarea]);

  const queueGuidancePrompt = useCallback(() => {
    const text = value.trim();
    if (!canQueueGuidance || (!text && readyImages.length === 0)) return;
    if (utf8Bytes(formatQuotedUserMessage(text, normalizedQuotedContext)) > maxTextBytes) {
      setInlineError(textTooLargeMessage());
      return;
    }
    const queuedImages = readyImagesToQueuedImages(readyImages);
    queuedPromptCounterRef.current += 1;
    const id = `queued-prompt-${Date.now()}-${queuedPromptCounterRef.current}`;
    const prompt: QueuedPrompt = {
      id,
      text,
      submission: { version: 1, state: "ready" },
      ...(queuedImages.length > 0 ? { images: queuedImages } : {}),
      ...(normalizedQuotedContext ? { quotedContext: normalizedQuotedContext } : {}),
      ...(activeSessionMentions.length > 0
        ? { sessionMentions: activeSessionMentions }
        : {}),
    };
    if (!commitQueuedPrompts([...queuedPromptsRef.current, prompt])) return;
    clear();
    clearComposerText();
    onQuotedContextChange?.(null);
  }, [
    activeSessionMentions,
    canQueueGuidance,
    clear,
    clearComposerText,
    commitQueuedPrompts,
    maxTextBytes,
    normalizedQuotedContext,
    onQuotedContextChange,
    readyImages,
    textTooLargeMessage,
    value,
  ]);

  const removeQueuedPrompt = useCallback((id: string) => {
    const current = queuedPromptsRef.current;
    const prompt = current.find((item) => item.id === id);
    if (!prompt || prompt.submission.state === "submitted" || prompt.submission.state === "unknown") {
      return;
    }
    if (!commitQueuedPrompts(current.filter((item) => item.id !== id))) return;
    heldQueuedPromptIdsRef.current.delete(id);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, [commitQueuedPrompts]);

  const editQueuedPrompt = useCallback((prompt: QueuedPrompt) => {
    const current = queuedPromptsRef.current;
    const livePrompt = current.find((item) => item.id === prompt.id);
    if (
      !livePrompt
      || livePrompt.submission.state === "submitted"
      || livePrompt.submission.state === "unknown"
      || disabled
    ) return;
    // With a separate draft, edit only this row in place. Moving the draft
    // into the queue would silently make it eligible for automatic sending.
    if (value || images.length > 0 || normalizedQuotedContext || sendPending) {
      setQueuedEdit({ id: livePrompt.id, text: livePrompt.text });
      return;
    }
    setQueuedEdit(null);
    if (!commitQueuedPrompts(current.filter((item) => item.id !== prompt.id))) return;
    heldQueuedPromptIdsRef.current.delete(prompt.id);
    setValue(livePrompt.text);
    setSelectedSessionMentions(livePrompt.sessionMentions ?? []);
    setInlineError(null);
    setSlashMenuDismissed(false);
    setCliAppMenuDismissed(false);
    setCursorPosition(prompt.text.length);
    onQuotedContextChange?.(livePrompt.quotedContext ?? null);
    if (livePrompt.images?.length) {
      restoreReadyImages(livePrompt.images as RestoredReadyImage[]);
    } else {
      clear();
    }
    resizeTextarea();
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(livePrompt.text.length, livePrompt.text.length);
    });
  }, [clear, commitQueuedPrompts, disabled, images.length, normalizedQuotedContext,
    onQuotedContextChange, resizeTextarea, restoreReadyImages, sendPending, value]);

  const saveQueuedEdit = useCallback(() => {
    if (!queuedEdit || disabled) return;
    const current = queuedPromptsRef.current;
    const prompt = current.find((item) => item.id === queuedEdit.id);
    if (!prompt || (prompt.submission.state !== "ready" && prompt.submission.state !== "not_admitted")) return;
    const text = queuedEdit.text.trim();
    if (!text && !prompt.images?.length) return;
    if (utf8Bytes(formatQuotedUserMessage(text, prompt.quotedContext)) > maxTextBytes) {
      setInlineError(textTooLargeMessage());
      return;
    }
    if (commitQueuedPrompts(current.map((item) => item === prompt ? { ...item, text } : item))) {
      setInlineError(null);
      setQueuedEdit(null);
    }
  }, [commitQueuedPrompts, disabled, maxTextBytes, queuedEdit, textTooLargeMessage]);

  const moveQueuedPrompt = useCallback((dragId: string, targetId: string) => {
    if (dragId === targetId) return;
    const current = queuedPromptsRef.current;
    const dragPrompt = current.find((item) => item.id === dragId);
    const targetPrompt = current.find((item) => item.id === targetId);
    if (!dragPrompt || !targetPrompt) return;
    if (dragPrompt.submission.state === "submitted" || dragPrompt.submission.state === "unknown") return;
    if (targetPrompt.submission.state === "submitted" || targetPrompt.submission.state === "unknown") return;
    const from = current.findIndex((item) => item.id === dragId);
    const to = current.findIndex((item) => item.id === targetId);
    if (from < 0 || to < 0) return;
    const next = [...current];
    const [moved] = next.splice(from, 1);
    if (!moved) return;
    next.splice(to, 0, moved);
    commitQueuedPrompts(next);
  }, [commitQueuedPrompts]);

  const prepareQueuedPrompt = useCallback((promptId: string): {
    prompt: QueuedPrompt;
    binding: QueuedAttemptBinding;
  } | null => {
    const queueKey = currentPendingQueueKeyRef.current;
    const current = queuedPromptsRef.current;
    const prompt = current.find((item) => item.id === promptId);
    if (
      !prompt
      || (prompt.submission.state !== "ready" && prompt.submission.state !== "not_admitted")
    ) return null;
    const binding: QueuedAttemptBinding = {
      queueKey,
      promptId: prompt.id,
      turnId: crypto.randomUUID(),
    };
    const next = current.map((item) => item.id === prompt.id
      ? {
          ...item,
          submission: { version: 1 as const, state: "submitted" as const, binding },
        }
      : item);
    if (!commitQueuedPrompts(next)) return null;
    const protectedPrompt = next.find((item) => item.id === prompt.id);
    if (!protectedPrompt) return null;
    heldQueuedPromptIdsRef.current.add(prompt.id);
    setPromptAwaitingConfirmation(prompt.id, queueKey, true);
    return { prompt: protectedPrompt, binding };
  }, [commitQueuedPrompts, setPromptAwaitingConfirmation]);

  const queuedActionDisabled = interactionDisabled || workspaceFullRetiring
    || (isStreaming ? steerDisabled || stopPending : modelNeedsSetup || newTurnBlocked);
  const sendQueuedPrompt = useCallback(
    (prompt: QueuedPrompt) => {
      if (
        queuedActionDisabled
        || prompt.id === queuedEdit?.id
        || queuedSubmissionInProgressRef.current.has(prompt.id)
      ) return;
      queuedSubmissionInProgressRef.current.add(prompt.id);
      requestAnimationFrame(() => queuedSubmissionInProgressRef.current.delete(prompt.id));
      const prepared = prepareQueuedPrompt(prompt.id);
      if (!prepared) return;
      const { prompt: protectedPrompt, binding } = prepared;
      const text = protectedPrompt.text.trim();
      const queuedImages = queuedImagesToSendImages(protectedPrompt.images);
      if (text || queuedImages?.length) {
        const options: SendOptions = {
          ...(protectedPrompt.quotedContext
            ? { quotedContext: protectedPrompt.quotedContext }
            : {}),
          ...(protectedPrompt.sessionMentions?.length
            ? { sessionMentions: protectedPrompt.sessionMentions }
            : {}),
          ...(isStreaming ? { continueActiveTurn: true } : {}),
          queuedAttempt: binding,
        };
        try {
          const submitted = onSend(text, queuedImages, options);
          if (submitted instanceof Promise) {
            void submitted.catch(() => {
              // Generic boolean failures never change the persisted hold.
            });
          }
        } catch {
          // The protected row stays held when its send owner is unavailable.
          requestAnimationFrame(() => textareaRef.current?.focus());
          observeQueuedAttempt(binding);
          return;
        }
        observeQueuedAttempt(binding);
      }
      requestAnimationFrame(() => textareaRef.current?.focus());
    },
    [
      isStreaming,
      onSend,
      observeQueuedAttempt,
      prepareQueuedPrompt,
      queuedActionDisabled,
      queuedEdit?.id,
    ],
  );

  /**
   * Admit one queued row only after the next-turn setting acknowledgement.
   *
   * A refused choice keeps the row ready and the draft intact; it never sends
   * under the previous folder/model/mode.
   */
  const sendQueuedPromptAfterSettings = useCallback((
    prompt: QueuedPrompt,
  ): Promise<void> => {
    if (!onBeforeNewTurnSend || isStreaming) {
      sendQueuedPrompt(prompt);
      return Promise.resolve();
    }
    return onBeforeNewTurnSend().then(() => {
      sendQueuedPrompt(prompt);
    }).catch(() => {
      // The owner already surfaced the refusal; the row stays retryable.
    });
  }, [isStreaming, onBeforeNewTurnSend, sendQueuedPrompt]);

  useEffect(() => {
    if (!queueCompletionTurnId || handledQueueCompletionRef.current === queueCompletionTurnId) return;
    const head = queuedPromptsRef.current[0];
    const eligible = head?.submission.state === "ready" && head.id !== queuedEdit?.id
      && !isStreaming && !automaticQueueSendBlocked && !interactionDisabled
      && !workspaceFullRetiring && !modelNeedsSetup && !stopPending;
    // A setting acknowledgement can make this exact natural completion
    // eligible later. Reconnect, Stop, or an uncertain head never do.
    if (eligible && newTurnBlocked) return;
    handledQueueCompletionRef.current = queueCompletionTurnId;
    if (eligible) void sendQueuedPromptAfterSettings(head);
  }, [automaticQueueSendBlocked, interactionDisabled, isStreaming, modelNeedsSetup, newTurnBlocked,
    queueCompletionTurnId, queuedEdit?.id, sendQueuedPromptAfterSettings, stopPending,
    workspaceFullRetiring]);

  const submit = useCallback((steer = false) => {
    const content = value.trim();
    if (
      !steer
      && !disabled
      && hasStopAction
      && onStop
      && content.toLowerCase() === "/stop"
    ) {
      if (!handleStop()) return;
      // Only the typed command was consumed. Keep staged attachments and
      // quoted context, as well as unrelated queued prompts.
      clearComposerText();
      return;
    }
    if (pendingSendRef.current !== null) return;
    if (modelNeedsSetup && !steer) {
      if (hasComposerContent) {
        setModelSetupAttentionRequest((request) => request + 1);
      }
      return;
    }
    if (steer ? !canQueueGuidance || steerDisabled || stopPending : !canSend) return;
    if (utf8Bytes(formatQuotedUserMessage(content, normalizedQuotedContext)) > maxTextBytes) {
      setInlineError(textTooLargeMessage());
      return;
    }
    // Share the same ``data:`` URL with both the wire payload and the
    // optimistic bubble preview: data URLs are self-contained (no blob
    // lifetime, safe under React StrictMode double-mount) and keep the bubble
    // in sync with whatever the backend actually sees.
    const payload: SendAttachment[] | undefined =
      readyImages.length > 0
        ? readyImages.map((img) => ({
            media: {
              data_url: img.dataUrl,
              name: img.file.name,
            },
            preview: { kind: img.kind, url: img.dataUrl, name: img.file.name },
          }))
        : undefined;
    const attachedCliApps = activeCliMentionApps.map(cliAppMentionPayload);
    const attachedMcpPresets = activeMcpPresetMentions.map(mcpPresetMentionPayload);
    const options: SendOptions | undefined =
      attachedCliApps.length > 0
      || attachedMcpPresets.length > 0
      || activeSessionMentions.length > 0
      || normalizedQuotedContext
        ? {
            ...(attachedCliApps.length > 0 ? { cliApps: attachedCliApps } : {}),
            ...(attachedMcpPresets.length > 0 ? { mcpPresets: attachedMcpPresets } : {}),
            ...(activeSessionMentions.length > 0
              ? { sessionMentions: activeSessionMentions }
              : {}),
            ...(normalizedQuotedContext ? { quotedContext: normalizedQuotedContext } : {}),
          }
        : undefined;
    const hasPlainTextCommandPayload =
      payload === undefined
      && attachedCliApps.length === 0
      && attachedMcpPresets.length === 0
      && activeSessionMentions.length === 0;
    const slashLifecycle = hasPlainTextCommandPayload
      ? slashCommandLifecycle(content, slashCommands)
      : null;
    const isSlashSideChannel = isSideChannelLifecycle(slashLifecycle);
    const finalizeActiveTurn =
      slashLifecycle === "finalize_active_turn";
    const sendDraftScope = draftScopeRef.current;
    const finishSend = () => {
      // The owner took custody of the full payload: drop the stored draft for
      // the chat that sent it, even if the editor has since unmounted or moved.
      saveComposerDraft(sendDraftScope, {
        text: "", quotedContext: null, mentions: [], attachments: [], reattachNames: [],
        selection: null, focused: false,
      });
      if (hasTouchPrimaryPointer) textareaRef.current?.blur();
      // Bubble owns the data URL copy; safe to revoke every staged blob
      // preview here without affecting the rendered message.
      clear();
      clearComposerText(!hasTouchPrimaryPointer);
      reattachNamesRef.current = [];
      setReattachNames([]);
      onQuotedContextChange?.(null);
    };
    const result = onSend(
      content,
      payload,
      steer ? { ...options, continueActiveTurn: true } : isSlashSideChannel
        ? {
            ...options,
            sideChannel: true,
            ...(finalizeActiveTurn ? { finalizeActiveTurn } : {}),
          }
        : options,
    );
    if (result instanceof Promise) {
      const attempt = Symbol();
      pendingSendRef.current = attempt;
      setSendPending(true);
      void result
        .then((accepted) => {
          if (accepted === false || accepted === null) return;
          saveComposerDraft(sendDraftScope, {
            text: "", quotedContext: null, mentions: [], attachments: [], reattachNames: [],
            selection: null, focused: false,
          });
          if (pendingSendRef.current === attempt) finishSend();
        })
        .catch((error: unknown) => {
          console.error("Failed to send message", error);
        })
        .finally(() => {
          if (pendingSendRef.current !== attempt) return;
          pendingSendRef.current = null;
          setSendPending(false);
        });
      return;
    }
    if (result !== false && result !== null) finishSend();
  }, [
    activeCliMentionApps,
    activeMcpPresetMentions,
    activeSessionMentions,
    canQueueGuidance,
    canSend,
    clear,
    clearComposerText,
    disabled,
    hasTouchPrimaryPointer,
    hasComposerContent,
    hasStopAction,
    handleStop,
    maxTextBytes,
    modelNeedsSetup,
    onSend,
    onStop,
    onQuotedContextChange,
    normalizedQuotedContext,
    readyImages,
    slashCommands,
    steerDisabled,
    stopPending,
    textTooLargeMessage,
    value,
  ]);

  const onKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    // Do not treat an IME commit or a held Enter as an action, including
    // accepting palette entries.
    if (e.nativeEvent.isComposing || (e.key === "Enter" && e.repeat)) return;
    if (showCliAppMenu) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedCliAppIndex((idx) => (idx + 1) % filteredMentionCandidates.length);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedCliAppIndex(
          (idx) => (idx - 1 + filteredMentionCandidates.length) % filteredMentionCandidates.length,
        );
        return;
      }
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
        e.preventDefault();
        chooseMentionCandidate(filteredMentionCandidates[selectedCliAppIndex]);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setCliAppMenuDismissed(true);
        return;
      }
    }
    if (showSlashMenu) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedCommandIndex((idx) => (idx + 1) % filteredSlashCommands.length);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedCommandIndex(
          (idx) => (idx - 1 + filteredSlashCommands.length) % filteredSlashCommands.length,
        );
        return;
      }
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
        e.preventDefault();
        chooseSlashCommand(filteredSlashCommands[selectedCommandIndex]);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setSlashMenuDismissed(true);
        return;
      }
    }
    if (e.key === "Enter" && !e.shiftKey && !e.altKey) {
      const modifiedEnter = e.metaKey || e.ctrlKey;
      if (modifiedEnter && !isStreaming) return;
      e.preventDefault();
      if (modifiedEnter) {
        submit(true);
        return;
      }
      if (canQueueGuidance) {
        queueGuidancePrompt();
        return;
      }
      submit();
    }
  };

  const onInput: React.FormEventHandler<HTMLTextAreaElement> = (e) => {
    if ((e.nativeEvent as InputEvent).isComposing) return;
    const el = e.currentTarget;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 260)}px`;
  };

  const onFilePick: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    addFiles(files);
  };

  const removeChip = useCallback(
    (id: string) => {
      const { nextFocusId } = remove(id);
      setInlineError(null);
      requestAnimationFrame(() => {
        const el = nextFocusId ? chipRefs.current.get(nextFocusId) : null;
        if (el) {
          el.focus();
        } else {
          textareaRef.current?.focus();
        }
      });
    },
    [remove],
  );

  const onChipKey = useCallback(
    (id: string) => (e: ReactKeyboardEvent<HTMLButtonElement>) => {
      if (
        e.key === "Delete" ||
        e.key === "Backspace" ||
        e.key === "Enter" ||
        e.key === " "
      ) {
        e.preventDefault();
        removeChip(id);
      }
    },
    [removeChip],
  );

  const attachButtonDisabled = interactionDisabled || full;
  const showVoiceButton = Boolean(onTranscribeAudio);
  const voiceRecordingStatusLabel = t("thread.composer.voice.recordingStatus", {
    time: voiceRecorder.elapsedLabel,
    defaultValue: `Recording ${voiceRecorder.elapsedLabel}`,
  });
  const voiceButtonLabel =
    voiceRecorder.state === "recording"
      ? t("thread.composer.voice.stop")
      : voiceRecorder.state === "transcribing"
        ? t("thread.composer.voice.transcribing")
        : t("thread.composer.tools.voice");
  const voiceButtonTooltip =
    voiceRecorder.state === "recording"
      ? t("thread.composer.voice.stop")
      : voiceRecorder.state === "transcribing"
        ? t("thread.composer.voice.transcribing")
        : t("thread.composer.voice.hint");
  // One primary button. While a turn runs it stops an empty composer and sends
  // (queueing guidance) when there is something to say, matching the reference
  // harness. Stop stays reachable from the queue rows and the slash command.
  // A secondary Stop icon is only shown when the main turn is idle yet
  // background child work can still be stopped.
  const showGuidanceSendButton = canQueueGuidance;
  const showStopButton = isStreaming && !!onStop && !hasComposerContent;
  const showSecondaryStopButton = !isStreaming && stopAvailable && !!onStop;
  const relaxedHeroInput = isHero && images.length === 0 && !isStreaming;
  const inputTextClasses = cn(
    "w-full resize-none bg-transparent",
    isHero
      ? cn(
          "min-h-[78px] px-4 text-[16px] leading-6 sm:px-5",
          relaxedHeroInput ? "pb-2 pt-[27px]" : "pb-1.5 pt-4",
        )
      : "min-h-[50px] px-3.5 pb-1.5 pt-3 text-[16px] leading-5 sm:px-4",
  );
  const projectPickerInline = (
    <WorkspaceProjectPicker
      inline
      isHero={isHero}
      ownerKey={pendingQueueKey}
      disabled={interactionDisabled || workspaceScopeDisabled || workspaceControlsHidden}
      scope={workspaceScope}
      defaultScope={workspaceDefaultScope}
      controls={workspaceControls}
      workspaces={workspaces}
      error={workspaceError}
      onDismissError={onDismissWorkspaceError}
      nextTurnScope={nextWorkspaceScope}
      nextTurnPending={nextTurnSettingsPending}
      turnActive={isStreaming}
      onPickFolder={onPickWorkspaceFolder}
      onRefreshWorkspaces={onRefreshWorkspaces}
      onChange={onWorkspaceScopeChange}
    />
  );
  const projectPickerLine = (
    <WorkspaceProjectPicker
      line
      ownerKey={pendingQueueKey}
      disabled={interactionDisabled || workspaceScopeDisabled || workspaceControlsHidden}
      scope={workspaceScope}
      defaultScope={workspaceDefaultScope}
      controls={workspaceControls}
      workspaces={workspaces}
      error={workspaceError}
      onDismissError={onDismissWorkspaceError}
      nextTurnScope={nextWorkspaceScope}
      nextTurnPending={nextTurnSettingsPending}
      turnActive={isStreaming}
      onPickFolder={onPickWorkspaceFolder}
      onRefreshWorkspaces={onRefreshWorkspaces}
      onChange={onWorkspaceScopeChange}
    />
  );
  const showMobileWorkspaceLine = mobileComposer
    && !voiceRecorder.isRecording
    && !workspaceControlsHidden
    && projectPickerAvailable;
  const hasMobileUsage = contextUsage != null
    || hasConfiguredModel
    || recentRoundUsage.length > 0;
  const showMobileWorkspaceRow = mobileComposer
    && !voiceRecorder.isRecording
    && (showMobileWorkspaceLine || hasMobileUsage);

  const accessControl = workspaceScope && !workspaceControlsHidden ? (
    <WorkspacePermissionV2Control
      enabled={workspaces?.controls.permission_v2 === true}
      permissions={workspacePermissions}
      loading={workspacePermissionsLoading}
      error={workspacePermissionsError}
      disabled={persistentWorkspaceChat && (interactionDisabled || workspaceAccessModeUnknownBlocked)}
      persistentChat={persistentWorkspaceChat}
      draftFullAccess={!persistentWorkspaceChat && !!onDraftFullAccess}
      onRequestFullAccess={
        !persistentWorkspaceChat && onDraftFullAccess
          ? () => onDraftFullAccess(value)
          : undefined
      }
      isHero={mobileComposer ? false : isHero}
      onModeChange={onWorkspaceAccessModeChange}
      onRetryRestricted={onRetryWorkspaceRestricted}
    />
  ) : null;
  const accessControlSlot = accessControl ? (
    <div
      className={cn(
        "thread-composer-access-slot flex min-w-0",
        showWorkspacePicker && !mobileComposer && !voiceRecorder.isRecording
          && "thread-composer-access-slot-after-workspace",
      )}
    >
      {accessControl}
    </div>
  ) : null;

  const mobileModelLabel = (modelLabel ?? "").replace(/\s*\([^)]*\)\s*$/, "").trim();
  const modelControl = modelLabel && !voiceRecorder.isRecording ? (
    <ModelPresetBadge
      label={mobileComposer ? mobileModelLabel || modelLabel : modelLabel}
      modelDetail={modelDetail}
      modelPreset={modelPreset}
      modelPresets={modelPresets}
      onPresetChange={onModelPresetChange}
      reasoning={modelReasoning}
      onManageModels={onManageModels}
      onRequestComposerFocus={() => {
        // A touch keyboard would cover the thread; only restore composer focus
        // when the user has a hardware keyboard.
        if (!hasTouchPrimaryPointer) textareaRef.current?.focus();
      }}
      provider={modelProvider}
      providerLabel={modelProviderLabel}
      needsSetup={modelNeedsSetup}
      attentionRequest={modelSetupAttentionRequest}
      fallbackModelName={fallbackModelName}
      isHero={mobileComposer ? false : isHero}
      onClick={modelNeedsSetup ? onModelBadgeClick : undefined}
    />
  ) : null;

  return (
    <form
      ref={formRef}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      onDragEnter={(event) => {
        if (!previewSessionDrop(event)) onDragEnter(event);
      }}
      onDragOver={(event) => {
        if (!previewSessionDrop(event)) onDragOver(event);
      }}
      onDragLeave={(event) => {
        if (!hasDraggedSession(event.dataTransfer)) {
          onDragLeave(event);
          return;
        }
        const nextTarget = event.relatedTarget;
        if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
          setSessionDragPreview(null);
        }
      }}
      onDrop={(event) => {
        if (!handleSessionDrop(event)) onDrop(event);
      }}
      className={cn(
        "relative w-full",
        isHero ? "px-0" : "px-1 pb-1.5 sm:px-0",
        mobileComposer ? "pt-3" : isHero ? "" : "pt-1",
      )}
    >
      {showSlashMenu ? (
        <SlashCommandPalette
          commands={filteredSlashCommands}
          selectedIndex={selectedCommandIndex}
          layout={slashPaletteLayout}
          isHero={isHero}
          onHover={setSelectedCommandIndex}
          onChoose={chooseSlashCommand}
        />
      ) : null}
      {showCliAppMenu ? (
        <CliAppMentionPalette
          candidates={filteredMentionCandidates}
          selectedIndex={selectedCliAppIndex}
          layout={slashPaletteLayout}
          isHero={isHero}
          onHover={setSelectedCliAppIndex}
          onChoose={chooseMentionCandidate}
        />
      ) : null}
      {showMobileWorkspaceRow && !composerChromeHidden ? (
        <div className="thread-composer-mobile-workspace-row">
          {showMobileWorkspaceLine ? projectPickerLine : null}
          {hasMobileUsage ? (
            <div className="thread-composer-mobile-workspace-meta flex min-w-0 items-center">
              <ComposerUsagePopover
                context={contextUsage}
                hasConfiguredModel={hasConfiguredModel}
                rounds={recentRoundUsage}
              />
            </div>
          ) : null}
        </div>
      ) : null}
      {queuedPrompts.length > 0 ? (
        <QueuedPromptStack
          prompts={queuedPrompts}
          edit={queuedEdit ? (
            <div key={queuedEdit.id} data-queued-prompt-row="true" className="flex flex-wrap items-center gap-1 px-2 py-1">
              <textarea
                autoFocus
                aria-label={t("thread.composer.queued.edit")}
                className="min-w-24 flex-1 rounded-control border border-border bg-background p-1.5 text-sm"
                rows={2}
                value={queuedEdit.text}
                onChange={(event) => setQueuedEdit({ id: queuedEdit.id, text: event.target.value })}
                onKeyDown={(event) => {
                  if (event.key === "Escape" && !event.nativeEvent.isComposing) setQueuedEdit(null);
                }}
              />
              <Button type="button" size="sm" onClick={saveQueuedEdit}>
                {t("thread.composer.queued.saveEdit", { defaultValue: "Save edit" })}
              </Button>
              <Button type="button" variant="ghost" size="sm" onClick={() => setQueuedEdit(null)}>
                {t("thread.composer.queued.cancelEdit", { defaultValue: "Cancel" })}
              </Button>
            </div>
          ) : null}
          awaitingConfirmationPromptIds={awaitingConfirmationPromptIds}
          isHero={isHero}
          className={interactionDisabled ? "opacity-60" : undefined}
          label={t("thread.composer.queued.label")}
          awaitingConfirmationLabel={t("thread.composer.queued.awaitingConfirmation")}
          actionLabel={isStreaming
            ? t("thread.composer.queued.guide")
            : t("thread.composer.queued.send", { defaultValue: "Send" })}
          actionDisabled={queuedActionDisabled}
          deleteLabel={t("thread.composer.queued.delete")}
          dragLabel={t("thread.composer.queued.drag")}
          editLabel={t("thread.composer.queued.edit")}
          onAction={(prompt) => void sendQueuedPromptAfterSettings(prompt)}
          onDelete={removeQueuedPrompt}
          onEdit={editQueuedPrompt}
          onDragStart={(id) => {
            draggedQueuedPromptIdRef.current = id;
          }}
          onDragEnd={() => {
            draggedQueuedPromptIdRef.current = null;
          }}
          onDrop={(targetId) => {
            const dragId = draggedQueuedPromptIdRef.current;
            if (dragId) moveQueuedPrompt(dragId, targetId);
          }}
        />
      ) : null}
      <div
        ref={surfaceRef}
        className={cn(
          "thread-composer-surface group/composer relative mx-auto flex w-full flex-col overflow-visible transition-all duration-200",
          isHero
            ? "max-w-[58rem] rounded-prominent bg-muted/30 focus-within:bg-muted/50 dark:bg-card dark:focus-within:bg-white/[0.06]"
            : "max-w-[49.5rem] rounded-panel bg-muted/30 focus-within:bg-muted/50 dark:bg-card dark:focus-within:bg-white/[0.06]",
          interactionDisabled && "opacity-60",
          sessionDragPreview && "ring-1 ring-primary/25",
          isDragging && "ring-2 ring-primary/40 motion-reduce:ring-0 motion-reduce:border-primary",
          goalState?.active &&
            "goal-shell-glow ring-1 ring-sky-400/35 motion-reduce:ring-sky-400/25 dark:ring-sky-400/45",
        )}
      >
        {images.length > 0 ? (
          <div
            className="flex flex-wrap gap-2 px-3 pt-3"
            aria-label={t("thread.composer.attachImage")}
          >
            {images.map((img) => (
              <AttachmentChip
                key={img.id}
                image={img}
                labelRemove={t("thread.composer.remove")}
                labelEncoding={t("thread.composer.encoding")}
                normalizedHint={(orig, current) =>
                  t("thread.composer.normalizedSizeHint", {
                    orig: formatBytes(orig),
                    current: formatBytes(current),
                  })
                }
                formatError={formatRejection}
                onRemove={() => removeChip(img.id)}
                onKeyDown={onChipKey(img.id)}
                registerRef={(el) => {
                  if (el) chipRefs.current.set(img.id, el);
                  else chipRefs.current.delete(img.id);
                }}
              />
            ))}
          </div>
        ) : null}
        {normalizedQuotedContext ? (
          <div
            className="mx-3 mt-3 flex min-w-0 items-start gap-2 border-l-2 border-muted-foreground/25 pl-3 pr-1 text-muted-foreground"
            aria-label={t("thread.composer.quotedContext")}
          >
            <Quote className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
            <p className="line-clamp-2 min-w-0 flex-1 text-[13px]/[1.45]">
              {normalizedQuotedContext}
            </p>
            <button
              type="button"
              className="touch-target -mr-1 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full transition-colors hover:bg-muted/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label={t("thread.composer.removeQuotedContext")}
              onClick={() => {
                onQuotedContextChange?.(null);
                requestAnimationFrame(() => textareaRef.current?.focus());
              }}
            >
              <X className="h-3.5 w-3.5" aria-hidden />
            </button>
          </div>
        ) : null}
        <GoalStateStrip goalState={goalState} />
        <div className="relative">
          {hasMentionDecorations ? (
            <ComposerCliMentionOverlay
              segments={displayMentionSegments}
              isHero={isHero}
              className={inputTextClasses}
              ghostRange={sessionDragInsertion
                ? { start: sessionDragInsertion.tokenStart, end: sessionDragInsertion.tokenEnd }
                : null}
            />
          ) : null}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setSlashMenuDismissed(false);
              setCliAppMenuDismissed(false);
              setCursorPosition(e.target.selectionStart ?? e.target.value.length);
            }}
            onInput={onInput}
            onKeyDown={onKeyDown}
            onFocus={(e) => {
              updateComposerDraftSelection(
                draftScopeRef.current,
                { start: e.currentTarget.selectionStart, end: e.currentTarget.selectionEnd },
                true,
              );
            }}
            onKeyUp={(e) => {
              setCursorPosition(e.currentTarget.selectionStart ?? e.currentTarget.value.length);
              updateComposerDraftSelection(
                draftScopeRef.current,
                { start: e.currentTarget.selectionStart, end: e.currentTarget.selectionEnd },
                document.activeElement === e.currentTarget,
              );
            }}
            onSelect={(e) => {
              setCursorPosition(e.currentTarget.selectionStart ?? e.currentTarget.value.length);
              updateComposerDraftSelection(
                draftScopeRef.current,
                { start: e.currentTarget.selectionStart, end: e.currentTarget.selectionEnd },
                document.activeElement === e.currentTarget,
              );
            }}
            onClick={(e) => {
              setCursorPosition(e.currentTarget.selectionStart ?? e.currentTarget.value.length);
              updateComposerDraftSelection(
                draftScopeRef.current,
                { start: e.currentTarget.selectionStart, end: e.currentTarget.selectionEnd },
                document.activeElement === e.currentTarget,
              );
            }}
            onPaste={onPaste}
            rows={1}
            placeholder={sessionDragPreview ? "" : resolvedPlaceholder}
            disabled={interactionDisabled}
            aria-label={inputAriaLabel ?? t("thread.composer.inputAria")}
            className={cn(
              inputTextClasses,
              "relative z-10 overflow-x-hidden caret-foreground placeholder:text-muted-foreground/70",
              "focus:outline-none focus-visible:outline-none",
              "disabled:cursor-not-allowed",
              hasMentionDecorations && "text-transparent selection:bg-primary/20",
            )}
          />
        </div>
        {reattachNames.length > 0 ? (
          <div
            role="status"
            data-testid="composer-reattach-notice"
            className="mx-3 mb-1 flex items-start justify-between gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-[11.5px] font-medium text-amber-700 dark:text-amber-300"
          >
            <span>
              {t("thread.composer.reattachNotice", {
                names: reattachNames.join(", "),
                defaultValue: `Attachments were not restored after relaunch. Reattach: ${reattachNames.join(", ")}`,
              })}
            </span>
            <button
              type="button"
              className="shrink-0 underline"
              onClick={() => {
                reattachNamesRef.current = [];
                setReattachNames([]);
              }}
            >
              {t("thread.composer.reattachDismiss", { defaultValue: "Dismiss" })}
            </button>
          </div>
        ) : null}
        {visibleNextTurnSettingsError && !inlineError ? (
          <div
            role="alert"
            data-testid="composer-next-turn-error"
            className="mx-3 mb-1 flex items-center justify-between gap-2 rounded-md border border-destructive/40 bg-destructive/8 px-2.5 py-1 text-[11.5px] font-medium text-destructive"
          >
            <span className="min-w-0 flex-1">
              {t("workspace.picker.nextTurnFailed", {
                defaultValue: "The folder change was not saved. The previous choice is still in effect; your draft is intact.",
              })}
            </span>
            <button
              type="button"
              aria-label={t("thread.composer.reattachDismiss", { defaultValue: "Dismiss" })}
              onClick={() => {
                setDismissedNextTurnError(visibleNextTurnSettingsError);
                onDismissNextTurnSettingsError?.();
              }}
              className="grid h-5 w-5 shrink-0 place-items-center rounded text-destructive/80 transition-colors hover:bg-destructive/15 hover:text-destructive"
            >
              <X className="h-3.5 w-3.5" aria-hidden />
            </button>
          </div>
        ) : null}
        {inlineError ? (
          <div
            role="alert"
            className={cn(
              "mx-3 mb-1 max-h-24 overflow-y-auto [overflow-wrap:anywhere] rounded-md border border-destructive/40 bg-destructive/8 px-2.5 py-1",
              "text-[11.5px] font-medium text-destructive transition-[max-height,margin,padding,opacity] [transition-duration:220ms] ease-out motion-reduce:transition-none",
              voiceErrorFading && "mb-0 max-h-0 border-transparent py-0 opacity-0",
            )}
          >
            {inlineError}
          </div>
        ) : null}
        <div
          className={cn(
            "thread-composer-footer flex flex-nowrap items-center motion-safe:transition-[padding-bottom] motion-safe:[transition-duration:220ms] motion-safe:ease-in-out",
            isHero
              ? "gap-x-1.5 px-3 pb-3.5 sm:px-4"
              : "gap-x-2 px-2.5 pb-2 sm:px-3",
          )}
        >
          <div
            className={cn(
              "thread-composer-footer-primary flex min-w-0 flex-1 basis-0 items-center",
              isHero ? "gap-1.5" : "gap-2",
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPT_ATTR}
              multiple
              hidden
              onChange={onFilePick}
            />
            {mobileTransport?.pickCamera ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    disabled={attachButtonDisabled}
                    aria-label={t("thread.composer.attachImage")}
                    className={cn(
                      "thread-composer-action touch-target shrink-0 rounded-full text-muted-foreground hover:text-foreground",
                      isHero
                        ? "h-8 w-8 border border-border/55 bg-card shadow-[0_2px_8px_rgba(15,23,42,0.05)] hover:bg-card"
                        : "h-9 w-9 border border-border/55 bg-card shadow-[0_2px_8px_rgba(15,23,42,0.05)] hover:bg-card",
                    )}
                  >
                    <Plus className={cn(isHero ? "h-[18px] w-[18px]" : "h-4 w-4")} />
                    <span className="sr-only">{t("thread.composer.attachImage")}</span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent side="top" align="start">
                  <DropdownMenuItem onSelect={() => void pickCamera()}>
                    Take a photo
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => fileInputRef.current?.click()}>
                    Choose from library
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Button
                type="button"
                size="icon"
                variant="ghost"
                disabled={attachButtonDisabled}
                aria-label={t("thread.composer.attachImage")}
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  "thread-composer-action touch-target shrink-0 rounded-full text-muted-foreground hover:text-foreground",
                  isHero
                    ? "h-8 w-8 border border-border/55 bg-card shadow-[0_2px_8px_rgba(15,23,42,0.05)] hover:bg-card"
                    : "h-9 w-9 border border-border/55 bg-card shadow-[0_2px_8px_rgba(15,23,42,0.05)] hover:bg-card",
                )}
              >
                <Plus className={cn(isHero ? "h-[18px] w-[18px]" : "h-4 w-4")} />
                <span className="sr-only">{t("thread.composer.attachImage")}</span>
              </Button>
            )}
            {showWorkspacePicker && !mobileComposer && !voiceRecorder.isRecording ? projectPickerInline : null}
            {mobileComposer && !voiceRecorder.isRecording ? accessControlSlot : null}
            {voiceRecorder.isRecording ? (
              <VoiceRecordingMeter
                ariaLabel={voiceRecordingStatusLabel}
                className="mx-1 flex-1"
                elapsedLabel={voiceRecorder.elapsedLabel}
                isHero={isHero}
                levels={voiceRecorder.levels}
              />
            ) : !mobileComposer ? accessControlSlot : null}
          </div>
          <div
            className={cn(
              "thread-composer-footer-actions ml-auto flex min-w-0 items-center justify-end",
              isHero ? "gap-1.5" : "gap-2",
            )}
          >
            {modelControl}
            {!voiceRecorder.isRecording && !mobileComposer ? (
              <ComposerUsagePopover
                context={contextUsage}
                hasConfiguredModel={hasConfiguredModel}
                rounds={recentRoundUsage}
                isHero={isHero}
              />
            ) : null}
            {showVoiceButton ? (
              <TooltipProvider delayDuration={220} skipDelayDuration={80}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      disabled={voiceRecorder.buttonDisabled}
                      aria-label={voiceButtonLabel}
                      aria-keyshortcuts={VOICE_SHORTCUT_ARIA}
                      title={voiceButtonTooltip}
                      onPointerDown={voiceRecorder.beginPress}
                      onPointerUp={voiceRecorder.endPress}
                      onPointerCancel={voiceRecorder.endPress}
                      onClick={voiceRecorder.handleClick}
                      className={cn(
                        "thread-composer-action touch-target shrink-0 rounded-full border border-transparent text-muted-foreground hover:bg-muted/65 hover:text-foreground",
                        isHero ? "h-8 w-8" : "h-9 w-9",
                        voiceRecorder.isRecording &&
                          "bg-red-500 text-white shadow-[0_8px_20px_rgba(239,68,68,0.22)] hover:bg-red-500 hover:text-white",
                      )}
                    >
                      {voiceRecorder.state === "transcribing" ? (
                        <Loader2 className={cn(isHero ? "h-4 w-4" : "h-4 w-4", "animate-spin")} />
                      ) : voiceRecorder.isRecording ? (
                        <Square className={cn(isHero ? "h-3.5 w-3.5" : "h-3.5 w-3.5")} fill="currentColor" />
                      ) : (
                        <Mic className="h-[18px] w-[18px]" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent
                    side="top"
                    align="center"
                    className="flex items-center gap-2 rounded-full border border-border/70 bg-popover px-3 py-1.5 text-[13px] font-medium text-popover-foreground shadow-[0_8px_24px_rgba(15,23,42,0.13)] dark:border-white/10"
                  >
                    <span>{voiceButtonTooltip}</span>
                    {voiceRecorder.state === "idle" ? (
                      <kbd className="rounded-full bg-muted px-2 py-0.5 font-sans text-[12px] font-semibold leading-none text-muted-foreground dark:bg-white/10 dark:text-white/80">
                        {voiceShortcutLabel}
                      </kbd>
                    ) : null}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            ) : null}
            {showSecondaryStopButton ? (
              <Button
                type="button"
                size="icon"
                variant="outline"
                disabled={disabled || stopDisabled}
                aria-label={stopPending
                  ? t("thread.composer.stopping", { defaultValue: "Stopping" })
                  : t("thread.composer.stop")}
                onClick={handleStop}
                className="thread-composer-background-stop touch-target h-9 w-9 shrink-0 rounded-full"
              >
                {stopPending
                  ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                  : <Square className="h-3.5 w-3.5 fill-current stroke-current" aria-hidden />}
              </Button>
            ) : null}
            <Button
              type={showStopButton || showGuidanceSendButton || modelNeedsSetup
                ? "button"
                : "submit"}
              size="icon"
              disabled={showStopButton
                ? disabled || stopDisabled
                : !canSend && !showGuidanceSendButton && !canOpenModelSettings}
              aria-label={
                showStopButton
                  ? stopPending
                    ? t("thread.composer.stopping", { defaultValue: "Stopping" })
                    : t("thread.composer.stop")
                  : modelNeedsSetup
                    ? t("thread.composer.configureModel", { defaultValue: "Configure model" })
                    : showGuidanceSendButton
                      ? t("thread.composer.queueGuidance")
                      : t("thread.composer.send")
              }
              onClick={showStopButton
                ? handleStop
                : showGuidanceSendButton
                  ? queueGuidancePrompt
                  : modelNeedsSetup
                    ? onModelBadgeClick
                    : undefined}
              className={cn(
                "thread-composer-action touch-target shrink-0 rounded-full transition-transform",
                showStopButton
                  ? "border border-border/70 bg-card text-foreground/85 shadow-[0_3px_10px_rgba(15,23,42,0.08)] hover:bg-muted/65 hover:text-foreground disabled:text-muted-foreground/50"
                  : isHero
                    ? "border border-foreground bg-foreground text-background shadow-[0_4px_12px_rgba(15,23,42,0.20)] hover:bg-foreground/90 disabled:border-foreground disabled:bg-foreground disabled:text-background"
                    : "border border-foreground bg-foreground text-background shadow-[0_3px_10px_rgba(15,23,42,0.18)] hover:bg-foreground/90 disabled:border-foreground disabled:bg-foreground disabled:text-background",
                isHero ? "h-8 w-8" : "h-9 w-9",
                (canSend || canOpenModelSettings || showStopButton) && "hover:scale-[1.03] active:scale-95",
              )}
            >
              {showStopButton ? stopPending ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              ) : (
                <Square className={cn("fill-current stroke-current", isHero ? "h-3 w-3" : "h-3.5 w-3.5")} />
              ) : (
                <ArrowUp className={cn(isHero ? "h-4 w-4" : "h-4 w-4")} />
              )}
            </Button>
          </div>
        </div>
      </div>
    </form>
  );
}

function QueuedPromptStack({
  prompts,
  edit,
  awaitingConfirmationPromptIds,
  isHero,
  className,
  label,
  awaitingConfirmationLabel,
  actionLabel,
  actionDisabled,
  deleteLabel,
  dragLabel,
  editLabel,
  onAction,
  onDelete,
  onEdit,
  onDragStart,
  onDragEnd,
  onDrop,
}: {
  prompts: QueuedPrompt[];
  edit: ReactElement | null;
  awaitingConfirmationPromptIds: ReadonlySet<string>;
  isHero: boolean;
  className?: string;
  label: string;
  awaitingConfirmationLabel: string;
  actionLabel: string;
  actionDisabled: boolean;
  deleteLabel: string;
  dragLabel: string;
  editLabel: string;
  onAction: (prompt: QueuedPrompt) => void;
  onDelete: (id: string) => void;
  onEdit: (prompt: QueuedPrompt) => void;
  onDragStart: (id: string) => void;
  onDragEnd: () => void;
  onDrop: (targetId: string) => void;
}) {
  return (
    <div
      role="group"
      data-state="enter"
      className={cn(
        "composer-status-strip relative z-0 mx-auto -mb-px w-[calc(100%-1.5rem)] overflow-hidden rounded-t-floating",
        "border border-b-0 border-black/[0.06] bg-muted p-1.5",
        "dark:border-white/[0.08]",
        isHero ? "max-w-[56.5rem]" : "max-w-[48rem]",
        className,
      )}
      style={{ "--composer-strip-max-height": "260px" } as CSSProperties}
      aria-label={label}
    >
      <div className="queued-prompt-list flex max-h-[216px] flex-col gap-1 overflow-y-auto">
        {prompts.map((prompt) => edit?.key === prompt.id ? edit : (
          <QueuedPromptRow
            key={prompt.id}
            prompt={prompt}
            awaitingConfirmation={awaitingConfirmationPromptIds.has(prompt.id)}
            isHero={isHero}
            actionLabel={actionLabel}
            actionDisabled={actionDisabled}
            awaitingConfirmationLabel={awaitingConfirmationLabel}
            deleteLabel={deleteLabel}
            dragLabel={dragLabel}
            editLabel={editLabel}
            onAction={onAction}
            onDelete={onDelete}
            onEdit={onEdit}
            onDragStart={onDragStart}
            onDragEnd={onDragEnd}
            onDrop={onDrop}
          />
        ))}
      </div>
    </div>
  );
}

function QueuedPromptRow({
  prompt,
  awaitingConfirmation,
  isHero,
  actionLabel,
  actionDisabled,
  awaitingConfirmationLabel,
  deleteLabel,
  dragLabel,
  editLabel,
  onAction,
  onDelete,
  onEdit,
  onDragStart,
  onDragEnd,
  onDrop,
}: {
  prompt: QueuedPrompt;
  awaitingConfirmation: boolean;
  isHero: boolean;
  actionLabel: string;
  actionDisabled: boolean;
  awaitingConfirmationLabel: string;
  deleteLabel: string;
  dragLabel: string;
  editLabel: string;
  onAction: (prompt: QueuedPrompt) => void;
  onDelete: (id: string) => void;
  onEdit: (prompt: QueuedPrompt) => void;
  onDragStart: (id: string) => void;
  onDragEnd: () => void;
  onDrop: (targetId: string) => void;
}) {
  const displayLabel = queuedPromptLabel(prompt);

  return (
    <div
      data-queued-prompt-row="true"
      onDragEnter={(event) => {
        event.preventDefault();
        if (!awaitingConfirmation) onDrop(prompt.id);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = awaitingConfirmation ? "none" : "move";
      }}
      onDrop={(event) => {
        event.preventDefault();
        if (!awaitingConfirmation) onDrop(prompt.id);
      }}
      onDragEnd={onDragEnd}
      aria-busy={awaitingConfirmation || undefined}
      className={cn(
        "queued-prompt-row group/queued flex min-h-8 shrink-0 items-start gap-1.5 rounded-control px-2 py-0.5",
        "text-[13px] transition-colors hover:bg-muted/55 dark:hover:bg-white/[0.055]",
        isHero && "text-[13.5px]",
      )}
    >
      <span
        draggable={!awaitingConfirmation}
        role="button"
        tabIndex={awaitingConfirmation ? -1 : 0}
        aria-label={dragLabel}
        aria-disabled={awaitingConfirmation || undefined}
        onDragStart={(event) => {
          if (awaitingConfirmation) {
            event.preventDefault();
            return;
          }
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", prompt.id);
          suppressNativeDragPreview(event.dataTransfer);
          onDragStart(prompt.id);
        }}
        onDragEnd={onDragEnd}
        className={cn(
          "inline-flex h-7 w-7 shrink-0 cursor-grab items-center justify-center rounded-lg",
          "text-muted-foreground/45 transition-colors hover:bg-background/80 hover:text-muted-foreground",
          "active:cursor-grabbing dark:hover:bg-white/[0.06]",
          awaitingConfirmation && "cursor-default opacity-50",
        )}
      >
        <GripVertical className="pointer-events-none h-3.5 w-3.5" aria-hidden />
      </span>
      <div className="flex min-h-7 min-w-0 flex-1 flex-col items-start justify-center">
        <p
          title={displayLabel}
          className={cn(
            "w-full truncate font-medium leading-[1.28] text-foreground/88",
            isHero && "text-[13.5px]",
          )}
        >
          {displayLabel}
        </p>
        {awaitingConfirmation ? (
          <span role="status" className="pt-0.5 text-[10px] leading-3 text-muted-foreground">
            {awaitingConfirmationLabel}
          </span>
        ) : null}
      </div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={awaitingConfirmation || actionDisabled}
        className="h-7 shrink-0 rounded-full px-2 text-[11.5px] font-medium text-muted-foreground hover:bg-muted/70 hover:text-foreground dark:hover:bg-white/[0.07]"
        onClick={() => onAction(prompt)}
      >
        <CornerDownRight className="mr-1 h-3 w-3" aria-hidden />
        {actionLabel}
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        disabled={awaitingConfirmation}
        aria-label={editLabel}
        title={editLabel}
        className="h-7 w-7 shrink-0 rounded-full text-muted-foreground hover:bg-background/85 hover:text-foreground dark:hover:bg-white/[0.07]"
        onClick={() => onEdit(prompt)}
      >
        <SquarePen className="h-3.5 w-3.5" aria-hidden />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        disabled={awaitingConfirmation}
        aria-label={deleteLabel}
        className="h-7 w-7 shrink-0 rounded-full text-muted-foreground hover:bg-background/85 hover:text-destructive dark:hover:bg-white/[0.07]"
        onClick={() => onDelete(prompt.id)}
      >
        <Trash2 className="h-3 w-3" aria-hidden />
      </Button>
    </div>
  );
}

function ComposerCliMentionOverlay({
  segments,
  isHero,
  className,
  ghostRange,
}: {
  segments: CapabilityMentionSegment[];
  isHero: boolean;
  className: string;
  ghostRange?: { start: number; end: number } | null;
}) {
  let offset = 0;
  return (
    <div
      aria-hidden
      className={cn(
        className,
        "pointer-events-none absolute inset-0 z-0 overflow-hidden whitespace-pre-wrap break-words text-foreground",
      )}
    >
      {segments.map((segment, index) => {
        const start = offset;
        offset += segment.text.length;
        if (segment.kind === "text") {
          return <span key={`text-${index}`}>{segment.text}</span>;
        }
        const isGhost = ghostRange?.start === start && ghostRange.end === offset;
        return (
          <span
            key={`${segment.kind}-${index}`}
            data-testid={isGhost ? "composer-session-drag-preview" : undefined}
            className={cn(isGhost && "opacity-45 transition-opacity duration-100")}
          >
            <CapabilityMentionToken
              segment={segment}
              variant="composer"
              isHero={isHero}
            />
          </span>
        );
      })}
    </div>
  );
}
interface SlashCommandPaletteProps {
  commands: SlashPaletteCommand[];
  selectedIndex: number;
  layout: SlashPaletteLayout;
  isHero: boolean;
  onHover: (index: number) => void;
  onChoose: (command: SlashPaletteCommand) => void;
}

interface CliAppMentionPaletteProps {
  candidates: MentionCandidate[];
  selectedIndex: number;
  layout: SlashPaletteLayout;
  isHero: boolean;
  onHover: (index: number) => void;
  onChoose: (candidate: MentionCandidate) => void;
}

function useSelectedOptionScroll(selectedIndex: number) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const option = container.querySelector<HTMLElement>(
      `[data-palette-index="${selectedIndex}"]`,
    );
    if (typeof option?.scrollIntoView === "function") {
      option.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  return containerRef;
}

function CliAppMentionPalette({
  candidates,
  selectedIndex,
  layout,
  isHero,
  onHover,
  onChoose,
}: CliAppMentionPaletteProps) {
  const { t } = useTranslation();
  const listMaxHeight = Math.max(
    0,
    layout.maxHeight - SLASH_PALETTE_CHROME_PX,
  );
  const listRef = useSelectedOptionScroll(selectedIndex);
  const groupedCandidates = (["session", "cli", "mcp"] as const)
    .map((kind) => ({
      kind,
      label: kind === "session"
        ? t("thread.composer.mentions.sessionGroup")
        : kind === "cli"
          ? t("thread.composer.mentions.cliGroup")
          : t("thread.composer.mentions.mcpGroup"),
      items: candidates
        .map((candidate, index) => ({ candidate, index }))
        .filter(({ candidate }) => candidate.kind === kind),
    }))
    .filter((group) => group.items.length > 0);
  return (
    <div
      role="listbox"
      aria-label={t("thread.composer.mentions.ariaLabel")}
      style={{ maxHeight: layout.maxHeight }}
      className={cn(
        floatingSurfaceVisualClassName,
        "absolute left-1/2 z-30 w-[calc(100%-0.5rem)] -translate-x-1/2 overflow-hidden",
        layout.placement === "above" ? "bottom-full mb-2" : "top-full mt-2",
        isHero ? "max-w-[58rem]" : "max-w-[49.5rem]",
      )}
    >
      <div ref={listRef} className="overflow-y-auto" style={{ maxHeight: listMaxHeight }}>
        {groupedCandidates.map((group) => (
          <div key={group.kind} role="group" aria-label={group.label} className="mt-1.5 first:mt-0">
            <div className="px-2 pb-1 pt-1 text-[12px] font-medium text-muted-foreground/72">
              {group.label}
            </div>
            {group.items.map(({ candidate, index }) => {
              const selected = index === selectedIndex;
              const name = candidate.name;
              const typeLabel = candidate.kind === "cli"
                ? t("thread.composer.mentions.cliBadge")
                : candidate.kind === "mcp"
                  ? t("thread.composer.mentions.mcpBadge")
                  : t("thread.composer.mentions.sessionBadge");
              const ariaDescription = candidate.kind === "cli"
                ? t("thread.composer.mentions.cliDescription", { name })
                : candidate.kind === "mcp"
                  ? t("thread.composer.mentions.mcpDescription", { name })
                  : t("thread.composer.mentions.sessionDescription", { name });
              return (
                <button
                  key={`${candidate.kind}-${name}`}
                  type="button"
                  role="option"
                  data-palette-index={index}
                  aria-selected={selected}
                  aria-label={`${candidate.displayName} @${name} ${ariaDescription} ${typeLabel}`}
                  onMouseEnter={() => onHover(index)}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    onChoose(candidate);
                  }}
                  className={cn(
                    floatingItemClassName,
                    "flex min-h-10 w-full items-center gap-2.5 px-2.5 py-1.5 text-left transition-colors",
                    selected
                      ? "bg-foreground/[0.055] text-foreground"
                      : "text-foreground/90 hover:bg-foreground/[0.04]",
                  )}
                >
                  <MentionCandidateLogo candidate={candidate} selected={selected} />
                  <span className="flex min-w-0 flex-1 items-baseline gap-2">
                    <span className="min-w-0 truncate text-[15px] font-medium tracking-normal text-foreground">
                      {candidate.displayName}
                    </span>
                    <span className="truncate text-[15px] font-normal tracking-normal text-muted-foreground/72">
                      @{name}
                    </span>
                  </span>
                  {candidate.kind !== "session" ? (
                    <span
                      className={cn(
                        "ml-2 shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold tracking-normal",
                        candidate.kind === "cli"
                          ? "bg-orange-500/10 text-orange-600 dark:text-orange-300"
                          : "bg-sky-500/10 text-sky-600 dark:text-sky-300",
                      )}
                    >
                      {typeLabel}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function MentionCandidateLogo({
  candidate,
  selected,
}: {
  candidate: MentionCandidate;
  selected: boolean;
}) {
  const color = candidate.kind === "session"
    ? candidate.mention.id
      ? sessionHandleColor(candidate.mention.id)
      : INLINE_TOKEN_HIGHLIGHT_COLOR
    : candidate.brandColor || INLINE_TOKEN_HIGHLIGHT_COLOR;
  const rawLogoUrl = candidate.kind === "session" ? null : candidate.logoUrl;
  const logoUrls = useMemo(() => logoFallbackUrls(rawLogoUrl), [rawLogoUrl]);
  const { logoUrl, onLogoError, onLogoLoad } = useLogoFallback(logoUrls);

  if (candidate.kind === "session") {
    return (
      <span
        className="flex h-5 w-5 shrink-0 items-center justify-center"
        style={{ color }}
      >
        <MessageCircle className="h-4 w-4" aria-hidden />
      </span>
    );
  }
  if (logoUrl) {
    return (
      <span
        className={cn(
          "flex h-5 w-5 shrink-0 items-center justify-center overflow-hidden rounded-compact",
          selected ? "bg-background/55" : "bg-transparent",
        )}
      >
        <img
          src={logoUrl}
          alt=""
          decoding="async"
          loading="lazy"
          className="h-5 w-5 object-contain"
          onLoad={onLogoLoad}
          onError={onLogoError}
        />
      </span>
    );
  }
  return (
    <span
      className="flex h-5 w-5 shrink-0 items-center justify-center rounded-compact text-[7.5px] font-semibold text-white"
      style={{ backgroundColor: color }}
    >
      {candidate.initials}
    </span>
  );
}

function SlashCommandPalette({
  commands,
  selectedIndex,
  layout,
  isHero,
  onHover,
  onChoose,
}: SlashCommandPaletteProps) {
  const { t } = useTranslation();
  const listMaxHeight = Math.max(
    0,
    layout.maxHeight - SLASH_PALETTE_CHROME_PX,
  );
  const listRef = useSelectedOptionScroll(selectedIndex);
  return (
    <div
      role="listbox"
      aria-label={t("thread.composer.slash.ariaLabel")}
      style={{ maxHeight: layout.maxHeight }}
      className={cn(
        floatingSurfaceVisualClassName,
        "absolute left-1/2 z-30 w-[calc(100%-0.5rem)] -translate-x-1/2 overflow-hidden",
        layout.placement === "above" ? "bottom-full mb-2" : "top-full mt-2",
        isHero ? "max-w-[58rem]" : "max-w-[49.5rem]",
      )}
    >
      <div ref={listRef} className="overflow-y-auto pr-0.5" style={{ maxHeight: listMaxHeight }}>
        {commands.map((command, index) => {
          const Icon = COMMAND_ICONS[command.icon] ?? CircleHelp;
          const selected = index === selectedIndex;
          const isSkill = command.kind === "skill";
          const commandKey = slashCommandI18nKey(command.command);
          const title = t(`thread.composer.slash.commands.${commandKey}.title`, {
            defaultValue: command.title,
          });
          const description = t(`thread.composer.slash.commands.${commandKey}.description`, {
            defaultValue: command.description,
          });
          return (
            <button
              key={command.command}
              type="button"
              role="option"
              data-palette-index={index}
              aria-selected={selected}
              onMouseEnter={() => onHover(index)}
              onMouseDown={(e) => {
                e.preventDefault();
                onChoose(command);
              }}
              className={cn(
                floatingItemClassName,
                "flex min-h-[44px] w-full items-center gap-3 px-3 py-2 text-left transition-colors",
                selected
                  ? "bg-foreground/[0.065] text-foreground dark:bg-white/[0.09]"
                  : "text-foreground/86 hover:bg-foreground/[0.045] dark:hover:bg-white/[0.065]",
              )}
            >
              <span
                className={cn(
                  "flex h-7 w-7 shrink-0 items-center justify-center text-muted-foreground transition-colors",
                  selected && "text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
              </span>
              <span className="flex min-w-0 flex-1 flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-2">
                <span
                  className={cn(
                    "text-[13.5px] font-semibold tracking-normal text-foreground",
                    isSkill
                      ? "max-w-full shrink-0 break-all sm:max-w-[55%]"
                      : "min-w-0 truncate",
                  )}
                >
                  {title}
                </span>
                <span className="min-w-0 truncate text-[13px] text-muted-foreground">
                  {command.detail || description}
                </span>
              </span>
              {!isSkill || command.badge || command.recent ? (
                <span className="ml-2 flex max-w-[42%] shrink-0 items-center gap-1.5 sm:max-w-none">
                  {command.badge || command.recent ? (
                    <span className="hidden rounded-full bg-foreground/[0.055] px-2 py-1 text-[11px] font-medium text-muted-foreground sm:inline-flex">
                      {command.badge ?? t("thread.composer.slash.badges.recent")}
                    </span>
                  ) : null}
                  {!isSkill ? (
                    <span className="font-mono text-[12px] text-muted-foreground/60">
                      {command.argHint ? `${command.command} ${command.argHint}` : command.command}
                    </span>
                  ) : null}
                </span>
              ) : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}

interface AttachmentChipProps {
  image: AttachedImage;
  labelRemove: string;
  labelEncoding: string;
  normalizedHint: (origBytes: number, currentBytes: number) => string;
  formatError: (reason: AttachmentError) => string;
  onRemove: () => void;
  onKeyDown: (e: ReactKeyboardEvent<HTMLButtonElement>) => void;
  registerRef: (el: HTMLButtonElement | null) => void;
}

function AttachmentChip({
  image,
  labelRemove,
  labelEncoding,
  normalizedHint,
  formatError,
  onRemove,
  onKeyDown,
  registerRef,
}: AttachmentChipProps) {
  const sizeLabel =
    image.status === "ready" && image.normalized && image.encodedBytes
      ? normalizedHint(image.file.size, image.encodedBytes)
      : formatBytes(image.file.size);
  const tone =
    image.status === "error"
      ? "border-destructive/40 bg-destructive/5 text-destructive"
      : "border-border/70 bg-muted/60";

  return (
    <div
      className={cn(
        "group relative flex items-center gap-2 rounded-control border px-2 py-1.5",
        "transition-colors motion-reduce:transition-none",
        tone,
      )}
      data-testid="composer-chip"
    >
      <div className="relative h-10 w-10 overflow-hidden rounded-md bg-background">
        {image.kind === "image" && image.previewUrl ? (
          <img
            src={image.previewUrl}
            alt=""
            aria-hidden
            loading="eager"
            draggable={false}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            {image.kind === "image" ? (
              <ImageIcon className="h-4 w-4 text-muted-foreground" aria-hidden />
            ) : (
              <FileText className="h-4 w-4 text-muted-foreground" aria-hidden />
            )}
          </div>
        )}
        {image.status === "encoding" ? (
          <div
            className="absolute inset-0 flex items-center justify-center bg-background/60"
            aria-label={labelEncoding}
          >
            <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden />
          </div>
        ) : null}
      </div>
      <div className="flex min-w-0 flex-col text-[11.5px] leading-4">
        <span className="max-w-[min(14rem,calc(100vw-8rem))] truncate font-medium" title={image.file.name}>
          {image.file.name}
        </span>
        <span className="truncate text-muted-foreground">
          {image.status === "error" && image.error
            ? formatError(image.error)
            : sizeLabel}
        </span>
      </div>
      <button
        type="button"
        ref={registerRef}
        onClick={onRemove}
        onKeyDown={onKeyDown}
        aria-label={labelRemove}
        className={cn(
          "ml-1 grid h-5 w-5 flex-none place-items-center rounded-full",
          "text-muted-foreground/80 hover:bg-foreground/8 hover:text-foreground",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground/30",
        )}
      >
        <X className="h-3.5 w-3.5" aria-hidden />
      </button>
    </div>
  );
}
