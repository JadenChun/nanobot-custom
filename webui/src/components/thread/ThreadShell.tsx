import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { Bot, FileText, Images, ListTodo, PanelRight } from "lucide-react";
import { useTranslation } from "react-i18next";

import { FilePreviewAvailabilityProvider } from "@/components/FilePreviewAvailabilityContext";
import { FilePreviewPanel } from "@/components/FilePreviewPanel";
import { splitFilePath } from "@/components/FileReferenceChip";
import { ApprovalRequestCard } from "@/components/thread/ApprovalRequestCard";
import {
  DelegatedChildHistoryOverlay,
  delegatedChildHasValidAncestry,
  delegatedChildSummaryIsAuthorized,
  delegatedChildSource,
  findDelegatedChildApprovalPrompt,
} from "@/components/thread/DelegatedChildHistoryOverlay";
import {
  FolderAccessRequestCard,
  type CorrelatedFolderAccessResult,
  type FolderAccessResolveIntent,
} from "@/components/thread/FolderAccessRequestCard";
import { SessionHandleLabel } from "@/components/SessionHandleLabel";
import { GeneratedAssetsPanel } from "@/components/thread/GeneratedAssetsSheet";
import { Button } from "@/components/ui/button";
import { PromptNavigator } from "@/components/thread/PromptNavigator";
import { RecoveryNotice } from "@/components/thread/RecoveryNotice";
import { SessionSummaryCard } from "@/components/thread/SessionInfoPopover";
import { cn } from "@/lib/utils";
import { resolveRetainedContextUsage } from "@/lib/context-usage-retention";
import {
  clearComposerDraftsForSession,
  draftNamespaceFor,
  handoffComposerDraft,
  loadComposerDraft,
  NEW_CHAT_DRAFT_KEY,
} from "@/lib/composer-drafts";
import { ThreadComposer } from "@/components/thread/ThreadComposer";
import type {
  ComposerContextUsage,
  ComposerRoundUsage,
} from "@/components/thread/ComposerUsagePopover";
import type { ModelPresetOption } from "@/components/thread/ModelPresetBadge";
import { ThreadHeader } from "@/components/thread/ThreadHeader";
import { MIN_INSPECTOR_DOCK_WIDTH, ThreadInspector, type InspectorTab, type InspectorTabId } from "@/components/thread/ThreadInspector";
import { StreamErrorNotice } from "@/components/thread/StreamErrorNotice";
import {
  ThreadViewport,
  type ThreadViewportHandle,
  type ThreadViewportState,
} from "@/components/thread/ThreadViewport";
import { useKohakuStream, type SendAttachment, type SendOptions } from "@/hooks/useKohakuStream";
import { useSessionHistory, type RevealPromptTarget } from "@/hooks/useSessions";
import { NextTurnSettingsTracker, type NextTurnSettingKind } from "@/lib/next-turn-settings";
import {
  ApiError,
  fetchFilePreviewAvailability,
  fetchInstalledCliApps,
  fetchMcpPresets,
  isCanonicalAbsolutePath,
  listSlashCommands,
  prepareWorkspaceFullAccess,
  setWorkspaceAccessMode,
  WorkspaceModeConflictError,
} from "@/lib/api";
import {
  CLI_APPS_CHANGED_EVENT,
  installedCliAppsFromPayload,
  isCliAppsPayload,
} from "@/lib/cli-app-events";
import { isMobileRuntimeActive, mobileOwnerCapability } from "@/lib/mobile-runtime";
import {
  MCP_PRESETS_CHANGED_EVENT,
  installedMcpPresetsFromPayload,
  isMcpPresetsPayload,
} from "@/lib/mcp-preset-events";
import { messageDeliveryErrorKind, type CanonicalRunSnapshot, type StreamError } from "@/lib/kohaku-client";
import { inferProviderFromModelName, providerDisplayLabel } from "@/lib/provider-brand";
import { parseRiskyApprovalPrompt } from "@/lib/risky-approval";
import {
  delegatedChildReferences,
  delegatedChildRows,
  mergeDelegatedChildRows,
  parseDelegatedChildSummaryPayload,
  parseDelegatedChildPayload,
  sameSourceOrigin,
} from "@/lib/delegated-child-projection";
import type {
  ApprovalDecisionPayload,
  ChatSummary,
  ChatHarness,
  LiveExternalApproval,
  DelegatedChildReference,
  DelegatedChildPayload,
  DelegatedChildSummaryPayload,
  FolderAccessRequest,
  RoundUsage,
  SettingsPayload,
  SlashCommand,
  SkillSummary,
  UIMessage,
  SourceOrigin,
  WorkspaceScopePayload,
  WorkspaceAccessMode,
  WorkspaceOnceReceipt,
  WorkspacesPayload,
} from "@/lib/types";
import { projectWebuiThreadMessages } from "@/lib/thread-display-compat";
import { useClient } from "@/providers/ClientProvider";

type MessageShape = Pick<UIMessage, "role" | "kind" | "content" | "isStreaming" | "turnId">;

/**
 * Matches the `@container thread-viewport (max-width: 59rem)` rule that hides
 * the left prompt rail. Below this width the rail is hidden, so the header must
 * offer the prompt navigator instead.
 */
const PROMPT_RAIL_MIN_WIDTH = 944;
/**
 * Minimum chat pane width to display the Session Summary card beside the chat
 * history without colliding with messages. Below this width the card hides.
 */
const SESSION_SUMMARY_MIN_PANE_WIDTH = 1040;
/**
 * Chat pane width where the empty right gutter next to the centered 49.5rem
 * message column already fits the Session Summary card without shifting chat.
 */
const SESSION_SUMMARY_GUTTER_MIN_WIDTH = 1328;

interface ChildHistoryInspection {
  parentKey: string;
  child: DelegatedChildReference;
  expectedRoot: SourceOrigin | null;
}

interface ChildApprovalPrompt {
  child: DelegatedChildPayload;
  requestKey: string;
  status: "loading" | "ready" | "unavailable";
  approvalId: string | null;
  summary: string;
  saveExclusionCommand?: string;
}

interface PendingChildApprovalDecision {
  requestKey: string;
  turnId: string;
  admitted: boolean;
}

interface PendingCanonicalHydrate {
  historyLineage: number;
  historyVersion: number;
  runGeneration: number;
  uiBaseline: MessageShape[];
  uiLineage: number | null;
  uiRevision: number;
}

interface PendingHistoryLineageCommit {
  lineage: number;
  messages: UIMessage[];
}

interface PendingCanonicalCommit {
  canonicalSnapshot: CanonicalRunSnapshot;
  completedTurnIds: string[];
  expectedUiRevision: number;
  historyLineage: number;
  historyVersion: number;
  hydrate: PendingCanonicalHydrate;
  messages: UIMessage[];
  previousMessages: UIMessage[];
}

type PendingRecoveryInvalidation =
  | {
      kind: "input";
      chatId: string;
      turnId: string;
      requestId: string;
      revision: number;
    }
  | {
      kind: "focus";
      chatId: string;
      requestId: string;
      revision: number;
    }
  | {
      kind: "replacement";
      chatId: string;
      requestId: string;
      revision: number;
    };

type WorkspaceFocusIdentity = Pick<WorkspaceScopePayload, "project_path" | "access_mode">;

interface PendingRecoveryResolve {
  intent: FolderAccessResolveIntent;
  result: CorrelatedFolderAccessResult | null;
  unconfirmedOnceProjection: CorrelatedFolderAccessResult | null;
  baselineFocus: WorkspaceFocusIdentity | null;
}

function workspaceFocusIdentity(scope: WorkspaceScopePayload | null | undefined): WorkspaceFocusIdentity | null {
  if (!scope || !scope.project_path) return null;
  return { project_path: scope.project_path, access_mode: scope.access_mode };
}

function sameWorkspaceFocus(
  left: WorkspaceFocusIdentity | null,
  right: WorkspaceFocusIdentity | null,
): boolean {
  return left !== null
    && right !== null
    && left.project_path === right.project_path
    && left.access_mode === right.access_mode;
}

function sameDelegatedChildProvenance(
  left: DelegatedChildPayload,
  right: DelegatedChildPayload,
): boolean {
  const leftSource = delegatedChildSource(left);
  const rightSource = delegatedChildSource(right);
  return left.parent_session_key === right.parent_session_key
    && left.parent_store_identity === right.parent_store_identity
    && left.parent_incarnation === right.parent_incarnation
    && left.parent_generation === right.parent_generation
    && left.root_key === right.root_key
    && left.root_store_identity === right.root_store_identity
    && left.root_incarnation === right.root_incarnation
    && left.root_generation === right.root_generation
    && leftSource !== null
    && rightSource !== null
    && sameSourceOrigin(leftSource, rightSource);
}

function sameDelegatedChildSummary(
  left: DelegatedChildSummaryPayload,
  right: DelegatedChildSummaryPayload,
): boolean {
  return JSON.stringify(parseDelegatedChildSummaryPayload(left))
    === JSON.stringify(parseDelegatedChildSummaryPayload(right));
}

function childApprovalCandidateKey(child: DelegatedChildPayload): string {
  return JSON.stringify([
    child.child_id,
    child.pending_approval_id,
    child.blocked_run_id,
    child.state,
    child.run_sequence,
    child.state_sequence,
    child.parent_session_key,
    child.parent_store_identity,
    child.parent_incarnation,
    child.parent_generation,
    child.root_key,
    child.root_store_identity,
    child.root_incarnation,
    child.root_generation,
    child.child_session_key,
    child.child_store_identity,
    child.child_incarnation,
    child.child_generation,
  ]);
}

function sameManagedAccess(
  left: NonNullable<FolderAccessRequest["selected_access"]>,
  right: NonNullable<WorkspaceScopePayload["effective_access"]>,
): boolean {
  return left.policy === "managed"
    && right.policy === "managed"
    && left.source === right.source
    && left.grant_root === right.grant_root
    && left.exists === right.exists
    && left.read === right.read
    && left.write === right.write
    && left.boundary === right.boundary
    && left.commands.requested === right.commands.requested
    && left.commands.available === right.commands.available
    && left.commands.backend === right.commands.backend
    && left.commands.reason === right.commands.reason;
}

function folderResolveProjectionMatches(
  intent: FolderAccessResolveIntent,
  result: CorrelatedFolderAccessResult,
  expectedAccessMode: WorkspaceFocusIdentity["access_mode"],
  expectedCanonicalPath: string | null = intent.canonicalPath,
): boolean {
  if (typeof expectedCanonicalPath !== "string" || !expectedCanonicalPath
    || expectedCanonicalPath.length > 2048 || expectedCanonicalPath.includes("\0")
    || !isCanonicalAbsolutePath(expectedCanonicalPath)) return false;
  const request = result.workspace_access_request;
  const scope = result.workspace_scope;
  const commonMatches = request.chat_id === intent.chatId
    && request.access_request_id === intent.accessRequestId
    && request.revision > intent.expectedRevision
    && request.task.turn_id === intent.taskTurnId
    && request.status === "ready"
    && request.can_continue
    && request.selected_path === expectedCanonicalPath
    && request.selected_access !== null
    && scope.project_path === expectedCanonicalPath
    && scope.access_mode === expectedAccessMode
    && scope.effective_access !== undefined
    && sameManagedAccess(request.selected_access, scope.effective_access)
    && (intent.expectedFileAccess === undefined
      || request.selected_access.write === (intent.expectedFileAccess === "read_write"))
    && (intent.expectedCommands === undefined
      || (request.selected_access.commands.requested === intent.expectedCommands
        || (intent.expectedCommands === false && request.selected_access.commands.requested)))
    && (request.selected_duration == null || request.selected_duration === intent.duration);
  if (!commonMatches) return false;
  if (intent.duration !== "once") return true;
  const permissions = result.workspace_permissions;
  const taskAccess = permissions?.task_access;
  const selectedAccess = request.selected_access;
  const folder = taskAccess?.folders.find((candidate) => candidate.path === expectedCanonicalPath);
  return !!permissions
    && !!taskAccess
    && !!selectedAccess
    && typeof intent.expectedPermissionRevision === "number"
    && intent.expectedFileAccess !== undefined
    && typeof intent.expectedCommands === "boolean"
    && permissions.runtime_id === intent.expectedRuntimeId
    && permissions.revision > intent.expectedPermissionRevision
    && permissions.mode === "restricted"
    && permissions.state === "active"
    && taskAccess.task_turn_id === intent.taskTurnId
    && taskAccess.state === "active"
    && folder?.file_access === intent.expectedFileAccess
    && folder?.commands === intent.expectedCommands
    && request.selected_duration === "once"
    && request.task_access_state === "active"
    && scope.access_mode === "restricted"
    && selectedAccess.source === "task"
    && selectedAccess.grant_root === expectedCanonicalPath
    && selectedAccess.read
    && selectedAccess.write === (intent.expectedFileAccess === "read_write")
    && (selectedAccess.commands.requested === intent.expectedCommands
      || (intent.expectedCommands === false && selectedAccess.commands.requested));
}

function folderResolveResultMatches(
  intent: FolderAccessResolveIntent,
  result: CorrelatedFolderAccessResult,
  expectedAccessMode: WorkspaceFocusIdentity["access_mode"],
): boolean {
  const canonicalPath = intent.duration === "once"
    ? result.once_receipt?.canonical_path ?? null
    : intent.canonicalPath;
  if (!folderResolveProjectionMatches(intent, result, expectedAccessMode, canonicalPath)) return false;
  if (intent.duration !== "once") return true;
  const receipt: WorkspaceOnceReceipt | undefined = result.once_receipt;
  const request = result.workspace_access_request;
  const permissions = result.workspace_permissions;
  return !!receipt
    && !!permissions
    && receipt.duration === "once"
    && receipt.access_request_id === intent.accessRequestId
    && receipt.request_revision === request.revision
    && receipt.task_turn_id === intent.taskTurnId
    && receipt.submitted_path === intent.submittedPath
    && receipt.canonical_path === canonicalPath
    && receipt.runtime_id === intent.expectedRuntimeId
    && receipt.permission_revision === permissions.revision
    && receipt.file_access === intent.expectedFileAccess
    && receipt.commands === intent.expectedCommands;
}

function sameFolderResolveAcknowledgement(
  left: CorrelatedFolderAccessResult,
  right: CorrelatedFolderAccessResult,
): boolean {
  const leftPermissions = left.workspace_permissions;
  const rightPermissions = right.workspace_permissions;
  return left.workspace_access_request.chat_id === right.workspace_access_request.chat_id
    && left.workspace_access_request.access_request_id === right.workspace_access_request.access_request_id
    && left.workspace_access_request.revision === right.workspace_access_request.revision
    && left.workspace_access_request.task.turn_id === right.workspace_access_request.task.turn_id
    && left.workspace_access_request.selected_path === right.workspace_access_request.selected_path
    && left.workspace_scope.project_path === right.workspace_scope.project_path
    && left.workspace_scope.access_mode === right.workspace_scope.access_mode
    && !!left.workspace_access_request.selected_access
    && !!right.workspace_access_request.selected_access
    && !!left.workspace_scope.effective_access
    && !!right.workspace_scope.effective_access
    && sameManagedAccess(left.workspace_access_request.selected_access, right.workspace_scope.effective_access)
    && sameManagedAccess(right.workspace_access_request.selected_access, left.workspace_scope.effective_access)
    && (leftPermissions === undefined
      ? rightPermissions === undefined
      : !!rightPermissions
        && leftPermissions.runtime_id === rightPermissions.runtime_id
        && leftPermissions.revision === rightPermissions.revision
        && leftPermissions.mode === rightPermissions.mode
        && leftPermissions.state === rightPermissions.state
        && leftPermissions.task_access?.task_turn_id === rightPermissions.task_access?.task_turn_id
        && leftPermissions.task_access?.state === rightPermissions.task_access?.state);
}

function requestMatchesResolveAcknowledgement(
  request: FolderAccessRequest,
  result: CorrelatedFolderAccessResult,
): boolean {
  const acknowledged = result.workspace_access_request;
  const effectiveAccess = result.workspace_scope.effective_access;
  return request.chat_id === acknowledged.chat_id
    && request.access_request_id === acknowledged.access_request_id
    && request.revision === acknowledged.revision
    && request.task.turn_id === acknowledged.task.turn_id
    && request.status === "ready"
    && request.can_continue
    && request.selected_path === acknowledged.selected_path
    && request.selected_access !== null
    && (acknowledged.selected_duration == null || request.selected_duration === acknowledged.selected_duration)
    && (acknowledged.selected_duration !== "once"
      || request.task_access_state === "active")
    && effectiveAccess !== undefined
    && sameManagedAccess(request.selected_access, effectiveAccess);
}

function sameFolderResolveIntent(
  left: FolderAccessResolveIntent,
  right: FolderAccessResolveIntent,
): boolean {
  return left.chatId === right.chatId
    && left.accessRequestId === right.accessRequestId
    && left.expectedRevision === right.expectedRevision
    && left.taskTurnId === right.taskTurnId
    && left.duration === right.duration
    && left.expectedRuntimeId === right.expectedRuntimeId
    && left.expectedPermissionRevision === right.expectedPermissionRevision
    && left.expectedFileAccess === right.expectedFileAccess
    && left.expectedCommands === right.expectedCommands
    && left.submittedPath === right.submittedPath
    && left.canonicalPath === right.canonicalPath;
}

function sameMessageShape(a: MessageShape, b: MessageShape): boolean {
  return (
    a.role === b.role
    && (a.kind ?? "") === (b.kind ?? "")
    && a.content === b.content
    && (!a.turnId || !b.turnId || a.turnId === b.turnId)
  );
}

function latestComposerContextUsage(messages: UIMessage[]): ComposerContextUsage | null {
  // Completion metadata may live on an earlier cumulative answer, a notice, or
  // the exact user row. Row placement is not a request-generation receipt.
  let end = messages.length;
  while (end > 0) {
    let start = end - 1;
    while (start > 0 && messages[start].role !== "user") start -= 1;
    const turn = messages.slice(start, end);
    const successes = turn.filter((m) => m.kind === "compaction" && m.compaction?.phase === "succeeded");
    let carrier: UIMessage | undefined;
    for (let i = turn.length - 1; i >= 0; i -= 1) {
      const row = turn[i];
      if (row.contextUsage === undefined) continue;
      const seq = row.contextUsage?.completionTurnSeq;
      if (!row.turnId || typeof seq !== "number" || !Number.isInteger(seq) || seq < 0) return null;
      if (!carrier || (carrier.turnId === row.turnId && seq > carrier.contextUsage!.completionTurnSeq!)) {
        carrier = row;
      }
    }
    if (carrier) {
      const snapshot = carrier.contextUsage!;
      const count = snapshot.contextTokens;
      const window = snapshot.contextWindowTokens;
      if (
        carrier.isStreaming || typeof count !== "number" || !Number.isInteger(count) || count < 0
        || (window !== null && (typeof window !== "number" || !Number.isInteger(window) || window <= 0))
        || successes.some((boundary) => (
          boundary.turnId !== carrier.turnId
          || typeof boundary.turnSeq !== "number" || !Number.isInteger(boundary.turnSeq) || boundary.turnSeq < 0
          || snapshot.completionTurnSeq! <= boundary.turnSeq
        ))
      ) return null;
      return { contextTokens: count, ...(window !== null ? { contextWindowTokens: window } : {}) };
    }
    // Legacy additive usage has no dispatch proof across *any* success in its
    // turn, even if a later canonical rewrite happens to carry it after notice.
    if (successes.length) return null;
    for (let i = turn.length - 1; i >= 0; i -= 1) {
      const row = turn[i];
      const count = row.usage?.context_tokens;
      if (row.role === "assistant" && row.kind !== "trace" && !row.isStreaming
        && typeof count === "number" && Number.isFinite(count) && count >= 0) {
        return {
          contextTokens: count,
          ...(typeof row.contextWindowTokens === "number" ? { contextWindowTokens: row.contextWindowTokens } : {}),
        };
      }
    }
    end = start;
  }
  return null;
}

function recentComposerRoundUsage(messages: UIMessage[]): ComposerRoundUsage[] {
  const recent: ComposerRoundUsage[] = [];
  const seenTurns = new Set<string>();
  for (let index = messages.length - 1; index >= 0 && recent.length < 8; index -= 1) {
    const message = messages[index];
    const turnKey = message.turnId || message.id;
    if (
      message.role !== "assistant"
      || message.kind === "trace"
      || message.isStreaming
      || seenTurns.has(turnKey)
    ) {
      continue;
    }

    seenTurns.add(turnKey);
    const rounds: RoundUsage[] = message.roundUsages ?? [];
    for (let roundIndex = rounds.length - 1; roundIndex >= 0; roundIndex -= 1) {
      const round = rounds[roundIndex];
      const inputTokens = round.prompt_tokens;
      if (
        recent.length >= 8
        || typeof inputTokens !== "number"
        || !Number.isFinite(inputTokens)
        || inputTokens <= 0
      ) {
        continue;
      }
      const outputTokens = round.completion_tokens;
      const cachedTokens = round.cached_tokens;
      const estimatedTokens = round.estimated_tokens;
      const generationMs = round.generation_ms;
      recent.push({
        id: `${turnKey}:${roundIndex}`,
        timestamp: message.completedAt ?? message.createdAt,
        inputTokens,
        ...(typeof outputTokens === "number" && Number.isFinite(outputTokens)
          ? { outputTokens }
          : {}),
        ...(typeof cachedTokens === "number" && Number.isFinite(cachedTokens)
          ? { cachedTokens }
          : {}),
        ...(typeof estimatedTokens === "number" && Number.isFinite(estimatedTokens)
          ? { estimatedTokens }
          : {}),
        ...(typeof generationMs === "number" && Number.isFinite(generationMs)
          ? { generationMs }
          : {}),
      });
    }
  }
  return recent.reverse();
}

function snapshotPreservesMessage(
  current: MessageShape,
  candidate: MessageShape,
  allowCompletedTurnReplacement: boolean,
): boolean {
  if (sameMessageShape(current, candidate)) return true;
  if (
    allowCompletedTurnReplacement
    && current.role === "assistant"
    && candidate.role === current.role
    && (candidate.kind ?? "") === (current.kind ?? "")
    && !!current.turnId
    && candidate.turnId === current.turnId
  ) {
    return true;
  }
  return (
    current.role === "assistant"
    && current.isStreaming === true
    && candidate.role === current.role
    && (candidate.kind ?? "") === (current.kind ?? "")
    && (!current.turnId || !candidate.turnId || candidate.turnId === current.turnId)
    && candidate.content.startsWith(current.content)
  );
}

function durableMessageShape(message: UIMessage): MessageShape | null {
  // A local uncertain input is retained for display, not durable-tail proof.
  if (message.role === "user" && message.deliveryStatus === "unknown") return null;
  if (message.kind === "trace") return null;
  if (message.role !== "user" && message.role !== "assistant") return null;
  if (message.role === "assistant" && !message.content.trim() && !message.media?.length) {
    return null;
  }
  return {
    role: message.role,
    kind: message.kind,
    content: message.content,
    isStreaming: message.isStreaming,
    turnId: message.turnId,
  };
}

function durableMessageShapes(messages: UIMessage[]): MessageShape[] {
  return messages
    .map(durableMessageShape)
    .filter((message): message is MessageShape => message !== null);
}

function preservesMessageShapes(
  expected: MessageShape[],
  candidates: MessageShape[],
  allowCompletedTurnReplacement: boolean,
): boolean {
  let cursor = 0;
  let previousCandidate: MessageShape | null = null;
  for (const message of expected) {
    if (
      allowCompletedTurnReplacement
      && previousCandidate?.role === "assistant"
      && message.role === "assistant"
      && !!message.turnId
      && message.turnId === previousCandidate.turnId
    ) {
      // A delayed websocket delta can briefly create a second bubble after an
      // HTTP completion snapshot. The completed replay is authoritative for
      // that turn, so both local fragments may map to its single assistant row.
      continue;
    }
    let found = false;
    while (cursor < candidates.length) {
      const candidate = candidates[cursor];
      cursor += 1;
      if (snapshotPreservesMessage(message, candidate, allowCompletedTurnReplacement)) {
        found = true;
        previousCandidate = candidate;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

function preservesDurableMessages(
  current: UIMessage[],
  snapshot: UIMessage[],
  allowCompletedTurnReplacement = false,
): boolean {
  // Canonical history refreshes can race with live websocket messages after fork/send.
  // Never accept a refreshed snapshot that drops a user/assistant message already shown.
  const expected = durableMessageShapes(current);
  if (expected.length === 0) return true;
  return preservesMessageShapes(
    expected,
    durableMessageShapes(snapshot),
    allowCompletedTurnReplacement,
  );
}

function resetDropsPostRequestDurableTail(
  baseline: MessageShape[],
  current: UIMessage[],
  snapshot: UIMessage[],
): boolean {
  const currentDurable = durableMessageShapes(current);
  let stablePrefixLength = 0;
  while (
    stablePrefixLength < baseline.length
    && stablePrefixLength < currentDurable.length
    && sameMessageShape(baseline[stablePrefixLength], currentDurable[stablePrefixLength])
  ) {
    stablePrefixLength += 1;
  }
  const postRequestTail = currentDurable.slice(stablePrefixLength);
  if (postRequestTail.length === 0) return false;
  return !preservesMessageShapes(
    postRequestTail,
    durableMessageShapes(snapshot),
    true,
  );
}

function isStaleThreadSnapshot(
  current: UIMessage[],
  snapshot: UIMessage[],
  allowCompletedTurnReplacement = false,
): boolean {
  if (current.length === 0) return false;
  if (snapshot.length === 0) return true;
  if (!preservesDurableMessages(current, snapshot, allowCompletedTurnReplacement)) return true;
  if (snapshot.length >= current.length) return false;
  if (allowCompletedTurnReplacement) return false;
  return snapshot.every((message, index) => sameMessageShape(current[index], message));
}

function prependOlderHistoryPrefix(
  current: UIMessage[],
  normalizedHistory: UIMessage[],
): UIMessage[] | null {
  const currentTranscript = current.filter((row) => row.kind !== "delegated_child");
  if (currentTranscript.length === 0 || normalizedHistory.length === 0) return null;
  const firstCurrent = currentTranscript[0];
  const prefixCount = normalizedHistory.findIndex(
    (candidate) => candidate.id === firstCurrent.id,
  );
  if (prefixCount <= 0) return null;
  const overlapCount = normalizedHistory.length - prefixCount;
  if (overlapCount <= 0 || overlapCount > currentTranscript.length) return null;
  for (let index = 0; index < overlapCount; index += 1) {
    const historyRow = normalizedHistory[prefixCount + index];
    const currentRow = currentTranscript[index];
    if (historyRow.id !== currentRow.id && !sameMessageShape(historyRow, currentRow)) {
      return null;
    }
  }
  return [...normalizedHistory.slice(0, prefixCount), ...current];
}

function latestActiveTurnId(messages: UIMessage[], runStartedAt: number | null): string | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.isStreaming && message.turnId) return message.turnId;
  }
  if (runStartedAt === null) return null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (
      message.role !== "user"
      && message.turnId
      && message.createdAt >= runStartedAt * 1000
    ) return message.turnId;
  }
  return null;
}

function hasInlineDeliveryError(
  messages: UIMessage[],
  error: StreamError | null,
): boolean {
  if (!error?.turnId) return false;
  const kind = messageDeliveryErrorKind(error);
  if (!kind) return false;
  return messages.some((message) => (
    message.role === "user"
    && message.turnId === error.turnId
    && (message.deliveryStatus === "failed" || message.deliveryStatus === "unknown")
    && message.deliveryErrorKind === kind
  ));
}

function preserveUnknownDeliveryStatus(
  current: UIMessage[],
  snapshot: UIMessage[],
): UIMessage[] {
  const uncertainRows = new Map<string, UIMessage>();
  for (const message of current) {
    if (
      message.role === "user"
      && message.turnId
      && message.deliveryStatus === "unknown"
    ) {
      uncertainRows.set(message.turnId, message);
    }
  }
  if (uncertainRows.size === 0) return snapshot;

  const restored = new Set<string>();
  const projected = snapshot.map<UIMessage>((message) => {
    const turnId = message.role === "user" ? message.turnId : undefined;
    if (!turnId || restored.has(turnId)) return message;
    const uncertain = uncertainRows.get(turnId);
    if (!uncertain) return message;
    restored.add(turnId);
    return {
      ...message,
      deliveryStatus: "unknown",
      deliveryErrorKind: uncertain.deliveryErrorKind,
    };
  });
  // A bounded/reset page can omit an UNKNOWN attempt. Retain its local row
  // only for display; canonical execution/source evidence still uses snapshot.
  return [...projected, ...[...uncertainRows].flatMap(([turnId, message]) => (
    restored.has(turnId) ? [] : [message]
  ))];
}

function completedAssistantTurnIds(messages: UIMessage[]): string[] {
  return Array.from(new Set(
    messages
      .filter((message) => message.role === "assistant" && !!message.turnId)
      .map((message) => message.turnId as string),
  ));
}

function canonicalRunSnapshot(
  messages: UIMessage[],
  hasPendingToolCalls: boolean,
  activeTurnId: string | null,
  activitySnapshotAvailable: boolean,
): CanonicalRunSnapshot {
  return {
    observedTurnIds: Array.from(new Set(
      messages
        .filter((message) => message.role === "user" && !!message.turnId)
        .map((message) => message.turnId as string),
    )),
    hasPendingToolCalls,
    activeTurnId,
    activitySnapshotAvailable,
  };
}

type FilePreviewAvailabilityCacheEntry = {
  available?: boolean;
  promise: Promise<boolean>;
  revision: string;
};

interface ThreadShellProps {
  session: ChatSummary | null;
  routeSessionKey?: string | null;
  openingConversation?: boolean;
  sessions?: ChatSummary[];
  title: string;
  temporary?: boolean;
  temporaryChatIds?: readonly string[];
  temporaryChatEnabled?: boolean;
  onTemporaryChatEnabledChange?: (enabled: boolean) => void;
  onToggleSidebar: () => void;
  onGoHome?: () => void;
  onNewChat?: () => void;
  onCreateChat?: (
    workspaceScope?: WorkspaceScopePayload | null,
    initialMessage?: string,
    modelPreset?: string | null,
    options?: { fullAccess?: boolean; harness?: ChatHarness },
  ) => Promise<string | null>;
  onForkChat?: (sourceChatId: string, beforeUserIndex: number) => Promise<string | null>;
  onTurnEnd?: () => void;
  theme?: "light" | "dark";
  onToggleTheme?: () => void;
  hideSidebarToggleForHostChrome?: boolean;
  hideSidebarToggle?: boolean;
  hideThemeButton?: boolean;
  hideHeaderTitle?: boolean;
  inlineHandle?: boolean;
  hideHeader?: boolean;
  headerActions?: ReactNode;
  headerPortalTarget?: HTMLElement | null;
  headerActive?: boolean;
  composerActive?: boolean;
  composerInputAriaLabel?: string;
  emptyComposerVariant?: "hero" | "thread";
  workspaceScope?: WorkspaceScopePayload | null;
  workspaceDefaultScope?: WorkspaceScopePayload | null;
  workspaceControls?: WorkspacesPayload["controls"] | null;
  workspaces?: WorkspacesPayload | null;
  onRefreshWorkspaces?: () => Promise<void>;
  workspaceScopeDisabled?: boolean;
  workspaceError?: string | null;
  onDismissWorkspaceError?: () => void;
  onWorkspaceScopeChange?: (scope: WorkspaceScopePayload) => void;
  onWorkspaceScopeAcknowledged?: (scope: WorkspaceScopePayload) => void;
  settingsSnapshot?: SettingsPayload | null;
  onOpenModelSettings?: () => void;
  skills?: SkillSummary[];
}

function toModelBadgeLabel(modelName: string | null): string | null {
  if (!modelName) return null;
  const trimmed = modelName.trim();
  if (!trimmed) return null;
  const leaf = trimmed.split("/").pop() ?? trimmed;
  return leaf || trimmed;
}

interface ModelBadgeInfo {
  label: string | null;
  model: string | null;
  provider: string | null;
  providerLabel: string | null;
  needsSetup: boolean;
}

function modelPresetForBadge(
  settings: SettingsPayload | null,
  scopedPreset: string | null,
): SettingsPayload["model_presets"][number] | null {
  if (!settings) return null;
  if (scopedPreset) {
    return settings.model_presets.find((preset) => preset.name === scopedPreset) ?? null;
  }
  const configured = settings.agent.model_preset || "default";
  return (
    settings.model_presets.find((preset) => preset.name === configured)
    ?? settings.model_presets.find((preset) => preset.active)
    ?? null
  );
}

function toModelBadgeInfo(
  modelName: string | null,
  settings: SettingsPayload | null,
  modelPreset: string | null = null,
): ModelBadgeInfo {
  const scopedPreset = modelPreset?.trim() || null;
  const preset = modelPresetForBadge(settings, scopedPreset);
  const model = scopedPreset
    ? preset?.model || null
    : settings?.agent.model || modelName || null;
  const label = preset
    ? preset.is_default
      ? preset.label?.trim() || "Default"
      : preset.name.trim()
    : scopedPreset || toModelBadgeLabel(model);
  const rawProvider = preset?.provider
    || (!scopedPreset ? settings?.agent.provider : null)
    || null;
  const provider = rawProvider === "auto"
    ? preset?.resolved_provider
      || (!scopedPreset ? settings?.agent.resolved_provider : null)
      || null
    : rawProvider || inferProviderFromModelName(model);
  const providerRow = provider
    ? settings?.providers.find((item) => item.name === provider)
    : null;
  const needsSetup = Boolean(
    settings && (!model || !provider || !providerRow || !providerRow.configured),
  );
  return {
    label,
    model: toModelBadgeLabel(model),
    provider,
    providerLabel: provider ? providerDisplayLabel(settings?.providers ?? [], provider) : null,
    needsSetup,
  };
}

function modelPresetOptionsFromSettings(
  settings: SettingsPayload | null,
): ModelPresetOption[] {
  if (!settings) return [];
  const order = new Map(
    (settings.model_call_order ?? []).map((name, index) => [name.trim(), index]),
  );
  return settings.model_presets
    .filter((preset) => !preset.is_default && preset.name.trim())
    .sort((a, b) => (
      (order.get(a.name.trim()) ?? Number.POSITIVE_INFINITY)
      - (order.get(b.name.trim()) ?? Number.POSITIVE_INFINITY)
    ))
    .map((preset) => {
      const name = preset.name.trim();
      return {
        name,
        model: preset.model,
        provider: preset.resolved_provider || preset.provider,
      };
    });
}

function modelReasoningCommand(preset: string, effort: string | null): string {
  return `/model --effort=${effort ?? "default"} ${preset}`;
}

const HERO_GREETING_KEYS = [
  "thread.empty.greetings.workOn",
  "thread.empty.greetings.start",
  "thread.empty.greetings.build",
  "thread.empty.greetings.tackle",
] as const;

function HeroGreeting({ text }: { text: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);

  useLayoutEffect(() => {
    const container = containerRef.current;
    const heading = headingRef.current;
    if (!container || !heading) return;

    const fitToWidth = () => {
      heading.style.removeProperty("font-size");
      const availableWidth = container.clientWidth;
      if (availableWidth <= 0) return;

      const naturalWidth = heading.scrollWidth;
      const maximumFontSize = Number.parseFloat(window.getComputedStyle(heading).fontSize);
      if (
        naturalWidth <= availableWidth
        || !Number.isFinite(maximumFontSize)
        || maximumFontSize <= 0
      ) {
        return;
      }

      const fittedFontSize = Math.max(
        12,
        Math.floor(maximumFontSize * ((availableWidth - 2) / naturalWidth) * 100) / 100,
      );
      heading.style.fontSize = `${fittedFontSize}px`;
    };

    fitToWidth();

    let lastObservedWidth = container.clientWidth;
    const resizeObserver = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(([entry]) => {
          const nextWidth = entry?.contentRect.width ?? container.clientWidth;
          if (nextWidth === lastObservedWidth) return;
          lastObservedWidth = nextWidth;
          fitToWidth();
        });
    resizeObserver?.observe(container);
    window.addEventListener("resize", fitToWidth);

    let cancelled = false;
    void document.fonts?.ready.then(() => {
      if (!cancelled) fitToWidth();
    });

    return () => {
      cancelled = true;
      resizeObserver?.disconnect();
      window.removeEventListener("resize", fitToWidth);
    };
  }, [text]);

  return (
    <div ref={containerRef} className="min-w-0 w-full max-w-[44rem]">
      <h1
        ref={headingRef}
        data-testid="hero-greeting"
        className="select-none whitespace-nowrap text-[34px] font-normal leading-[1.08] tracking-normal text-foreground sm:text-[48px] sm:leading-tight"
      >
        {text}
      </h1>
    </div>
  );
}

function randomHeroGreetingKey(): (typeof HERO_GREETING_KEYS)[number] {
  const index = Math.floor(Math.random() * HERO_GREETING_KEYS.length);
  return HERO_GREETING_KEYS[index] ?? HERO_GREETING_KEYS[0];
}

interface PendingFirstMessage {
  content: string;
  images?: SendAttachment[];
  options?: SendOptions;
}

interface InstalledSettingItemsOptions<Payload, Item> {
  getToken: () => string;
  eventName: string;
  fetchPayload: (token: string) => Promise<Payload>;
  isPayload: (value: unknown) => value is Payload;
  selectItems: (payload: Payload) => Item[];
}

function useInstalledSettingItems<Payload, Item>({
  getToken,
  eventName,
  fetchPayload,
  isPayload,
  selectItems,
}: InstalledSettingItemsOptions<Payload, Item>): Item[] {
  const [items, setItems] = useState<Item[]>([]);

  useEffect(() => {
    let cancelled = false;
    let refreshQueued = false;
    let refreshAfterFlight = false;
    let refreshing = false;
    let payloadVersion = 0;
    const refresh = async (): Promise<void> => {
      if (refreshing) return;
      refreshing = true;
      const version = payloadVersion;
      try {
        const payload = await fetchPayload(getToken());
        if (!cancelled && version === payloadVersion) {
          setItems(selectItems(payload));
        }
      } catch {
        // Keep the last successful catalog during transient refresh failures.
      } finally {
        refreshing = false;
        if (refreshAfterFlight && !cancelled) {
          refreshAfterFlight = false;
          void refresh();
        }
      }
    };
    const queueRefresh = () => {
      if (document.visibilityState === "hidden" || refreshQueued) return;
      refreshQueued = true;
      queueMicrotask(() => {
        refreshQueued = false;
        if (!cancelled) void refresh();
      });
    };
    void refresh();

    const refreshOnChanged = (event: Event) => {
      const payload = (event as CustomEvent<unknown>).detail;
      if (isPayload(payload)) {
        payloadVersion += 1;
        setItems(selectItems(payload));
        return;
      }
      if (refreshing) {
        refreshAfterFlight = true;
        return;
      }
      queueRefresh();
    };

    window.addEventListener("focus", queueRefresh);
    document.addEventListener("visibilitychange", queueRefresh);
    window.addEventListener(eventName, refreshOnChanged);
    return () => {
      cancelled = true;
      window.removeEventListener("focus", queueRefresh);
      document.removeEventListener("visibilitychange", queueRefresh);
      window.removeEventListener(eventName, refreshOnChanged);
    };
  }, [eventName, fetchPayload, getToken, isPayload, selectItems]);

  return items;
}

export function ThreadShell({
  session,
  routeSessionKey = null,
  openingConversation = false,
  sessions = [],
  title,
  temporary = false,
  temporaryChatIds = [],
  temporaryChatEnabled = false,
  onTemporaryChatEnabledChange,
  onToggleSidebar,
  onCreateChat,
  onForkChat,
  onTurnEnd,
  theme = "light",
  onToggleTheme = () => {},
  hideSidebarToggleForHostChrome = false,
  hideSidebarToggle = false,
  hideThemeButton = true,
  hideHeaderTitle = false,
  inlineHandle = false,
  hideHeader = false,
  headerActions,
  headerPortalTarget,
  headerActive = true,
  composerActive = true,
  composerInputAriaLabel,
  emptyComposerVariant = "hero",
  workspaceScope = null,
  workspaceDefaultScope = null,
  workspaceControls = null,
  workspaces = null,
  onRefreshWorkspaces,
  workspaceScopeDisabled = false,
  workspaceError = null,
  onDismissWorkspaceError,
  onWorkspaceScopeChange,
  onWorkspaceScopeAcknowledged,
  settingsSnapshot = null,
  onOpenModelSettings,
  skills = [],
}: ThreadShellProps) {
  const { t } = useTranslation();
  const chatId = session?.chatId ?? null;
  const claudeChat = session?.harnessBinding?.harness === "claude_sdk";
  const workspaceLocked = claudeChat && !!session?.harnessBinding?.locked_workspace;
  // A route can select a read before the sidebar supplies the chat summary.
  // Only the verified/optimistically created session can own a live composer.
  const historyKey = temporary ? null : session?.key ?? routeSessionKey;
  const permissionOwnerChatIdRef = useRef(chatId);
  permissionOwnerChatIdRef.current = chatId;
  const currentScopeFocus = workspaceFocusIdentity(workspaceScope);
  const acknowledgedWorkspaceFocusRef = useRef<{
    chatId: string | null;
    focus: WorkspaceFocusIdentity | null;
  }>({ chatId, focus: currentScopeFocus });
  if (acknowledgedWorkspaceFocusRef.current.chatId !== chatId) {
    acknowledgedWorkspaceFocusRef.current = { chatId, focus: currentScopeFocus };
  } else if (!acknowledgedWorkspaceFocusRef.current.focus && currentScopeFocus) {
    // The selected canonical scope may load after the session shell mounts.
    acknowledgedWorkspaceFocusRef.current.focus = currentScopeFocus;
  }
  const mentionSessions = useMemo(
    () => sessions.filter((candidate) => candidate.key !== historyKey),
    [historyKey, sessions],
  );
  const {
    messages: historical,
    loading,
    error: historyError,
    historyAccessDenied,
    sessionIncarnation,
    loadingOlder,
    loadingNewer,
    loadOlder,
    loadNewer,
    hasMoreBefore,
    hasMoreAfter,
    tailActivityPending,
    userMessageOffset,
    hasPendingToolCalls,
    completedTurnIds,
    continuity: historyContinuity,
    lineage: historyLineage,
    activeTurnId: historyActiveTurnId,
    activitySnapshotAvailable: historyActivitySnapshotAvailable,
    stopTarget: historyStopTarget,
    stopTargetSnapshotAvailable: historyStopTargetSnapshotAvailable,
    delegatedChildren,
    delegatedChildrenLoading,
    delegatedChildrenError,
    workspaceAccessRequest,
    workspaceAccessRequestLoading,
    workspaceAccessRequestError,
    workspacePermissions,
    workspacePermissionsLoading,
    workspacePermissionsError,
    nextWorkspaceScope: liveNextWorkspaceScope,
    nextModelPreset: liveNextModelPreset,
    nextReasoningEffort: liveNextReasoningEffort,
    nextTurnRevision: liveNextTurnRevision,
    markWorkspaceAccessRequestStale,
    updateWorkspaceAccessRequest,
    refreshWorkspaceAccessRequest,
    markWorkspacePermissionsStale,
    updateWorkspacePermissions,
    refreshWorkspacePermissions,
    separateControls,
    refreshLiveControls,
    refresh: refreshHistory,
    version: historyVersion,
    forkBoundaryMessageCount,
    turnIndex: sessionTurns,
    promptIndex: sessionPrompts,
    totalTurnCount: sessionTurnTotal,
    totalPromptCount: sessionPromptTotal,
    promptIndexHasMoreBefore,
    loadingPromptIndex,
    loadTurnIndex,
    loadMorePromptIndex,
    revealPrompt,
    pendingReveal,
    clearPendingReveal,
    goToLatest,
    loadTurnAt,
  } = useSessionHistory(historyKey);
  const viewingHistoricalWindow = historyKey !== null && hasMoreAfter;
  const effectivePromptTotal = Math.max(sessionPromptTotal, sessionTurnTotal);
  const refreshStopState = separateControls ? refreshLiveControls : refreshHistory;
  const nextTurnTrackerRef = useRef(new NextTurnSettingsTracker());
  const [nextTurnScopeOverride, setNextTurnScopeOverride] =
    useState<WorkspaceScopePayload | null>(null);
  const [nextTurnSettingsError, setNextTurnSettingsError] = useState<string | null>(null);
  const [nextTurnPendingKinds, setNextTurnPendingKinds] =
    useState<ReadonlySet<NextTurnSettingKind>>(() => new Set());
  const nextTurnChatIdRef = useRef<string | null>(chatId);
  const nextTurnSettings = nextTurnTrackerRef.current;
  const workspaceNextTurnPending = nextTurnPendingKinds.has("workspace");
  // Prefer the latest local/acknowledged override over the HTTP history snapshot
  // so repeated next-turn folder changes during a running turn react immediately.
  const nextWorkspaceScope = nextTurnScopeOverride ?? liveNextWorkspaceScope;
  const trackNextTurnSetting = useCallback((
    ownerChatId: string,
    kind: NextTurnSettingKind,
    operation: Promise<unknown>,
  ): number => {
    const seq = nextTurnSettings.begin(ownerChatId, kind, operation);
    if (nextTurnChatIdRef.current === ownerChatId) {
      setNextTurnPendingKinds(nextTurnSettings.pendingKinds(ownerChatId));
    }
    const settle = () => {
      if (nextTurnChatIdRef.current !== ownerChatId) return;
      setNextTurnPendingKinds(nextTurnSettings.pendingKinds(ownerChatId));
    };
    void operation.then(settle, settle);
    return seq;
  }, [nextTurnSettings]);
  useEffect(() => {
    if (nextTurnChatIdRef.current === chatId) return;
    nextTurnChatIdRef.current = chatId;
    setNextTurnScopeOverride(null);
    setNextTurnSettingsError(null);
    setNextTurnPendingKinds(chatId ? nextTurnSettings.pendingKinds(chatId) : new Set());
  }, [chatId, nextTurnSettings]);
  useEffect(() => {
    if (!chatId) return;
    nextTurnSettings.bindIncarnation(chatId, sessionIncarnation);
    setNextTurnPendingKinds(nextTurnSettings.pendingKinds(chatId));
  }, [chatId, nextTurnSettings, sessionIncarnation]);
  useEffect(() => {
    if (!liveNextWorkspaceScope) return;
    setNextTurnScopeOverride((current) => (
      current && current.project_path === liveNextWorkspaceScope.project_path
        && current.access_mode === liveNextWorkspaceScope.access_mode
        ? null
        : current
    ));
  }, [liveNextWorkspaceScope]);
  const [appliedHistoryVersion, setAppliedHistoryVersion] = useState(0);
  const pendingRecoveryInvalidationRef = useRef<PendingRecoveryInvalidation | null>(null);
  const pendingRecoveryResolveRef = useRef<PendingRecoveryResolve | null>(null);
  const recoveryRefreshGenerationRef = useRef(0);
  useEffect(() => {
    pendingRecoveryInvalidationRef.current = null;
    pendingRecoveryResolveRef.current = null;
  }, [chatId]);
  const pendingRecoveryInvalidationSettled = useCallback((request: typeof workspaceAccessRequest) => {
    const pending = pendingRecoveryInvalidationRef.current;
    if (!pending) return true;
    if (pending.chatId !== chatId) {
      pendingRecoveryInvalidationRef.current = null;
      return true;
    }
    if (!request) {
      pendingRecoveryInvalidationRef.current = null;
      return true;
    }
    if (
      pending.kind === "replacement"
      && request.chat_id === chatId
      && request.access_request_id !== pending.requestId
    ) {
      pendingRecoveryInvalidationRef.current = null;
      return true;
    }
    if (
      pending.kind === "input"
      && request.task.turn_id === pending.turnId
    ) {
      pendingRecoveryInvalidationRef.current = null;
      return true;
    }
    if (
      request.access_request_id === pending.requestId
      && request.revision > pending.revision
      && request.status !== "pending"
      && request.status !== "ready"
    ) {
      pendingRecoveryInvalidationRef.current = null;
      return true;
    }
    return false;
  }, [chatId]);
  const startRecoveryResolve = useCallback((intent: FolderAccessResolveIntent) => {
    if (
      intent.chatId !== chatId
      || !workspaceAccessRequest
      || workspaceAccessRequest.chat_id !== chatId
      || workspaceAccessRequest.access_request_id !== intent.accessRequestId
      || workspaceAccessRequest.revision !== intent.expectedRevision
      || workspaceAccessRequest.task.turn_id !== intent.taskTurnId
      || (workspaceAccessRequest.status !== "pending" && workspaceAccessRequest.status !== "ready")
      || (intent.duration === "once" && (
        !workspacePermissions
        || workspacePermissions.runtime_id !== intent.expectedRuntimeId
        || workspacePermissions.revision !== intent.expectedPermissionRevision
        || workspacePermissions.mode !== "restricted"
        || workspacePermissions.state !== "active"
      ))
      || pendingRecoveryInvalidationRef.current !== null
    ) {
      pendingRecoveryResolveRef.current = null;
      return;
    }
    const focusRecord = acknowledgedWorkspaceFocusRef.current;
    const baselineFocus = focusRecord.chatId === chatId ? focusRecord.focus : null;
    pendingRecoveryResolveRef.current = { intent, result: null, unconfirmedOnceProjection: null, baselineFocus };
  }, [chatId, workspaceAccessRequest, workspacePermissions]);
  const settleRecoveryResolve = useCallback((
    intent: FolderAccessResolveIntent,
    result: CorrelatedFolderAccessResult | null,
  ): boolean => {
    const pending = pendingRecoveryResolveRef.current;
    if (!pending || !sameFolderResolveIntent(pending.intent, intent)) return false;
    if (!result) {
      pendingRecoveryResolveRef.current = null;
      return false;
    }
    if (
      intent.chatId !== chatId
      || pendingRecoveryInvalidationRef.current !== null
      || !pending.baselineFocus
      || !folderResolveResultMatches(intent, result, pending.baselineFocus.access_mode)
    ) return false;
    const alreadyAcknowledged = pending.result;
    if (pending.unconfirmedOnceProjection
      && !sameFolderResolveAcknowledgement(pending.unconfirmedOnceProjection, result)) {
      pendingRecoveryResolveRef.current = null;
      return false;
    }
    if (
      alreadyAcknowledged
      && !sameFolderResolveAcknowledgement(alreadyAcknowledged, result)
    ) return false;
    if (
      !alreadyAcknowledged
      && !sameWorkspaceFocus(
        acknowledgedWorkspaceFocusRef.current.chatId === chatId
          ? acknowledgedWorkspaceFocusRef.current.focus
          : null,
        pending.baselineFocus,
      )
    ) return false;
    pending.result = result;
    pending.unconfirmedOnceProjection = null;
    acknowledgedWorkspaceFocusRef.current = {
      chatId,
      focus: workspaceFocusIdentity(result.workspace_scope),
    };
    return true;
  }, [chatId]);
  const refreshRecoveryProjection = useCallback(async () => {
    const generation = ++recoveryRefreshGenerationRef.current;
    try {
      const refreshed = await refreshWorkspaceAccessRequest();
      if (generation !== recoveryRefreshGenerationRef.current) return;
      if (!pendingRecoveryInvalidationSettled(refreshed)) {
        markWorkspaceAccessRequestStale();
      }
    } catch {
      // A failed repair read keeps the old card visible with actions disabled.
    }
  }, [markWorkspaceAccessRequestStale, pendingRecoveryInvalidationSettled, refreshWorkspaceAccessRequest]);
  const { client, getToken, ingressLimits, modelName, token } = useClient();
  const [claudeAvailable, setClaudeAvailable] = useState(false);
  const [draftHarness, setDraftHarness] = useState<ChatHarness>("native");
  const [externalApprovals, setExternalApprovals] = useState<Record<string, LiveExternalApproval & { childId?: string }>>({});
  const activeHarness = !temporaryChatEnabled && !temporary && claudeAvailable ? draftHarness : "native";

  useEffect(() => client.onHarnessAvailability?.(setClaudeAvailable), [client]);
  const pickWorkspaceFolder = useCallback(async (): Promise<string | null> => {
    const response = await client.requestMutation<{ path: unknown }>(
      "workspace.pick_folder",
      {},
      300_000,
    );
    return typeof response.path === "string" ? response.path : null;
  }, [client]);
  const [fallbackModelName, setFallbackModelName] = useState<string | null>(null);
  const [booting, setBooting] = useState(false);
  const [slashCommands, setSlashCommands] = useState<SlashCommand[]>([]);
  const cliApps = useInstalledSettingItems({
    getToken,
    eventName: CLI_APPS_CHANGED_EVENT,
    fetchPayload: fetchInstalledCliApps,
    isPayload: isCliAppsPayload,
    selectItems: installedCliAppsFromPayload,
  });
  const mcpPresets = useInstalledSettingItems({
    getToken,
    eventName: MCP_PRESETS_CHANGED_EVENT,
    fetchPayload: fetchMcpPresets,
    isPayload: isMcpPresetsPayload,
    selectItems: installedMcpPresetsFromPayload,
  });
  const settings = settingsSnapshot;
  const [heroGreetingKey, setHeroGreetingKey] = useState(randomHeroGreetingKey);
  const [submittedViewportTurnId, setSubmittedViewportTurnId] = useState<string | null>(null);
  const [childHistoryInspection, setChildHistoryInspection] =
    useState<ChildHistoryInspection | null>(null);
  const [childHistoryError, setChildHistoryError] = useState<string | null>(null);
  const [childApprovalPrompts, setChildApprovalPrompts] = useState<ChildApprovalPrompt[]>([]);
  const [pendingChildApprovalIds, setPendingChildApprovalIds] = useState<Set<string>>(
    () => new Set(),
  );
  const pendingChildApprovalTurnsRef = useRef<Map<string, PendingChildApprovalDecision>>(
    new Map(),
  );
  const previousInspectionActiveRef = useRef(false);
  const previousChildHistoryModalRef = useRef(false);
  const inspectionReturnRef = useRef<{
    parentKey: string;
    viewportState: ThreadViewportState | null;
    trigger: HTMLButtonElement;
    wasModal: boolean;
  } | null>(null);
  const [shellWidth, setShellWidth] = useState(0);
  const [chatPaneWidth, setChatPaneWidth] = useState(0);
  const [sessionSummaryOpen, setSessionSummaryOpen] = useState(false);
  const [selectedInspector, setSelectedInspector] = useState<InspectorTabId>("subagents");
  const [filePreview, setFilePreview] = useState<{ parentKey: string; path: string } | null>(null);
  const filePreviewPath = filePreview?.parentKey === historyKey ? filePreview.path : null;
  const [filePreviewRefreshKey, setFilePreviewRefreshKey] = useState(0);
  const [assetsInspectorOpen, setAssetsInspectorOpen] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const filePreviewEditKeysRef = useRef<Set<string>>(new Set());
  // Token renewal keeps the same client and draft owner.
  const [draftOwner, setDraftOwner] = useState(() => ({
    client,
    namespace: draftNamespaceFor(token),
  }));
  let draftNamespace = draftOwner.namespace;
  if (draftOwner.client !== client) {
    draftNamespace = draftNamespaceFor(token);
    setDraftOwner({ client, namespace: draftNamespace });
  }
  const [quotedContext, setQuotedContext] = useState<string | null>(null);
  const [composerFocusSignal, setComposerFocusSignal] = useState(0);
  const shellRef = useRef<HTMLElement | null>(null);
  const chatPaneRef = useRef<HTMLDivElement | null>(null);
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null);
  const pendingFirstRef = useRef<PendingFirstMessage | null>(null);
  const [pendingFirstTargetChatId, setPendingFirstTargetChatId] = useState<string | null>(null);
  const [pendingFullDraft, setPendingFullDraft] = useState<{
    chatId: string;
    text: string;
  } | null>(null);
  const [preparingFull, setPreparingFull] = useState(false);
  const [fullSetupError, setFullSetupError] = useState<string | null>(null);
  const viewportRef = useRef<ThreadViewportHandle | null>(null);

  useLayoutEffect(() => {
    const shell = shellRef.current;
    const chatPane = chatPaneRef.current;
    if (!shell) return;
    const measure = () => {
      setShellWidth(shell.clientWidth);
      setChatPaneWidth(chatPane ? chatPane.clientWidth : 0);
    };
    measure();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }
    const observer = new ResizeObserver(measure);
    observer.observe(shell);
    if (chatPane) observer.observe(chatPane);
    return () => observer.disconnect();
  }, []);
  const activeViewportTurnByChatIdRef = useRef<Map<string, string>>(new Map());
  const messageCacheRef = useRef<Map<string, UIMessage[]>>(new Map());
  const knownTemporaryChatIdsRef = useRef(new Set<string>());
  /** Last chatId we associated with the in-memory thread (for cache-on-switch). */
  const prevChatIdForCacheRef = useRef<string | null>(null);
  /** Skip one message-cache write right after chatId changes (messages may not match yet). */
  const skipLayoutCacheRef = useRef(false);
  const pendingCanonicalHydrateRef = useRef<Map<string, PendingCanonicalHydrate>>(new Map());
  const pendingCanonicalCommitRef = useRef<Map<string, PendingCanonicalCommit>>(new Map());
  const pendingHistoryLineageCommitRef = useRef<Map<string, PendingHistoryLineageCommit>>(
    new Map(),
  );
  const committedHistoryLineageRef = useRef<Map<string, number>>(new Map());
  const sessionKeyByChatIdRef = useRef<Map<string, string>>(new Map());
  const currentUiMessagesRef = useRef<UIMessage[] | null>(null);
  const uiRevisionRef = useRef(0);
  const showTemporaryChatControl =
    !hideHeader && !session && !openingConversation && !!onTemporaryChatEnabledChange;

  const clearPendingChildApproval = useCallback((approvalId: string): void => {
    pendingChildApprovalTurnsRef.current.delete(approvalId);
    setPendingChildApprovalIds((current) => {
      if (!current.has(approvalId)) return current;
      const next = new Set(current);
      next.delete(approvalId);
      return next;
    });
  }, []);
  const rejectChildApprovalPresentation = useCallback((
    requestKey: string,
    detail: string,
  ): void => {
    setChildApprovalPrompts((current) => current.map((prompt) => (
      prompt.requestKey === requestKey
        ? { ...prompt, status: "unavailable", approvalId: null, summary: detail }
        : prompt
    )));
  }, []);

  const initial = useMemo(() => {
    if (!chatId) return historical;
    return messageCacheRef.current.get(chatId) ?? historical;
  }, [chatId, historical]);
  const handleTurnEnd = useCallback(() => {
    if (chatId) activeViewportTurnByChatIdRef.current.delete(chatId);
    if (workspaceAccessRequest || pendingRecoveryInvalidationRef.current) {
      void refreshRecoveryProjection();
    }
    setSubmittedViewportTurnId(null);
    setFallbackModelName(null);
    onTurnEnd?.();
  }, [
    chatId,
    onTurnEnd,
    pendingRecoveryInvalidationRef,
    refreshRecoveryProjection,
    workspaceAccessRequest,
  ]);
  const {
    messages,
    messagesReady,
    isStreaming,
    queueCompletionTurnId,
    hasOwnedWork,
    stopCanBeSent,
    stopPending,
    stopNotice,
    dismissStopNotice,
    stopTarget,
    connectionOpen,
    hasUnknownMessageSend,
    runStartedAt,
    retryStatus,
    goalState,
    recoveryState,
    continueRecovery,
    dismissRecovery,
    send,
    joinQueuedAttempt,
    consumeQueuedAttempt,
    transcribeAudio,
    stop,
    reconcileTurnComplete,
    setMessages,
    streamError,
    dismissStreamError,
  } = useKohakuStream(
    chatId,
    initial,
    hasPendingToolCalls,
    handleTurnEnd,
    historyKey,
    historyStopTarget,
    refreshStopState,
    historyStopTargetSnapshotAvailable,
    separateControls,
  );

  useEffect(() => {
    setExternalApprovals({});
    if (!chatId || temporary) return;
    return client.onChat(chatId, (event) => {
      if (event.event === "attached") {
        setExternalApprovals({});
      } else if (event.event === "turn_end") {
        setExternalApprovals((current) => {
          const entries = Object.entries(current);
          const children = entries.filter(([, approval]) => approval.childId);
          return children.length === entries.length ? current : Object.fromEntries(children);
        });
      } else if (event.event === "external_approval") {
        setExternalApprovals((current) => {
          const next = { ...current };
          if (event.approval) next[event.approval_id] = { ...event.approval, childId: event.child_id };
          else delete next[event.approval_id];
          return next;
        });
      }
    });
  }, [chatId, client, temporary]);

  const admissionRejectedTurn = useMemo(() => {
    if (streamError?.kind !== "turn_rejected" || streamError.detail !== "not_admitted") {
      return null;
    }
    if (!streamError.reason) return null;
    return { turnId: streamError.turnId, reason: streamError.reason };
  }, [streamError]);

  useLayoutEffect(() => {
    if (currentUiMessagesRef.current === messages) return;
    currentUiMessagesRef.current = messages;
    uiRevisionRef.current += 1;
    if (!chatId) return;
    const lineageCommit = pendingHistoryLineageCommitRef.current.get(chatId);
    if (!lineageCommit) return;
    pendingHistoryLineageCommitRef.current.delete(chatId);
    if (lineageCommit.messages === messages) {
      committedHistoryLineageRef.current.set(chatId, lineageCommit.lineage);
    }
  }, [chatId, messages]);

  useEffect(() => {
    // Revoked read access invalidates device-local drafts for this session.
    if (historyAccessDenied && historyKey) {
      clearComposerDraftsForSession(draftNamespace, historyKey);
    }
  }, [draftNamespace, historyAccessDenied, historyKey]);

  useEffect(() => {
    if (chatId && historyKey) sessionKeyByChatIdRef.current.set(chatId, historyKey);
  }, [chatId, historyKey]);

  useEffect(() => {
    setFilePreview(null);
    // The composer owns quote save/restore per chat; it resets or restores it.
    setSubmittedViewportTurnId(null);
  }, [historyKey]);

  useEffect(() => {
    const retained = new Set(temporaryChatIds);
    for (const chatId of retained) knownTemporaryChatIdsRef.current.add(chatId);
    for (const cachedChatId of knownTemporaryChatIdsRef.current) {
      if (!retained.has(cachedChatId)) {
        messageCacheRef.current.delete(cachedChatId);
        activeViewportTurnByChatIdRef.current.delete(cachedChatId);
        knownTemporaryChatIdsRef.current.delete(cachedChatId);
      }
    }
  }, [temporaryChatIds]);

  const handleQuoteSelection = useCallback((text: string) => {
    setQuotedContext(text);
    setComposerFocusSignal((value) => value + 1);
  }, []);

  const displayMessages = useMemo(() => projectWebuiThreadMessages(messages), [messages]);
  const currentDelegatedChildren = useMemo(
    () => delegatedChildRows(displayMessages),
    [displayMessages],
  );
  const currentDelegatedChildReferences = useMemo(
    () => delegatedChildReferences(displayMessages),
    [displayMessages],
  );
  const childApprovalCandidates = useMemo(() => {
    if (!historyKey || temporary) return [];
    const expectedRoot = stopTarget?.source ?? historyStopTarget?.source ?? null;
    if (!expectedRoot) return [];
    return currentDelegatedChildren.filter((child) => (
      child.state === "waiting_approval"
      && typeof child.pending_approval_id === "string"
      && child.pending_approval_id.length > 0
      && typeof child.blocked_run_id === "string"
      && child.blocked_run_id.length > 0
      && child.blocked_run_id === child.child_run_id
      && delegatedChildHasValidAncestry(
        child,
        currentDelegatedChildren,
        historyKey,
        expectedRoot,
      )
    ));
  }, [
    currentDelegatedChildren,
    historyKey,
    historyStopTarget?.source,
    stopTarget?.source,
    temporary,
  ]);
  const childApprovalCandidateSignature = JSON.stringify(
    childApprovalCandidates.map(childApprovalCandidateKey),
  );
  const childApprovalCandidatesRef = useRef(childApprovalCandidates);
  childApprovalCandidatesRef.current = childApprovalCandidates;
  useEffect(() => {
    const candidates = childApprovalCandidatesRef.current;
    for (const [approvalId, pending] of pendingChildApprovalTurnsRef.current) {
      const stillCurrent = candidates.some((child) => (
        child.pending_approval_id === approvalId
        && childApprovalCandidateKey(child) === pending.requestKey
      ));
      if (!stillCurrent) clearPendingChildApproval(approvalId);
    }
    setChildApprovalPrompts(candidates.map((child) => ({
      child,
      requestKey: childApprovalCandidateKey(child),
      status: "loading",
      approvalId: null,
      summary: "",
    })));
    if (candidates.length === 0 || !historyKey) return;

    const controller = new AbortController();
    let active = true;
    for (const child of candidates) {
      const requestKey = childApprovalCandidateKey(child);
      void findDelegatedChildApprovalPrompt(
        token,
        child,
        historyKey,
        controller.signal,
      ).then((prompt) => {
        if (!active || controller.signal.aborted) return;
        setChildApprovalPrompts((current) => current.map((entry) => (
          entry.requestKey !== requestKey
            ? entry
            : prompt
              ? {
                  ...entry,
                  status: "ready",
                  approvalId: prompt.approvalId,
                  summary: prompt.summary,
                  saveExclusionCommand: prompt.save_exclusion_command,
                }
              : {
                  ...entry,
                  status: "unavailable",
                  summary: "Approval details could not be verified. Refresh the parent chat before deciding.",
                }
        )));
      }).catch(() => {
        if (!active || controller.signal.aborted) return;
        setChildApprovalPrompts((current) => current.map((entry) => (
          entry.requestKey === requestKey
            ? {
                ...entry,
                status: "unavailable",
                summary: "Approval details could not be verified. Refresh the parent chat before deciding.",
              }
            : entry
        )));
      });
    }
    return () => {
      active = false;
      controller.abort();
    };
  }, [
    childApprovalCandidateSignature,
    clearPendingChildApproval,
    historyVersion,
    historyKey,
    token,
  ]);
  // The meter follows the latest qualified completion observation, not the
  // history window being viewed. While an older page is shown the window cannot
  // see the tail, so keep the last tail-derived value (null if none yet).
  const retainedContextUsageRef = useRef<{
    key: string | null;
    usage: ComposerContextUsage | null;
  }>({ key: null, usage: null });
  const composerContextUsage = useMemo(() => {
    const resolved = resolveRetainedContextUsage(retainedContextUsageRef.current, {
      key: historyKey,
      viewingHistoricalWindow,
      computeFresh: () => latestComposerContextUsage(displayMessages),
    });
    retainedContextUsageRef.current = resolved.retained;
    return resolved.usage;
  }, [displayMessages, historyKey, viewingHistoricalWindow]);
  const composerRoundUsage = useMemo(
    () => recentComposerRoundUsage(displayMessages),
    [displayMessages],
  );
  const currentGoalState = messagesReady ? goalState : undefined;
  // Decision states freeze the interrupted turn and hand the next action to
  // the recovery notice. ``resuming`` remains active; ``recovered`` is only
  // historical metadata and must not suppress a later normal turn.
  const recoveryNeedsDecision = recoveryState?.status === "awaiting_user"
    || recoveryState?.status === "failed";
  const currentRunStartedAt = messagesReady && !recoveryNeedsDecision ? runStartedAt : null;
  const turnActive = messagesReady
    && !recoveryNeedsDecision
    && (isStreaming || currentRunStartedAt !== null);
  useEffect(() => {
    if (!chatId || typeof client.onNextTurnSettings !== "function") return;
    return client.onNextTurnSettings(() => {
      if (nextTurnChatIdRef.current !== chatId) return;
      const latest = client.getNextTurnSettings?.(chatId)?.workspaceScope;
      if (latest) setNextTurnScopeOverride(latest);
    });
  }, [chatId, client]);
  useEffect(() => {
    if (turnActive) return;
    // A confirmed scope equal to the pending choice means admission consumed
    // it; the picker now shows the active folder, not a pending one.
    setNextTurnScopeOverride((current) => (
      current && workspaceScope
        && current.project_path === workspaceScope.project_path
        && current.access_mode === workspaceScope.access_mode
        ? null
        : current
    ));
  }, [turnActive, workspaceScope]);
  const activeChildHistoryInspection = childHistoryInspection?.parentKey === historyKey
    ? childHistoryInspection
    : null;
  const assetsInspectorAvailable = Boolean(historyKey) && !temporary && assetsInspectorOpen;
  // Prefer the tab the user selected when it is still open. Otherwise fall back
  // to whatever remains open, in the same order the tabs are rendered.
  const inspectorTab = selectedInspector === "file" && filePreviewPath
    ? "file"
    : selectedInspector === "assets" && assetsInspectorAvailable
      ? "assets"
      : selectedInspector === "subagents" && activeChildHistoryInspection
        ? "subagents"
        : filePreviewPath
          ? "file"
          : assetsInspectorAvailable
            ? "assets"
            : activeChildHistoryInspection ? "subagents" : null;
  const inspectorDocked = shellWidth >= MIN_INSPECTOR_DOCK_WIDTH;
  const inspectorVisible = !inspectorCollapsed && (inspectorDocked || composerActive);
  const inspectorOpen = inspectorVisible && inspectorTab !== null;
  const inspectorModal = inspectorVisible && inspectorTab !== null && !inspectorDocked;
  const childHistoryModal = inspectorModal && inspectorTab === "subagents";
  const delegatedStatusAvailable = connectionOpen && !delegatedChildrenLoading && !delegatedChildrenError;
  const inspectedCurrentChild = activeChildHistoryInspection
    ? currentDelegatedChildReferences.find((child) => (
        child.child_id === activeChildHistoryInspection.child.child_id
      )) ?? null
    : null;
  const inspectedExpectedRoot = stopTarget?.source
    ?? historyStopTarget?.source
    ?? activeChildHistoryInspection?.expectedRoot
    ?? null;
  const childHistoryProvenanceComplete = Boolean(activeChildHistoryInspection
    && inspectedCurrentChild
    && !delegatedChildrenLoading
    && !delegatedChildrenError
    && (activeChildHistoryInspection.child.history_kind === "summary"
      ? inspectedCurrentChild.history_kind === "summary"
        && sameDelegatedChildSummary(
          activeChildHistoryInspection.child,
          inspectedCurrentChild,
        )
        && delegatedChildSummaryIsAuthorized(
          inspectedCurrentChild,
          historyKey ?? "",
        )
      : inspectedCurrentChild.history_kind !== "summary"
        && sameDelegatedChildProvenance(
          activeChildHistoryInspection.child,
          inspectedCurrentChild,
        )
        && delegatedChildHasValidAncestry(
          activeChildHistoryInspection.child,
          currentDelegatedChildren,
          historyKey ?? "",
          inspectedExpectedRoot,
        )));
  const handleOpenChildHistory = useCallback((
    childValue: DelegatedChildReference,
    trigger: HTMLButtonElement,
  ) => {
    if (!historyKey || temporary) {
      setChildHistoryError("Delegated history is only available inside its original parent chat.");
      return;
    }
    const isSummary = childValue.history_kind === "summary";
    const child = isSummary
      ? parseDelegatedChildSummaryPayload(childValue)
      : parseDelegatedChildPayload(childValue);
    const current = child
      ? currentDelegatedChildReferences.find((row) => row.child_id === child.child_id)
      : undefined;
    const expectedRoot = isSummary
      ? null
      : stopTarget?.source ?? historyStopTarget?.source ?? null;
    const candidate = isSummary
      ? current?.history_kind === "summary"
        ? parseDelegatedChildSummaryPayload(current)
        : null
      : current?.history_kind !== "summary"
        ? parseDelegatedChildPayload(current)
        : null;
    const authorized = isSummary
      ? child?.history_kind === "summary"
        && candidate?.history_kind === "summary"
        && sameDelegatedChildSummary(child, candidate)
        && delegatedChildSummaryIsAuthorized(candidate, historyKey)
      : child?.history_kind !== "summary"
        && candidate?.history_kind !== "summary"
        && !!child
        && !!candidate
        && sameDelegatedChildProvenance(child, candidate)
        && delegatedChildHasValidAncestry(
          candidate,
          currentDelegatedChildren,
          historyKey,
          expectedRoot,
        );
    if (!authorized || !child || !candidate || delegatedChildrenLoading || delegatedChildrenError) {
      setChildHistoryError("This delegated history cannot be verified from complete current parent records. Refresh the parent chat and try again.");
      return;
    }
    const viewportState = viewportRef.current?.captureViewState() ?? null;
    inspectionReturnRef.current = { parentKey: historyKey, viewportState, trigger, wasModal: !inspectorDocked };
    setSelectedInspector("subagents");
    setInspectorCollapsed(false);
    setChildHistoryError(null);
    setChildHistoryInspection({
      parentKey: historyKey,
      child: candidate,
      expectedRoot,
    });
  }, [
    currentDelegatedChildren,
    currentDelegatedChildReferences,
    delegatedChildrenError,
    delegatedChildrenLoading,
    historyKey,
    historyStopTarget?.source,
    inspectorDocked,
    stopTarget?.source,
    temporary,
  ]);
  const handleCloseChildHistory = useCallback(() => {
    setChildHistoryInspection(null);
    setChildHistoryError(null);
  }, []);
  useEffect(() => {
    if (childHistoryInspection && childHistoryInspection.parentKey !== historyKey) {
      setChildHistoryInspection(null);
      inspectionReturnRef.current = null;
    }
  }, [childHistoryInspection, historyKey]);
  useLayoutEffect(() => {
    const isActive = activeChildHistoryInspection !== null;
    const returnState = inspectionReturnRef.current;
    const wasModal = previousChildHistoryModalRef.current;
    if (returnState?.parentKey === historyKey) {
      if (childHistoryModal && !wasModal) {
        if (previousInspectionActiveRef.current) {
          returnState.viewportState = viewportRef.current?.captureViewState() ?? null;
        }
        returnState.wasModal = true;
      } else if (isActive && wasModal && !childHistoryModal) {
        viewportRef.current?.restoreViewState(returnState.viewportState);
        returnState.wasModal = false;
      }
    }
    if (previousInspectionActiveRef.current && !isActive) {
      if (returnState?.parentKey === historyKey) {
        if (returnState.wasModal) viewportRef.current?.restoreViewState(returnState.viewportState);
        const focus = document.activeElement;
        if (focus === null || focus === document.body || !focus.isConnected) {
          window.requestAnimationFrame(() => {
            if (returnState.trigger.isConnected && document.activeElement === document.body) {
              returnState.trigger.focus({ preventScroll: true });
            }
          });
        }
      }
      inspectionReturnRef.current = null;
    }
    previousInspectionActiveRef.current = isActive;
    previousChildHistoryModalRef.current = childHistoryModal;
  }, [activeChildHistoryInspection, childHistoryModal, historyKey]);
  const pendingRootApprovalRef = useRef<{ turnId: string; approvalId: string } | null>(null);
  const [rootApprovalResetSignal, setRootApprovalResetSignal] = useState(0);
  const [dismissedRootApprovalIds, setDismissedRootApprovalIds] = useState<string[]>([]);
  const riskyApproval = useMemo(() => {
    if (claudeChat || turnActive || booting) return null;
    for (let index = displayMessages.length - 1; index >= 0; index -= 1) {
      const message = displayMessages[index];
      if (message.role === "user") return null;
      if (message.role === "assistant" && message.kind !== "trace") {
        const parsed = parseRiskyApprovalPrompt(message.content);
        if (parsed?.approvalId && dismissedRootApprovalIds.includes(parsed.approvalId)) {
          return null;
        }
        return parsed;
      }
    }
    return null;
  }, [booting, claudeChat, dismissedRootApprovalIds, displayMessages, turnActive]);
  const submitRootApprovalDecision = (decision: ApprovalDecisionPayload["decision"]) => {
    const approvalId = riskyApproval?.approvalId;
    if (!chatId || !approvalId) return;
    const turnId = client.sendApprovalDecision(chatId, approvalId, decision, session?.key);
    pendingRootApprovalRef.current = { turnId, approvalId };
  };
  const submitChildApprovalDecision = useCallback((
    prompt: ChildApprovalPrompt,
    decision: ApprovalDecisionPayload["decision"],
  ) => {
    const current = currentDelegatedChildren.find(
      (child) => child.child_id === prompt.child.child_id,
    );
    const expectedRoot = stopTarget?.source ?? historyStopTarget?.source ?? null;
    const rejectGuard = (detail: string) => {
      rejectChildApprovalPresentation(prompt.requestKey, detail);
      void refreshHistory();
    };
    if (
      !chatId
      || !historyKey
      || !messagesReady
      || !delegatedStatusAvailable
      || prompt.status !== "ready"
      || !prompt.approvalId
      || !current
      || childApprovalCandidateKey(current) !== prompt.requestKey
      || current.pending_approval_id !== prompt.approvalId
      || !delegatedChildHasValidAncestry(
        current,
        currentDelegatedChildren,
        historyKey,
        expectedRoot,
      )
      || !current.child_incarnation
      || typeof current.child_generation !== "number"
      || !current.blocked_run_id
    ) {
      rejectGuard("The exact child approval is no longer available. Refresh status before deciding again.");
      return;
    }
    if (pendingChildApprovalIds.has(prompt.approvalId)) return;

    try {
      const turnId = client.sendApprovalDecision(chatId, prompt.approvalId, decision, {
        sessionKey: current.child_session_key,
        sessionIncarnation: current.child_incarnation,
        runId: current.blocked_run_id,
        generation: current.child_generation,
      });
      pendingChildApprovalTurnsRef.current.set(prompt.approvalId, {
        requestKey: prompt.requestKey,
        turnId,
        admitted: false,
      });
      setPendingChildApprovalIds((existing) => new Set(existing).add(prompt.approvalId!));
    } catch {
      rejectGuard("The approval decision could not be sent. Refresh status before deciding again.");
    }
  }, [
    chatId,
    client,
    delegatedStatusAvailable,
    currentDelegatedChildren,
    historyKey,
    historyStopTarget?.source,
    messagesReady,
    pendingChildApprovalIds,
    refreshHistory,
    rejectChildApprovalPresentation,
    stopTarget?.source,
  ]);
  const restoredViewportTurnId = useMemo(
    () => turnActive ? latestActiveTurnId(displayMessages, currentRunStartedAt) : null,
    [currentRunStartedAt, displayMessages, turnActive],
  );
  const rememberedViewportTurnId = chatId
    ? activeViewportTurnByChatIdRef.current.get(chatId) ?? null
    : null;
  const canonicalRunTurnId = chatId && messagesReady && turnActive
    ? client.getRunTurnId(chatId)
    : null;
  const viewportTurnId = messagesReady && turnActive
    ? canonicalRunTurnId
      ?? rememberedViewportTurnId
      ?? historyActiveTurnId
      ?? restoredViewportTurnId
    : null;
  const activeTurnStartedHere =
    viewportTurnId !== null && viewportTurnId === submittedViewportTurnId;
  useEffect(() => {
    if (!chatId || !messagesReady || turnActive) return;
    activeViewportTurnByChatIdRef.current.delete(chatId);
    setSubmittedViewportTurnId((current) =>
      current === rememberedViewportTurnId ? null : current,
    );
  }, [chatId, messagesReady, rememberedViewportTurnId, turnActive]);
  const filePreviewAvailabilityCache = useMemo(
    () => new Map<string, FilePreviewAvailabilityCacheEntry>(),
    [historyKey],
  );
  const filePreviewAvailabilityRevision = `${displayMessages.length}:${turnActive}`;
  const filePreviewAvailabilityRevisionRef = useRef(filePreviewAvailabilityRevision);
  filePreviewAvailabilityRevisionRef.current = filePreviewAvailabilityRevision;
  const resolveFilePreviewAvailability = useCallback((path: string) => {
    if (!historyKey) return Promise.resolve(false);
    const revision = filePreviewAvailabilityRevisionRef.current;
    const cached = filePreviewAvailabilityCache.get(path);
    if (
      cached
      && (cached.available !== false || cached.revision === revision)
    ) {
      return cached.promise;
    }
    const pending = fetchFilePreviewAvailability(getToken(), historyKey, path).catch(
      (error: unknown) => {
        if (error instanceof ApiError) {
          if (error.status === 404 && /API route not found/i.test(error.message)) {
            return true;
          }
          if ([400, 403, 404, 415].includes(error.status)) return false;
        }
        return false;
      },
    );
    const entry: FilePreviewAvailabilityCacheEntry = {
      promise: pending,
      revision,
    };
    filePreviewAvailabilityCache.set(path, entry);
    void pending.then((available) => {
      if (filePreviewAvailabilityCache.get(path) === entry) {
        entry.available = available;
      }
    });
    return pending;
  }, [
    filePreviewAvailabilityCache,
    getToken,
    historyKey,
    turnActive,
  ]);

  const [appliedHistoryMeta, setAppliedHistoryMeta] = useState<{
    chatId: string | null;
    userMessageOffset: number;
    forkBoundaryMessageCount: number | null;
  }>({
    chatId: null,
    userMessageOffset: 0,
    forkBoundaryMessageCount: null,
  });

  // Fetch completion comes before the history effect updates the displayed messages.
  const historyIsPending = historyKey !== null && (
    loading || !messagesReady || historyVersion !== appliedHistoryVersion
  );
  const showingOpeningPage = historyKey !== null && (
    !chatId || (historyIsPending && messagesReady && displayMessages.length === 0)
  );
  const projectedHistorical = useMemo(
    () => projectWebuiThreadMessages(historical),
    [historical],
  );
  const openingMessages = showingOpeningPage ? projectedHistorical : [];
  const historicalWindowMessages = viewingHistoricalWindow ? projectedHistorical : [];
  const displayWithOlderHistory = useMemo(
    () =>
      !viewingHistoricalWindow && messagesReady
        ? (prependOlderHistoryPrefix(displayMessages, projectedHistorical) ?? displayMessages)
        : displayMessages,
    [displayMessages, messagesReady, projectedHistorical, viewingHistoricalWindow],
  );
  const viewportMessages = historyAccessDenied
    ? []
    : viewingHistoricalWindow
      ? historicalWindowMessages
      : showingOpeningPage || !messagesReady ? openingMessages : displayWithOlderHistory;
  const useDirectHistoryMeta =
    viewingHistoricalWindow
    || showingOpeningPage
    || !messagesReady
    || displayWithOlderHistory !== displayMessages
    || appliedHistoryMeta.chatId !== chatId;
  const viewportUserMessageOffset = useDirectHistoryMeta
    ? userMessageOffset
    : appliedHistoryMeta.userMessageOffset;
  const viewportForkBoundaryMessageCount = useDirectHistoryMeta
    ? forkBoundaryMessageCount
    : appliedHistoryMeta.forkBoundaryMessageCount;
  const conversationLoading = (openingConversation || historyIsPending) && viewportMessages.length === 0;
  const showHeroComposer = viewportMessages.length === 0 && !conversationLoading;
  const composerVariant = showHeroComposer ? emptyComposerVariant : "thread";
  const wasShowingHeroComposerRef = useRef(showHeroComposer);
  const sessionModelPreset = session?.modelPreset?.trim() || null;
  const [localModelPreset, setLocalModelPreset] = useState<string | null>(null);
  const [localReasoning, setLocalReasoning] = useState<{
    chatId: string | null;
    preset: string;
    effort: string | null;
  } | null>(null);
  const [reasoningUpdate, setReasoningUpdate] = useState<{
    chatId: string;
    pending: boolean;
  } | null>(null);
  useEffect(() => {
    setLocalModelPreset(null);
  }, [session?.key, sessionModelPreset]);
  useEffect(() => {
    if (!temporary && localReasoning && localReasoning.chatId === chatId
      && localReasoning.preset === sessionModelPreset
      && localReasoning.effort === (session?.reasoningEffort ?? null)) {
      setLocalReasoning(null);
    }
  }, [chatId, localReasoning, session?.reasoningEffort, sessionModelPreset, temporary]);
  const configuredPresetNames = useMemo(
    () => new Set(settings?.model_presets.map((preset) => preset.name) ?? []),
    [settings],
  );
  const activeModelPreset = (
    (localModelPreset && (!settings || configuredPresetNames.has(localModelPreset))
      ? localModelPreset
      : null)
    || (sessionModelPreset && (!settings || configuredPresetNames.has(sessionModelPreset))
      ? sessionModelPreset
      : null)
    || settings?.agent.model_preset
    || "default"
  );
  const configuredAgentPreset = typeof settings?.agent.model_preset === "string"
    ? settings.agent.model_preset.trim()
    : "";
  const configuredPresetName = localModelPreset?.trim()
    || sessionModelPreset
    || configuredAgentPreset
    || "default";
  const configuredModel = settings?.model_presets.find(
    (candidate) => candidate.name === configuredPresetName,
  )?.model;
  const hasConfiguredModel = typeof configuredModel === "string" && configuredModel.trim().length > 0;
  const handleModelPresetChange = useCallback((name: string) => {
    setLocalModelPreset(name);
    setLocalReasoning((current) => current?.chatId === chatId ? null : current);
    setReasoningUpdate((current) => current?.chatId === chatId ? null : current);
    setNextTurnSettingsError(null);
    if (!chatId) return;
    const ownerChatId = chatId;
    const operation = client.sendSystemCommand(ownerChatId, `/model ${name}`).then(() => undefined);
    const seq = trackNextTurnSetting(ownerChatId, "model", operation);
    void operation.catch((error: unknown) => {
      if (
        nextTurnChatIdRef.current !== ownerChatId
        || !nextTurnSettings.isCurrent(ownerChatId, "model", seq)
      ) {
        return;
      }
      setLocalModelPreset(null);
      setNextTurnSettingsError(
        error instanceof Error ? error.message : "model_preset_rejected",
      );
    });
  }, [chatId, client, nextTurnSettings, trackNextTurnSetting]);
  const reasoningPreset = modelPresetForBadge(settings, activeModelPreset);
  const reasoningOptions = reasoningPreset?.reasoning_effort_values?.filter(Boolean) ?? [];
  const selectedReasoning = localReasoning?.chatId === chatId && localReasoning.preset === activeModelPreset
    ? localReasoning.effort
    : !localModelPreset ? session?.reasoningEffort ?? null : null;
  const reasoningEffort = selectedReasoning && reasoningOptions.includes(selectedReasoning)
    ? selectedReasoning : null;
  const reasoningPending = reasoningUpdate?.chatId === chatId && reasoningUpdate.pending;
  useEffect(() => {
    // A detached next-turn model/effort choice is what the picker shows after a
    // revisit, while the admitted turn keeps its confirmed pair.
    if (liveNextTurnRevision === null) return;
    if (liveNextModelPreset) setLocalModelPreset(liveNextModelPreset);
    if (liveNextReasoningEffort && liveNextModelPreset) {
      setLocalReasoning({
        chatId,
        preset: liveNextModelPreset,
        effort: liveNextReasoningEffort,
      });
    }
  }, [chatId, liveNextModelPreset, liveNextReasoningEffort, liveNextTurnRevision]);
  const handleReasoningChange = (effort: string | null) => {
    if (effort && !reasoningOptions.includes(effort)) return;
    const selection = { chatId, preset: activeModelPreset, effort };
    const previousReasoning = localReasoning;
    setLocalReasoning(selection);
    if (!chatId) {
      setReasoningUpdate(null);
      return;
    }
    const ownerChatId = chatId;
    const update = { chatId: ownerChatId, pending: true };
    setReasoningUpdate(update);
    const rawOperation = client.sendSystemCommand(
      ownerChatId,
      modelReasoningCommand(activeModelPreset, effort),
    );
    const seq = trackNextTurnSetting(ownerChatId, "reasoning", rawOperation);
    void rawOperation.then(() => {
      if (
        nextTurnChatIdRef.current !== ownerChatId
        || !nextTurnSettings.isCurrent(ownerChatId, "reasoning", seq)
      ) {
        return;
      }
      setLocalReasoning(selection);
      setReasoningUpdate((current) => current === update ? null : current);
    }).catch(() => {
      if (
        nextTurnChatIdRef.current !== ownerChatId
        || !nextTurnSettings.isCurrent(ownerChatId, "reasoning", seq)
      ) {
        return;
      }
      setLocalReasoning(previousReasoning);
      setReasoningUpdate((current) => current === update ? { chatId: ownerChatId, pending: false } : current);
    });
  };
  const modelPresetOptions = useMemo(
    () => modelPresetOptionsFromSettings(settings),
    [settings],
  );
  const availableSlashCommands = useMemo(
    () => temporary
      ? slashCommands.filter(({ command }) => command === "/model" || command === "/stop")
      : slashCommands,
    [slashCommands, temporary],
  );
  const modelBadge = useMemo(
    () => toModelBadgeInfo(modelName, settings, activeModelPreset),
    [activeModelPreset, modelName, settings],
  );
  const modelBadgeLabel = modelBadge.needsSetup
    ? t("thread.composer.chooseAI", { defaultValue: "Choose your AI" })
    : modelBadge.label;
  const modelReasoning = !modelBadge.needsSetup && reasoningPreset?.chat_reasoning_effort && reasoningOptions.length
    ? {
      value: reasoningEffort,
      defaultValue: reasoningPreset?.reasoning_effort ?? null,
      options: reasoningOptions,
      onChange: handleReasoningChange,
      pending: reasoningPending,
      failed: reasoningUpdate?.chatId === chatId && !reasoningUpdate.pending,
    }
    : undefined;
  useEffect(() => {
    if (showHeroComposer && !wasShowingHeroComposerRef.current) {
      setHeroGreetingKey(randomHeroGreetingKey());
    }
    wasShowingHeroComposerRef.current = showHeroComposer;
  }, [showHeroComposer]);

  useEffect(() => {
    if (!chatId) {
      setFallbackModelName(null);
      return;
    }
    setFallbackModelName(null);
    return client.onChat(chatId, (event) => {
      if (event.event !== "turn_model_updated" || event.fallback !== true) return;
      setFallbackModelName(event.model_name);
    });
  }, [chatId, client]);

  useEffect(() => {
    if (!chatId || !historyKey) return;
    return client.onChat(chatId, (event) => {
      const turnId = "turn_id" in event && typeof event.turn_id === "string"
        ? event.turn_id
        : null;
      if (!turnId) return;
      const pending = [...pendingChildApprovalTurnsRef.current.entries()]
        .find(([, decision]) => decision.turnId === turnId);
      if (!pending) return;
      const [approvalId, decision] = pending;

      if (event.event === "message_accepted" || event.event === "message_admitted") {
        // An outer starts_turn:false acknowledgement confirms admission only.
        // The exact child row remains the authority for the decision outcome.
        pendingChildApprovalTurnsRef.current.set(approvalId, {
          ...decision,
          admitted: true,
        });
        return;
      }
      if (event.event === "error") {
        clearPendingChildApproval(approvalId);
        rejectChildApprovalPresentation(
          decision.requestKey,
          "The approval decision was refused. Refresh the parent chat before deciding again.",
        );
        void refreshHistory();
        return;
      }
      if (event.event === "turn_end") {
        clearPendingChildApproval(approvalId);
        void refreshHistory();
      }
    });
  }, [
    chatId,
    clearPendingChildApproval,
    client,
    historyKey,
    refreshHistory,
    rejectChildApprovalPresentation,
  ]);

  useEffect(() => {
    if (connectionOpen || pendingChildApprovalTurnsRef.current.size === 0) return;
    for (const [approvalId, decision] of pendingChildApprovalTurnsRef.current) {
      clearPendingChildApproval(approvalId);
      rejectChildApprovalPresentation(
        decision.requestKey,
        "Approval outcome is unknown after disconnect. Refresh status before deciding again.",
      );
    }
  }, [
    clearPendingChildApproval,
    connectionOpen,
    rejectChildApprovalPresentation,
  ]);

  useEffect(() => {
    if (!historyKey || !chatId || loading) return;
    setAppliedHistoryVersion(historyVersion);
    setAppliedHistoryMeta((current) => (
      current.chatId === chatId
      && current.userMessageOffset === userMessageOffset
      && current.forkBoundaryMessageCount === forkBoundaryMessageCount
        ? current
        : { chatId, userMessageOffset, forkBoundaryMessageCount }
    ));
    if (historyAccessDenied) {
      messageCacheRef.current.delete(chatId);
      pendingCanonicalHydrateRef.current.delete(chatId);
      pendingCanonicalCommitRef.current.delete(chatId);
      setMessages([]);
      return;
    }
    // Keep detached historical windows isolated from the live stream owner so
    // an active run at the tail never splices into or overwrites an older page.
    if (viewingHistoricalWindow) return;
    const cached = messageCacheRef.current.get(chatId);
    const pendingCanonicalHydrate = pendingCanonicalHydrateRef.current.get(chatId);
    const hasNewCanonicalHistory = (
      pendingCanonicalHydrate !== undefined
      && historyVersion > pendingCanonicalHydrate.historyVersion
    );
    // When the user switches away and back, keep the local in-memory thread
    // state (including not-yet-persisted messages) instead of replacing it with
    // whatever the history endpoint currently knows about. Once a fresh
    // canonical replay arrives (e.g. after ``session_updated`` refresh), prefer it
    // so rendering converges to the same shape as a manual refresh.
    const historySource = messagesReady ? messages : cached ?? [];
    const normalizedHistory = projectWebuiThreadMessages(historical);
    const keepLiveMessages = (current: UIMessage[]) => projectWebuiThreadMessages(current);
    if (hasNewCanonicalHistory && pendingCanonicalHydrate) {
      // Transcript replay strips streaming metadata and uses persisted ids.
      // Never adopt it while the turn is active: even if no assistant delta
      // arrived locally yet, the next resumed delta must create/continue the
      // live cursor rather than append to an immutable replay row.
      if (
        !historyActivitySnapshotAvailable
        || historyActiveTurnId !== null
        || hasPendingToolCalls
      ) {
        setMessages((current) => keepLiveMessages(current));
        return;
      }
      const authoritativeReset = (
        pendingCanonicalHydrate.uiLineage !== null
        && historyLineage !== pendingCanonicalHydrate.uiLineage
        && (
          historyContinuity === "reset"
          || (
            historyContinuity === "overlap"
            && historyLineage === pendingCanonicalHydrate.historyLineage
          )
        )
      );
      const responseUiRevision = uiRevisionRef.current;
      const resetDropsRenderedTail = (
        authoritativeReset
        && responseUiRevision !== pendingCanonicalHydrate.uiRevision
        && resetDropsPostRequestDurableTail(
          pendingCanonicalHydrate.uiBaseline,
          messages,
          normalizedHistory,
        )
      );
      if (
        authoritativeReset
          ? resetDropsRenderedTail
          : isStaleThreadSnapshot(messages, normalizedHistory, true)
      ) {
        setMessages((current) => keepLiveMessages(current));
        return;
      }
      const canonicalCompletedTurnIds = Array.from(new Set([
        ...completedTurnIds,
        ...completedAssistantTurnIds(normalizedHistory),
      ]));
      const canonicalSnapshot = canonicalRunSnapshot(
        normalizedHistory,
        hasPendingToolCalls,
        historyActiveTurnId,
        historyActivitySnapshotAvailable,
      );
      if (!client.canReconcileCanonicalCompletion(
        chatId,
        pendingCanonicalHydrate.runGeneration,
        canonicalCompletedTurnIds,
        canonicalSnapshot,
      )) {
        setMessages((current) => keepLiveMessages(current));
        return;
      }
      const canonicalMessages = preserveUnknownDeliveryStatus(
        historySource,
        normalizedHistory,
      );
      pendingCanonicalCommitRef.current.set(chatId, {
        canonicalSnapshot,
        completedTurnIds: canonicalCompletedTurnIds,
        expectedUiRevision: responseUiRevision + 1,
        historyLineage,
        historyVersion,
        hydrate: pendingCanonicalHydrate,
        messages: canonicalMessages,
        previousMessages: messages,
      });
      setMessages((current) => {
        if (current !== messages) return current;
        if (
          authoritativeReset
            ? resetDropsRenderedTail
            : isStaleThreadSnapshot(current, normalizedHistory, true)
        ) {
          return keepLiveMessages(current);
        }
        return canonicalMessages;
      });
      return;
    }
    // History no longer includes delegated activity. Compare like with like,
    // and keep the current child rows for the independent hydration to update.
    const transcriptRows = (rows: UIMessage[]) => rows.filter((row) => row.kind !== "delegated_child");
    const cachedTranscriptLength = cached ? transcriptRows(cached).length : 0;
    const isExplicitReveal = pendingReveal?.status === "ready";
    const prependedFromSource = prependOlderHistoryPrefix(historySource, normalizedHistory);
    const canAdoptHistory = (current: UIMessage[]) => (
      isExplicitReveal
      || (
        (cachedTranscriptLength === 0 || normalizedHistory.length > cachedTranscriptLength)
        && !isStaleThreadSnapshot(transcriptRows(current), normalizedHistory)
      )
    );
    const mergeHistory = (current: UIMessage[]) => mergeDelegatedChildRows(
      preserveUnknownDeliveryStatus(current, normalizedHistory),
      delegatedChildReferences(current),
      {
        expectedRootKey: historyKey,
        expectedRoot: historyStopTarget?.source ?? null,
        mode: "hydration",
      },
    ).messages;
    const mergedHistory = prependedFromSource ?? mergeHistory(historySource);
    if (prependedFromSource !== null || canAdoptHistory(messages)) {
      pendingHistoryLineageCommitRef.current.set(chatId, {
        lineage: historyLineage,
        messages: mergedHistory,
      });
    }
    setMessages((current) => {
      const prepended =
        current === historySource
          ? prependedFromSource
          : prependOlderHistoryPrefix(current, normalizedHistory);
      if (prepended !== null) {
        return prepended;
      }
      if (canAdoptHistory(current)) {
        return messagesReady && current !== historySource ? mergeHistory(current) : mergedHistory;
      }
      return cached && !isStaleThreadSnapshot(current, cached) ? cached : keepLiveMessages(current);
    });
  }, [
    loading,
    chatId,
    client,
    completedTurnIds,
    historical,
    historyAccessDenied,
    historyVersion,
    historyContinuity,
    historyLineage,
    messagesReady,
    historyActiveTurnId,
    historyActivitySnapshotAvailable,
    hasPendingToolCalls,
    historyKey,
    viewingHistoricalWindow,
    pendingReveal?.status,
    userMessageOffset,
    forkBoundaryMessageCount,
  ]);

  useEffect(() => {
    if (!chatId || !historyKey || !messagesReady || delegatedChildren.length === 0) return;
    // Late child pages may enrich the current thread, but must never replay its
    // transcript or overwrite a newer live child version.
    setMessages((current) => mergeDelegatedChildRows(current, delegatedChildren, {
      expectedRootKey: historyKey,
      expectedRoot: stopTarget?.source ?? historyStopTarget?.source ?? null,
      mode: "hydration",
      allowLifetimeReplacement: false,
    }).messages);
  }, [chatId, delegatedChildren, historyKey, historyStopTarget?.source, messagesReady, setMessages,
    stopTarget?.source]);

  useLayoutEffect(() => {
    if (!historyKey || !chatId) return;
    const commit = pendingCanonicalCommitRef.current.get(chatId);
    if (!commit) return;
    if (
      commit.historyVersion !== historyVersion
      || commit.historyLineage !== historyLineage
      || commit.messages !== messages
    ) {
      pendingCanonicalCommitRef.current.delete(chatId);
      return;
    }
    if (pendingCanonicalHydrateRef.current.get(chatId) !== commit.hydrate) {
      pendingCanonicalCommitRef.current.delete(chatId);
      return;
    }
    if (uiRevisionRef.current !== commit.expectedUiRevision) {
      pendingCanonicalCommitRef.current.delete(chatId);
      const fallback = messageCacheRef.current.get(chatId) ?? commit.previousMessages;
      messageCacheRef.current.set(chatId, fallback);
      setMessages((current) => current === commit.messages ? fallback : current);
      return;
    }
    if (!client.reconcileCanonicalCompletion(
      chatId,
      commit.hydrate.runGeneration,
      commit.completedTurnIds,
      commit.canonicalSnapshot,
    )) {
      pendingCanonicalCommitRef.current.delete(chatId);
      const fallback = messageCacheRef.current.get(chatId) ?? commit.previousMessages;
      messageCacheRef.current.set(chatId, fallback);
      setMessages((current) => current === commit.messages ? fallback : current);
      return;
    }
    pendingCanonicalHydrateRef.current.delete(chatId);
    pendingCanonicalCommitRef.current.delete(chatId);
    committedHistoryLineageRef.current.set(chatId, historyLineage);
    // Clear the cursor in the same accepted final commit, before effects can
    // start an independent send and make this idle projection stale.
    reconcileTurnComplete();
  }, [chatId, client, historyKey, historyLineage, historyVersion, messages,
    reconcileTurnComplete, setMessages]);

  const refreshCanonicalHistory = useCallback(() => {
    if (!historyKey || !chatId) return;
    pendingCanonicalHydrateRef.current.set(chatId, {
      historyLineage,
      historyVersion,
      runGeneration: client.getRunGeneration(chatId),
      uiBaseline: durableMessageShapes(currentUiMessagesRef.current ?? []),
      uiLineage: committedHistoryLineageRef.current.get(chatId) ?? null,
      uiRevision: uiRevisionRef.current,
    });
    refreshHistory();
  }, [chatId, client, historyKey, historyLineage, historyVersion, refreshHistory]);

  useEffect(() => {
    if (!historyKey || !chatId) return;
    return client.onSessionUpdate((updatedChatId, scope, workspaceScopeUpdate, accessRequestUpdate, permissionsUpdate) => {
      if (updatedChatId !== chatId) return;
      if (permissionsUpdate) updateWorkspacePermissions(permissionsUpdate);
      let pendingResolve = pendingRecoveryResolveRef.current;
      if (pendingResolve?.intent.duration === "once" && permissionsUpdate
        && (permissionsUpdate.runtime_id !== pendingResolve.intent.expectedRuntimeId
          || permissionsUpdate.mode !== "restricted" || permissionsUpdate.state !== "active")) {
        pendingRecoveryResolveRef.current = null;
        pendingResolve = null;
      }
      const eventResult = workspaceScopeUpdate && accessRequestUpdate
        ? {
            workspace_access_request: accessRequestUpdate,
            workspace_scope: workspaceScopeUpdate,
            ...(permissionsUpdate ? { workspace_permissions: permissionsUpdate } : {}),
          }
        : null;
      const incomingRequest = eventResult?.workspace_access_request;
      const focusRecord = acknowledgedWorkspaceFocusRef.current;
      const previousFocus = focusRecord.chatId === chatId ? focusRecord.focus : null;
      const incomingFocus = workspaceFocusIdentity(workspaceScopeUpdate);
      const sameCanonicalFocus = sameWorkspaceFocus(previousFocus, incomingFocus);
      const matchingResolveEvent = !!(
        eventResult
        && incomingRequest
        && pendingResolve
        && pendingResolve.intent.chatId === chatId
        && pendingResolve.baselineFocus
        && pendingRecoveryInvalidationRef.current === null
        && workspaceAccessRequest?.chat_id === chatId
        && workspaceAccessRequest.access_request_id === pendingResolve.intent.accessRequestId
        && workspaceAccessRequest.task.turn_id === pendingResolve.intent.taskTurnId
        && folderResolveProjectionMatches(
          pendingResolve.intent,
          eventResult,
          pendingResolve.baselineFocus.access_mode,
          pendingResolve.intent.duration === "once"
            ? pendingResolve.result?.once_receipt?.canonical_path
              ?? pendingResolve.unconfirmedOnceProjection?.workspace_scope.project_path
              ?? eventResult.workspace_scope.project_path
            : pendingResolve.intent.canonicalPath,
        )
        && sameWorkspaceFocus(
          previousFocus,
          pendingResolve.result
            ? workspaceFocusIdentity(pendingResolve.result.workspace_scope)
            : pendingResolve.baselineFocus,
        )
        && (
          pendingResolve.result
            ? sameFolderResolveAcknowledgement(pendingResolve.result, eventResult)
            : workspaceAccessRequest.revision === pendingResolve.intent.expectedRevision
              && incomingRequest.revision > workspaceAccessRequest.revision
              && (!pendingResolve.unconfirmedOnceProjection
                || sameFolderResolveAcknowledgement(pendingResolve.unconfirmedOnceProjection, eventResult))
        )
      );
      const requestStillMatchesResolve = !!(
        pendingResolve
        && accessRequestUpdate
        && accessRequestUpdate.chat_id === pendingResolve.intent.chatId
        && accessRequestUpdate.access_request_id === pendingResolve.intent.accessRequestId
        && accessRequestUpdate.task.turn_id === pendingResolve.intent.taskTurnId
        && (
          (
            accessRequestUpdate.revision === pendingResolve.intent.expectedRevision
            && (accessRequestUpdate.status === "pending" || accessRequestUpdate.status === "ready")
          )
          || (
            pendingResolve.result
            && requestMatchesResolveAcknowledgement(accessRequestUpdate, pendingResolve.result)
          )
        )
      );
      if (pendingResolve && accessRequestUpdate && !matchingResolveEvent && !requestStillMatchesResolve) {
        // A replaced or advanced request cannot reuse an old resolve receipt.
        pendingRecoveryResolveRef.current = null;
      }
      if (matchingResolveEvent && eventResult) {
        // This precise newer request + selected scope pair is the user's
        // in-flight resolve acknowledgement, not an unrelated focus change.
        // An event has no Once receipt. Keep the validated HTTP acknowledgement
        // intact so an equal-revision projection cannot turn it back into a
        // stale request while the owning reads are still in flight.
        if (pendingResolve && !pendingResolve.result?.once_receipt) {
          if (pendingResolve.intent.duration === "once") {
            // A projection cannot acknowledge the canonical folder on its own.
            pendingResolve.unconfirmedOnceProjection = eventResult;
          } else {
            pendingResolve.result = eventResult;
          }
        }
      } else if (workspaceScopeUpdate && !sameCanonicalFocus) {
        // Compare the canonical folder and access mode before advancing the
        // baseline. Effective policy/backend changes are capability refreshes.
        pendingRecoveryResolveRef.current = null;
        if (workspaceAccessRequest && !pendingRecoveryInvalidationRef.current) {
          pendingRecoveryInvalidationRef.current = {
            kind: "focus",
            chatId,
            requestId: workspaceAccessRequest.access_request_id,
            revision: workspaceAccessRequest.revision,
          };
        }
      }
      if (workspaceScopeUpdate && !(matchingResolveEvent
        && pendingResolve?.intent.duration === "once" && !pendingResolve.result?.once_receipt)) {
        acknowledgedWorkspaceFocusRef.current = { chatId, focus: incomingFocus };
      }
      const replacedRequest = !!(
        accessRequestUpdate
        && workspaceAccessRequest?.chat_id === chatId
        && accessRequestUpdate.chat_id === chatId
        && accessRequestUpdate.access_request_id !== workspaceAccessRequest.access_request_id
      );
      const existingInvalidation = pendingRecoveryInvalidationRef.current;
      if (
        replacedRequest
        && workspaceAccessRequest
        && (!existingInvalidation || existingInvalidation.kind === "focus")
      ) {
        pendingRecoveryInvalidationRef.current = {
          kind: "replacement",
          chatId: workspaceAccessRequest.chat_id,
          requestId: workspaceAccessRequest.access_request_id,
          revision: workspaceAccessRequest.revision,
        };
      }
      if (accessRequestUpdate) {
        if (matchingResolveEvent) {
          // Once receipts bind this projection already; equal revisions are
          // idempotent and must not invalidate the acknowledgement's reads.
          if (
            workspaceAccessRequest?.access_request_id === accessRequestUpdate.access_request_id
            && accessRequestUpdate.revision > workspaceAccessRequest.revision
            && (
              pendingResolve?.intent.duration !== "once"
              || !!pendingResolve.result?.once_receipt
            )
          ) {
            updateWorkspaceAccessRequest(accessRequestUpdate);
          } else if (
            pendingResolve?.intent.duration === "once"
            && !pendingResolve.result?.once_receipt
          ) {
            // The session event carries the exact projection, but only the
            // HTTP acknowledgement carries the Once receipt. Keep the old
            // card stale and locked until both channels agree.
            markWorkspaceAccessRequestStale();
          }
        } else if (pendingRecoveryInvalidationRef.current?.kind === "replacement") {
          // A different request id must be confirmed by the current history read.
          markWorkspaceAccessRequestStale();
          void refreshRecoveryProjection();
        } else if (workspaceScopeUpdate && sameCanonicalFocus) {
          // Refresh capabilities and the request projection, but never infer
          // that a same-folder policy/backend change is a reviewed selection.
          markWorkspaceAccessRequestStale();
          void refreshRecoveryProjection();
        } else if (
          pendingRecoveryInvalidationRef.current
          && workspaceAccessRequest?.access_request_id === accessRequestUpdate.access_request_id
          && accessRequestUpdate.revision <= workspaceAccessRequest.revision
        ) {
          // An equal/older event cannot settle an invalidation created from a
          // later user or focus action, even if its status is terminal.
          markWorkspaceAccessRequestStale();
          void refreshRecoveryProjection();
        } else if (!pendingRecoveryInvalidationSettled(accessRequestUpdate)) {
          markWorkspaceAccessRequestStale();
          void refreshRecoveryProjection();
        } else {
          if (workspaceAccessRequest?.access_request_id === accessRequestUpdate.access_request_id) {
            if (accessRequestUpdate.revision > workspaceAccessRequest.revision) {
              updateWorkspaceAccessRequest(accessRequestUpdate);
            }
          } else {
            markWorkspaceAccessRequestStale();
            void refreshRecoveryProjection();
          }
        }
      } else if (
        workspaceAccessRequest
        && (scope === "metadata" || workspaceScopeUpdate || scope === "thread")
      ) {
        markWorkspaceAccessRequestStale();
        void refreshRecoveryProjection();
      }
      if (scope === "metadata") return;
      // A turn-end thread refresh can arrive while the viewport is easing the
      // final layout change. User-driven scrolling already disables following,
      // so keep an active programmatic follow alive across canonical hydration.
      refreshCanonicalHistory();
    });
  }, [
    chatId,
    client,
    historyKey,
    markWorkspaceAccessRequestStale,
    pendingRecoveryInvalidationRef,
    pendingRecoveryResolveRef,
    pendingRecoveryInvalidationSettled,
    refreshCanonicalHistory,
    refreshRecoveryProjection,
    updateWorkspaceAccessRequest,
    updateWorkspacePermissions,
    workspaceAccessRequest,
  ]);

  useEffect(() => {
    if (!historyKey || !chatId) return;
    return client.onError((error) => {
      if (
        (error.kind === "connection_closed" || error.kind === "message_too_big")
        && error.chatId === chatId
        && error.turnId
      ) {
        const pending = [...pendingChildApprovalTurnsRef.current.entries()]
          .find(([, decision]) => decision.turnId === error.turnId);
        if (pending) {
          const [approvalId, decision] = pending;
          clearPendingChildApproval(approvalId);
          rejectChildApprovalPresentation(
            decision.requestKey,
            error.kind === "connection_closed"
              ? "Approval outcome is unknown after disconnect. Refresh status before deciding again."
              : "The approval decision was not sent. Refresh status before deciding again.",
          );
          void refreshHistory();
        }
      }
      if (error.kind === "workspace_scope_rejected" && error.chatId === chatId) {
        if (
          pendingRecoveryInvalidationRef.current?.kind === "focus"
          && pendingRecoveryInvalidationRef.current.chatId === chatId
        ) {
          pendingRecoveryInvalidationRef.current = null;
        }
        void refreshRecoveryProjection();
      }
      if (
        (error.kind === "turn_rejected"
          || error.kind === "connection_closed"
          || error.kind === "message_too_big")
        && error.chatId === chatId
        && error.turnId
        && pendingRootApprovalRef.current?.turnId === error.turnId
      ) {
        const failedApprovalId = pendingRootApprovalRef.current.approvalId;
        pendingRootApprovalRef.current = null;
        setRootApprovalResetSignal((count) => count + 1);
        if (
          error.kind === "turn_rejected"
          && (error.reason === "approval_stale"
            || error.reason === "invalid_approval_decision"
            || error.reason === "approval_source_unavailable")
        ) {
          setDismissedRootApprovalIds((prev) =>
            prev.includes(failedApprovalId) ? prev : [...prev, failedApprovalId],
          );
        }
      }
      if (
        error.kind === "turn_rejected"
        && error.chatId === chatId
        && pendingRecoveryInvalidationRef.current?.kind === "input"
        && pendingRecoveryInvalidationRef.current.chatId === chatId
        && pendingRecoveryInvalidationRef.current.turnId === error.turnId
      ) {
        pendingRecoveryInvalidationRef.current = null;
        void refreshRecoveryProjection();
      }
    });
  }, [
    chatId,
    clearPendingChildApproval,
    client,
    historyKey,
    pendingRecoveryInvalidationRef,
    refreshHistory,
    refreshRecoveryProjection,
    rejectChildApprovalPresentation,
  ]);

  const wasPageHiddenRef = useRef(document.visibilityState === "hidden");
  useEffect(() => {
    const refreshOnReturn = () => {
      if (document.visibilityState === "hidden") {
        wasPageHiddenRef.current = true;
        return;
      }
      if (!wasPageHiddenRef.current) return;
      wasPageHiddenRef.current = false;
      if (!historyKey || !chatId || client.status !== "open" || loading) return;
      if (
        !isMobileRuntimeActive()
        && !turnActive
        && !hasPendingToolCalls
        && !client.hasUnsettledRun(chatId)
        && !historyError
      ) {
        return;
      }
      refreshCanonicalHistory();
    };
    document.addEventListener("visibilitychange", refreshOnReturn);
    return () => document.removeEventListener("visibilitychange", refreshOnReturn);
  }, [
    chatId,
    client,
    hasPendingToolCalls,
    historyKey,
    historyError,
    loading,
    refreshCanonicalHistory,
    turnActive,
  ]);

  useEffect(() => {
    let refreshOnNextOpen = client.status !== "open";
    return client.onStatus((status) => {
      if (status !== "open") {
        refreshOnNextOpen = true;
        return;
      }
      if (refreshOnNextOpen) refreshCanonicalHistory();
      refreshOnNextOpen = false;
    });
  }, [client, refreshCanonicalHistory]);

  useEffect(() => {
    if (chatId || historyKey) return;
    setMessages(projectWebuiThreadMessages(historical));
  }, [chatId, historical, historyKey, setMessages]);

  useLayoutEffect(() => {
    if (chatId) {
      const prev = prevChatIdForCacheRef.current;
      if (prev && prev !== chatId) {
        messageCacheRef.current.set(prev, displayMessages);
        skipLayoutCacheRef.current = true;
      }
      prevChatIdForCacheRef.current = chatId;
    } else {
      if (prevChatIdForCacheRef.current) {
        messageCacheRef.current.set(
          prevChatIdForCacheRef.current,
          displayMessages,
        );
        skipLayoutCacheRef.current = true;
      }
      prevChatIdForCacheRef.current = null;
    }
  }, [chatId, displayMessages]);

  // Persist thread to in-memory cache after paint so ``useKohakuStream``'s chat switch
  // ``useEffect`` reset has flushed; ``skipLayoutCacheRef`` drops the first run that still
  // sees the *previous* chat's ``messages`` (avoids stale rows leaking across sessions).
  useEffect(() => {
    if (!chatId) {
      return;
    }
    if (skipLayoutCacheRef.current) {
      skipLayoutCacheRef.current = false;
      return;
    }
    if (loading || historyAccessDenied) {
      return;
    }
    messageCacheRef.current.set(chatId, displayMessages);
  }, [chatId, displayMessages, historyAccessDenied, loading]);

  // The landing composer queues the first message while `new_chat` is in flight.
  // Only the chat created for that send may consume it; selecting another chat
  // while creation is pending must not leak the message there.
  useEffect(() => {
    if (!chatId || pendingFirstTargetChatId !== chatId) return;
    const pending = pendingFirstRef.current;
    if (!pending) {
      setPendingFirstTargetChatId(null);
      return;
    }
    pendingFirstRef.current = null;
    setPendingFirstTargetChatId(null);
    const submitted = send(pending.content, pending.images, pending.options);
    if (submitted && !submitted.sideChannel) {
      activeViewportTurnByChatIdRef.current.set(chatId, submitted.turnId);
      setSubmittedViewportTurnId(submitted.turnId);
    }
    setBooting(false);
  }, [chatId, pendingFirstTargetChatId, send]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const commands = await listSlashCommands(getToken());
        if (!cancelled) setSlashCommands(commands);
      } catch {
        if (!cancelled) setSlashCommands([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [getToken]);

  const handleWelcomeSend = useCallback(
    async (content: string, images?: SendAttachment[], options?: SendOptions) => {
      if (booting) return false;
      setBooting(true);
      const snapshot = loadComposerDraft({
        namespace: draftNamespace,
        key: NEW_CHAT_DRAFT_KEY,
        persist: false,
      });
      pendingFirstRef.current = { content, images, options };
      setPendingFirstTargetChatId(null);
      const newId = await (activeHarness === "claude_sdk"
        ? onCreateChat?.(workspaceScope, content, localModelPreset, { harness: activeHarness })
        : onCreateChat?.(workspaceScope, content, localModelPreset));
      if (!newId) {
        pendingFirstRef.current = null;
        setPendingFirstTargetChatId(null);
        setBooting(false);
        return false;
      }
      setDraftHarness("native");
      if (reasoningEffort) {
        try {
          await client.sendSystemCommand(newId, modelReasoningCommand(activeModelPreset, reasoningEffort));
          setLocalReasoning({ chatId: newId, preset: activeModelPreset, effort: reasoningEffort });
        } catch {
          setReasoningUpdate({ chatId: newId, pending: false });
          handoffComposerDraft(draftNamespace, `websocket:${newId}`, snapshot, content);
          setPendingFullDraft({ chatId: newId, text: content });
          pendingFirstRef.current = null;
          setBooting(false);
          return false;
        }
      } else if (localModelPreset) {
        await client.sendSystemCommand(newId, `/model ${localModelPreset}`).catch(() => {});
      }
      setPendingFirstTargetChatId(newId);
      return true;
    },
    [activeHarness, activeModelPreset, booting, client, draftNamespace, localModelPreset, onCreateChat, reasoningEffort, workspaceScope],
  );

  const handleDraftFullAccess = useCallback(async (draft: string) => {
    if (preparingFull || booting) return;
    setPreparingFull(true);
    setFullSetupError(null);
    const snapshot = loadComposerDraft({
      namespace: draftNamespace,
      key: NEW_CHAT_DRAFT_KEY,
      persist: false,
    });
    const draftToTransfer = draft.trim() ? draft : null;
    try {
      const newId = await onCreateChat?.(workspaceScope, draft, localModelPreset, {
        fullAccess: true,
        ...(activeHarness === "claude_sdk" ? { harness: activeHarness } : {}),
      });
      if (!newId) {
        setFullSetupError(t("workspace.permissionV2.actionFailed", {
          defaultValue: "Access could not be updated. Refresh this chat before retrying.",
        }));
      } else {
        setDraftHarness("native");
        if (reasoningEffort) {
          setLocalReasoning({ chatId: newId, preset: activeModelPreset, effort: reasoningEffort });
          void client
            .sendSystemCommand(newId, modelReasoningCommand(activeModelPreset, reasoningEffort))
            .catch(() => {});
        } else if (activeHarness === "claude_sdk" && localModelPreset) {
          await client.sendSystemCommand(newId, `/model ${localModelPreset}`).catch(() => {});
        }
        handoffComposerDraft(draftNamespace, `websocket:${newId}`, snapshot, draftToTransfer);
        setPendingFullDraft(draftToTransfer ? { chatId: newId, text: draftToTransfer } : null);
      }
    } catch {
      setFullSetupError(t("workspace.permissionV2.actionFailed", {
        defaultValue: "Access could not be updated. Refresh this chat before retrying.",
      }));
    } finally {
      setPreparingFull(false);
    }
  }, [activeHarness, activeModelPreset, booting, client, draftNamespace, localModelPreset, onCreateChat, preparingFull, reasoningEffort, t, workspaceScope]);

  const awaitNextTurnSettings = useCallback(async (): Promise<void> => {
    if (!chatId) return;
    try {
      await nextTurnTrackerRef.current.pending(chatId);
    } catch (error) {
      const reason = error instanceof Error ? error.message : "next_turn_setting_failed";
      setNextTurnSettingsError(reason);
      throw error;
    }
  }, [chatId]);

  const submitWithNextTurnSettings = useCallback((
    chat: string,
    submit: () => ReturnType<typeof send>,
  ): ReturnType<typeof send> | Promise<ReturnType<typeof send>> => {
    const tracker = nextTurnTrackerRef.current;
    if (!tracker.hasPending(chat)) return submit();
    // Order admission behind the latest setting acknowledgement. Stop and
    // Steer never reach this path: they address the admitted turn directly.
    return tracker.pending(chat).then(submit).catch((error: unknown) => {
      const reason = error instanceof Error ? error.message : "next_turn_setting_failed";
      setNextTurnSettingsError(reason);
      // Never admit under the previous folder/model: the draft or queued row
      // stays exactly as it was until the user retries. `null` means "not sent"
      // for both the editor and a queued row.
      return null;
    });
  }, []);

  const handleThreadSend = useCallback(
    (content: string, images?: SendAttachment[], options?: SendOptions) => {
      if (options?.continueActiveTurn) {
        // Steer addresses the admitted turn immediately; it never waits for a
        // next-turn setting acknowledgement.
        if (!canonicalRunTurnId) return null;
        options = { ...options, steerTargetTurnId: canonicalRunTurnId };
      }
      setFallbackModelName(null);
      const submit = () => send(content, images, options);
      const settleSubmitted = (submitted: ReturnType<typeof send>) => {
        if (submitted && viewingHistoricalWindow) {
          void goToLatest();
        }
        if (
          chatId
          && submitted
          && !submitted.sideChannel
          && options?.continueActiveTurn !== true
        ) {
          if (workspaceAccessRequest) {
            pendingRecoveryResolveRef.current = null;
            pendingRecoveryInvalidationRef.current = {
              kind: "input",
              chatId,
              turnId: submitted.turnId,
              requestId: workspaceAccessRequest.access_request_id,
              revision: workspaceAccessRequest.revision,
            };
            markWorkspaceAccessRequestStale();
          }
          activeViewportTurnByChatIdRef.current.set(chatId, submitted.turnId);
          setSubmittedViewportTurnId(submitted.turnId);
        }
        return submitted;
      };
      const isStopCommand = content.trim() === "/stop" && !(images && images.length > 0);
      const waitForNextTurnSettings = (
        Boolean(chatId)
        && options?.continueActiveTurn !== true
        && !isStopCommand
      );
      const outcome = waitForNextTurnSettings && chatId
        ? submitWithNextTurnSettings(chatId, submit)
        : submit();
      if (outcome instanceof Promise) {
        return outcome.then((submitted) => {
          const settled = settleSubmitted(submitted);
          return settled !== null;
        });
      }
      return settleSubmitted(outcome);
    },
    [canonicalRunTurnId, chatId, goToLatest, markWorkspaceAccessRequestStale, pendingRecoveryInvalidationRef,
      pendingRecoveryResolveRef, send, submitWithNextTurnSettings, viewingHistoricalWindow,
      workspaceAccessRequest],
  );

  const handleAdmissionRejectedPromptRestored = useCallback((turnId: string) => {
    setMessages((prev) => prev.filter(
      (message) => !(message.role === "user" && message.turnId === turnId),
    ));
    dismissStreamError();
  }, [dismissStreamError, setMessages]);

  const handleWorkspaceScopeChange = useCallback((scope: WorkspaceScopePayload) => {
    if (turnActive && chatId && !temporary) {
      const ownerChatId = chatId;
      // The running turn keeps folder A. Persist B for the next turn through
      // the detached preference owner; do not touch A's recovery card, Stop,
      // Steer target, or the active scope projection.
      setNextTurnScopeOverride(scope);
      setNextTurnSettingsError(null);
      const rawOperation = client.setNextTurnWorkspaceScope(ownerChatId, scope);
      const seq = trackNextTurnSetting(ownerChatId, "workspace", rawOperation);
      void rawOperation
        .then((ack) => {
          if (
            nextTurnChatIdRef.current !== ownerChatId
            || !nextTurnSettings.isCurrent(ownerChatId, "workspace", seq)
          ) {
            return;
          }
          if (ack.next_workspace_scope) {
            setNextTurnScopeOverride(ack.next_workspace_scope);
          }
        })
        .catch((error: unknown) => {
          if (
            nextTurnChatIdRef.current !== ownerChatId
            || !nextTurnSettings.isCurrent(ownerChatId, "workspace", seq)
          ) {
            return;
          }
          const reason = error instanceof Error ? error.message : "workspace_scope_rejected";
          // Restore the confirmed display; the draft and queue are untouched.
          setNextTurnScopeOverride(null);
          setNextTurnSettingsError(reason);
        });
      return;
    }
    if (workspaceAccessRequest && chatId) {
      pendingRecoveryResolveRef.current = null;
      pendingRecoveryInvalidationRef.current = {
        kind: "focus",
        chatId,
        requestId: workspaceAccessRequest.access_request_id,
        revision: workspaceAccessRequest.revision,
      };
      markWorkspaceAccessRequestStale();
    }
    onWorkspaceScopeChange?.(scope);
  }, [chatId, client, markWorkspaceAccessRequestStale, nextTurnSettings, onWorkspaceScopeChange,
    pendingRecoveryInvalidationRef, pendingRecoveryResolveRef, temporary, trackNextTurnSetting,
    turnActive, workspaceAccessRequest]);

  const mutateWorkspaceAccessMode = useCallback(async (mode: WorkspaceAccessMode, retryRetirement = false) => {
    const ownerChatId = chatId;
    if (!ownerChatId || temporary) throw new Error("session_required");
    // Reuse the live projection when it already describes this chat. The
    // pre-read exists only to obtain the runtime id, revision and source
    // token, and the server re-validates all three before any effect.
    const current = workspacePermissions ?? await refreshWorkspacePermissions();
    if (!workspacePermissions) markWorkspacePermissionsStale();
    if (permissionOwnerChatIdRef.current !== ownerChatId || temporary) return;
    if (!current) throw new Error("permission_state_unavailable");
    if (retryRetirement && (current.mode !== "full" || current.state !== "retiring")) {
      updateWorkspacePermissions(current);
      return;
    }
    if (mode === "full" && current.mode !== "full" && (
      current.mode !== "restricted"
      || current.state !== "active"
      || !current.can_enable_full
    )) {
      updateWorkspacePermissions(current);
      throw new Error(current.full_access_unavailable_reason ?? "permission_state_changed");
    }
    try {
      const prepared = mode === "full"
        ? await prepareWorkspaceFullAccess(client, ownerChatId, current.source_token)
        : null;
      if (prepared && permissionOwnerChatIdRef.current !== ownerChatId) return;
      const result = await setWorkspaceAccessMode(client, {
        chat_id: ownerChatId,
        mode,
        expected_runtime_id: current.runtime_id,
        expected_permission_revision: current.revision,
        source_token: current.source_token,
        ...(prepared?.full_confirmation_id
          ? { full_confirmation_id: prepared.full_confirmation_id }
          : {}),
      });
      if (permissionOwnerChatIdRef.current !== ownerChatId || temporary) return;
      updateWorkspacePermissions(result.workspace_permissions);
      onWorkspaceScopeAcknowledged?.(result.workspace_scope);
      void refreshWorkspaceAccessRequest().catch(() => null);
    } catch (reason) {
      if (reason instanceof WorkspaceModeConflictError) {
        if (permissionOwnerChatIdRef.current === ownerChatId && !temporary) {
          updateWorkspacePermissions(reason.conflict.workspace_permissions);
          onWorkspaceScopeAcknowledged?.(reason.conflict.workspace_scope);
          void refreshWorkspaceAccessRequest().catch(() => null);
          return;
        }
      }
      throw reason;
    }
  }, [
    chatId,
    client,
    markWorkspacePermissionsStale,
    onWorkspaceScopeAcknowledged,
    refreshWorkspaceAccessRequest,
    refreshWorkspacePermissions,
    temporary,
    updateWorkspacePermissions,
    workspacePermissions,
  ]);

  const handleWorkspaceAccessModeChange = useCallback((mode: WorkspaceAccessMode) => {
    return mutateWorkspaceAccessMode(mode);
  }, [mutateWorkspaceAccessMode]);

  const retryWorkspaceRestricted = useCallback(() => {
    return mutateWorkspaceAccessMode("restricted", true);
  }, [mutateWorkspaceAccessMode]);

  const handleOpenFilePreview = useCallback((path: string) => {
    if (!historyKey) return;
    setSelectedInspector("file");
    setInspectorCollapsed(false);
    setFilePreview({ parentKey: historyKey, path });
  }, [historyKey]);

  // The whole-session rail loads bounded prompt summaries on demand and
  // reveals unloaded prompts by identity without clobbering the current view.
  const handleNeedSessionTurns = useCallback(() => {
    if (!historyKey) return;
    void loadTurnIndex();
  }, [historyKey, loadTurnIndex]);

  const handleLoadMorePrompts = useCallback(() => {
    if (!historyKey) return;
    void loadMorePromptIndex();
  }, [historyKey, loadMorePromptIndex]);

  const handleRevealPrompt = useCallback((target: RevealPromptTarget) => {
    if (!historyKey) return;
    void revealPrompt(target);
  }, [historyKey, revealPrompt]);

  const handleRetryReveal = useCallback(() => {
    if (!historyKey || !pendingReveal) return;
    void revealPrompt({
      promptId: pendingReveal.promptId,
      locator: pendingReveal.locator ?? null,
      cursor: pendingReveal.cursor ?? null,
    });
  }, [historyKey, pendingReveal, revealPrompt]);

  const handleDismissRevealError = useCallback(() => {
    clearPendingReveal();
  }, [clearPendingReveal]);

  const handleGoToLatest = useCallback(() => {
    if (!historyKey) return Promise.resolve(false);
    return goToLatest();
  }, [goToLatest, historyKey]);

  const handleJumpToTurn = useCallback((cursor: string) => {
    if (!historyKey || !cursor) return;
    void loadTurnAt(cursor);
  }, [historyKey, loadTurnAt]);

  useEffect(() => {
    filePreviewEditKeysRef.current = new Set();
  }, [filePreviewPath]);

  // The assets tab only exists inside a saved session. Drop the open flag when
  // the shell moves to a new or temporary chat so it cannot reappear later.
  useEffect(() => {
    if (!historyKey || temporary) setAssetsInspectorOpen(false);
  }, [historyKey, temporary]);

  useEffect(() => {
    if (!filePreviewPath) return;
    const normalize = (value: string) => value.replace(/\\/g, "/");
    const target = normalize(filePreviewPath);
    let refreshed = false;
    for (const message of messages) {
      for (const edit of message.fileEdits ?? []) {
        if (edit.status !== "done" || edit.pending) continue;
        const candidate = normalize(edit.absolute_path || edit.path);
        const samePath = candidate === target
          || candidate.endsWith(`/${target}`)
          || target.endsWith(`/${candidate}`);
        const editKey = `${edit.call_id}|${candidate}|${edit.status}`;
        if (!samePath || filePreviewEditKeysRef.current.has(editKey)) continue;
        filePreviewEditKeysRef.current.add(editKey);
        refreshed = true;
      }
    }
    if (refreshed) setFilePreviewRefreshKey((value) => value + 1);
  }, [filePreviewPath, messages]);

  const handleCloseFilePreview = useCallback(() => {
    setFilePreview(null);
  }, []);
  const handleCloseAssets = useCallback(() => {
    setAssetsInspectorOpen(false);
  }, []);
  const handleCloseInspectorDialog = useCallback(() => {
    handleCloseChildHistory();
    handleCloseFilePreview();
    handleCloseAssets();
  }, [handleCloseAssets, handleCloseChildHistory, handleCloseFilePreview]);

  const handleToggleAssetsInspector = useCallback(() => {
    if (!historyKey || temporary) return;
    if (assetsInspectorOpen && inspectorTab === "assets" && !inspectorCollapsed) {
      setAssetsInspectorOpen(false);
      return;
    }
    setSelectedInspector("assets");
    setAssetsInspectorOpen(true);
    setInspectorCollapsed(false);
  }, [assetsInspectorOpen, historyKey, inspectorCollapsed, inspectorTab, temporary]);

  const handleToggleInspectorPanel = useCallback(() => {
    if (inspectorOpen) {
      setInspectorCollapsed(true);
      return;
    }
    if (inspectorTab !== null) {
      setInspectorCollapsed(false);
      return;
    }
    if (!historyKey || temporary) return;
    setSelectedInspector("assets");
    setAssetsInspectorOpen(true);
    setInspectorCollapsed(false);
  }, [historyKey, inspectorOpen, inspectorTab, temporary]);

  const handleForkFromMessage = useCallback(
    async (beforeUserIndex: number) => {
      if (!chatId || !onForkChat) return;
      const forkedChatId = await onForkChat(chatId, beforeUserIndex);
      if (!forkedChatId) return;
      messageCacheRef.current.delete(forkedChatId);
      pendingCanonicalHydrateRef.current.delete(forkedChatId);
    },
    [chatId, onForkChat],
  );

  const composer = (
    <>
      {recoveryState ? (
        <RecoveryNotice
          state={recoveryState}
          onContinue={continueRecovery}
          onDismiss={dismissRecovery}
        />
      ) : null}
      {streamError && !hasInlineDeliveryError(messages, streamError) ? (
        <StreamErrorNotice
          error={streamError}
          onDismiss={dismissStreamError}
        />
      ) : null}
      {stopNotice ? (
        <div role="alert" className="mx-4 flex items-center gap-2 rounded-control border border-destructive/25 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          <span className="min-w-0 flex-1">
            {stopNotice === "stale_target"
              ? t("thread.stop.targetChanged", {
                  defaultValue: "The owned-work snapshot changed. Refresh status, then click Stop again.",
                })
              : t("thread.stop.unavailable", {
                  defaultValue: "Stop is unavailable until the current owned-work identity can be verified.",
                })}
          </span>
          <button
            type="button"
            className="shrink-0 rounded px-2 py-1 font-medium underline underline-offset-2"
            onClick={refreshStopState}
          >
            {t("thread.stop.refresh", { defaultValue: "Refresh status" })}
          </button>
          <button
            type="button"
            className="shrink-0 rounded px-2 py-1"
            aria-label={t("thread.stop.dismiss", { defaultValue: "Dismiss Stop status" })}
            onClick={dismissStopNotice}
          >
            ×
          </button>
        </div>
      ) : null}
      {delegatedChildrenError ? (
        <p role="status" className="mx-4 rounded-control border border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
          {t("thread.delegatedChild.refreshRequired", {
            defaultValue: "Delegated activity could not be fully verified. Refresh the conversation to update task status.",
          })}
        </p>
      ) : null}
      {childHistoryError ? (
        <p role="alert" className="mx-4 rounded-control border border-destructive/25 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          {childHistoryError}
        </p>
      ) : null}
      {session || historyKey ? (
        <>
        {Object.entries(externalApprovals).map(([approvalId, approval]) => {
          if (!chatId || (!claudeChat && !approval.childId)
            || dismissedRootApprovalIds.includes(approvalId)) return null;
          const child = currentDelegatedChildren.find((row) => row.child_id === approval.childId);
          const decide = (decision: ApprovalDecisionPayload["decision"]) => {
            const turnId = client.sendApprovalDecision(chatId, approvalId, decision, {
              sessionKey: approval.session_key,
              sessionIncarnation: approval.session_incarnation,
              runId: approval.run_id,
              generation: approval.generation,
              effectId: approval.effect_id,
            });
            pendingRootApprovalRef.current = { turnId, approvalId };
          };
          return (
            <ApprovalRequestCard
              key={approvalId}
              summary={child?.task_label ? `${child.task_label} — ${approval.summary}` : approval.summary}
              command={approval.command}
              approvalId={approvalId}
              busy={!connectionOpen}
              resetSignal={rootApprovalResetSignal}
              saveExclusionCommand={approval.save_exclusion_command ?? undefined}
              onApprove={() => decide("approve")}
              onApproveAndSave={() => decide("approve_and_save")}
              onCancel={() => decide("deny")}
            />
          );
        })}
        {riskyApproval ? (
          <ApprovalRequestCard
            summary={riskyApproval.summary}
            approvalId={riskyApproval.approvalId}
            busy={turnActive || booting || preparingFull}
            resetSignal={rootApprovalResetSignal}
            saveExclusionCommand={riskyApproval.save_exclusion_command}
            onApprove={() => submitRootApprovalDecision("approve")}
            onApproveAndSave={() => submitRootApprovalDecision("approve_and_save")}
            onCancel={() => submitRootApprovalDecision("deny")}
          />
        ) : null}
        {!temporary ? childApprovalPrompts.map((prompt) => {
          const taskLabel = prompt.child.task_label?.trim() || prompt.child.profile || "Delegated task";
          const requestLabel = t("thread.delegatedChild.approvalFor", {
            defaultValue: "Approval for {{task}}",
            task: taskLabel,
          });
          const description = prompt.status === "loading"
            ? t("thread.delegatedChild.verifyingApproval", {
                defaultValue: "Verifying this exact child request…",
              })
            : prompt.status === "unavailable"
              ? t("thread.delegatedChild.approvalUnavailable", {
                  defaultValue: "Approval details could not be verified. Refresh the parent chat before deciding.",
                })
              : undefined;
          return (
            <ApprovalRequestCard
              key={`${prompt.child.child_id}:${prompt.approvalId ?? prompt.requestKey}`}
              summary={prompt.summary || taskLabel}
              approvalId={prompt.status === "ready" ? prompt.approvalId : null}
              saveExclusionCommand={prompt.saveExclusionCommand}
              requestLabel={requestLabel}
              contextLabel={t("thread.delegatedChild.approvalTask", {
                defaultValue: "Delegated task: {{task}}",
                task: taskLabel,
              })}
              description={description}
              busy={
                prompt.status !== "ready"
                || pendingChildApprovalIds.has(prompt.approvalId ?? "")
                || !messagesReady
                || !delegatedStatusAvailable
                || !delegatedChildHasValidAncestry(
                  prompt.child,
                  currentDelegatedChildren,
                  historyKey ?? "",
                  stopTarget?.source ?? historyStopTarget?.source ?? null,
                )
              }
              onApprove={() => submitChildApprovalDecision(prompt, "approve")}
              onApproveAndSave={() => submitChildApprovalDecision(prompt, "approve_and_save")}
              onCancel={() => submitChildApprovalDecision(prompt, "deny")}
            />
          );
        }) : null}
        {!temporary && workspaceAccessRequest ? (
          <FolderAccessRequestCard
            chatId={chatId}
            request={workspaceAccessRequest}
            requestLoading={workspaceAccessRequestLoading}
            requestRefreshFailed={workspaceAccessRequestError}
            busy={turnActive || hasUnknownMessageSend || workspaceScopeDisabled}
            workspaces={workspaces}
            onUpdateRequest={updateWorkspaceAccessRequest}
            workspacePermissions={workspacePermissions}
            workspacePermissionsLoading={workspacePermissionsLoading}
            workspacePermissionsError={workspacePermissionsError}
            onUpdateWorkspacePermissions={updateWorkspacePermissions}
            onRefreshWorkspacePermissions={refreshWorkspacePermissions}
            onRefreshRequest={refreshWorkspaceAccessRequest}
            onRetryRequestRefresh={async () => {
              await Promise.all([
                refreshWorkspaceAccessRequest(),
                onRefreshWorkspaces?.(),
              ]);
            }}
            onApplyWorkspaceScope={(scope) => onWorkspaceScopeAcknowledged?.(scope)}
            onResolveStarted={startRecoveryResolve}
            onResolveResult={settleRecoveryResolve}
            onRefreshWorkspaces={onRefreshWorkspaces ?? (async () => {})}
            onPickFolder={workspaceControls?.can_pick_folder ? pickWorkspaceFolder : undefined}
          />
        ) : null}
        </>
      ) : (
        <>
        {fullSetupError ? (
          <p role="alert" className="px-2 pb-2 text-[11.5px] text-destructive">
            {fullSetupError}
          </p>
        ) : null}
        </>
      )}
      <ThreadComposer
        aboveInput={!session && !historyKey && !openingConversation
          && (!isMobileRuntimeActive() || mobileOwnerCapability()) ? (
          <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 px-2 text-xs">
            <label className="flex min-w-0 max-w-full items-center gap-2 font-medium text-muted-foreground">
              {t("thread.harness.label", { defaultValue: "Harness" })}
              <select
                className="min-w-0 rounded-control border border-border bg-background px-2 py-1 text-foreground"
                value={activeHarness}
                disabled={booting || preparingFull}
                onChange={(event) => setDraftHarness(event.target.value === "claude_sdk" ? "claude_sdk" : "native")}
              >
                <option value="native">Kohaku</option>
                <option value="claude_sdk" disabled={temporaryChatEnabled || temporary || !claudeAvailable}>
                  Claude Code
                </option>
              </select>
            </label>
            {temporaryChatEnabled || temporary ? (
              <span className="text-muted-foreground">
                {t("thread.harness.temporary", { defaultValue: "Claude Code requires a saved chat." })}
              </span>
            ) : !claudeAvailable ? (
              <span className="text-muted-foreground">
                {t("thread.harness.unavailable", { defaultValue: "Claude Code requires the Claude Agent SDK." })}
              </span>
            ) : null}
          </div>
        ) : null}
        onSend={session || historyKey ? handleThreadSend : handleWelcomeSend}
        onJoinQueuedAttempt={session || historyKey ? joinQueuedAttempt : undefined}
        onConsumeQueuedAttempt={session || historyKey ? consumeQueuedAttempt : undefined}
        admissionRejectedTurn={session || historyKey ? admissionRejectedTurn : undefined}
        onAdmissionRejectedPromptRestored={
          session || historyKey ? handleAdmissionRejectedPromptRestored : undefined
        }
        disabled={
          session || historyKey
            ? historyAccessDenied
            : (openingConversation || booting || preparingFull)
        }
        newTurnBlocked={session || historyKey ? (!chatId || reasoningPending) : false}
        steerDisabled={session || historyKey ? (claudeChat || !connectionOpen || !canonicalRunTurnId) : false}
        inputAriaLabel={composerInputAriaLabel}
        isStreaming={turnActive}
        queueCompletionTurnId={session || historyKey ? queueCompletionTurnId : null}
        automaticQueueSendBlocked={hasUnknownMessageSend || !connectionOpen}
        stopAvailable={session || historyKey ? hasOwnedWork : false}
        stopDisabled={
          session || historyKey
            ? (historyAccessDenied || !stopCanBeSent || (delegatedChildrenLoading && !stopTarget))
            : false
        }
        stopPending={session || historyKey ? stopPending : false}
        onDraftFullAccess={
          session || historyKey || temporary || (isMobileRuntimeActive() && !mobileOwnerCapability())
            ? undefined
            : handleDraftFullAccess
        }
        placeholder={
          session || historyKey
            ? (composerVariant === "hero"
              ? t("thread.composer.placeholderHero")
              : t("thread.composer.placeholderThread"))
            : (booting || preparingFull
              ? t("thread.composer.placeholderOpening")
              : t(openingConversation
                ? "thread.composer.placeholderThread"
                : "thread.composer.placeholderHero"))
        }
        modelLabel={modelBadgeLabel}
        modelDetail={modelBadge.model}
        modelPreset={activeModelPreset}
        modelPresets={modelPresetOptions}
        onModelPresetChange={
          session || historyKey
            ? (chatId ? handleModelPresetChange : undefined)
            : handleModelPresetChange
        }
        modelReasoning={
          session || historyKey
            ? (chatId ? modelReasoning : undefined)
            : modelReasoning
        }
        modelProvider={modelBadge.provider}
        modelProviderLabel={modelBadge.providerLabel}
        modelNeedsSetup={modelBadge.needsSetup}
        fallbackModelName={fallbackModelName}
        onModelBadgeClick={modelBadge.needsSetup ? onOpenModelSettings : undefined}
        onManageModels={
          isMobileRuntimeActive() && !mobileOwnerCapability() ? undefined : onOpenModelSettings
        }
        contextUsage={composerContextUsage}
        hasConfiguredModel={hasConfiguredModel}
        recentRoundUsage={composerRoundUsage}
        variant={composerVariant}
        slashCommands={availableSlashCommands}
        cliApps={cliApps}
        mcpPresets={mcpPresets}
        sessions={mentionSessions}
        skills={skills}
        onStop={session || historyKey ? stop : undefined}
        surfaceRef={session || historyKey ? undefined : composerSurfaceRef}
        onTranscribeAudio={transcribeAudio}
        goalState={currentGoalState}
        workspaceScope={workspaceScope}
        workspaceControlsHidden={
          session || historyKey ? (temporary || !chatId) : (temporary || openingConversation)
        }
        workspaceDefaultScope={workspaceDefaultScope}
        workspaceControls={workspaceControls}
        workspaces={workspaces}
        onRefreshWorkspaces={onRefreshWorkspaces}
        workspaceScopeDisabled={workspaceScopeDisabled || hasUnknownMessageSend}
        workspaceLocked={workspaceLocked}
        nextWorkspaceScope={workspaceLocked ? null : nextWorkspaceScope}
        nextTurnSettingsPending={workspaceNextTurnPending}
        nextTurnSettingsError={nextTurnSettingsError}
        onDismissNextTurnSettingsError={() => setNextTurnSettingsError(null)}
        onBeforeNewTurnSend={session || historyKey ? awaitNextTurnSettings : undefined}
        workspaceAccessModeUnknownBlocked={hasUnknownMessageSend}
        workspaceError={workspaceError}
        onDismissWorkspaceError={onDismissWorkspaceError}
        onPickWorkspaceFolder={
          workspaceControls?.can_pick_folder ? pickWorkspaceFolder : undefined
        }
        onWorkspaceScopeChange={handleWorkspaceScopeChange}
        workspacePermissions={session || historyKey ? workspacePermissions : null}
        workspacePermissionsLoading={session || historyKey ? workspacePermissionsLoading : false}
        workspacePermissionsError={session || historyKey ? workspacePermissionsError : false}
        persistentWorkspaceChat={Boolean((session || historyKey) && !temporary && chatId)}
        onWorkspaceAccessModeChange={
          session || historyKey ? handleWorkspaceAccessModeChange : undefined
        }
        onRetryWorkspaceRestricted={
          session || historyKey ? retryWorkspaceRestricted : undefined
        }
        initialDraft={
          (session || historyKey) && pendingFullDraft?.chatId === chatId
            ? pendingFullDraft.text
            : null
        }
        onInitialDraftConsumed={
          session || historyKey
            ? () => {
                clearComposerDraftsForSession(draftNamespace, NEW_CHAT_DRAFT_KEY);
                setPendingFullDraft((current) => (
                  current?.chatId === chatId ? null : current
                ));
              }
            : undefined
        }
        pendingQueueKey={(session || historyKey) && !temporary ? chatId : null}
        draftKey={session || historyKey ? (historyKey ?? chatId) : null}
        draftNamespace={draftNamespace}
        draftRevoked={Boolean((session || historyKey) && historyAccessDenied)}
        draftIncarnation={session || historyKey ? sessionIncarnation : null}
        transcriptionProvider={settingsSnapshot?.transcription?.provider}
        ingressLimits={ingressLimits}
        quotedContext={quotedContext}
        focusRequest={composerFocusSignal}
        onQuotedContextChange={setQuotedContext}
      />
    </>
  );

  const emptyState = historyAccessDenied ? (
    <div role="status" className="text-sm text-muted-foreground">
      {t("thread.conversationUnavailable", { defaultValue: "This conversation is unavailable." })}
    </div>
  ) : conversationLoading ? (
    <div role="status" className="text-sm text-muted-foreground">
      {t("thread.loadingConversation")}
    </div>
  ) : (
    <div className="flex w-full flex-col items-center text-center animate-in fade-in-0 slide-in-from-bottom-2 [animation-duration:220ms] motion-reduce:animate-none">
      <HeroGreeting text={t(heroGreetingKey)} />
    </div>
  );
  const canToggleInspector = Boolean(historyKey && !temporary) || inspectorTab !== null;
  const generatedAssetsAction = canToggleInspector ? (
    <>
      {historyKey && !temporary ? (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label={t("thread.generatedAssets.open")}
          aria-pressed={assetsInspectorAvailable && inspectorOpen && inspectorTab === "assets"}
          onClick={handleToggleAssetsInspector}
          className="host-no-drag h-8 w-8 shrink-0 rounded-full text-muted-foreground/85 hover:bg-accent/40 hover:text-foreground"
        >
          <Images className="h-4 w-4" aria-hidden />
        </Button>
      ) : null}
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-label={t("thread.inspector.toggle", { defaultValue: "Toggle side panel" })}
        aria-pressed={inspectorOpen}
        onClick={handleToggleInspectorPanel}
        className="host-no-drag h-8 w-8 shrink-0 rounded-full text-muted-foreground/85 hover:bg-accent/40 hover:text-foreground"
      >
        <PanelRight className="h-4 w-4" aria-hidden />
      </Button>
    </>
  ) : undefined;
  const effectiveChatPaneWidth =
    chatPaneWidth > 0
      ? chatPaneWidth
      : shellWidth > 0
        ? Math.max(0, shellWidth - (inspectorOpen && inspectorDocked ? 440 : 0))
        : 0;
  const sessionSummaryAvailable = Boolean(historyKey)
    && !isMobileRuntimeActive()
    && effectiveChatPaneWidth >= SESSION_SUMMARY_MIN_PANE_WIDTH
    && !inspectorModal;
  const sessionSummaryInGutter = effectiveChatPaneWidth >= SESSION_SUMMARY_GUTTER_MIN_WIDTH;
  const sessionSummaryVisible = sessionSummaryAvailable && sessionSummaryOpen;

  // Only offer the toggle where the summary can appear beside the transcript.
  const sessionInfoAction = sessionSummaryAvailable ? (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      aria-label={t("thread.header.sessionInfo")}
      aria-pressed={sessionSummaryVisible}
      onClick={() => setSessionSummaryOpen((open) => !open)}
      className={cn(
        "host-no-drag h-8 w-8 shrink-0 rounded-full text-muted-foreground/85",
        "hover:bg-accent/40 hover:text-foreground",
        sessionSummaryVisible && "bg-accent/45 text-foreground",
      )}
    >
      <ListTodo className="h-4 w-4 stroke-[1.75]" aria-hidden />
    </Button>
  ) : undefined;
  // `shellWidth` starts at 0 before the resize observer measures the shell.
  // Both cases keep the navigator, which matches the hidden left rail.
  const promptNavigatorAction = historyKey && shellWidth < PROMPT_RAIL_MIN_WIDTH ? (
    <PromptNavigator
      messages={viewportMessages}
      onJumpToPrompt={(promptId) => viewportRef.current?.jumpToUserPrompt(promptId)}
      sessionPrompts={sessionPrompts}
      sessionTurns={sessionTurns}
      totalPromptCount={effectivePromptTotal}
      hasMoreBefore={hasMoreBefore}
      hasMorePromptsBefore={promptIndexHasMoreBefore}
      loadingMorePrompts={loadingPromptIndex}
      onNeedSessionPrompts={handleNeedSessionTurns}
      onLoadMorePrompts={handleLoadMorePrompts}
      onRevealPrompt={handleRevealPrompt}
    />
  ) : undefined;
  const childHistoryOverlay = activeChildHistoryInspection ? (
    <DelegatedChildHistoryOverlay
      key={`${activeChildHistoryInspection.child.child_id}:${activeChildHistoryInspection.child.child_session_key}:${activeChildHistoryInspection.child.child_generation}`}
      child={activeChildHistoryInspection.child}
      rootKey={activeChildHistoryInspection.parentKey}
      rootRows={currentDelegatedChildren}
      expectedRoot={activeChildHistoryInspection.expectedRoot}
      provenanceComplete={childHistoryProvenanceComplete}
      currentChild={childHistoryProvenanceComplete ? inspectedCurrentChild : null}
      statusAvailable={delegatedStatusAvailable}
      modal={!inspectorDocked}
      visible={inspectorVisible && inspectorTab === "subagents"}
      active={composerActive && inspectorTab === "subagents"}
      token={token}
      onClose={inspectorDocked ? handleCloseChildHistory : handleCloseInspectorDialog}
    />
  ) : null;
  const inspectorTabs: InspectorTab[] = [];
  if (childHistoryOverlay) {
    inspectorTabs.push({
      id: "subagents",
      icon: Bot,
      label: t("thread.inspector.subagents", "Subagents"),
      closeLabel: t("thread.delegatedChild.closeActivity", "Close subagent activity"),
      content: childHistoryOverlay,
    });
  }
  if (filePreviewPath && historyKey) {
    inspectorTabs.push({
      id: "file",
      icon: FileText,
      label: splitFilePath(filePreviewPath).name || filePreviewPath,
      closeLabel: t("filePreview.close", { defaultValue: "Close file preview" }),
      content: (
        <FilePreviewPanel
          key={historyKey}
          sessionKey={historyKey}
          path={filePreviewPath}
          token={token}
          refreshKey={filePreviewRefreshKey}
          modal={!inspectorDocked}
          visible={inspectorVisible && inspectorTab === "file"}
          active={composerActive && inspectorTab === "file"}
          onClose={inspectorDocked ? handleCloseFilePreview : handleCloseInspectorDialog}
        />
      ),
    });
  }
  if (assetsInspectorAvailable && historyKey) {
    inspectorTabs.push({
      id: "assets",
      icon: Images,
      label: t("thread.generatedAssets.title"),
      closeLabel: t("thread.generatedAssets.close", { defaultValue: "Close generated assets" }),
      content: (
        <GeneratedAssetsPanel
          key={historyKey}
          sessionKey={historyKey}
          token={token}
          refreshKey={turnActive ? "running" : "idle"}
          modal={!inspectorDocked}
          visible={inspectorVisible && inspectorTab === "assets"}
          active={composerActive && inspectorTab === "assets"}
          onClose={inspectorDocked ? handleCloseAssets : handleCloseInspectorDialog}
        />
      ),
    });
  }

  const rightPanelActionsInInspector =
    inspectorOpen && inspectorDocked && headerPortalTarget === undefined;

  const threadHeader = !hideHeader ? (
    <ThreadHeader
      title={title}
      claude={claudeChat}
      handle={temporary || (hideHeaderTitle && inlineHandle) ? null : session?.handle}
      onToggleSidebar={onToggleSidebar}
      theme={theme}
      onToggleTheme={onToggleTheme}
      hideSidebarToggleForHostChrome={hideSidebarToggleForHostChrome}
      hideSidebarToggle={hideSidebarToggle}
      hideThemeButton={hideThemeButton}
      hideTitle={hideHeaderTitle}
      actions={headerActions}
      minimal={!session && !conversationLoading}
      promptNavigatorAction={promptNavigatorAction}
      sessionInfoAction={sessionInfoAction}
      generatedAssetsAction={rightPanelActionsInInspector ? undefined : generatedAssetsAction}
      temporaryChatEnabled={temporaryChatEnabled}
      temporaryChatDisabled={booting || turnActive}
      onTemporaryChatEnabledChange={
        showTemporaryChatControl ? onTemporaryChatEnabledChange : undefined
      }
    />
  ) : null;

  return (
    <section ref={shellRef} className="relative flex min-h-0 min-w-0 flex-1 overflow-hidden">
      <div
        ref={chatPaneRef}
        className="relative flex min-w-0 flex-1 flex-col overflow-hidden"
      >
        {hideHeaderTitle && inlineHandle && !temporary && session?.handle ? (
          <div
            aria-label={`Session @${session.handle.name}`}
            className="flex h-8 shrink-0 items-center px-3 text-[12px]"
          >
            <span
              className="shrink-0"
            >
              <SessionHandleLabel id={session.handle.id}>
                @{session.handle.name}
              </SessionHandleLabel>
            </span>
          </div>
        ) : null}
        {headerPortalTarget === undefined ? threadHeader : null}
        <div className="relative flex min-h-0 flex-1 overflow-hidden">
          <FilePreviewAvailabilityProvider
            resolve={historyKey ? resolveFilePreviewAvailability : undefined}
          >
            <ThreadViewport
              ref={viewportRef}
              messages={viewportMessages}
              temporary={temporary}
              obscured={inspectorModal}
              isStreaming={viewingHistoricalWindow ? false : turnActive}
              runStartedAt={viewingHistoricalWindow ? null : currentRunStartedAt}
              retryStatus={viewingHistoricalWindow ? null : retryStatus}
              emptyState={emptyState}
              composer={(
                <div hidden={!composerActive} data-testid={composerActive ? "active-pane-composer" : undefined}>
                  {composer}
                </div>
              )}
              activeTurnId={viewingHistoricalWindow ? null : viewportTurnId}
              activeTurnStartedHere={viewingHistoricalWindow ? false : activeTurnStartedHere}
              conversationKey={historyKey}
              conversationReady={messagesReady}
              conversationLoading={conversationLoading}
              showScrollToBottomButton={!!session}
              cliApps={cliApps}
              mcpPresets={mcpPresets}
              slashCommands={availableSlashCommands}
              forkBoundaryMessageCount={viewportForkBoundaryMessageCount}
              hasMoreBefore={hasMoreBefore}
              loadingOlder={loadingOlder}
              hasMoreAfter={hasMoreAfter}
              loadingNewer={loadingNewer}
              tailActivityPending={tailActivityPending || (viewingHistoricalWindow && turnActive)}
              userMessageOffset={viewportUserMessageOffset}
              onLoadOlder={loadOlder}
              onLoadNewer={loadNewer}
              onGoToLatest={handleGoToLatest}
              sessionPrompts={sessionPrompts}
              sessionPromptTotal={effectivePromptTotal}
              hasMorePromptsBefore={promptIndexHasMoreBefore}
              loadingMorePrompts={loadingPromptIndex}
              onLoadMorePrompts={handleLoadMorePrompts}
              onRevealPrompt={handleRevealPrompt}
              pendingReveal={pendingReveal}
              onPendingRevealHandled={clearPendingReveal}
              onRetryReveal={handleRetryReveal}
              onDismissRevealError={handleDismissRevealError}
              sessionTurns={sessionTurns}
              sessionTurnTotal={sessionTurnTotal}
              onNeedSessionTurns={handleNeedSessionTurns}
              onJumpToTurn={handleJumpToTurn}
              onOpenFilePreview={historyKey ? handleOpenFilePreview : undefined}
              onForkFromMessage={onForkChat && !claudeChat && !showingOpeningPage ? handleForkFromMessage : undefined}
              onOpenChildHistory={handleOpenChildHistory}
              delegatedStatusAvailable={delegatedStatusAvailable}
              onQuoteSelection={session ? handleQuoteSelection : undefined}
            />
          </FilePreviewAvailabilityProvider>
          {sessionSummaryVisible && historyKey ? (
            sessionSummaryInGutter ? (
              <div
                data-testid="session-summary-gutter"
                className="pointer-events-none absolute inset-y-0 right-0 z-20 flex w-[16.5rem] items-start justify-end py-3 pr-4 pl-2"
              >
                <div className="pointer-events-auto max-h-full w-full">
                  <SessionSummaryCard
                    sessionKey={historyKey}
                    token={token}
                    title={title}
                    active={sessionSummaryVisible}
                    onClose={() => setSessionSummaryOpen(false)}
                  />
                </div>
              </div>
            ) : (
              <aside
                data-testid="session-summary-rail"
                className="flex w-[16.5rem] shrink-0 items-start py-3 pr-4 pl-2"
              >
                <SessionSummaryCard
                  sessionKey={historyKey}
                  token={token}
                  title={title}
                  active={sessionSummaryVisible}
                  onClose={() => setSessionSummaryOpen(false)}
                />
              </aside>
            )
          ) : null}
        </div>
      </div>
      {headerPortalTarget && headerActive
        ? createPortal(threadHeader, headerPortalTarget)
        : null}
      <ThreadInspector
        tabs={inspectorTabs}
        selected={inspectorTab}
        containerWidth={shellWidth}
        visible={inspectorVisible}
        headerActions={rightPanelActionsInInspector ? generatedAssetsAction : undefined}
        onSelect={setSelectedInspector}
        onClose={(id) => {
          if (id === "file") handleCloseFilePreview();
          else if (id === "assets") handleCloseAssets();
          else handleCloseChildHistory();
        }}
      />
    </section>
  );
}
