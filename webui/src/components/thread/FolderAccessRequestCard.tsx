import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Folder, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { allowWorkspaceAccessOnce, cancelWorkspaceAccessRequest, continueWorkspaceAccessRequest, resolveWorkspaceAccessRequest, upsertWorkspaceGrant } from "@/lib/api";
import type { FolderAccessActionResult, FolderAccessDuration, FolderAccessRequest, FolderFileAccess, WorkspaceOnceReceipt, WorkspacePermissionProjection, WorkspacesPayload, WorkspaceScopePayload } from "@/lib/types";
import { folderAccessErrorMessage, folderForKnownTarget, folderOperationCanContinue, requiredFolderCapabilities } from "@/lib/folderAccess";
import { isAbsoluteWorkspacePath, sameWorkspacePath } from "@/lib/workspace";
import { getRuntimeHost } from "@/lib/runtime";
import { useClient } from "@/providers/ClientProvider";

type BusyAction = "grant" | "resolve" | "continue" | "cancel" | "refresh" | null;
type ActionIdentity = {
  chatId: string | null;
  requestId: string | null;
  revision: number | null;
  taskTurnId: string | null;
  status: FolderAccessRequest["status"] | null;
};

export interface FolderAccessResolveIntent {
  chatId: string;
  accessRequestId: string;
  expectedRevision: number;
  taskTurnId: string;
  duration: FolderAccessDuration;
  expectedRuntimeId?: string;
  expectedPermissionRevision?: number;
  expectedFileAccess?: FolderFileAccess;
  expectedCommands?: boolean;
  submittedPath: string;
  /** Always uses its grant receipt; Once awaits its acknowledgement receipt. */
  canonicalPath: string | null;
}

export type CorrelatedFolderAccessResult = FolderAccessActionResult & {
  workspace_permissions?: WorkspacePermissionProjection;
  once_receipt?: WorkspaceOnceReceipt;
};

const operationLabels: Record<string, string> = {
  read: "Read",
  write: "Write",
  execute: "Run a command",
  discover: "Find a project",
  select: "Select a project",
  locate: "Choose a folder",
};

const unavailableReasonLabels: Record<string, string> = {
  task_changed: "The task changed. Start a new request to continue.",
  focus_changed: "The working folder changed. Review the current folder before continuing.",
  source_changed: "The task source changed and cannot be continued safely.",
  session_reset: "This conversation was reset. This task cannot be continued.",
  session_deleted: "This conversation is no longer available.",
  access_changed: "Folder access changed. Review the current access settings before continuing.",
  path_missing: "The selected folder is no longer available.",
  path_changed: "The selected folder changed. Choose a folder again.",
  context_unavailable: "The original task context is unavailable.",
  continuation_unknown: "Nanobot could not confirm whether the task continued. Check the conversation before trying again.",
};

export function FolderAccessRequestCard({
  chatId,
  request,
  requestLoading,
  requestRefreshFailed = false,
  busy,
  workspaces,
  onUpdateRequest,
  workspacePermissions = null,
  workspacePermissionsLoading = false,
  workspacePermissionsError = false,
  onUpdateWorkspacePermissions,
  onRefreshWorkspacePermissions,
  onRefreshRequest,
  onRetryRequestRefresh,
  onApplyWorkspaceScope,
  onResolveStarted,
  onResolveResult,
  onRefreshWorkspaces,
  onPickFolder,
}: {
  chatId: string | null;
  request: FolderAccessRequest | null;
  requestLoading: boolean;
  requestRefreshFailed?: boolean;
  busy: boolean;
  workspaces: WorkspacesPayload | null;
  onUpdateRequest: (request: FolderAccessRequest | null) => void;
  workspacePermissions?: WorkspacePermissionProjection | null;
  workspacePermissionsLoading?: boolean;
  workspacePermissionsError?: boolean;
  onUpdateWorkspacePermissions?: (permissions: WorkspacePermissionProjection | null) => void;
  onRefreshWorkspacePermissions?: () => Promise<WorkspacePermissionProjection | null>;
  onRefreshRequest: () => Promise<FolderAccessRequest | null>;
  onRetryRequestRefresh?: () => Promise<void>;
  onApplyWorkspaceScope: (scope: WorkspaceScopePayload) => void;
  onResolveStarted?: (intent: FolderAccessResolveIntent) => void;
  onResolveResult?: (intent: FolderAccessResolveIntent, result: CorrelatedFolderAccessResult | null) => boolean;
  onRefreshWorkspaces: () => Promise<void>;
  onPickFolder?: () => Promise<string | null>;
}) {
  const { t } = useTranslation();
  const { client } = useClient();
  const [reviewIdentity, setReviewIdentity] = useState<ActionIdentity | null>(null);
  const [path, setPath] = useState("");
  const [fileAccess, setFileAccess] = useState<FolderFileAccess>("read_only");
  const [commands, setCommands] = useState(false);
  const [duration, setDuration] = useState<FolderAccessDuration>("once");
  const [settingsTouched, setSettingsTouched] = useState(false);
  const [busyAction, setBusyAction] = useState<BusyAction>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [picking, setPicking] = useState(false);
  const [refreshFailed, setRefreshFailed] = useState(false);
  const submissionLock = useRef(false);
  const pickerRunRef = useRef(0);
  const reviewTriggerRef = useRef<HTMLButtonElement | null>(null);
  const cardHeadingRef = useRef<HTMLHeadingElement | null>(null);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      pickerRunRef.current += 1;
    };
  }, []);
  const management = workspaces?.management ?? null;
  const requestedFolderPath = request ? folderForKnownTarget(request.target) : null;
  const canManage = !!management && workspaces?.controls.can_manage_folders === true;
  const permissionV2Enabled = workspaces?.controls.permission_v2 === true;
  const currentIdentity: ActionIdentity = {
    chatId,
    requestId: request?.access_request_id ?? null,
    revision: request?.revision ?? null,
    taskTurnId: request?.task.turn_id ?? null,
    status: request?.status ?? null,
  };
  const actionIdentityRef = useRef(currentIdentity);
  actionIdentityRef.current = currentIdentity;
  const actionStillCurrent = (captured: ActionIdentity) => {
    if (!mountedRef.current) return false;
    const current = actionIdentityRef.current;
    return current.chatId === captured.chatId
      && current.requestId === captured.requestId
      && current.revision === captured.revision
      && current.taskTurnId === captured.taskTurnId
      && current.status === captured.status;
  };
  const reviewing = !!reviewIdentity && actionStillCurrent(reviewIdentity);
  const openReview = (trigger: HTMLButtonElement, clearPath = false) => {
    reviewTriggerRef.current = trigger;
    if (clearPath) {
      setPath("");
    }
    setError(null);
    setReviewIdentity(currentIdentity);
  };
  const closeReview = () => {
    setReviewIdentity(null);
    setError(null);
  };
  const actionDisabled = busy || requestLoading || requestRefreshFailed || refreshFailed || !canManage
    || busyAction !== null || submissionLock.current;
  const canAllowOnce = permissionV2Enabled
    && !workspacePermissionsLoading
    && !workspacePermissionsError
    && workspacePermissions?.mode === "restricted"
    && workspacePermissions.state === "active";
  const canAct = !!request && request.chat_id === chatId && !!chatId && !actionDisabled;
  const isPending = request?.status === "pending";
  const isReady = request?.status === "ready";
  const isReviewable = isPending || isReady;
  const approvedCapabilitiesSufficient = !!request
    && folderOperationCanContinue(request.operation, request.selected_access);
  const currentTaskAccess = workspacePermissions?.task_access
    && workspacePermissions.task_access.task_turn_id === request?.task.turn_id
    ? workspacePermissions.task_access
    : null;
  const onceTaskSelected = permissionV2Enabled && request?.selected_duration === "once";
  const canContinueTask = !!request?.can_continue && approvedCapabilitiesSufficient && (!onceTaskSelected || (
    !workspacePermissionsLoading
    && !workspacePermissionsError
    && workspacePermissions?.state === "active"
    && currentTaskAccess?.state === "active"
  ));
  const onceEndingMessage = t("workspace.recovery.onceEnding", {
    defaultValue: "One-time access is ending.",
  });
  const retiringProjection = workspacePermissions?.retirement;
  const retiringOnceMessage = retiringProjection?.unknown_processes
    ? `${onceEndingMessage} ${t("workspace.recovery.onceRetiringUnknown", { defaultValue: "Nanobot cannot yet confirm whether earlier commands have finished. Access changes are pending." })}`
    : retiringProjection?.active_processes || retiringProjection?.active_turn
      ? `${onceEndingMessage} ${t("workspace.recovery.onceRetiringActive", { defaultValue: "A command is still running with the previous access. Wait for it to finish or stop it, then retry." })}`
      : retiringProjection
        ? t("workspace.recovery.onceRetiringFinished", { defaultValue: "Earlier work has finished. One-time access is still ending." })
        : `${onceEndingMessage} ${t("workspace.recovery.onceRetiringUnknown", { defaultValue: "Nanobot cannot yet confirm whether earlier commands have finished. Access changes are pending." })}`;

  const allowedFolders = useMemo(() => {
    if (!workspaces) return [];
    const candidates = [
      ...(workspaces.default_scope ? [{
        name: t("workspace.agentHome", { defaultValue: "Agent home" }),
        path: workspaces.default_scope.project_path,
      }] : []),
      ...(management?.grants ?? [])
        .filter((grant) => grant.exists && grant.effective_access.read)
        .map((grant) => ({ name: grant.path.split(/[\\/]/).filter(Boolean).at(-1) ?? grant.path, path: grant.path })),
      ...(management?.repositories ?? [])
        .filter((repository) => repository.exists && repository.effective_access.read)
        .map((repository) => ({ name: repository.name, path: repository.path })),
    ];
    return candidates.filter((entry, index) => (
      candidates.findIndex((candidate) => sameWorkspacePath(candidate.path, entry.path)) === index
    ));
  }, [management?.grants, management?.repositories, t, workspaces]);

  const exactGrant = useMemo(() => management?.grants.find((grant) => sameWorkspacePath(grant.path, path)) ?? null,
    [management?.grants, path]);

  const selectPath = (selectedPath: string) => {
    setPath(selectedPath);
    setSettingsTouched(true);
    if (!request) return;
    const existingGrant = management?.grants.find((grant) => sameWorkspacePath(grant.path, selectedPath));
    const required = requiredFolderCapabilities(request.operation);
    setFileAccess(existingGrant?.file_access === "read_write" || required.fileAccess === "read_write"
      ? "read_write"
      : "read_only");
    setCommands(!!existingGrant?.commands || required.commands);
  };

  const capabilitiesSatisfyOperation = (
    selectedPath: string,
    selectedFileAccess: FolderFileAccess,
    selectedCommands: boolean,
    selectedDuration: FolderAccessDuration,
  ) => {
    if (!request) return false;
    const existingGrant = management?.grants.find((grant) => sameWorkspacePath(grant.path, selectedPath));
    const commandAvailable = selectedCommands && (
      selectedDuration === "once"
      || !existingGrant?.commands
      || existingGrant.effective_access.commands.available
    );
    return folderOperationCanContinue(request.operation, {
      read: true,
      write: selectedFileAccess === "read_write",
      commands: { available: commandAvailable },
    });
  };
  const configuredCapabilitiesSufficient = capabilitiesSatisfyOperation(
    path.trim(), fileAccess, commands, duration,
  );
  const knownFolderCapabilitiesSufficient = !!requestedFolderPath
    && capabilitiesSatisfyOperation(requestedFolderPath, fileAccess, commands, duration);
  const knownFolderCanAllowOnce = !!requestedFolderPath
    && capabilitiesSatisfyOperation(requestedFolderPath, fileAccess, commands, "once");
  const knownFolderCanAlwaysAllow = !!requestedFolderPath
    && capabilitiesSatisfyOperation(requestedFolderPath, fileAccess, commands, "always");

  useEffect(() => {
    pickerRunRef.current += 1;
    setPicking(false);
    setReviewIdentity(null);
    setError(null);
    setNotice(null);
    setPath(requestedFolderPath ?? "");
    const required = request
      ? requiredFolderCapabilities(request.operation)
      : { fileAccess: "read_only" as const, commands: false };
    setFileAccess(required.fileAccess);
    setCommands(required.commands);
    setSettingsTouched(false);
    setDuration(request?.selected_duration === "always" ? "always" : "once");
  }, [chatId, request?.access_request_id, request?.revision, request?.status, request?.operation, requestedFolderPath]);

  useEffect(() => {
    if (!request || settingsTouched || !requestedFolderPath) return;
    const existingGrant = management?.grants.find((grant) => sameWorkspacePath(grant.path, requestedFolderPath));
    const required = requiredFolderCapabilities(request.operation);
    setFileAccess(existingGrant?.file_access === "read_write" || required.fileAccess === "read_write"
      ? "read_write"
      : "read_only");
    setCommands(!!existingGrant?.commands || required.commands);
  }, [chatId, management?.grants, request?.access_request_id, request?.revision, request?.status, request?.operation, requestedFolderPath, settingsTouched]);

  useEffect(() => {
    setRefreshFailed(false);
  }, [request]);

  if (!request || !chatId || request.chat_id !== chatId) return null;

  const pickFolder = getRuntimeHost().pickFolder ?? onPickFolder;
  const hasAcknowledgedSelection = request.status !== "pending"
    && !!request.selected_path;
  const targetDescription = request.target.kind === "unknown"
    ? hasAcknowledgedSelection
      ? null
      : t("workspace.recovery.unknownTarget", { defaultValue: "Nanobot has not identified the exact folder yet. Choose a folder explicitly." })
    : request.target.kind === "file"
      ? requestedFolderPath
        ? t("workspace.recovery.fileTargetWithFolder", {
            path: request.target.path,
            folder: requestedFolderPath,
            defaultValue: `Blocked file: ${request.target.path}. Folder to authorize: ${requestedFolderPath}`,
          })
        : t("workspace.recovery.fileTarget", {
            path: request.target.path,
            defaultValue: `Blocked file: ${request.target.path}. Choose its containing folder explicitly.`,
          })
      : t("workspace.recovery.directoryTarget", { path: request.target.path, defaultValue: `Requested folder: ${request.target.path}` });
  const operation = t(`workspace.recovery.operations.${request.operation}`, {
    defaultValue: operationLabels[request.operation] ?? request.operation,
  });
  const capabilityRequirement = request.operation === "read"
    ? t("workspace.recovery.requirements.read", { defaultValue: "Required: read-only file access; commands off." })
    : request.operation === "write"
      ? t("workspace.recovery.requirements.write", { defaultValue: "Required: read and write file access; commands off." })
      : request.operation === "execute"
        ? t("workspace.recovery.requirements.execute", { defaultValue: "Required: command access; file changes are not required." })
        : t("workspace.recovery.requirements.other", { defaultValue: "Review the file and command access needed for this request." });
  const selectedFileAccessLabel = fileAccess === "read_write"
    ? t("workspace.management.readWrite", { defaultValue: "Read and write" })
    : t("workspace.management.readOnly", { defaultValue: "Read only" });
  const selectedCapabilities = t("workspace.recovery.selectedCapabilities", {
    fileAccess: selectedFileAccessLabel,
    commands: commands
      ? t("workspace.recovery.commandsRequested", { defaultValue: "Commands requested" })
      : t("workspace.management.commandsOff", { defaultValue: "Commands off" }),
    defaultValue: `Selected access: ${selectedFileAccessLabel} · ${commands ? "Commands requested" : "Commands off"}`,
  });

  const refreshAfterFailure = async (identity: ActionIdentity) => {
    try {
      await Promise.all([onRefreshWorkspaces(), onRefreshRequest()]);
      if (actionStillCurrent(identity)) setRefreshFailed(false);
    } catch {
      // Keep the existing card visible and non-actionable until the next refresh.
      if (actionStillCurrent(identity)) setRefreshFailed(true);
    }
  };

  const retryRefresh = async () => {
    if (
      !onRetryRequestRefresh
      || !request
      || busy
      || requestLoading
      || !canManage
      || busyAction !== null
      || submissionLock.current
    ) return;
    const identity = currentIdentity;
    submissionLock.current = true;
    setBusyAction("refresh");
    setError(null);
    try {
      await onRetryRequestRefresh();
      if (actionStillCurrent(identity)) setRefreshFailed(false);
    } catch {
      if (actionStillCurrent(identity)) setRefreshFailed(true);
    } finally {
      submissionLock.current = false;
      if (actionStillCurrent(identity)) setBusyAction(null);
    }
  };

  const resolve = async (
    selectedPath: string,
    grantsRevision: number,
    identity: ActionIdentity,
    submittedPath: string,
    expectedFileAccess?: FolderFileAccess,
    expectedCommands?: boolean,
  ) => {
    const intent: FolderAccessResolveIntent = {
      chatId,
      accessRequestId: request.access_request_id,
      expectedRevision: request.revision,
      taskTurnId: request.task.turn_id,
      duration: "always",
      expectedFileAccess,
      expectedCommands,
      submittedPath,
      canonicalPath: selectedPath,
    };
    onResolveStarted?.(intent);
    let result: CorrelatedFolderAccessResult;
    try {
      result = await resolveWorkspaceAccessRequest(
        client,
        chatId,
        request.access_request_id,
        request.revision,
        selectedPath,
        grantsRevision,
      );
    } catch (reason) {
      onResolveResult?.(intent, null);
      throw reason;
    }
    if (!actionStillCurrent(identity)) return;
    if (onResolveResult && !onResolveResult(intent, result)) return;
    onApplyWorkspaceScope(result.workspace_scope);
    onUpdateRequest(result.workspace_access_request);
    setReviewIdentity(null);
    setNotice(null);
    await onRefreshWorkspaces();
  };

  const resolveAllowedFolder = async (selectedPath: string) => {
    if (!canAct || !management) return;
    const identity = currentIdentity;
    submissionLock.current = true;
    setBusyAction("resolve");
    setError(null);
    setNotice(null);
    try {
      await resolve(selectedPath, management.grants_revision, identity, selectedPath);
    } catch (reason) {
      if (!actionStillCurrent(identity)) return;
      setError(folderAccessErrorMessage(reason, t));
      await refreshAfterFailure(identity);
    } finally {
      submissionLock.current = false;
      setBusyAction(null);
    }
  };

  const applyFolderAccess = async (
    selectedPath: string,
    selectedFileAccess: FolderFileAccess,
    selectedCommands: boolean,
    selectedDuration: FolderAccessDuration,
  ) => {
    if (!canAct || !management) return;
    if (!isAbsoluteWorkspacePath(selectedPath)) {
      setError(t("workspace.recovery.absolutePathRequired", {
        defaultValue: "Enter an absolute folder path on this machine, then try again.",
      }));
      return;
    }
    if (!capabilitiesSatisfyOperation(
      selectedPath, selectedFileAccess, selectedCommands, selectedDuration,
    )) {
      setError(t("workspace.recovery.insufficientCapabilities", {
        required: capabilityRequirement,
        defaultValue: `${capabilityRequirement} Select the required access before allowing this folder.`,
      }));
      return;
    }
    const identity = currentIdentity;
    submissionLock.current = true;
    setBusyAction(permissionV2Enabled && selectedDuration === "once" ? "resolve" : "grant");
    setError(null);
    setNotice(null);
    let grantSaved = false;
    try {
      if (permissionV2Enabled && selectedDuration === "once") {
        if (!canAllowOnce || !workspacePermissions || workspacePermissionsLoading || workspacePermissionsError) {
          setError(t("workspace.recovery.onceUnavailable", {
            defaultValue: "One-time access is unavailable until Nanobot confirms current restricted access.",
          }));
          return;
        }
        const intent: FolderAccessResolveIntent = {
          chatId,
          accessRequestId: request.access_request_id,
          expectedRevision: request.revision,
          taskTurnId: request.task.turn_id,
          duration: "once",
          expectedRuntimeId: workspacePermissions.runtime_id,
          expectedPermissionRevision: workspacePermissions.revision,
          expectedFileAccess: selectedFileAccess,
          expectedCommands: selectedCommands,
          submittedPath: selectedPath,
          canonicalPath: null,
        };
        onResolveStarted?.(intent);
        let result: Awaited<ReturnType<typeof allowWorkspaceAccessOnce>>;
        try {
          result = await allowWorkspaceAccessOnce(client, {
            chat_id: chatId,
            access_request_id: request.access_request_id,
            expected_revision: request.revision,
            path: selectedPath,
            file_access: selectedFileAccess,
            commands: selectedCommands,
            expected_grants_revision: management.grants_revision,
            expected_runtime_id: workspacePermissions.runtime_id,
            expected_permission_revision: workspacePermissions.revision,
          }, request.task.turn_id);
        } catch (reason) {
          onResolveResult?.(intent, null);
          throw reason;
        }
        if (!actionStillCurrent(identity)) return;
        if (onResolveResult && !onResolveResult(intent, result)) return;
        onApplyWorkspaceScope(result.workspace_scope);
        onUpdateRequest(result.workspace_access_request);
        onUpdateWorkspacePermissions?.(result.workspace_permissions);
        setReviewIdentity(null);
        setNotice(null);
        await Promise.all([
          onRefreshRequest(),
          onRefreshWorkspacePermissions?.() ?? Promise.resolve(null),
          onRefreshWorkspaces(),
        ]);
        return;
      }
      let grantsRevision = management.grants_revision;
      const selectedGrant = management.grants.find((grant) => sameWorkspacePath(grant.path, selectedPath)) ?? null;
      let canonicalPath = selectedGrant?.path ?? selectedPath;
      if (
        !selectedGrant
        || selectedGrant.file_access !== selectedFileAccess
        || selectedGrant.commands !== selectedCommands
      ) {
        const saved = await upsertWorkspaceGrant(client, {
          path: selectedPath,
          file_access: selectedFileAccess,
          commands: selectedCommands,
        }, management.grants_revision);
        if (!actionStillCurrent(identity)) return;
        grantSaved = true;
        const receipt = saved.grant_receipt;
        if (!receipt) {
          setNotice(t("workspace.recovery.accessSavedPending", {
            defaultValue: "Folder access was saved, but this task is still waiting for you to use the folder.",
          }));
          setError(t("workspace.recovery.grantPathUnconfirmed", {
            defaultValue: "Nanobot could not confirm which folder this task should use. Refresh the task before trying again.",
          }));
          await refreshAfterFailure(identity);
          return;
        }
        grantsRevision = receipt.grants_revision;
        canonicalPath = receipt.canonical_path;
        setNotice(t("workspace.recovery.accessSaved", {
          defaultValue: "Folder access was saved for this Nanobot instance.",
        }));
        setBusyAction("resolve");
      }
      await resolve(canonicalPath, grantsRevision, identity, selectedPath, selectedFileAccess, selectedCommands);
    } catch (reason) {
      if (!actionStillCurrent(identity)) return;
      if (grantSaved) {
        setNotice(t("workspace.recovery.accessSavedPending", {
          defaultValue: "Folder access was saved, but this task is still waiting for you to use the folder.",
        }));
      }
      setError(folderAccessErrorMessage(reason, t));
      await refreshAfterFailure(identity);
    } finally {
      submissionLock.current = false;
      setBusyAction(null);
    }
  };

  const saveAndUseFolder = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await applyFolderAccess(path.trim(), fileAccess, commands, duration);
  };

  const allowKnownFolder = (selectedDuration: FolderAccessDuration) => {
    if (!requestedFolderPath) return;
    void applyFolderAccess(requestedFolderPath, fileAccess, commands, selectedDuration);
  };

  const cancelTask = async () => {
    if (!canAct || !isReviewable) return;
    const identity = currentIdentity;
    submissionLock.current = true;
    setBusyAction("cancel");
    setError(null);
    try {
      const result = await cancelWorkspaceAccessRequest(
        client,
        chatId,
        request.access_request_id,
        request.revision,
      );
      if (!actionStillCurrent(identity)) return;
      onUpdateRequest(result.workspace_access_request);
      setReviewIdentity(null);
    } catch (reason) {
      if (!actionStillCurrent(identity)) return;
      setError(folderAccessErrorMessage(reason, t));
      await refreshAfterFailure(identity);
    } finally {
      submissionLock.current = false;
      setBusyAction(null);
    }
  };

  const continueTask = async () => {
    if (!canAct || !isReady || !canContinueTask) return;
    const identity = currentIdentity;
    submissionLock.current = true;
    setBusyAction("continue");
    setError(null);
    try {
      if (permissionV2Enabled) {
        const [latestRequest, latestPermissions] = await Promise.all([
          onRefreshRequest(),
          onRefreshWorkspacePermissions?.() ?? Promise.resolve(workspacePermissions),
        ]);
        if (!actionStillCurrent(identity)) return;
        if (
          !latestRequest
          || latestRequest.access_request_id !== identity.requestId
          || latestRequest.revision !== identity.revision
          || latestRequest.task.turn_id !== request.task.turn_id
        ) {
          if (latestRequest) onUpdateRequest(latestRequest);
          if (latestPermissions) onUpdateWorkspacePermissions?.(latestPermissions);
          setError(t("workspace.recovery.changed", { defaultValue: "This task or folder access changed. Review the current task before trying again." }));
          return;
        }
        if (
          request.selected_duration === "once"
          && (!latestPermissions
            || latestPermissions.state !== "active"
            || latestPermissions.task_access?.task_turn_id !== request.task.turn_id
            || latestPermissions.task_access.state !== "active")
        ) {
          if (latestPermissions) onUpdateWorkspacePermissions?.(latestPermissions);
          setError(t("workspace.recovery.onceNotActive", { defaultValue: "One-time access is no longer active. Review the folder before continuing." }));
          return;
        }
      }
      const result = await continueWorkspaceAccessRequest(
        client,
        chatId,
        request.access_request_id,
        request.revision,
      );
      if (!actionStillCurrent(identity)) return;
      onUpdateRequest(result.workspace_access_request);
      onApplyWorkspaceScope(result.workspace_scope);
    } catch (reason) {
      if (!actionStillCurrent(identity)) return;
      setError(folderAccessErrorMessage(reason, t));
      await refreshAfterFailure(identity);
    } finally {
      submissionLock.current = false;
      setBusyAction(null);
    }
  };

  const browse = async () => {
    if (!pickFolder || actionDisabled || picking) return;
    const identity = currentIdentity;
    const pickerRun = ++pickerRunRef.current;
    const pickerStillCurrent = () => pickerRun === pickerRunRef.current && actionStillCurrent(identity);
    setPicking(true);
    setError(null);
    try {
      const selected = await pickFolder();
      if (selected && pickerStillCurrent()) selectPath(selected);
    } catch {
      if (pickerStillCurrent()) {
        setError(t("workspace.recovery.browseFailed", {
          defaultValue: "Folder browsing is unavailable. Enter an absolute path instead.",
        }));
      }
    } finally {
      if (pickerStillCurrent()) setPicking(false);
    }
  };

  const terminalOnceStatus = onceTaskSelected
    && (request.status === "continuing" || request.status === "continued"
      || request.status === "cancelled" || request.status === "invalidated")
    ? currentTaskAccess?.state === "expired" || request.task_access_state === "expired"
      ? t("workspace.recovery.onceExpired", { defaultValue: "One-time access has ended. Review the folder again before continuing." })
      : currentTaskAccess?.state === "retiring" || request.task_access_state === "retiring"
        ? retiringOnceMessage
        : null
    : null;
  const insufficientCapabilitiesMessage = t("workspace.recovery.insufficientCapabilities", {
    required: capabilityRequirement,
    defaultValue: `${capabilityRequirement} Select the required access before allowing this folder.`,
  });
  const status = terminalOnceStatus ?? (isReady && !approvedCapabilitiesSufficient
    ? t("workspace.recovery.readyInsufficient", {
        path: request.selected_path ?? "the selected folder",
        defaultValue: `The selected access for ${request.selected_path ?? "the selected folder"} cannot satisfy this request. Choose another folder or review its access.`,
      })
    : !canManage
    ? t("workspace.recovery.localOnly", { defaultValue: "Open the trusted local Nanobot WebUI to manage folder access for this task." })
    : request.status === "pending"
    ? requestedFolderPath
      ? t("workspace.recovery.pendingKnown", { defaultValue: "Review the identified folder and required access, then choose how to allow it." })
      : t("workspace.recovery.pending", { defaultValue: "Choose a folder to continue this task." })
      : request.status === "ready"
      ? onceTaskSelected && currentTaskAccess?.state === "retiring"
        ? retiringOnceMessage
        : onceTaskSelected && (currentTaskAccess?.state === "expired" || request.task_access_state === "expired")
          ? t("workspace.recovery.onceExpired", { defaultValue: "One-time access has ended. Review the folder again before continuing." })
          : onceTaskSelected && (!currentTaskAccess || workspacePermissions?.state !== "active")
            ? workspacePermissionsLoading
              ? t("workspace.recovery.permissionRefreshing", { defaultValue: "Checking current access…" })
              : t("workspace.recovery.onceUnavailable", { defaultValue: "One-time access is unavailable until Nanobot confirms current restricted access." })
            : canContinueTask
              ? request.selected_duration === "once"
                ? t("workspace.recovery.readyOnce", { path: request.selected_path ?? "", defaultValue: `Access allowed for this task. Ready to continue using ${request.selected_path ?? "the selected folder"}.` })
                : request.selected_duration === "always"
                  ? t("workspace.recovery.readyAlways", { path: request.selected_path ?? "", defaultValue: `Saved access. Ready to continue using ${request.selected_path ?? "the selected folder"}.` })
                  : t("workspace.recovery.ready", { path: request.selected_path ?? "", defaultValue: `Ready to continue using ${request.selected_path ?? "the selected folder"}.` })
        : t(
            `workspace.recovery.unavailable.${request.unavailable_reason ?? "context_unavailable"}`,
            {
              defaultValue: unavailableReasonLabels[request.unavailable_reason ?? "context_unavailable"]
                ?? unavailableReasonLabels.context_unavailable,
            },
          )
      : request.status === "continuing"
        ? t("workspace.recovery.continuing", { defaultValue: "Continuing the task…" })
        : request.status === "continued"
          ? t("workspace.recovery.continued", { defaultValue: "This task has been continued." })
          : request.status === "cancelled"
            ? t("workspace.recovery.cancelled", { defaultValue: "This task was cancelled." })
          : t("workspace.recovery.invalidated", { defaultValue: "This task can no longer continue from this request." }));

  const heading = request.status === "pending"
    ? t("workspace.recovery.headingNeeded", { defaultValue: "Folder access needed" })
    : request.status === "ready"
      ? canContinueTask
        ? t("workspace.recovery.headingReady", { defaultValue: "Ready to continue" })
        : t("workspace.recovery.headingInvalidated", { defaultValue: "Task can't continue" })
      : request.status === "continuing"
        ? t("workspace.recovery.headingContinuing", { defaultValue: "Continuing task" })
        : request.status === "continued"
          ? t("workspace.recovery.headingContinued", { defaultValue: "Task continued" })
          : request.status === "cancelled"
            ? t("workspace.recovery.headingCancelled", { defaultValue: "Task cancelled" })
            : t("workspace.recovery.headingInvalidated", { defaultValue: "Task can't continue" });

  return (
    <section
      aria-label={heading}
      aria-busy={busyAction !== null || requestLoading}
      className="mx-auto mb-2 w-full max-w-3xl rounded-panel border border-border/60 bg-card shadow-sm"
    >
      <div className="space-y-3 p-3 sm:p-4">
        <div className="flex items-start gap-3">
          <Folder className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
          <div className="min-w-0 flex-1 space-y-1">
            <h2 ref={cardHeadingRef} tabIndex={-1} className="text-[13px] font-semibold text-foreground focus:outline-none">
              {heading}
            </h2>
            <p className="text-[12px] leading-relaxed text-muted-foreground">{request.task.excerpt}</p>
            {request.target.kind === "unknown" ? targetDescription ? (
              <p className="break-words text-[11px] text-muted-foreground">{targetDescription} · {operation}</p>
            ) : null : (
              <div className="space-y-2 py-1">
                <p className="text-[11px] text-muted-foreground">
                  {request.target.kind === "file"
                    ? t("workspace.recovery.requestedFileLabel", { defaultValue: "Requested file" })
                    : t("workspace.recovery.requestedFolderLabel", { defaultValue: "Requested folder" })}
                  {" · "}{operation}
                  <code
                    aria-label={request.target.kind === "file"
                      ? t("workspace.recovery.requestedFileLabel", { defaultValue: "Requested file" })
                      : t("workspace.recovery.requestedFolderLabel", { defaultValue: "Requested folder" })}
                    className="mt-1 block break-all font-mono text-[12px] leading-relaxed text-foreground"
                  >{request.target.path}</code>
                </p>
                {request.target.kind === "file" && requestedFolderPath ? (
                  <p className="text-[11px] text-muted-foreground">
                    {t("workspace.recovery.folderToAuthorizeLabel", { defaultValue: "Folder to authorize" })}
                    <code
                      aria-label={t("workspace.recovery.folderToAuthorizeLabel", { defaultValue: "Folder to authorize" })}
                      className="mt-1 block break-all font-mono text-[12px] leading-relaxed text-foreground"
                    >{requestedFolderPath}</code>
                  </p>
                ) : null}
              </div>
            )}
            {request.status !== "pending" && request.selected_path ? (
              <p className="break-all text-[11px] text-muted-foreground">
                {t("workspace.recovery.selectedFolderScope", {
                  path: request.selected_path,
                  defaultValue: `Selected folder scope: ${request.selected_path}`,
                })}
              </p>
            ) : null}
            <p className="break-words text-[11px] text-muted-foreground">
              {t("workspace.recovery.purpose", {
                purpose: request.purpose,
                defaultValue: `Purpose: ${request.purpose}`,
              })}
            </p>
            <p className="break-words text-[11px] text-muted-foreground">{capabilityRequirement}</p>
            {isPending ? <p className="break-words text-[11px] text-muted-foreground">{selectedCapabilities}</p> : null}
          </div>
        </div>

        <p role={!requestLoading && isReady && !approvedCapabilitiesSufficient ? "alert" : "status"}
          className={`text-[12px] ${!requestLoading && isReady && !approvedCapabilitiesSufficient ? "text-destructive" : "text-muted-foreground"}`}>{requestLoading
          ? t("workspace.recovery.refreshing", { defaultValue: "Checking the task status…" })
          : status}</p>

        {requestRefreshFailed || refreshFailed ? (
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-control border border-destructive/30 bg-destructive/5 px-3 py-2">
            <p role="alert" className="text-[11.5px] text-destructive">
              {t("workspace.recovery.refreshFailed", { defaultValue: "Task status could not be refreshed. Actions are disabled until it is checked again." })}
            </p>
            {onRetryRequestRefresh ? (
              <Button type="button" variant="outline" disabled={busy || requestLoading || !canManage || busyAction !== null} onClick={() => void retryRefresh()} className="h-8 rounded-control">
                {busyAction === "refresh" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : null}
                {t("workspace.recovery.retryRefresh", { defaultValue: "Refresh task status" })}
              </Button>
            ) : null}
          </div>
        ) : null}

        {!reviewing && notice ? <p role="status" className="text-[11.5px] text-muted-foreground">{notice}</p> : null}
        {!reviewing && error ? <p role="alert" className="text-[11.5px] text-destructive">{error}</p> : null}

        {isPending ? (
          <div className="space-y-3">
            {request.target.kind === "unknown" && allowedFolders.length > 0 ? (
              <div className="flex flex-wrap items-end gap-2">
                <label className="min-w-[12rem] flex-1 space-y-1 text-[11px] font-medium">
                  <span>{t("workspace.recovery.allowedFolder", { defaultValue: "Use an allowed project" })}</span>
                  <select
                    aria-label={t("workspace.recovery.allowedFolder", { defaultValue: "Use an allowed project" })}
                    defaultValue=""
                    disabled={actionDisabled}
                    onChange={(event) => selectPath(event.target.value)}
                    className="h-9 w-full rounded-control border border-border/60 bg-background px-2 text-[12px]"
                  >
                    <option value="" disabled>{t("workspace.recovery.chooseAllowed", { defaultValue: "Choose an allowed folder" })}</option>
                    {allowedFolders.map((folder) => <option key={folder.path} value={folder.path}>{folder.name} — {folder.path}</option>)}
                  </select>
                </label>
                <Button
                  type="button"
                  variant="outline"
                  disabled={!canAct || !path || !allowedFolders.some((folder) => sameWorkspacePath(folder.path, path))}
                  onClick={() => void resolveAllowedFolder(path)}
                  className="h-9 rounded-control"
                >
                  {t("workspace.recovery.useAllowedFolder", { defaultValue: "Use allowed folder" })}
                </Button>
              </div>
            ) : null}
            <div className="flex flex-wrap gap-2">
              {requestedFolderPath && isAbsoluteWorkspacePath(requestedFolderPath) ? (
                <>
                  {permissionV2Enabled ? (
                    <>
                      <Button
                        type="button"
                        disabled={!canAct || !canAllowOnce || !knownFolderCanAllowOnce}
                        onClick={() => allowKnownFolder("once")}
                        className="h-9 rounded-control"
                      >
                        {busyAction === "resolve" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : null}
                        {t("workspace.recovery.allowForTask", { defaultValue: "Allow for this task" })}
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        disabled={!canAct || !knownFolderCanAlwaysAllow}
                        onClick={() => allowKnownFolder("always")}
                        className="h-9 rounded-control"
                      >
                        {busyAction === "grant" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : null}
                        {t("workspace.recovery.alwaysAllow", { defaultValue: "Always allow" })}
                      </Button>
                    </>
                  ) : (
                    <Button
                      type="button"
                      disabled={!canAct || !knownFolderCanAlwaysAllow}
                      onClick={() => allowKnownFolder("always")}
                      className="h-9 rounded-control"
                    >
                      {busyAction ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : null}
                      {t("workspace.recovery.allowFolder", { defaultValue: "Allow folder" })}
                    </Button>
                  )}
                  <Button type="button" variant="outline" disabled={!canAct} onClick={(event) => openReview(event.currentTarget)} className="h-9 rounded-control">
                    {t("workspace.recovery.reviewAccess", { defaultValue: "Review access" })}
                  </Button>
                  <Button type="button" variant="ghost" disabled={!canAct} onClick={(event) => openReview(event.currentTarget, true)} className="h-9 rounded-control">
                    {t("workspace.recovery.chooseAnotherFolder", { defaultValue: "Choose another folder" })}
                  </Button>
                  {!knownFolderCapabilitiesSufficient ? (
                    <p role="alert" className="basis-full text-[11px] text-destructive">{insufficientCapabilitiesMessage}</p>
                  ) : null}
                </>
              ) : (
                <Button type="button" disabled={!canAct} onClick={(event) => openReview(event.currentTarget, true)} className="h-9 rounded-control">
                  {t("workspace.recovery.chooseFolder", { defaultValue: "Choose folder" })}
                </Button>
              )}
              <Button type="button" variant="ghost" disabled={!canAct} onClick={() => void cancelTask()} className="h-9 rounded-control">
                {busyAction === "cancel" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : null}
                {t("workspace.recovery.cancelTask", { defaultValue: "Cancel task" })}
              </Button>
            </div>
          </div>
        ) : null}

        {isReady ? (
          <div className="flex flex-wrap gap-2">
            {request.can_continue && approvedCapabilitiesSufficient ? (
              <Button type="button" disabled={!canAct || !canContinueTask} onClick={() => void continueTask()} className="h-9 rounded-control">
                {busyAction === "continue" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : null}
                {t("workspace.recovery.continueTask", { defaultValue: "Continue task" })}
              </Button>
            ) : null}
            <Button type="button" variant="outline" disabled={!canAct} onClick={(event) => openReview(event.currentTarget, true)} className="h-9 rounded-control">
              {t("workspace.recovery.chooseAnotherFolder", { defaultValue: "Choose another folder" })}
            </Button>
            <Button type="button" variant="ghost" disabled={!canAct} onClick={() => void cancelTask()} className="h-9 rounded-control">
              {t("workspace.recovery.cancelTask", { defaultValue: "Cancel task" })}
            </Button>
          </div>
        ) : null}

        {request.status === "ready" && request.selected_access ? (
          <p className="text-[11px] text-muted-foreground">
            {!request.selected_access.read
              ? t("workspace.management.noFileAccess", { defaultValue: "No file access" })
              : request.selected_access.write
                ? t("workspace.management.readWrite", { defaultValue: "Read and write" })
                : t("workspace.management.readOnly", { defaultValue: "Read only" })}
            {" · "}
            {request.selected_access.commands.available
              ? t("workspace.management.commandsAvailable", { defaultValue: "Commands available" })
              : !request.selected_access.commands.requested
                ? t("workspace.management.commandsOff", { defaultValue: "Commands off" })
                : t("workspace.management.commandUnavailable", { defaultValue: "Commands unavailable" })}
          </p>
        ) : null}

        {request.status === "ready" && onceTaskSelected && currentTaskAccess?.state === "retiring" ? (
          <p role="status" className="text-[11px] text-muted-foreground">
            {retiringOnceMessage}
          </p>
        ) : null}

        <Dialog
          open={reviewing && isReviewable}
          onOpenChange={(open) => {
            if (open) setReviewIdentity(currentIdentity);
            else closeReview();
          }}
        >
          <DialogContent
            showCloseButton={false}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              const trigger = reviewTriggerRef.current;
              reviewTriggerRef.current = null;
              if (trigger?.isConnected && !trigger.disabled) trigger.focus();
              else cardHeadingRef.current?.focus();
            }}
            className="max-h-[calc(100dvh-32px)] w-[min(calc(100vw-2rem),32rem)] max-w-none overflow-x-hidden overflow-y-auto p-4 sm:p-6"
          >
            <DialogHeader className="pr-2 text-left">
              <DialogTitle>
                {t("workspace.recovery.reviewTitle", { defaultValue: "Review folder access" })}
              </DialogTitle>
              <DialogDescription>
                {t("workspace.recovery.reviewDescription", { defaultValue: "Choose what Nanobot can do in this folder." })}
              </DialogDescription>
            </DialogHeader>
            {reviewing && isReviewable ? (
              <form onSubmit={(event) => void saveAndUseFolder(event)} className="min-w-0 space-y-4">
                <div className="min-w-0 space-y-1.5 rounded-control border border-border/60 bg-muted/25 p-3">
                  <p className="text-[12px] font-medium text-foreground">
                    {t("workspace.recovery.selectedFolderPath", { defaultValue: "Selected folder path" })}
                  </p>
                  <p
                    aria-label={t("workspace.recovery.selectedFolderPath", { defaultValue: "Selected folder path" })}
                    className="break-all text-[12px] leading-relaxed text-muted-foreground"
                  >
                    {path.trim() || t("workspace.recovery.noFolderSelected", { defaultValue: "No folder selected yet." })}
                  </p>
                </div>

                <div className="min-w-0 space-y-1.5">
                  <label htmlFor={`folder-access-path-${request.access_request_id}`} className="text-[12px] font-medium">
                    {t("workspace.management.folderPath", { defaultValue: "Folder path" })}
                  </label>
                  <div className="flex min-w-0 flex-col gap-2 sm:flex-row">
                    <Input
                      id={`folder-access-path-${request.access_request_id}`}
                      aria-label={t("workspace.management.folderPath", { defaultValue: "Folder path" })}
                      value={path}
                      onChange={(event) => selectPath(event.target.value)}
                      placeholder={t("workspace.dialog.manualPlaceholder")}
                      disabled={actionDisabled}
                      className="h-9 min-w-0 flex-1 rounded-control"
                    />
                    <Button type="button" variant="outline" disabled={!pickFolder || actionDisabled || picking} onClick={() => void browse()} className="h-9 shrink-0 rounded-control">
                      {picking ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : null}
                      {t("workspace.dialog.browse", { defaultValue: "Browse…" })}
                    </Button>
                  </div>
                  {!pickFolder ? (
                    <p className="text-[11px] text-muted-foreground">
                      {t("workspace.recovery.browseUnavailable", { defaultValue: "Browsing is unavailable here. Enter an absolute path instead." })}
                    </p>
                  ) : null}
                </div>

                {permissionV2Enabled ? (
                  <fieldset className="min-w-0 space-y-2" disabled={actionDisabled}>
                    <legend className="mb-2 text-[12px] font-medium">
                      {t("workspace.recovery.durationTitle", { defaultValue: "How long should this access last?" })}
                    </legend>
                    <div className="grid min-w-0 gap-2 sm:grid-cols-2">
                      {([
                        {
                          value: "once",
                          title: t("workspace.recovery.allowOnce", { defaultValue: "Allow once" }),
                          description: t("workspace.recovery.allowOnceDescription", { defaultValue: "For this task, including Continue steps. Not saved for future tasks." }),
                          disabled: !canAllowOnce,
                        },
                        {
                          value: "always",
                          title: t("workspace.recovery.alwaysAllow", { defaultValue: "Always allow" }),
                          description: t("workspace.recovery.alwaysAllowDescription", { defaultValue: "Saved for future use. Manage in Folders & projects." }),
                          disabled: false,
                        },
                      ] as const).map((option) => {
                        const optionId = `folder-access-duration-${option.value}-${request.access_request_id}`;
                        return (
                          <label
                            key={option.value}
                            htmlFor={optionId}
                            className={`flex min-w-0 cursor-pointer items-start gap-3 rounded-control border p-3 transition-colors ${duration === option.value ? "border-primary bg-primary/5" : "border-border/60 bg-background"} ${option.disabled ? "cursor-not-allowed opacity-55" : ""}`}
                          >
                            <input
                              id={optionId}
                              type="radio"
                              name={`folder-access-duration-${request.access_request_id}`}
                              value={option.value}
                              checked={duration === option.value}
                              disabled={option.disabled}
                              aria-labelledby={`${optionId}-label`}
                              aria-describedby={`${optionId}-description`}
                              onChange={() => setDuration(option.value)}
                              className="mt-0.5 h-4 w-4 shrink-0 accent-foreground"
                            />
                            <span className="min-w-0 space-y-1">
                              <span id={`${optionId}-label`} className="block text-[12px] font-medium text-foreground">{option.title}</span>
                              <span id={`${optionId}-description`} className="block break-words text-[11px] leading-relaxed text-muted-foreground">{option.description}</span>
                            </span>
                          </label>
                        );
                      })}
                    </div>
                    {!canAllowOnce ? (
                      <p role={workspacePermissionsError ? "alert" : "status"} className="text-[11px] text-muted-foreground">
                        {workspacePermissionsError
                          ? t("workspace.recovery.permissionRefreshFailed", { defaultValue: "Current access could not be checked. Refresh before allowing this task." })
                          : workspacePermissionsLoading
                            ? t("workspace.recovery.permissionRefreshing", { defaultValue: "Checking current access…" })
                            : workspacePermissions?.mode === "full"
                              ? t("workspace.recovery.onceUnavailableFull", { defaultValue: "One-time folder access is unavailable while Full access is active." })
                              : t("workspace.recovery.onceUnavailable", { defaultValue: "One-time access is unavailable until Nanobot confirms current restricted access." })}
                      </p>
                    ) : null}
                  </fieldset>
                ) : null}

                <fieldset className="min-w-0 space-y-2" disabled={actionDisabled}>
                  <legend className="mb-2 text-[12px] font-medium">
                    {t("workspace.management.fileAccess", { defaultValue: "File access" })}
                  </legend>
                  <div className="grid min-w-0 gap-2 sm:grid-cols-2">
                    {([
                      {
                        value: "read_only",
                        title: t("workspace.management.readOnly", { defaultValue: "Read only" }),
                        description: t("workspace.recovery.readOnlyDescription", { defaultValue: "Nanobot can inspect files but cannot change them." }),
                      },
                      {
                        value: "read_write",
                        title: t("workspace.management.readWrite", { defaultValue: "Read and write" }),
                        description: t("workspace.recovery.readWriteDescription", { defaultValue: "Nanobot can read and change files in this folder." }),
                      },
                    ] as const).map((option) => {
                      const optionId = `folder-access-${option.value}-${request.access_request_id}`;
                      return (
                        <label
                          key={option.value}
                          htmlFor={optionId}
                          className={`flex min-w-0 cursor-pointer items-start gap-3 rounded-control border p-3 transition-colors ${fileAccess === option.value ? "border-primary bg-primary/5" : "border-border/60 bg-background"}`}
                        >
                          <input
                            id={optionId}
                            type="radio"
                            name={`folder-file-access-${request.access_request_id}`}
                            value={option.value}
                            checked={fileAccess === option.value}
                            aria-labelledby={`${optionId}-label`}
                            aria-describedby={`${optionId}-description`}
                            onChange={() => {
                              setFileAccess(option.value as FolderFileAccess);
                              setSettingsTouched(true);
                            }}
                            className="mt-0.5 h-4 w-4 shrink-0 accent-foreground"
                          />
                          <span className="min-w-0 space-y-1">
                            <span id={`${optionId}-label`} className="block text-[12px] font-medium text-foreground">{option.title}</span>
                            <span id={`${optionId}-description`} className="block break-words text-[11px] leading-relaxed text-muted-foreground">{option.description}</span>
                          </span>
                        </label>
                      );
                    })}
                  </div>
                </fieldset>

                <div className="space-y-2">
                  <label className="flex min-h-9 items-center gap-2 text-[12px]">
                    <input
                      type="checkbox"
                      checked={commands}
                      disabled={actionDisabled}
                      onChange={(event) => {
                        setCommands(event.target.checked);
                        setSettingsTouched(true);
                      }}
                      className="h-4 w-4 accent-foreground"
                    />
                    {t("workspace.management.allowCommands", { defaultValue: "Allow commands" })}
                  </label>
                  {exactGrant?.effective_access.commands.requested && !exactGrant.effective_access.commands.available ? (
                    <p className="text-[11px] text-muted-foreground">
                      {t("workspace.management.commandUnavailable", { defaultValue: "Command access may be unavailable on this system." })}
                    </p>
                  ) : null}
                </div>

                {!configuredCapabilitiesSufficient ? (
                  <p role="alert" className="text-[11.5px] text-destructive">{insufficientCapabilitiesMessage}</p>
                ) : null}

                <p className="rounded-control bg-muted/35 px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">
                  {permissionV2Enabled
                    ? duration === "once"
                      ? t("workspace.recovery.allowOnceDescription", { defaultValue: "For this task, including Continue steps. Not saved for future tasks." })
                      : t("workspace.recovery.alwaysAllowDescription", { defaultValue: "Saved for future use. Manage in Folders & projects." })
                    : t("workspace.recovery.persistence", { defaultValue: "Saved folder access applies to normal chats in this Nanobot instance. The folder becomes the current chat's working folder only after you save and use it." })}
                </p>
                {error ? <p role="alert" className="text-[11.5px] text-destructive">{error}</p> : null}
                {notice ? <p role="status" className="text-[11.5px] text-muted-foreground">{notice}</p> : null}
                <div className="flex flex-wrap justify-end gap-2">
                  <Button type="button" variant="ghost" disabled={actionDisabled} onClick={closeReview} className="h-9 rounded-control">
                    {t("common.cancel", { defaultValue: "Cancel" })}
                  </Button>
                  <Button type="submit" disabled={!canAct || !path.trim() || !configuredCapabilitiesSufficient || (permissionV2Enabled && duration === "once" && !canAllowOnce)} className="h-9 rounded-control">
                    {busyAction ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : null}
                    {permissionV2Enabled && duration === "once"
                      ? t("workspace.recovery.allowOnceAndUse", { defaultValue: "Allow once and use folder" })
                      : t("workspace.recovery.saveAndUse", { defaultValue: "Save access and use folder" })}
                  </Button>
                </div>
              </form>
            ) : null}
          </DialogContent>
        </Dialog>
      </div>
    </section>
  );
}
