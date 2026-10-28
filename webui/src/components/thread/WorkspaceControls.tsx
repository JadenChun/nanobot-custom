import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, Check, ChevronDown, Folder, Hand, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  floatingItemClassName,
  floatingItemFocusClassName,
} from "@/components/ui/floating-surface";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { WorkspaceManagementDialog } from "@/components/settings/folders/WorkspaceManagementDialog";
import type {
  EffectiveWorkspaceAccess,
  WorkspaceAccessMode,
  WorkspacePermissionProjection,
  WorkspaceScopePayload,
  WorkspacesPayload,
} from "@/lib/types";
import { getRuntimeHost } from "@/lib/runtime";
import { cn } from "@/lib/utils";
import {
  isAbsoluteWorkspacePath,
  projectNameFromPath,
  scopeWithAccessMode,
  sameWorkspacePath,
  shortWorkspacePath,
} from "@/lib/workspace";

function workspacePathPlaceholder(defaultWorkspacePath: string, macPlaceholder: string): string {
  const normalized = defaultWorkspacePath.trim().replace(/\\/g, "/");
  const windowsDrive = normalized.match(/^([A-Za-z]):\//)?.[1];
  if (windowsDrive) return `${windowsDrive.toUpperCase()}:\\path\\to\\project`;
  if (normalized.startsWith("/Users/")) return macPlaceholder;
  return "/home/name/project";
}

function commandAccessLabel(
  access: EffectiveWorkspaceAccess,
  t: (key: string, options?: Record<string, unknown>) => string,
): string {
  if (access.commands.available) {
    return t("workspace.management.commandsAvailable", {
      defaultValue: "Commands available",
    });
  }
  if (!access.commands.requested) {
    return t("workspace.management.commandsOff", { defaultValue: "Commands off" });
  }
  const reason = access.commands.reason;
  return reason
    ? t(`workspace.management.commandReasons.${reason}`, {
        defaultValue: reason.replaceAll("_", " "),
      })
    : t("workspace.management.commandUnavailable", { defaultValue: "Unavailable" });
}

function effectiveAccessLabel(
  access: EffectiveWorkspaceAccess,
  t: (key: string, options?: Record<string, unknown>) => string,
): string {
  const fileAccess = !access.read
    ? t("workspace.management.noFileAccess", { defaultValue: "No file access" })
    : access.write
      ? t("workspace.management.readWrite", { defaultValue: "Read and write" })
      : t("workspace.management.readOnly", { defaultValue: "Read only" });
  return `${fileAccess} · ${commandAccessLabel(access, t)}`;
}

export function WorkspaceProjectPicker({
  compact = false,
  connected = false,
  disabled,
  scope,
  defaultScope,
  controls,
  workspaces = null,
  error,
  onPickFolder,
  onRefreshWorkspaces = async () => {},
  onChange,
}: {
  compact?: boolean;
  connected?: boolean;
  disabled?: boolean;
  scope: WorkspaceScopePayload | null;
  defaultScope: WorkspaceScopePayload | null;
  controls: WorkspacesPayload["controls"] | null;
  workspaces?: WorkspacesPayload | null;
  error?: string | null;
  onPickFolder?: () => Promise<string | null>;
  onRefreshWorkspaces?: () => Promise<void>;
  onChange?: (scope: WorkspaceScopePayload) => void;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [managementOpen, setManagementOpen] = useState(false);
  const [pathDraft, setPathDraft] = useState("");
  const [pathError, setPathError] = useState<string | null>(null);
  const [pickingFolder, setPickingFolder] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const pathInputRef = useRef<HTMLInputElement>(null);
  const pathErrorId = useId();
  const activeScope = scope ?? defaultScope;
  const isAgentHome = !!activeScope && !!defaultScope
    && sameWorkspacePath(activeScope.project_path, defaultScope.project_path);
  const folderName = isAgentHome
    ? t("workspace.agentHome", { defaultValue: "Agent home" })
    : activeScope?.project_name || projectNameFromPath(activeScope?.project_path ?? "");
  const currentPath = activeScope?.project_path ?? defaultScope?.project_path ?? "";
  const visible = !!defaultScope
    && !!onChange;
  const canChangeFocus = controls?.can_change_project !== false;
  const pickFolder = getRuntimeHost().pickFolder ?? onPickFolder;
  const canManage = controls?.can_manage_folders === true || controls?.can_manage_catalog === true;

  useEffect(() => {
    if (!open) return;
    setPathDraft(isAgentHome ? "" : currentPath);
    setPathError(null);
  }, [currentPath, isAgentHome, open]);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (!error || !visible || disabled) return;
    const frame = window.requestAnimationFrame(() => triggerRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [disabled, error, visible]);

  useEffect(() => {
    if (!open || !(pathError || error)) return;
    const frame = window.requestAnimationFrame(() => pathInputRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [error, open, pathError]);

  const applyProjectPath = useCallback(
    (projectPath: string, projectName?: string) => {
      const base = scope ?? defaultScope;
      const trimmed = projectPath.trim();
      if (!base || !onChange) return;
      if (!trimmed || !isAbsoluteWorkspacePath(trimmed)) {
        setPathError(t("workspace.dialog.absolutePathRequired"));
        return;
      }
      const managed = workspaces?.managed_access === true
        || base.effective_access?.policy === "managed";
      const acknowledged = { ...base };
      delete acknowledged.effective_access;
      delete acknowledged.sandbox_status;
      onChange({
        ...acknowledged,
        project_path: trimmed,
        project_name: projectName || projectNameFromPath(trimmed),
        access_mode: base.access_mode === "full" ? "full" : managed ? "restricted" : base.access_mode,
        restrict_to_workspace: base.access_mode === "full"
          ? acknowledged.restrict_to_workspace
          : managed || base.access_mode === "restricted",
      });
      setPathError(null);
      setOpen(false);
    },
    [defaultScope, onChange, scope, t, workspaces?.managed_access],
  );

  const browseForFolder = useCallback(async () => {
    if (!pickFolder || disabled) return;
    setPickingFolder(true);
    setPathError(null);
    try {
      const picked = await pickFolder();
      if (picked) {
        setPathDraft(picked);
        applyProjectPath(picked);
      }
    } catch (reason) {
      setPathError(reason instanceof Error
        ? reason.message
        : t("workspace.dialog.browseFailed", { defaultValue: "Folder picker unavailable. Paste a path instead." }));
    } finally {
      setPickingFolder(false);
    }
  }, [applyProjectPath, disabled, pickFolder, t]);

  if (!visible || !defaultScope || !onChange) return null;

  const pathAccessibleLabel = t("workspace.picker.workingIn", {
    folder: folderName,
    defaultValue: `Working in: ${folderName}`,
  });
  const effectiveAccess = activeScope?.effective_access;
  const currentAccess = effectiveAccess
    ? effectiveAccessLabel(effectiveAccess, t)
    : workspaces?.managed_access === true
      ? t("workspace.picker.managedAccessUnavailable", { defaultValue: "Folder access is managed by the server." })
    : activeScope?.sandbox_status?.summary
      ?? (activeScope?.access_mode === "full"
        ? t("workspace.permissionV2.unavailable", { defaultValue: "Access unavailable" })
        : t("thread.composer.workspace.default", { defaultValue: "Restricted" }));
  const readableRepositories = (workspaces?.management?.repositories ?? [])
    .filter((repository) => repository.exists && repository.effective_access.read);

  return (
    <>
      <div className={cn(
        compact ? "inline-flex" : "flex min-w-0 items-center gap-2 rounded-b-[28px] bg-muted/45 px-3 py-1.5 dark:bg-white/[0.045] sm:px-4",
      )}>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              ref={triggerRef}
              type="button"
              disabled={disabled || pickingFolder || !canChangeFocus}
              aria-label={pathAccessibleLabel}
              title={currentPath}
              className={cn(
                compact
                  ? "thread-composer-action touch-target inline-flex h-8 items-center justify-center rounded-full border border-transparent px-2.5"
                  : "inline-flex h-8 min-w-0 max-w-full items-center gap-2 rounded-full px-2.5 sm:max-w-[28rem]",
                "text-[12px] font-medium text-muted-foreground/90 transition-colors hover:text-foreground disabled:pointer-events-none disabled:opacity-55",
                compact ? "hover:bg-muted/65" : "hover:bg-background/70",
                (connected || !isAgentHome) && "text-primary",
              )}
            >
              {pickingFolder
                ? <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden />
                : <Folder className={cn("shrink-0", compact ? "h-4 w-4" : "h-3.5 w-3.5")} aria-hidden />}
              <span className="min-w-0 truncate">
                {pathAccessibleLabel}
              </span>
              {!compact ? <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden /> : null}
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            side="bottom"
            sideOffset={8}
            className="w-[min(31rem,calc(100vw-1rem))] space-y-3 p-3 sm:p-4"
          >
            <div className="space-y-1 rounded-control bg-muted/35 px-3 py-2.5">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                {t("workspace.picker.currentFolder", { defaultValue: "Current folder" })}
              </p>
              <p className="break-all text-[12px] font-medium" title={currentPath}>{currentPath}</p>
              <p className="text-[11px] text-muted-foreground">{currentAccess}</p>
            </div>

            <button
              type="button"
              disabled={disabled || !canChangeFocus}
              onClick={() => applyProjectPath(defaultScope.project_path, defaultScope.project_name)}
              className={cn(
                floatingItemClassName,
                floatingItemFocusClassName,
                "flex min-h-[48px] w-full cursor-default gap-3 px-3 py-2.5 focus:bg-muted/55",
              )}
            >
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-control bg-muted text-foreground/80">
                <Folder className="h-4 w-4" aria-hidden />
              </span>
              <span className="min-w-0 flex-1 text-left">
                <span className="block truncate text-[13px] font-semibold text-foreground">
                  {t("workspace.dialog.defaultProject", { defaultValue: "Use agent home" })}
                </span>
                <span className="block break-all text-[11.5px] text-muted-foreground">
                  {defaultScope.project_path}
                </span>
              </span>
              {isAgentHome ? <Check className="h-4 w-4 text-foreground/80" aria-hidden /> : null}
            </button>

            {readableRepositories.length > 0 ? (
              <div className="space-y-1">
                <p className="px-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {t("workspace.picker.readableProjects", { defaultValue: "Readable known projects" })}
                </p>
                <div className="max-h-40 space-y-1 overflow-y-auto overscroll-contain pr-1">
                  {readableRepositories.map((repository) => (
                    <button
                      key={repository.name}
                      type="button"
                      disabled={disabled || !canChangeFocus}
                      onClick={() => applyProjectPath(repository.path, repository.name)}
                      title={repository.path}
                      className={cn(
                        floatingItemClassName,
                        floatingItemFocusClassName,
                        "flex min-h-[44px] w-full cursor-default gap-3 px-3 py-2 text-left focus:bg-muted/55",
                      )}
                    >
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[12px] font-medium">{repository.name}</span>
                        <span className="block break-all text-[10.5px] text-muted-foreground">{shortWorkspacePath(repository.path)}</span>
                      </span>
                      <span className="max-w-[45%] text-right text-[10px] text-muted-foreground">
                        {effectiveAccessLabel(repository.effective_access, t)}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="space-y-2 border-t border-border/50 pt-3">
              <div className="space-y-1.5">
                <label htmlFor="workspace-manual-path" className="text-[11.5px] font-medium">
                  {t("workspace.dialog.manual", { defaultValue: "Paste path" })}
                </label>
                <div className="flex items-center gap-2">
                  <Input
                    id="workspace-manual-path"
                    ref={pathInputRef}
                    value={pathDraft}
                    disabled={disabled}
                    onChange={(event) => {
                      setPathDraft(event.target.value);
                      setPathError(null);
                    }}
                    placeholder={workspacePathPlaceholder(
                      defaultScope.project_path,
                      t("workspace.dialog.manualPlaceholder"),
                    )}
                    aria-invalid={pathError || error ? true : undefined}
                    aria-describedby={pathError || error ? pathErrorId : undefined}
                    className="h-9 min-w-0 rounded-full border-border/55 bg-background/80 px-3 text-[12.5px]"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    disabled={!pickFolder || disabled || pickingFolder || !canChangeFocus}
                    title={!pickFolder
                      ? t("workspace.dialog.browseUnavailable", { defaultValue: "Browsing is unavailable here. Paste a path instead." })
                      : undefined}
                    onClick={() => void browseForFolder()}
                    className="h-9 shrink-0 rounded-full px-3 text-[12px]"
                  >
                    {t("workspace.dialog.browse", { defaultValue: "Browse…" })}
                  </Button>
                  <Button
                    type="button"
                    disabled={disabled || !canChangeFocus || !pathDraft.trim()}
                    onClick={() => applyProjectPath(pathDraft)}
                    className="h-9 shrink-0 rounded-full px-3 text-[12px]"
                  >
                    {t("workspace.dialog.usePath", { defaultValue: "Use folder" })}
                  </Button>
                </div>
              </div>
              {pathError || error ? (
                <p id={pathErrorId} role="alert" className="text-[11.5px] font-medium text-destructive">
                  {pathError ?? error}
                </p>
              ) : null}
              {disabled ? (
                <p className="text-[11px] text-muted-foreground">
                  {t("workspace.picker.busy", { defaultValue: "A response is running. Folder changes are available when it finishes." })}
                </p>
              ) : null}
              {!canChangeFocus && !disabled ? (
                <p className="text-[11px] text-muted-foreground">
                  {t("workspace.picker.focusUnavailable", { defaultValue: "Folder focus cannot be changed from this connection." })}
                </p>
              ) : null}
            </div>
            {canManage && workspaces?.management ? (
              <button
                type="button"
                onClick={() => {
                  setOpen(false);
                  setManagementOpen(true);
                }}
                className="flex min-h-9 w-full items-center justify-center rounded-control border border-border/55 px-3 text-[12px] font-medium text-foreground transition-colors hover:bg-muted/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {t("workspace.management.open", { defaultValue: "Manage folders and repositories" })}
              </button>
            ) : null}
          </PopoverContent>
        </Popover>
        <span className="min-w-0 truncate text-[10.5px] text-muted-foreground" title={currentAccess}>
          {currentAccess}
        </span>
        {disabled ? (
          <span className={cn("text-[10.5px] text-muted-foreground", compact && "sr-only")}>
            {t("workspace.picker.busy", { defaultValue: "A response is running. Folder changes are available when it finishes." })}
          </span>
        ) : null}
        {!canChangeFocus && !disabled ? (
          <span className={cn("text-[10.5px] text-muted-foreground", compact && "sr-only")}>
            {t("workspace.picker.focusUnavailable", { defaultValue: "Folder focus cannot be changed from this connection." })}
          </span>
        ) : null}
        {!open && error ? (
          <span role="alert" className="min-w-0 truncate text-[11.5px] font-medium text-destructive">
            {error}
          </span>
        ) : null}
      </div>
      <WorkspaceManagementDialog
        open={managementOpen}
        onOpenChange={setManagementOpen}
        workspaces={workspaces}
        onRefreshWorkspaces={onRefreshWorkspaces}
        onPickFolder={pickFolder}
      />
    </>
  );
}

export function WorkspaceAccessMenu({
  scope,
  disabled,
  canUseFullAccess,
  managedAccessEnabled = false,
  isHero,
  onChange,
}: {
  scope: WorkspaceScopePayload;
  disabled?: boolean;
  canUseFullAccess: boolean;
  managedAccessEnabled?: boolean;
  isHero: boolean;
  onChange?: (scope: WorkspaceScopePayload) => void;
}) {
  const { t } = useTranslation();
  const managedAccess = managedAccessEnabled || scope.effective_access?.policy === "managed";
  const mode = scope.access_mode;
  const isFull = mode === "full";
  const accessLabel = managedAccess
    ? scope.effective_access
      ? effectiveAccessLabel(scope.effective_access, t)
      : t("workspace.picker.managedAccessUnavailable", { defaultValue: "Folder access is managed by the server." })
    : t(isFull ? "thread.composer.workspace.full" : "thread.composer.workspace.default");
  const shortAccessLabel = managedAccess
    ? t("thread.composer.workspace.managedShort", { defaultValue: "Folder access" })
    : t(isFull ? "thread.composer.workspace.fullShort" : "thread.composer.workspace.defaultShort");
  const accessAriaLabel = managedAccess
    ? `${t("thread.composer.workspace.accessAria")}: ${accessLabel}`
    : `${t("thread.composer.workspace.accessAria")}: ${accessLabel}`;

  const setMode = (value: WorkspaceAccessMode) => {
    if (value === "full" && !canUseFullAccess) return;
    if (value === mode) return;
    onChange?.(scopeWithAccessMode(scope, value));
  };

  if (managedAccess) {
    return (
      <Button
        type="button"
        variant="ghost"
        aria-label={accessAriaLabel}
        title={accessLabel}
        disabled={disabled}
        className={cn(
          "thread-composer-access touch-target min-w-0 max-w-[min(15rem,45vw)] whitespace-nowrap rounded-control border border-transparent font-semibold shadow-none",
          isHero ? "h-8 px-2.5 text-[12px]" : "h-9 px-3 text-[12.5px]",
          "bg-transparent text-muted-foreground hover:bg-foreground/[0.045] hover:text-foreground dark:hover:bg-white/[0.06]",
        )}
      >
        <Hand className="thread-composer-access-icon mr-1.5 h-3.5 w-3.5 shrink-0" aria-hidden />
        <span className="thread-composer-access-label-full min-w-0 truncate">{accessLabel}</span>
        <span className="thread-composer-access-label-short hidden min-w-0 truncate">{shortAccessLabel}</span>
      </Button>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild disabled={disabled || !onChange}>
        <Button
          type="button"
          variant="ghost"
          aria-label={accessAriaLabel}
          title={accessLabel}
          className={cn(
            "thread-composer-access touch-target min-w-0 max-w-[min(12.5rem,42vw)] whitespace-nowrap rounded-control border border-transparent font-semibold shadow-none",
            isHero ? "h-8 px-2.5 text-[12px]" : "h-9 px-3 text-[12.5px]",
            isFull
              ? "bg-transparent text-orange-600 hover:bg-orange-500/8 dark:text-orange-300 dark:hover:bg-orange-400/10"
              : "bg-transparent text-muted-foreground hover:bg-foreground/[0.045] hover:text-foreground dark:hover:bg-white/[0.06]",
          )}
        >
          {isFull ? (
            <AlertTriangle className="thread-composer-access-icon mr-1.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          ) : (
            <Hand className="thread-composer-access-icon mr-1.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          )}
          <span aria-hidden className="thread-composer-access-label-full min-w-0 truncate">
            {accessLabel}
          </span>
          <span aria-hidden className="thread-composer-access-label-short hidden min-w-0 truncate">
            {shortAccessLabel}
          </span>
          <ChevronDown className="thread-composer-access-chevron ml-1.5 h-3 w-3 shrink-0" aria-hidden />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-56">
        <AccessMenuItem
          icon={<Hand className="h-4 w-4" />}
          label={t("thread.composer.workspace.default", { defaultValue: "Restricted" })}
          selected={mode === "restricted"}
          onSelect={() => setMode("restricted")}
        />
        <AccessMenuItem
          icon={<AlertTriangle className="h-4 w-4" />}
          label={t("thread.composer.workspace.full", { defaultValue: "Full access" })}
          selected={mode === "full"}
          disabled={!canUseFullAccess}
          warning
          onSelect={() => setMode("full")}
        />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function AccessMenuItem({
  icon,
  label,
  selected,
  disabled,
  warning,
  onSelect,
}: {
  icon: ReactNode;
  label: string;
  selected: boolean;
  disabled?: boolean;
  warning?: boolean;
  onSelect: () => void;
}) {
  return (
    <DropdownMenuItem
      disabled={disabled}
      onSelect={onSelect}
      className={cn(
        "flex h-10 items-center gap-3 px-3 text-[13.5px] font-semibold",
        warning && "text-orange-600 focus:text-orange-600 dark:text-orange-300 dark:focus:text-orange-300",
      )}
    >
      <span className="grid h-5 w-5 shrink-0 place-items-center text-current" aria-hidden>
        {icon}
      </span>
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {selected ? <Check className="h-4 w-4 shrink-0" aria-hidden /> : null}
    </DropdownMenuItem>
  );
}

/** Permission-v2 access control. It never changes access through a raw scope update. */
export function WorkspacePermissionV2Control({
  enabled = true,
  permissions,
  loading = false,
  error = false,
  disabled = false,
  persistentChat,
  isHero,
  draftFullAccess = false,
  onModeChange,
  onRetryRestricted,
  onRequestFullAccess,
}: {
  enabled?: boolean;
  permissions: WorkspacePermissionProjection | null;
  loading?: boolean;
  error?: boolean;
  disabled?: boolean;
  persistentChat: boolean;
  isHero: boolean;
  draftFullAccess?: boolean;
  onModeChange?: (mode: WorkspaceAccessMode) => void | Promise<void>;
  onRetryRestricted?: () => void | Promise<void>;
  onRequestFullAccess?: () => void | Promise<void>;
}) {
  const { t } = useTranslation();
  const [confirmFull, setConfirmFull] = useState(false);
  const [pending, setPending] = useState(false);
  const [actionFailed, setActionFailed] = useState(false);
  const unavailableReason = permissions?.full_access_unavailable_reason;
  const canEnableFull = enabled && !!permissions
    && permissions.mode === "restricted"
    && permissions.state === "active"
    && permissions.can_enable_full
    && persistentChat;
  const canRequestDraftFull = enabled
    && draftFullAccess
    && !permissions
    && !loading
    && !error
    && !disabled;
  const fullAvailable = persistentChat ? canEnableFull : canRequestDraftFull;
  const isFull = permissions?.mode === "full";
  const isRetiring = isFull && permissions?.state === "retiring";
  const isRestrictedRetiring = enabled
    && permissions?.mode === "restricted"
    && permissions.state === "retiring";
  const draftRestricted = enabled && !permissions && !persistentChat && !loading && !error;
  const unavailable = !enabled || (!permissions && !draftRestricted) || error;
  const busy = loading || pending || disabled;
  const label = unavailable
    ? t("workspace.permissionV2.unavailable", { defaultValue: "Access unavailable" })
    : draftRestricted
      ? t("thread.composer.workspace.default", { defaultValue: "Restricted" })
    : isRetiring
      ? t("workspace.permissionV2.fullEnding", { defaultValue: "Full access is ending" })
      : isRestrictedRetiring
        ? t("workspace.permissionV2.restrictedEnding", { defaultValue: "Restricted · previous access ending" })
      : isFull
        ? t("thread.composer.workspace.full", { defaultValue: "Full access" })
        : t("thread.composer.workspace.default", { defaultValue: "Restricted" });
  const compactLabel = isRetiring
    ? t("workspace.permissionV2.fullEndingShort", { defaultValue: "Ending" })
    : isRestrictedRetiring
      ? t("workspace.permissionV2.fullEndingShort", { defaultValue: "Ending" })
    : isFull
      ? t("thread.composer.workspace.fullShort", { defaultValue: "Full" })
      : t("thread.composer.workspace.defaultShort", { defaultValue: "Default" });
  const reasonCopy = isRestrictedRetiring
    ? null
    : unavailableReason
    ? t(`workspace.permissionV2.unavailableReasons.${unavailableReason}`, {
        defaultValue: unavailableReason === "configured_sandbox"
          ? "Full access is unavailable while command confinement is configured."
          : unavailableReason === "supervisor_unavailable"
            ? "Full access is unavailable because Nanobot cannot supervise commands safely."
            : unavailableReason === "session_required"
              ? "Send a message to create this chat before enabling Full access."
              : "Full access is unavailable with the current local controls.",
      })
    : !persistentChat && !draftFullAccess
      ? t("workspace.permissionV2.sessionRequired", { defaultValue: "Send a message to create this chat before enabling Full access." })
      : null;
  const restrictedRetirementMessage = permissions?.retirement?.unknown_processes
    ? t("workspace.permissionV2.restrictedEndingUnknown", {
        defaultValue: "Nanobot cannot confirm whether earlier commands have finished. New work remains Restricted; Full access stays unavailable.",
      })
    : permissions?.retirement?.active_processes || permissions?.retirement?.active_turn
      ? t("workspace.permissionV2.restrictedEndingActive", {
          defaultValue: "Earlier work is still active with the previous access. New work remains Restricted; Full access stays unavailable.",
        })
      : permissions?.retirement
        ? t("workspace.permissionV2.restrictedEndingUncounted", {
            defaultValue: "No earlier commands are reported active or unknown, but the previous access is still ending. Restricted remains the current mode.",
          })
        : t("workspace.permissionV2.restrictedEndingPending", {
            defaultValue: "Previous access is still ending. New work remains Restricted; Full access stays unavailable.",
          });
  const retirementMessage = permissions?.retirement?.unknown_processes
    ? t("workspace.permissionV2.retiringUnknown", { defaultValue: "Nanobot cannot yet confirm whether earlier commands have finished. Access changes are pending." })
    : permissions?.retirement?.active_processes || permissions?.retirement?.active_turn
      ? t("workspace.permissionV2.retiringActive", { defaultValue: "A command is still running with the previous access. Wait for it to finish or stop it, then retry." })
      : t("workspace.permissionV2.retiringFinished", { defaultValue: "Earlier work has finished. Retry to confirm Restricted access." });

  const changeMode = async (mode: WorkspaceAccessMode) => {
    if (!permissions || !onModeChange || busy) return;
    setPending(true);
    setActionFailed(false);
    try {
      await onModeChange(mode);
    } catch {
      setActionFailed(true);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex min-w-0 flex-col items-start">
      <DropdownMenu>
        <DropdownMenuTrigger asChild disabled={busy || unavailable}>
          <Button
            type="button"
            variant="ghost"
            aria-label={`${t("thread.composer.workspace.accessAria", { defaultValue: "Access" })}: ${label}`}
            title={label}
            disabled={busy || unavailable}
            className={cn(
              "thread-composer-access touch-target min-w-0 max-w-[min(15rem,45vw)] whitespace-nowrap rounded-control border border-transparent font-semibold shadow-none",
              isHero ? "h-8 px-2.5 text-[12px]" : "h-9 px-3 text-[12.5px]",
              isFull || isRestrictedRetiring
                ? "bg-transparent text-orange-600 hover:bg-orange-500/8 dark:text-orange-300 dark:hover:bg-orange-400/10"
                : "bg-transparent text-muted-foreground hover:bg-foreground/[0.045] hover:text-foreground dark:hover:bg-white/[0.06]",
            )}
          >
            {isFull || isRestrictedRetiring
              ? <AlertTriangle className="mr-1.5 h-3.5 w-3.5 shrink-0" aria-hidden />
              : <Hand className="mr-1.5 h-3.5 w-3.5 shrink-0" aria-hidden />}
            <span aria-hidden className="thread-composer-access-label-full min-w-0 truncate">{label}</span>
            <span aria-hidden className="thread-composer-access-label-short hidden min-w-0 truncate">{compactLabel}</span>
            {loading ? <Loader2 className="ml-1 h-3.5 w-3.5 animate-spin" aria-hidden /> : <ChevronDown className="ml-1.5 h-3 w-3 shrink-0" aria-hidden />}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-[min(22rem,calc(100vw-1rem))]">
          {isRetiring ? (
            <div className="space-y-2 px-3 py-2">
              <p className="text-[13px] font-semibold text-foreground">
                {t("workspace.permissionV2.fullEnding", { defaultValue: "Full access is ending" })}
              </p>
              <p className="text-[11.5px] leading-relaxed text-muted-foreground">
                {retirementMessage}
              </p>
              {actionFailed ? <p role="alert" className="text-[11px] text-destructive">{t("workspace.permissionV2.actionFailed", { defaultValue: "Access could not be updated. Refresh this chat before retrying." })}</p> : null}
              <Button
                type="button"
                variant="outline"
                disabled={busy || !onRetryRestricted}
                onClick={() => {
                  setPending(true);
                  setActionFailed(false);
                  void Promise.resolve(onRetryRestricted?.()).catch(() => setActionFailed(true)).finally(() => setPending(false));
                }}
                className="h-9 w-full rounded-control"
              >
                {t("workspace.permissionV2.retryRestricted", { defaultValue: "Retry returning to Restricted" })}
              </Button>
            </div>
          ) : isFull ? (
            <div className="space-y-2 px-3 py-2">
              <p className="text-[12px] font-semibold text-foreground">{t("workspace.permissionV2.fullIndicator", { defaultValue: "Full access" })}</p>
              <p className="text-[11.5px] leading-relaxed text-muted-foreground">{t("workspace.permissionV2.fullReach", { defaultValue: "Files and commands across this device under your OS account permissions." })}</p>
              {actionFailed ? <p role="alert" className="text-[11px] text-destructive">{t("workspace.permissionV2.actionFailed", { defaultValue: "Access could not be updated. Refresh this chat before retrying." })}</p> : null}
              <Button type="button" variant="outline" disabled={busy} onClick={() => void changeMode("restricted")} className="h-9 w-full rounded-control">
                {t("workspace.permissionV2.returnRestricted", { defaultValue: "Return to Restricted" })}
              </Button>
            </div>
          ) : (
            <div className="space-y-2 p-2">
              <AccessMenuItem
                icon={<Hand className="h-4 w-4" />}
                label={t("thread.composer.workspace.default", { defaultValue: "Restricted" })}
                selected={permissions?.mode === "restricted" || draftRestricted}
                onSelect={() => {}}
              />
              <AccessMenuItem
                icon={<AlertTriangle className="h-4 w-4" />}
                label={t("thread.composer.workspace.full", { defaultValue: "Full access" })}
                selected={false}
                disabled={!fullAvailable || busy}
                warning
                onSelect={() => setConfirmFull(true)}
              />
              {isRestrictedRetiring ? (
                <div role="status" className="space-y-1 rounded-control bg-muted/35 px-3 py-2">
                  <p className="text-[12px] font-semibold text-foreground">
                    {t("workspace.permissionV2.restrictedEndingHeading", { defaultValue: "Previous access is ending." })}
                  </p>
                  <p className="text-[11.5px] leading-relaxed text-muted-foreground">
                    {restrictedRetirementMessage}
                  </p>
                </div>
              ) : null}
              {!canEnableFull && reasonCopy ? <p className="px-2 py-1 text-[11px] leading-relaxed text-muted-foreground">{reasonCopy}</p> : null}
              {actionFailed ? <p role="alert" className="px-2 py-1 text-[11px] text-destructive">{t("workspace.permissionV2.actionFailed", { defaultValue: "Access could not be updated. Refresh this chat before retrying." })}</p> : null}
            </div>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      {!persistentChat && !draftFullAccess ? (
        <span role="status" className="max-w-[min(19rem,65vw)] px-2 text-[10px] leading-tight text-muted-foreground">
          {t("workspace.permissionV2.sessionRequired", { defaultValue: "Send a message to create this chat before enabling Full access." })}
        </span>
      ) : null}
      <Dialog open={confirmFull} onOpenChange={setConfirmFull}>
        <DialogContent className="w-[min(calc(100vw-2rem),32rem)]">
          <DialogHeader className="text-left">
            <DialogTitle>{t("workspace.permissionV2.confirmTitle", { defaultValue: "Enable Full access?" })}</DialogTitle>
            <DialogDescription>
              {t("workspace.permissionV2.confirmDescription", { defaultValue: "Nanobot can read and change files and run commands beyond the selected folder under your OS account permissions. Working in is the current directory and context. New chats start in Restricted mode." })}
            </DialogDescription>
          </DialogHeader>
          {!persistentChat && !draftFullAccess ? (
            <p className="rounded-control bg-muted/35 px-3 py-2 text-[11.5px] text-muted-foreground">
              {t("workspace.permissionV2.sessionRequired", { defaultValue: "Send a message to create this chat before enabling Full access." })}
            </p>
          ) : null}
          {!canEnableFull && reasonCopy && persistentChat ? (
            <p role="status" className="rounded-control bg-muted/35 px-3 py-2 text-[11.5px] text-muted-foreground">{reasonCopy}</p>
          ) : null}
          <DialogFooter className="flex-row justify-end gap-2 sm:justify-end">
            <Button type="button" variant="ghost" disabled={busy} onClick={() => setConfirmFull(false)} className="rounded-control">
              {t("common.cancel", { defaultValue: "Cancel" })}
            </Button>
            <Button
              type="button"
              disabled={!fullAvailable || busy}
              onClick={() => {
                setConfirmFull(false);
                if (!persistentChat) {
                  if (onRequestFullAccess) {
                    setPending(true);
                    setActionFailed(false);
                    void Promise.resolve(onRequestFullAccess())
                      .catch(() => setActionFailed(true))
                      .finally(() => setPending(false));
                  }
                  return;
                }
                void changeMode("full");
              }}
              className="rounded-control"
            >
              {t("workspace.permissionV2.enableFull", { defaultValue: "Enable Full access" })}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
