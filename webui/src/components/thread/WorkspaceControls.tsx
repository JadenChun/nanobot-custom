import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  Check,
  ChevronDown,
  Folder,
  FolderOpen,
  Loader2,
  LockKeyhole,
  Monitor,
  ShieldCheck,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
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
import { formControlFocusClassName } from "@/components/ui/form-control";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type {
  EffectiveWorkspaceAccess,
  WorkspaceAccessMode,
  WorkspacePermissionProjection,
  WorkspaceScopePayload,
  WorkspacesPayload,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { isMobileRuntimeActive } from "@/lib/mobile-runtime";
import {
  isAbsoluteWorkspacePath,
  projectNameFromPath,
  scopeWithAccessMode,
  sameWorkspacePath,
  shortWorkspacePath,
} from "@/lib/workspace";

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
  disabled,
  locked = false,
  scope,
  defaultScope,
  controls,
  workspaces = null,
  error,
  onDismissError,
  nextTurnScope = null,
  nextTurnPending = false,
  turnActive = false,
  onChange,
}: {
  disabled?: boolean;
  locked?: boolean;
  ownerKey?: string | null;
  scope: WorkspaceScopePayload | null;
  defaultScope: WorkspaceScopePayload | null;
  controls: WorkspacesPayload["controls"] | null;
  workspaces?: WorkspacesPayload | null;
  error?: string | null;
  onDismissError?: () => void;
  /** Detached choice for the next turn; the active turn keeps ``scope``. */
  nextTurnScope?: WorkspaceScopePayload | null;
  /** A next-turn acknowledgement is still in flight. */
  nextTurnPending?: boolean;
  /** A turn is running; the picker still edits the next-turn choice. */
  turnActive?: boolean;
  onPickFolder?: () => Promise<string | null>;
  onRefreshWorkspaces?: () => Promise<void>;
  onChange?: (scope: WorkspaceScopePayload) => void;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const pickerTitleId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const activeScope = scope ?? defaultScope;
  // While a turn runs the picker edits the *next* turn instead of the admitted
  // one, so its own selection is the readable label.
  const pendingNextScope = nextTurnScope && (
    !activeScope
    || !sameWorkspacePath(nextTurnScope.project_path, activeScope.project_path)
    || nextTurnScope.access_mode !== activeScope.access_mode
  )
    ? nextTurnScope
    : null;
  const showNextTurnSelection = turnActive && pendingNextScope !== null;
  const visible = !!defaultScope
    && !!onChange;
  const canChangeFocus = controls?.can_change_project !== false;

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (!error || !visible || disabled) return;
    const frame = window.requestAnimationFrame(() => triggerRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [disabled, error, visible]);

  const applyProjectPath = useCallback(
    (projectPath: string, projectName?: string) => {
      const base = scope ?? defaultScope;
      const trimmed = projectPath.trim();
      if (!base || !onChange || !trimmed || !isAbsoluteWorkspacePath(trimmed)) return;
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
      setOpen(false);
    },
    [defaultScope, onChange, scope, workspaces?.managed_access],
  );

  if (!visible || !defaultScope || !onChange) return null;

  const pickerTitle = t("workspace.picker.chooseProject", { defaultValue: "Choose project folder" });
  const readableRepositories = (workspaces?.management?.repositories ?? [])
    .filter((repository) => repository.exists && repository.effective_access.read);
  const displayedScope = showNextTurnSelection ? pendingNextScope : activeScope;
  const selectedScope = displayedScope
    && !sameWorkspacePath(displayedScope.project_path, defaultScope.project_path)
    ? displayedScope
    : null;
  const selectedRepository = selectedScope && readableRepositories.find(
    (repository) => sameWorkspacePath(repository.path, selectedScope.project_path),
  );
  const namedScope = locked ? displayedScope : selectedScope;
  const folderName = namedScope
    ? selectedRepository?.name.trim() || namedScope.project_name?.trim()
      || projectNameFromPath(namedScope.project_path)
    : "";
  // With no project chosen, the trigger is only a folder icon so the idle
  // composer does not repeat the picker prompt.
  const showTriggerLabel = Boolean(folderName);
  const triggerAccessibleLabel = folderName
    ? showNextTurnSelection
      ? t("workspace.picker.nextTurn", { folder: folderName, defaultValue: "Next turn: {{folder}}" })
      : t("workspace.picker.workingIn", { folder: folderName, defaultValue: "Working in: {{folder}}" })
    : pickerTitle;
  const nextTurnHint = showNextTurnSelection
    ? t("workspace.picker.nextTurnHint", {
        defaultValue: "The running turn keeps its folder. This choice applies to the next turn.",
      })
    : null;
  const lockedHint = locked
    ? t("thread.harness.workspaceLocked", { defaultValue: "Claude Code locked this working folder." })
    : null;
  const triggerTitle = [triggerAccessibleLabel, namedScope?.project_path, nextTurnHint, lockedHint]
    .filter(Boolean).join("\n");
  const effectiveAccess = activeScope?.effective_access;
  const currentAccess = effectiveAccess
    ? effectiveAccessLabel(effectiveAccess, t)
    : workspaces?.managed_access === true
      ? t("workspace.picker.managedAccessUnavailable", { defaultValue: "Folder access is managed by the server." })
    : activeScope?.sandbox_status?.summary
      ?? (activeScope?.access_mode === "full"
        ? t("workspace.permissionV2.unavailable", { defaultValue: "Access unavailable" })
        : t("thread.composer.workspace.default", { defaultValue: "Sandbox" }));
  const listItems: Array<{
    key: string;
    name: string;
    path: string;
    selected: boolean;
    subtitle?: string;
  }> = [
    ...(selectedScope && !selectedRepository
      ? [{
          key: `selected:${selectedScope.project_path}`,
          name: folderName,
          path: selectedScope.project_path,
          selected: true,
          subtitle: currentAccess,
        }]
      : []),
    ...readableRepositories.map((repository) => ({
      key: repository.name,
      name: repository.name.trim() || projectNameFromPath(repository.path),
      path: repository.path,
      selected: repository === selectedRepository,
    })),
  ];
  const canClearDirectly = !!selectedScope && !disabled && canChangeFocus;
  const clearProjectLabel = t("workspace.picker.clearProject", { defaultValue: "Clear project folder" });
  const clearProject = () => applyProjectPath(defaultScope.project_path, defaultScope.project_name);

  return (
    <>
      <div className="flex min-w-0 items-center gap-0.5">
        <Popover open={open} onOpenChange={setOpen}>
          <div className={cn(
            "inline-flex min-w-0 max-w-full rounded-compact",
            showTriggerLabel && "ring-1 ring-inset ring-border/70",
          )}>
            <PopoverTrigger asChild>
              <button
                ref={triggerRef}
                type="button"
                disabled={disabled || !canChangeFocus}
                aria-label={lockedHint ? `${triggerAccessibleLabel}. ${lockedHint}` : triggerAccessibleLabel}
                title={triggerTitle}
                className={cn(
                  "thread-composer-workspace-action relative inline-flex h-7 min-w-0 items-center gap-1.5 rounded-compact text-[11px] font-medium text-muted-foreground transition-colors dark:text-foreground/70 disabled:pointer-events-none disabled:opacity-55",
                  formControlFocusClassName,
                  showTriggerLabel ? "px-2" : "w-7 justify-center",
                  canClearDirectly && "rounded-e-none",
                )}
              >
                {nextTurnPending
                  ? <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden />
                  : selectedScope
                    ? <FolderOpen className="h-3.5 w-3.5 shrink-0" aria-hidden />
                    : <Folder className="h-3.5 w-3.5 shrink-0" aria-hidden />}
                {showTriggerLabel ? (
                  <span className="min-w-0 truncate">
                    {folderName}
                  </span>
                ) : null}
                {locked ? (
                  <LockKeyhole className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
                ) : showTriggerLabel ? (
                  <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
                ) : null}
              </button>
            </PopoverTrigger>
            {canClearDirectly ? (
              <button
                type="button"
                aria-label={clearProjectLabel}
                title={clearProjectLabel}
                onClick={() => {
                  clearProject();
                  triggerRef.current?.focus();
                }}
                className={cn(
                  "thread-composer-workspace-action relative grid w-7 shrink-0 place-items-center rounded-e-compact text-muted-foreground transition-colors",
                  "before:absolute before:inset-y-1.5 before:start-0 before:w-px before:bg-border/70",
                  formControlFocusClassName,
                )}
              >
                <X className="h-3 w-3" aria-hidden />
              </button>
            ) : null}
          </div>
          <PopoverContent
            align="start"
            side="top"
            sideOffset={8}
            aria-labelledby={pickerTitleId}
            onOpenAutoFocus={(event) => event.preventDefault()}
            className="max-h-[var(--radix-popover-content-available-height)] w-[min(20rem,calc(100vw-2rem))] space-y-1.5 overflow-y-auto overscroll-contain p-1.5"
          >
            <h2 id={pickerTitleId} className="border-b border-border/55 px-2.5 pb-2 pt-1.5 text-sm font-semibold">
              {pickerTitle}
            </h2>
            {listItems.length > 0 ? (
              <div className="max-h-48 space-y-0.5 overflow-y-auto overscroll-contain scrollbar-thin scrollbar-track-transparent">
                {listItems.map((item) => (
                  <div
                    key={item.key}
                    className={cn(
                      "flex min-h-9 w-full items-center gap-1 rounded-lg transition-colors",
                      item.selected
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-muted/55",
                    )}
                  >
                    <button
                      type="button"
                      disabled={disabled || !canChangeFocus}
                      onClick={() => applyProjectPath(item.path, item.name)}
                      title={item.path}
                      aria-current={item.selected ? "true" : undefined}
                      className={cn(
                        floatingItemClassName,
                        floatingItemFocusClassName,
                        "flex min-h-9 min-w-0 flex-1 cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left hover:bg-transparent focus:bg-muted/55",
                        item.selected && "text-primary focus:bg-primary/15",
                      )}
                    >
                      {item.selected ? (
                        nextTurnPending
                          ? <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-primary" aria-hidden />
                          : <FolderOpen className="h-3.5 w-3.5 shrink-0 text-primary" aria-hidden />
                      ) : (
                        <Folder className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
                      )}
                      <span className="min-w-0 flex-1">
                        <span className={cn(
                          "block truncate text-[13px] leading-snug",
                          item.selected ? "font-semibold text-primary" : "font-medium text-foreground",
                        )}>
                          {item.name}
                        </span>
                        <span className="block truncate font-mono text-[10.5px] leading-tight text-muted-foreground/80">
                          {shortWorkspacePath(item.path)}
                        </span>
                        {item.subtitle ? (
                          <span className="block truncate text-[10px] leading-tight text-muted-foreground/80">
                            {item.subtitle}
                          </span>
                        ) : null}
                      </span>
                    </button>
                    {item.selected ? (
                      <button
                        type="button"
                        disabled={disabled || !canChangeFocus}
                        aria-label={clearProjectLabel}
                        title={clearProjectLabel}
                        onClick={clearProject}
                        className="mr-1.5 grid h-6.5 w-6.5 shrink-0 place-items-center rounded-md text-primary/75 transition-colors hover:bg-primary/15 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-45"
                      >
                        <X className="h-3.5 w-3.5" aria-hidden />
                      </button>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
              <p className="px-2.5 py-2 text-[11.5px] leading-relaxed text-muted-foreground">
                {t("workspace.picker.noConfiguredFolders", {
                  defaultValue: "No configured folders yet. Add folders in Settings.",
                })}
              </p>
            )}

            {error ? (
              <div
                role="alert"
                className="flex items-start justify-between gap-2 rounded-md border border-destructive/35 bg-destructive/8 px-2.5 py-1.5 text-[11.5px] font-medium text-destructive"
              >
                <span className="min-w-0 flex-1 [overflow-wrap:anywhere]">{error}</span>
                {onDismissError ? (
                  <button
                    type="button"
                    aria-label={t("thread.composer.reattachDismiss", { defaultValue: "Dismiss" })}
                    onClick={onDismissError}
                    className="grid h-4 w-4 shrink-0 place-items-center rounded text-destructive/80 hover:bg-destructive/15 hover:text-destructive"
                  >
                    <X className="h-3 w-3" aria-hidden />
                  </button>
                ) : null}
              </div>
            ) : null}

            {!canChangeFocus && !disabled ? (
              <p className="px-2 text-[11px] text-muted-foreground">
                {t("workspace.picker.focusUnavailable", { defaultValue: "Folder focus cannot be changed from this connection." })}
              </p>
            ) : null}
          </PopoverContent>
        </Popover>
        {!open && error ? (
          <span role="alert" className="inline-flex min-w-0 items-center gap-1 text-[11.5px] font-medium text-destructive">
            <span className="min-w-0 max-h-24 overflow-y-auto [overflow-wrap:anywhere]">{error}</span>
            {onDismissError ? (
              <button
                type="button"
                aria-label={t("thread.composer.reattachDismiss", { defaultValue: "Dismiss" })}
                onClick={onDismissError}
                className="grid h-4 w-4 shrink-0 place-items-center rounded text-destructive/80 hover:bg-destructive/15 hover:text-destructive"
              >
                <X className="h-3 w-3" aria-hidden />
              </button>
            ) : null}
          </span>
        ) : null}
      </div>
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
        <ShieldCheck className="thread-composer-access-icon mr-1.5 h-3.5 w-3.5 shrink-0" aria-hidden />
        <span className="thread-composer-access-label-full min-w-0 truncate">{accessLabel}</span>
        <span className="thread-composer-access-label-short hidden min-w-0 truncate">{shortAccessLabel}</span>
      </Button>
    );
  }

  return (
    <DropdownMenu modal={false}>
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
            <Monitor className="thread-composer-access-icon mr-1.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          ) : (
            <ShieldCheck className="thread-composer-access-icon mr-1.5 h-3.5 w-3.5 shrink-0" aria-hidden />
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
          icon={<ShieldCheck className="h-4 w-4" />}
          label={t("thread.composer.workspace.default", { defaultValue: "Sandbox" })}
          selected={mode === "restricted"}
          onSelect={() => setMode("restricted")}
        />
        <AccessMenuItem
          icon={<Monitor className="h-4 w-4" />}
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
  const mobile = isMobileRuntimeActive();
  const reasonId = useId();
  const [optimisticMode, setOptimisticMode] = useState<WorkspaceAccessMode | null>(null);
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
  const isRetiring = permissions?.mode === "full" && permissions?.state === "retiring";
  const isRestrictedRetiring = enabled
    && permissions?.mode === "restricted"
    && permissions.state === "retiring";
  const draftRestricted = enabled && !permissions && !persistentChat && !loading && !error;
  const unavailable = !enabled || (!permissions && !draftRestricted) || error;
  const isFull = !unavailable && !isRetiring
    ? (optimisticMode ?? permissions?.mode) === "full"
    : permissions?.mode === "full";
  const busy = loading || pending || disabled;
  const triggerDisabled = loading || disabled;
  const label = unavailable
    ? t("workspace.permissionV2.unavailable", { defaultValue: "Access unavailable" })
    : isRetiring
      ? t("workspace.permissionV2.fullEnding", { defaultValue: "Full access is ending" })
      : isRestrictedRetiring
        ? t("workspace.permissionV2.restrictedEnding", { defaultValue: "Sandbox · previous access ending" })
      : isFull
        ? t("thread.composer.workspace.full", { defaultValue: "Full access" })
        : t("thread.composer.workspace.default", { defaultValue: "Sandbox" });
  const compactLabel = isRetiring
    ? t("workspace.permissionV2.fullEndingShort", { defaultValue: "Ending" })
    : isRestrictedRetiring
      ? t("workspace.permissionV2.fullEndingShort", { defaultValue: "Ending" })
    : isFull
      ? t("thread.composer.workspace.fullShort", { defaultValue: "Full" })
      : t("thread.composer.workspace.defaultShort", { defaultValue: "Sandbox" });
  const reasonCopy = isRestrictedRetiring
    ? null
    : unavailableReason
    ? t(`workspace.permissionV2.unavailableReasons.${unavailableReason}`, {
        defaultValue: unavailableReason === "configured_sandbox"
          ? "Full access is unavailable while command confinement is configured."
          : unavailableReason === "supervisor_unavailable"
            ? "Full access is unavailable because Kohaku cannot supervise commands safely."
            : unavailableReason === "session_required"
              ? "Send a message to create this chat before enabling Full access."
              : "Full access is unavailable with the current local controls.",
      })
    : !persistentChat && !draftFullAccess
      ? t("workspace.permissionV2.sessionRequired", { defaultValue: "Send a message to create this chat before enabling Full access." })
      : null;
  const restrictedRetirementMessage = permissions?.retirement?.unknown_processes
    ? t("workspace.permissionV2.restrictedEndingUnknown", {
        defaultValue: "Kohaku cannot confirm whether earlier commands have finished. New work remains Sandbox; Full access stays unavailable.",
      })
    : permissions?.retirement?.active_processes || permissions?.retirement?.active_turn
      ? t("workspace.permissionV2.restrictedEndingActive", {
          defaultValue: "Earlier work is still active with the previous access. New work remains Sandbox; Full access stays unavailable.",
        })
      : permissions?.retirement
        ? t("workspace.permissionV2.restrictedEndingUncounted", {
            defaultValue: "No earlier commands are reported active or unknown, but the previous access is still ending. Sandbox remains the current mode.",
          })
        : t("workspace.permissionV2.restrictedEndingPending", {
            defaultValue: "Previous access is still ending. New work remains Sandbox; Full access stays unavailable.",
          });
  const retirementMessage = permissions?.retirement?.unknown_processes
    ? t("workspace.permissionV2.retiringUnknown", { defaultValue: "Kohaku cannot yet confirm whether earlier commands have finished. Access changes are pending." })
    : permissions?.retirement?.active_processes || permissions?.retirement?.active_turn
      ? t("workspace.permissionV2.retiringActive", { defaultValue: "A command is still running with the previous access. Wait for it to finish or stop it, then retry." })
      : t("workspace.permissionV2.retiringFinished", { defaultValue: "Earlier work has finished. Retry to confirm Sandbox access." });

  const unavailableCopy = unavailable
    ? reasonCopy ?? t("workspace.permissionV2.unavailableDetail", {
        defaultValue: "Full access is unavailable with the current local controls.",
      })
    : null;
  const accessibleDescription = [label, unavailableCopy]
    .filter((part): part is string => typeof part === "string" && part.length > 0)
    .join(". ");
  const changeMode = async (mode: WorkspaceAccessMode) => {
    if (!permissions || !onModeChange || busy) return;
    setOptimisticMode(mode);
    setPending(true);
    setActionFailed(false);
    try {
      await onModeChange(mode);
    } catch {
      setActionFailed(true);
    } finally {
      setOptimisticMode(null);
      setPending(false);
    }
  };

  const selectFullAccess = () => {
    if (!fullAvailable || busy) return;
    if (!persistentChat) {
      if (onRequestFullAccess) {
        setOptimisticMode("full");
        setPending(true);
        setActionFailed(false);
        void Promise.resolve(onRequestFullAccess())
          .catch(() => setActionFailed(true))
          .finally(() => {
            setOptimisticMode(null);
            setPending(false);
          });
      }
      return;
    }
    void changeMode("full");
  };

  return (
    <div className="flex min-w-0 flex-col items-start">
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild disabled={triggerDisabled}>
          <Button
            type="button"
            variant="ghost"
            aria-label={`${t("thread.composer.workspace.accessAria", { defaultValue: "Access" })}: ${label}`}
            aria-describedby={reasonId}
            aria-busy={loading || pending}
            aria-pressed={isFull}
            title={unavailableCopy ? `${label}. ${unavailableCopy}` : label}
            disabled={triggerDisabled}
            className={cn(
              "thread-composer-access touch-target whitespace-nowrap rounded-control border border-transparent font-semibold shadow-none",
              mobile
                ? "h-9 w-9 shrink-0 justify-center rounded-full px-0"
                : "min-w-0 max-w-[min(15rem,45vw)]",
              isHero ? "h-8 px-1.5 text-[12px]" : "h-9 px-1.5 text-[12.5px]",
              mobile && "px-0",
              isFull || isRestrictedRetiring
                ? "bg-transparent text-orange-600 hover:bg-orange-500/8 dark:text-orange-300 dark:hover:bg-orange-400/10"
                : "bg-transparent text-muted-foreground hover:bg-foreground/[0.045] hover:text-foreground dark:text-foreground/70 dark:hover:bg-white/[0.06]",
            )}
          >
            {mobile && loading ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            ) : isFull ? (
              <Monitor className={cn("shrink-0", mobile ? "h-4 w-4" : "mr-1.5 h-3.5 w-3.5")} aria-hidden />
            ) : (
              <ShieldCheck className={cn("shrink-0", mobile ? "h-4 w-4" : "mr-1.5 h-3.5 w-3.5")} aria-hidden />
            )}
            {!mobile ? (
              <>
                <span aria-hidden className="thread-composer-access-label-full min-w-0 truncate">{label}</span>
                <span aria-hidden className="thread-composer-access-label-short hidden min-w-0 truncate">{compactLabel}</span>
                {loading ? <Loader2 className="ml-1 h-3.5 w-3.5 animate-spin" aria-hidden /> : <ChevronDown className="ml-1.5 h-3 w-3 shrink-0" aria-hidden />}
              </>
            ) : null}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          className={cn(
            unavailable || isRetiring || isRestrictedRetiring
              ? "w-[min(22rem,calc(100vw-1rem))]"
              : "w-56",
          )}
        >
          {unavailable ? (
            <div role="status" className="space-y-1 px-3 py-2">
              <p className="text-[13px] font-semibold text-foreground">{label}</p>
              <p className="text-[11.5px] leading-relaxed text-muted-foreground">
                {unavailableCopy}
              </p>
            </div>
          ) : isRetiring ? (
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
                {t("workspace.permissionV2.retryRestricted", { defaultValue: "Retry returning to Sandbox" })}
              </Button>
            </div>
          ) : (
            <div className="space-y-1 p-1.5">
              <AccessMenuItem
                icon={<ShieldCheck className="h-4 w-4" />}
                label={t("thread.composer.workspace.default", { defaultValue: "Sandbox" })}
                selected={!isFull && (permissions?.mode === "restricted" || draftRestricted)}
                disabled={busy}
                onSelect={() => {
                  if (isFull) void changeMode("restricted");
                }}
              />
              <AccessMenuItem
                icon={<Monitor className="h-4 w-4" />}
                label={t("thread.composer.workspace.full", { defaultValue: "Full access" })}
                selected={isFull}
                disabled={(!isFull && !fullAvailable) || busy}
                warning
                onSelect={() => {
                  if (!isFull) {
                    selectFullAccess();
                  } else if (mobile) {
                    void changeMode("full");
                  }
                }}
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
              {!isFull && !canEnableFull && reasonCopy ? (
                <p className="px-2 py-1 text-[11px] leading-relaxed text-muted-foreground">{reasonCopy}</p>
              ) : null}
              {actionFailed ? (
                <p role="alert" className="px-2 py-1 text-[11px] text-destructive">
                  {t("workspace.permissionV2.actionFailed", { defaultValue: "Access could not be updated. Refresh this chat before retrying." })}
                </p>
              ) : null}
            </div>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      <span id={reasonId} className="sr-only">
        {accessibleDescription}
      </span>
      {!persistentChat && !draftFullAccess && !mobile ? (
        <span role="status" className="max-w-[min(19rem,65vw)] px-2 text-[10px] leading-tight text-muted-foreground">
          {t("workspace.permissionV2.sessionRequired", { defaultValue: "Send a message to create this chat before enabling Full access." })}
        </span>
      ) : null}
    </div>
  );
}
