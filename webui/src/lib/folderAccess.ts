import { ApiError } from "@/lib/api";
import type { FolderAccessOperation, FolderAccessTarget } from "@/lib/types";

type Translate = (key: string, options?: Record<string, unknown>) => string;

/** Return a file's direct containing folder without probing the filesystem. */
export function containingFolderForFile(path: string): string | null {
  if (!path || path.includes("\0") || /[\\/]$/.test(path)) return null;

  if (path === "~") return null;
  if (path.startsWith("~/") || path.startsWith("~\\")) {
    const lastSeparator = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    return lastSeparator === 1 ? "~" : path.slice(0, lastSeparator);
  }

  if (/^[A-Za-z]:[\\/]/.test(path)) {
    const lastSeparator = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    return lastSeparator === 2 ? path.slice(0, 3) : path.slice(0, lastSeparator);
  }

  const isBackslashUnc = path.startsWith("\\\\");
  const normalized = path.replace(/\\/g, "/");
  if (normalized.startsWith("//")) {
    const parts = normalized.slice(2).split("/");
    if (parts.length < 3 || parts.some((part) => !part || part === "." || part === "..")) {
      return null;
    }
    const folder = parts.slice(0, -1).join(isBackslashUnc ? "\\" : "/");
    return isBackslashUnc ? `\\\\${folder}` : `//${folder}`;
  }

  if (path.startsWith("/")) {
    const lastSeparator = path.lastIndexOf("/");
    return lastSeparator === 0 ? "/" : path.slice(0, lastSeparator);
  }
  return null;
}

export function folderForKnownTarget(target: FolderAccessTarget): string | null {
  if (target.kind === "directory") return target.path;
  return target.kind === "file" ? containingFolderForFile(target.path) : null;
}

export function requiredFolderCapabilities(operation: FolderAccessOperation): {
  fileAccess: "read_only" | "read_write";
  commands: boolean;
} {
  return {
    fileAccess: operation === "write" ? "read_write" : "read_only",
    commands: operation === "execute",
  };
}

export function folderOperationCanContinue(
  operation: FolderAccessOperation,
  access: {
    read: boolean;
    write: boolean;
    commands: { available: boolean };
  } | null,
): boolean {
  if (!access) return true;
  if (operation === "write") return access.read && access.write;
  if (operation === "execute") return access.commands.available;
  if (operation === "read") return access.read;
  return true;
}

export function folderAccessErrorMessage(error: unknown, t: Translate): string {
  if (!(error instanceof ApiError)) {
    return t("workspace.recovery.actionFailed", {
      defaultValue: "The folder action did not finish. Refresh the task and try again.",
    });
  }

  if (error.message === "configured_repository_has_own_policy") {
    return t("workspace.folders.configuredPolicy", {
      defaultValue: "This folder has a configured repository policy. Folder grants cannot override its protected, read-only or proposal rules.",
    });
  }
  if (error.message === "folder_metadata_saved_access_pending") {
    return t("workspace.folders.partialSave", {
      defaultValue: "The folder metadata was saved, but its Sandbox access update did not finish. Retry the same save to complete it. If a command is still running, wait for it to finish first.",
    });
  }
  if (error.message === "invalid_access_request") {
    return t("workspace.recovery.invalidFolder", {
      defaultValue: "Nanobot could not verify that folder. Choose an existing folder and try again.",
    });
  }
  if (error.message === "folder path must be an existing directory" || error.message === "path_missing") {
    return t("workspace.recovery.folderMissing", {
      defaultValue: "Nanobot could not verify that folder. Choose an existing folder and try again.",
    });
  }
  if (error.message === "folder path must be absolute") {
    return t("workspace.recovery.absolutePathRequired", {
      defaultValue: "Enter an absolute folder path on this machine, then try again.",
    });
  }
  if (
    error.message === "folder grants cannot overlap private runtime storage"
  ) {
    return t("workspace.recovery.chooseNarrowerFolder", {
      defaultValue: "This folder includes Nanobot's private runtime files. Choose a narrower project folder.",
    });
  }
  if (error.message === "native_processes_active" || error.message === "permission_retiring") {
    return t("workspace.management.nativeProcessesActive", {
      defaultValue: "A command is still running with access it started with. Wait for it to finish before changing this folder.",
    });
  }
  if (error.message === "native_processes_unknown") {
    return t("workspace.management.nativeProcessesUnknown", {
      defaultValue: "Nanobot cannot verify whether a command from this device is still running. Restarting Nanobot will not clear this state; you may need to restart this device.",
    });
  }
  if (
    error.message === "permission_changed"
    || error.message === "grants_changed"
    || error.message === "access_changed"
    || error.message === "access_request_changed"
    || error.message === "access_request_inactive"
    || error.message === "source_changed"
    || error.message === "task_changed"
    || error.message === "focus_changed"
    || error.message === "full_access_active"
    || error.status === 409
  ) {
    return t("workspace.recovery.changed", {
      defaultValue: "This task or folder access changed. Refresh and review the current access before trying again.",
    });
  }
  if (error.status === 403) {
    return t("workspace.recovery.folderUnavailable", {
      defaultValue: "Nanobot cannot use that folder with the current access settings. Choose a different folder or review access.",
    });
  }
  return t("workspace.recovery.actionFailed", {
    defaultValue: "The folder action did not finish. Refresh the task and try again.",
  });
}
