import { useMemo, useState, type FormEvent } from "react";
import { Folder, FolderPlus, GitBranch, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ApiError, discoverRepositories, forgetRepository, rememberRepository, revokeWorkspaceGrant, upsertWorkspaceGrant } from "@/lib/api";
import type {
  EffectiveWorkspaceAccess,
  FolderFileAccess,
  FolderGrantInput,
  FolderGrantRow,
  RepositoryCandidate,
  RepositoryRecord,
  RepositoryRow,
  WorkspacesPayload,
} from "@/lib/types";
import { folderAccessErrorMessage } from "@/lib/folderAccess";
import { isAbsoluteWorkspacePath, normalizeWorkspacePath } from "@/lib/workspace";
import { getRuntimeHost } from "@/lib/runtime";
import { useClient } from "@/providers/ClientProvider";
import { cn } from "@/lib/utils";

type ManagerSection = "folders" | "repositories";

interface GrantFormState extends FolderGrantInput {
  editing: boolean;
}

interface RepositoryFormState extends RepositoryRecord {
  editing: boolean;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request could not be completed.";
}

function grantErrorMessage(
  error: unknown,
  t: (key: string, options?: Record<string, unknown>) => string,
): string {
  return folderAccessErrorMessage(error, t);
}

function isNativeAdmissionError(error: unknown): boolean {
  return error instanceof ApiError
    && (error.message === "native_processes_active" || error.message === "native_processes_unknown");
}

function commandStatus(
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
  const reasonKey = reason ? `workspace.management.commandReasons.${reason}` : "";
  const reasonText = reason
    ? t(reasonKey, { defaultValue: reason.replaceAll("_", " ") })
    : t("workspace.management.commandUnavailable", { defaultValue: "Unavailable" });
  return t("workspace.management.commandsUnavailable", {
    reason: reasonText,
    defaultValue: `Commands unavailable: ${reasonText}`,
  });
}

function accessSummary(
  access: EffectiveWorkspaceAccess,
  t: (key: string, options?: Record<string, unknown>) => string,
): string {
  const file = !access.read
    ? t("workspace.management.noFileAccess", { defaultValue: "No file access" })
    : access.write
      ? t("workspace.management.readWrite", { defaultValue: "Read and write" })
      : t("workspace.management.readOnly", { defaultValue: "Read only" });
  return `${file} · ${commandStatus(access, t)}`;
}

function nearestParentGrant(path: string, grants: FolderGrantRow[]): FolderGrantRow | null {
  const normalized = normalizeWorkspacePath(path);
  return grants
    .filter((grant) => {
      const root = normalizeWorkspacePath(grant.path);
      return root !== normalized && normalized.startsWith(`${root}/`);
    })
    .sort((a, b) => b.path.length - a.path.length)[0] ?? null;
}

function StatusLine({
  children,
  tone = "muted",
}: {
  children: string;
  tone?: "muted" | "error" | "success";
}) {
  return (
    <p
      role={tone === "error" ? "alert" : "status"}
      className={cn(
        "rounded-control border px-3 py-2 text-[12px]",
        tone === "error" && "border-destructive/30 bg-destructive/5 text-destructive",
        tone === "success" && "border-emerald-500/25 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300",
        tone === "muted" && "border-border/55 bg-muted/35 text-muted-foreground",
      )}
    >
      {children}
    </p>
  );
}

export function FoldersProjectsManager({
  workspaces,
  onRefreshWorkspaces,
  onPickFolder,
  className,
}: {
  workspaces: WorkspacesPayload | null;
  onRefreshWorkspaces: () => Promise<void>;
  onPickFolder?: () => Promise<string | null>;
  className?: string;
}) {
  const { t } = useTranslation();
  const { client } = useClient();
  const [section, setSection] = useState<ManagerSection>("folders");
  const [grantForm, setGrantForm] = useState<GrantFormState | null>(null);
  const [repositoryForm, setRepositoryForm] = useState<RepositoryFormState | null>(null);
  const [repositoryPathDraft, setRepositoryPathDraft] = useState("");
  const [discovery, setDiscovery] = useState<Awaited<ReturnType<typeof discoverRepositories>> | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const management = workspaces?.management ?? null;
  const canManageFolders = workspaces?.controls.can_manage_folders === true;
  const canManageCatalog = workspaces?.controls.can_manage_catalog === true;
  const canDiscover = workspaces?.controls.can_discover_repositories === true;
  const availableSections = useMemo(() => [
    ...(canManageFolders ? ["folders" as const] : []),
    ...(canManageCatalog ? ["repositories" as const] : []),
  ], [canManageCatalog, canManageFolders]);

  const refreshForConflict = async () => {
    await onRefreshWorkspaces();
    setError(t("workspace.management.grantsChanged", {
      defaultValue: "Folder access changed elsewhere. Review the updated values before trying again.",
    }));
  };

  const saveGrant = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!management || !grantForm) return;
    const path = grantForm.path.trim();
    if (!path || !isAbsoluteWorkspacePath(path)) {
      setError(t("workspace.dialog.absolutePathRequired"));
      return;
    }
    setPending("grant-save");
    setError(null);
    setNotice(null);
    try {
      await upsertWorkspaceGrant(client, {
        path,
        file_access: grantForm.file_access,
        commands: grantForm.commands,
      }, management.grants_revision);
      await onRefreshWorkspaces();
      setGrantForm(null);
      setNotice(t("workspace.management.folderSaved", { defaultValue: "Folder access saved." }));
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 409 && isNativeAdmissionError(reason)) {
        setError(grantErrorMessage(reason, t));
      } else if (reason instanceof ApiError && reason.status === 409) {
        await refreshForConflict();
      } else {
        setError(grantErrorMessage(reason, t));
      }
    } finally {
      setPending(null);
    }
  };

  const revokeGrant = async (grant: FolderGrantRow) => {
    if (!management) return;
    const parent = nearestParentGrant(grant.path, management.grants);
    setPending(`grant-revoke:${grant.path}`);
    setError(null);
    setNotice(null);
    try {
      await revokeWorkspaceGrant(client, grant.path, management.grants_revision);
      await onRefreshWorkspaces();
      setNotice(parent
        ? t("workspace.management.revokedInherited", {
            path: parent.path,
            access: accessSummary(parent.effective_access, t),
            defaultValue: `Folder grant removed. Access remains inherited from ${parent.path}: ${accessSummary(parent.effective_access, t)}.`,
          })
        : t("workspace.management.folderRevoked", { defaultValue: "Folder access removed." }));
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 409 && isNativeAdmissionError(reason)) {
        setError(grantErrorMessage(reason, t));
      } else if (reason instanceof ApiError && reason.status === 409) {
        await refreshForConflict();
      } else {
        setError(grantErrorMessage(reason, t));
      }
    } finally {
      setPending(null);
    }
  };

  const saveRepository = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!repositoryForm) return;
    const path = repositoryForm.path.trim();
    const name = repositoryForm.name.trim();
    const repositories = management?.repositories ?? [];
    if (!name || !path || !isAbsoluteWorkspacePath(path)) {
      setError(!path || !isAbsoluteWorkspacePath(path)
        ? t("workspace.dialog.absolutePathRequired")
        : t("workspace.management.repositoryNameRequired", { defaultValue: "Enter a repository name." }));
      return;
    }
    const nameConflict = repositories.find((row) => row.name === name && row.path !== path);
    if (nameConflict) {
      setError(t("workspace.management.repositoryNameConflict", {
        name,
        defaultValue: `The name “${name}” is already used by another repository.`,
      }));
      return;
    }
    const entry: RepositoryRecord = {
      name,
      path,
      purpose: repositoryForm.purpose.trim(),
      kind: repositoryForm.kind,
      knowledge_repos: repositoryForm.kind === "project"
        ? repositoryForm.knowledge_repos.map((item) => item.trim()).filter(Boolean)
        : [],
      auto_sync: repositoryForm.auto_sync,
      sync_paths: repositoryForm.sync_paths,
    };
    setPending("repository-save");
    setError(null);
    setNotice(null);
    try {
      await rememberRepository(client, entry);
      await onRefreshWorkspaces();
      setRepositoryForm(null);
      setNotice(t("workspace.management.repositorySaved", { defaultValue: "Repository details saved." }));
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setPending(null);
    }
  };

  const forget = async (repository: RepositoryRow) => {
    setPending(`repository-forget:${repository.name}`);
    setError(null);
    setNotice(null);
    try {
      await forgetRepository(client, repository.name);
      await onRefreshWorkspaces();
      setNotice(t("workspace.management.repositoryForgotten", {
        defaultValue: "Repository reference removed. Its files remain in place.",
      }));
    } catch (reason) {
      const message = errorMessage(reason);
      setError(message.includes("repository_in_use")
        ? t("workspace.management.repositoryInUse", {
            defaultValue: "This repository is linked from a project. Edit that project's knowledge associations before forgetting it.",
          })
        : message);
    } finally {
      setPending(null);
    }
  };

  const runDiscovery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const path = repositoryPathDraft.trim();
    if (!path || !isAbsoluteWorkspacePath(path)) {
      setError(t("workspace.dialog.absolutePathRequired"));
      return;
    }
    setPending("repository-discover");
    setError(null);
    setNotice(null);
    try {
      setDiscovery(await discoverRepositories(client, path));
    } catch (reason) {
      setDiscovery(null);
      setError(errorMessage(reason));
    } finally {
      setPending(null);
    }
  };

  const startRememberCandidate = (candidate: RepositoryCandidate) => {
    const rows = management?.repositories ?? [];
    const sameName = rows.find((row) => row.name === candidate.suggested_name);
    if (sameName && sameName.path !== candidate.path) {
      setError(t("workspace.management.repositoryNameConflict", {
        name: candidate.suggested_name,
        defaultValue: `The name “${candidate.suggested_name}” is already used by another repository.`,
      }));
      return;
    }
    setError(null);
    setRepositoryForm({
      name: sameName?.name ?? candidate.suggested_name,
      path: candidate.path,
      purpose: sameName?.purpose ?? "",
      kind: sameName?.kind ?? "project",
      knowledge_repos: sameName?.knowledge_repos ?? [],
      auto_sync: sameName?.auto_sync ?? false,
      sync_paths: sameName?.sync_paths ?? [],
      editing: !!sameName,
    });
  };

  const contentAvailable = management !== null && availableSections.length > 0;
  const selectedSection = availableSections.includes(section)
    ? section
    : availableSections[0] ?? "folders";

  return (
    <div className={cn("space-y-4", className)}>
      {contentAvailable ? (
        <SegmentedControl
          value={selectedSection}
          mode="tabs"
          ariaLabel={t("workspace.management.sectionsAria", { defaultValue: "Folder and repository sections" })}
          options={[
            ...(canManageFolders ? [{ value: "folders" as const, label: t("workspace.management.folders", { defaultValue: "Folder access" }) }] : []),
            ...(canManageCatalog ? [{ value: "repositories" as const, label: t("workspace.management.repositories", { defaultValue: "Known repositories" }) }] : []),
          ]}
          onChange={setSection}
          className="w-full"
        />
      ) : null}

      {error ? <StatusLine tone="error">{error}</StatusLine> : null}
      {notice ? <StatusLine tone="success">{notice}</StatusLine> : null}

      {!contentAvailable ? (
        <StatusLine>
          {t("workspace.management.unavailable", {
            defaultValue: "Folder and repository management is available only from a trusted local WebUI connection.",
          })}
        </StatusLine>
      ) : selectedSection === "folders" ? (
        <section aria-label={t("workspace.management.folders", { defaultValue: "Folder access" })} className="space-y-4">
          <div className="rounded-control border border-border/55 bg-muted/30 px-3 py-2.5 text-[12px] leading-relaxed text-muted-foreground">
            {t("workspace.management.folderExplanation", {
              defaultValue: "File access follows the most specific folder setting. Commands are available only when Bubblewrap can confine them; browser, device, and desktop actions are unavailable in managed mode.",
            })}
          </div>
          <p className="text-[12px] text-muted-foreground">
            {t("workspace.management.runningProcessNote", {
              defaultValue: "Already running confined commands keep their launch access. New commands follow updated folder access.",
            })}
          </p>
          {!grantForm ? (
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setError(null);
                setGrantForm({ path: "", file_access: "read_only", commands: false, editing: false });
              }}
              className="h-9 rounded-control"
            >
              <FolderPlus className="mr-2 h-4 w-4" aria-hidden />
              {t("workspace.management.addFolder", { defaultValue: "Add folder access" })}
            </Button>
          ) : (
            <form onSubmit={(event) => void saveGrant(event)} className="space-y-3 rounded-panel border border-border/60 bg-card p-3">
              <div className="space-y-1.5">
                <label htmlFor="workspace-grant-path" className="text-[12px] font-medium">
                  {t("workspace.management.folderPath", { defaultValue: "Folder path" })}
                </label>
                <div className="flex gap-2">
                  <Input
                    id="workspace-grant-path"
                    aria-label={t("workspace.management.folderPath", { defaultValue: "Folder path" })}
                    value={grantForm.path}
                    readOnly={grantForm.editing}
                    onChange={(event) => setGrantForm({ ...grantForm, path: event.target.value })}
                    placeholder={t("workspace.dialog.manualPlaceholder")}
                    className="h-9 min-w-0 rounded-control"
                  />
                  <BrowseButton
                    onPickFolder={onPickFolder}
                    onSelect={(path) => setGrantForm({ ...grantForm, path })}
                    onError={setError}
                  />
                </div>
              </div>
              <label className="flex items-center justify-between gap-3 text-[12px]">
                <span>{t("workspace.management.fileAccess", { defaultValue: "File access" })}</span>
                <select
                  aria-label={t("workspace.management.fileAccess", { defaultValue: "File access" })}
                  value={grantForm.file_access}
                  onChange={(event) => setGrantForm({
                    ...grantForm,
                    file_access: event.target.value as FolderFileAccess,
                  })}
                  className="h-9 rounded-control border border-border/60 bg-background px-2 text-[12px]"
                >
                  <option value="read_only">{t("workspace.management.readOnly", { defaultValue: "Read only" })}</option>
                  <option value="read_write">{t("workspace.management.readWrite", { defaultValue: "Read and write" })}</option>
                </select>
              </label>
              <label className="flex min-h-9 items-center gap-2 text-[12px]">
                <input
                  type="checkbox"
                  checked={grantForm.commands}
                  onChange={(event) => setGrantForm({ ...grantForm, commands: event.target.checked })}
                  className="h-4 w-4 accent-foreground"
                />
                {t("workspace.management.allowCommands", { defaultValue: "Allow commands" })}
              </label>
              <div className="flex flex-wrap justify-end gap-2">
                <Button type="button" variant="ghost" onClick={() => setGrantForm(null)} className="h-9 rounded-control">
                  {t("settings.actions.cancel", { defaultValue: "Cancel" })}
                </Button>
                <Button type="submit" disabled={pending === "grant-save" || !grantForm.path.trim()} className="h-9 rounded-control">
                  {pending === "grant-save" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                  {t("settings.actions.save", { defaultValue: "Save" })}
                </Button>
              </div>
            </form>
          )}

          {management.grants.length === 0 ? (
            <p className="rounded-control border border-dashed border-border/70 px-3 py-5 text-center text-[12px] text-muted-foreground">
              {t("workspace.management.noFolders", { defaultValue: "No folders have been granted yet." })}
            </p>
          ) : (
            <ul className="space-y-2">
              {management.grants.map((grant) => (
                <li key={grant.path} className="rounded-panel border border-border/60 bg-card p-3">
                  <div className="flex items-start gap-3">
                    <Folder className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
                    <div className="min-w-0 flex-1">
                      <p className="break-all text-[12px] font-medium">{grant.path}</p>
                      <p className="mt-1 text-[11px] text-muted-foreground">
                        {grant.exists
                          ? accessSummary(grant.effective_access, t)
                          : t("workspace.management.missingFolder", { defaultValue: "Folder missing" })}
                      </p>
                      {grant.effective_access.grant_root && grant.effective_access.grant_root !== grant.path ? (
                        <p className="mt-1 break-all text-[11px] text-muted-foreground">
                          {t("workspace.management.inheritedFrom", {
                            path: grant.effective_access.grant_root,
                            defaultValue: `Effective access inherited from ${grant.effective_access.grant_root}`,
                          })}
                        </p>
                      ) : null}
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      aria-label={t("workspace.management.editFolder", { path: grant.path, defaultValue: `Edit ${grant.path}` })}
                      disabled={!!pending}
                      onClick={() => {
                        setError(null);
                        setGrantForm({
                          path: grant.path,
                          file_access: grant.file_access,
                          commands: grant.commands,
                          editing: true,
                        });
                      }}
                      className="h-8 shrink-0 rounded-control px-2 text-[11px]"
                    >
                      {t("settings.actions.edit", { defaultValue: "Edit" })}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      aria-label={t("workspace.management.revokeFolder", { path: grant.path, defaultValue: `Revoke ${grant.path}` })}
                      disabled={!!pending}
                      onClick={() => void revokeGrant(grant)}
                      className="h-8 shrink-0 rounded-control px-2 text-destructive hover:text-destructive"
                    >
                      {pending === `grant-revoke:${grant.path}`
                        ? <Loader2 className="h-4 w-4 animate-spin" />
                        : <Trash2 className="h-4 w-4" aria-hidden />}
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      ) : (
        <section aria-label={t("workspace.management.repositories", { defaultValue: "Known repositories" })} className="space-y-4">
          <div className="rounded-control border border-border/55 bg-muted/30 px-3 py-2.5 text-[12px] leading-relaxed text-muted-foreground">
            {t("workspace.management.repositoryExplanation", {
              defaultValue: "Remembering a repository stores its name and purpose. It does not grant folder access or change the current conversation. Forgetting removes metadata only; files stay in place.",
            })}
          </div>

          <div className="flex flex-wrap gap-2">
            {!repositoryForm ? (
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setError(null);
                  setRepositoryForm({
                    name: "",
                    path: "",
                    purpose: "",
                    kind: "project",
                    knowledge_repos: [],
                    auto_sync: false,
                    sync_paths: [],
                    editing: false,
                  });
                }}
                className="h-9 rounded-control"
              >
                <GitBranch className="mr-2 h-4 w-4" aria-hidden />
                {t("workspace.management.addRepository", { defaultValue: "Remember repository" })}
              </Button>
            ) : null}
          </div>

          {canDiscover ? (
            <form onSubmit={(event) => void runDiscovery(event)} className="space-y-2 rounded-panel border border-border/60 bg-card p-3">
              <label htmlFor="workspace-discovery-root" className="text-[12px] font-medium">
                {t("workspace.management.discoverUnder", { defaultValue: "Discover repositories under a folder" })}
              </label>
              <div className="flex gap-2">
                <Input
                  id="workspace-discovery-root"
                  value={repositoryPathDraft}
                  onChange={(event) => setRepositoryPathDraft(event.target.value)}
                  placeholder={t("workspace.dialog.manualPlaceholder")}
                  className="h-9 min-w-0 rounded-control"
                />
                <BrowseButton
                  onPickFolder={onPickFolder}
                  onSelect={setRepositoryPathDraft}
                  onError={setError}
                />
                <Button type="submit" disabled={pending === "repository-discover" || !repositoryPathDraft.trim()} className="h-9 shrink-0 rounded-control">
                  {pending === "repository-discover" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                  {t("workspace.management.discover", { defaultValue: "Discover" })}
                </Button>
              </div>
            </form>
          ) : null}

          {discovery ? (
            <div className="space-y-2 rounded-panel border border-border/60 bg-card p-3">
              <p className="break-all text-[11px] text-muted-foreground">
                {t("workspace.management.discoverySummary", {
                  count: discovery.visited_directories,
                  root: discovery.root,
                  defaultValue: `Checked ${discovery.visited_directories} folders under ${discovery.root}.`,
                })}
                {discovery.truncated ? ` ${t("workspace.management.discoveryTruncated", { defaultValue: "Results were limited; narrow the folder and try again." })}` : ""}
              </p>
              {discovery.candidates.length === 0 ? (
                <p className="py-2 text-[12px] text-muted-foreground">
                  {t("workspace.management.noCandidates", { defaultValue: "No readable repository folders were found." })}
                </p>
              ) : (
                <ul className="space-y-2">
                  {discovery.candidates.map((candidate) => (
                    <li key={candidate.path} className="flex flex-wrap items-center gap-2 rounded-control border border-border/50 px-3 py-2">
                      <span className="min-w-0 flex-1">
                        <span className="block break-all text-[12px] font-medium">{candidate.path}</span>
                        <span className="block text-[11px] text-muted-foreground">
                          {accessSummary(candidate.effective_access, t)}
                        </span>
                      </span>
                      {candidate.known_name ? (
                        <span className="text-[11px] text-muted-foreground">
                          {t("workspace.management.alreadyKnown", {
                            name: candidate.known_name,
                            defaultValue: `Known as ${candidate.known_name}`,
                          })}
                        </span>
                      ) : (
                        <Button type="button" variant="outline" onClick={() => startRememberCandidate(candidate)} className="h-8 rounded-control px-2.5 text-[11px]">
                          {t("workspace.management.remember", { defaultValue: "Remember" })}
                        </Button>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ) : null}

          {repositoryForm ? (
            <form onSubmit={(event) => void saveRepository(event)} className="space-y-3 rounded-panel border border-border/60 bg-card p-3">
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1.5 text-[12px] font-medium">
                  <span>{t("workspace.management.repositoryName", { defaultValue: "Name" })}</span>
                  <Input
                    aria-label={t("workspace.management.repositoryName", { defaultValue: "Name" })}
                    value={repositoryForm.name}
                    readOnly={repositoryForm.editing}
                    onChange={(event) => setRepositoryForm({ ...repositoryForm, name: event.target.value })}
                    className="h-9 rounded-control"
                  />
                </label>
                <label className="space-y-1.5 text-[12px] font-medium">
                  <span>{t("workspace.management.repositoryKind", { defaultValue: "Kind" })}</span>
                  <select
                    aria-label={t("workspace.management.repositoryKind", { defaultValue: "Kind" })}
                    value={repositoryForm.kind}
                    onChange={(event) => setRepositoryForm({ ...repositoryForm, kind: event.target.value as RepositoryRecord["kind"] })}
                    className="h-9 w-full rounded-control border border-border/60 bg-background px-2 text-[12px]"
                  >
                    <option value="project">{t("workspace.management.projectKind", { defaultValue: "Project" })}</option>
                    <option value="knowledge">{t("workspace.management.knowledgeKind", { defaultValue: "Knowledge" })}</option>
                  </select>
                </label>
              </div>
              <div className="space-y-1.5 text-[12px] font-medium">
                <label htmlFor="workspace-repository-path">
                  {t("workspace.management.repositoryPath", { defaultValue: "Repository path" })}
                </label>
                <div className="flex gap-2">
                  <Input
                    id="workspace-repository-path"
                    value={repositoryForm.path}
                    readOnly={repositoryForm.editing}
                    onChange={(event) => setRepositoryForm({ ...repositoryForm, path: event.target.value })}
                    placeholder={t("workspace.dialog.manualPlaceholder")}
                    className="h-9 min-w-0 rounded-control"
                  />
                  {!repositoryForm.editing ? (
                    <BrowseButton onPickFolder={onPickFolder} onSelect={(path) => setRepositoryForm({ ...repositoryForm, path })} onError={setError} />
                  ) : null}
                </div>
              </div>
              <label className="block space-y-1.5 text-[12px] font-medium">
                <span>{t("workspace.management.repositoryPurpose", { defaultValue: "Purpose" })}</span>
                <textarea
                  aria-label={t("workspace.management.repositoryPurpose", { defaultValue: "Purpose" })}
                  value={repositoryForm.purpose}
                  onChange={(event) => setRepositoryForm({ ...repositoryForm, purpose: event.target.value })}
                  rows={2}
                  className="w-full resize-y rounded-control border border-border/60 bg-background px-3 py-2 text-[12px] font-normal"
                />
              </label>
              {repositoryForm.kind === "project" ? (
                <label className="block space-y-1.5 text-[12px] font-medium">
                  <span>{t("workspace.management.knowledgeAssociations", { defaultValue: "Knowledge repositories (comma separated)" })}</span>
                  <Input
                    aria-label={t("workspace.management.knowledgeAssociations", { defaultValue: "Knowledge repositories (comma separated)" })}
                    value={repositoryForm.knowledge_repos.join(", ")}
                    onChange={(event) => setRepositoryForm({
                      ...repositoryForm,
                      knowledge_repos: event.target.value.split(","),
                    })}
                    className="h-9 rounded-control"
                  />
                </label>
              ) : null}
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                {t("workspace.management.syncPreserved", {
                  defaultValue: "Existing knowledge sync choices are preserved when metadata is updated. Remembering a repository does not enable sync or grant access.",
                })}
              </p>
              <div className="flex flex-wrap justify-end gap-2">
                <Button type="button" variant="ghost" onClick={() => setRepositoryForm(null)} className="h-9 rounded-control">
                  {t("settings.actions.cancel", { defaultValue: "Cancel" })}
                </Button>
                <Button type="submit" disabled={pending === "repository-save" || !repositoryForm.name.trim() || !repositoryForm.path.trim()} className="h-9 rounded-control">
                  {pending === "repository-save" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                  {t("settings.actions.save", { defaultValue: "Save" })}
                </Button>
              </div>
            </form>
          ) : null}

          {management.repositories.length === 0 ? (
            <p className="rounded-control border border-dashed border-border/70 px-3 py-5 text-center text-[12px] text-muted-foreground">
              {t("workspace.management.noRepositories", { defaultValue: "No repositories are remembered yet." })}
            </p>
          ) : (
            <ul className="space-y-2">
              {management.repositories.map((repository) => (
                <li key={repository.name} className="rounded-panel border border-border/60 bg-card p-3">
                  <div className="flex items-start gap-3">
                    <GitBranch className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <p className="text-[12px] font-semibold">{repository.name}</p>
                        <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                          {repository.kind === "project"
                            ? t("workspace.management.projectKind", { defaultValue: "Project" })
                            : t("workspace.management.knowledgeKind", { defaultValue: "Knowledge" })}
                        </span>
                        {!repository.exists ? (
                          <span className="text-[10px] font-medium text-destructive">
                            {t("workspace.management.missingRepository", { defaultValue: "Missing" })}
                          </span>
                        ) : null}
                      </div>
                      <p className="mt-1 break-all text-[11px] text-muted-foreground">{repository.path}</p>
                      {repository.purpose ? <p className="mt-1 text-[11px] text-muted-foreground">{repository.purpose}</p> : null}
                      {repository.knowledge_repos.length > 0 ? (
                        <p className="mt-1 text-[11px] text-muted-foreground">
                          {t("workspace.management.knowledgeLinks", {
                            names: repository.knowledge_repos.join(", "),
                            defaultValue: `Knowledge: ${repository.knowledge_repos.join(", ")}`,
                          })}
                        </p>
                      ) : null}
                      <p className="mt-1 text-[11px] text-muted-foreground">
                        {accessSummary(repository.effective_access, t)}
                      </p>
                    </div>
                    <div className="flex shrink-0 flex-wrap justify-end gap-1">
                      {canManageFolders ? (
                        <Button
                          type="button"
                          variant="ghost"
                          onClick={() => {
                            setSection("folders");
                            setError(null);
                            setGrantForm({ path: repository.path, file_access: "read_only", commands: false, editing: false });
                          }}
                          className="h-8 rounded-control px-2 text-[11px]"
                        >
                          {t("workspace.management.grantAccess", { defaultValue: "Grant access…" })}
                        </Button>
                      ) : null}
                      <Button
                        type="button"
                        variant="ghost"
                        aria-label={t("workspace.management.editRepository", { name: repository.name, defaultValue: `Edit ${repository.name}` })}
                        onClick={() => {
                          setError(null);
                          setRepositoryForm({ ...repository, editing: true });
                        }}
                        className="h-8 rounded-control px-2 text-[11px]"
                      >
                        {t("settings.actions.edit", { defaultValue: "Edit" })}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        aria-label={t("workspace.management.forgetRepository", { name: repository.name, defaultValue: `Forget ${repository.name}` })}
                        disabled={!!pending}
                        onClick={() => void forget(repository)}
                        className="h-8 rounded-control px-2 text-[11px] text-destructive hover:text-destructive"
                      >
                        {pending === `repository-forget:${repository.name}`
                          ? <Loader2 className="h-4 w-4 animate-spin" />
                          : t("workspace.management.forget", { defaultValue: "Forget" })}
                      </Button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  );
}

function BrowseButton({
  onPickFolder,
  onSelect,
  onError,
}: {
  onPickFolder?: () => Promise<string | null>;
  onSelect: (path: string) => void;
  onError?: (message: string) => void;
}) {
  const { t } = useTranslation();
  const available = !!(getRuntimeFolderPicker() ?? onPickFolder);
  return (
    <Button
      type="button"
      variant="outline"
      disabled={!available}
      title={available
        ? undefined
        : t("workspace.dialog.browseUnavailable", { defaultValue: "Browsing is unavailable here. Paste a path instead." })}
      onClick={() => {
        const chooser = getRuntimeFolderPicker() ?? onPickFolder;
        if (!chooser) return;
        void chooser().then((path) => {
          if (path) onSelect(path);
        }).catch((reason: unknown) => {
          onError?.(errorMessage(reason));
        });
      }}
      className="h-9 shrink-0 rounded-control px-3 text-[12px]"
    >
      {t("workspace.dialog.browse", { defaultValue: "Browse…" })}
    </Button>
  );
}

function getRuntimeFolderPicker(): (() => Promise<string | null>) | undefined {
  try {
    return getRuntimeHost().pickFolder;
  } catch {
    return undefined;
  }
}
