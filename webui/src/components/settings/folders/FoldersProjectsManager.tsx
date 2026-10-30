import { useMemo, useRef, useState, type FormEvent } from "react";
import { Folder, FolderPlus, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ApiError, editFolderLocation, type FolderLocationEdit } from "@/lib/api";
import { folderAccessErrorMessage } from "@/lib/folderAccess";
import { folderLocationRows, folderOperationId, pathContains, type FolderLocationRow } from "@/lib/folderLocations";
import { getRuntimeHost } from "@/lib/runtime";
import type { RepositoryRecord, WorkspacesPayload } from "@/lib/types";
import { isAbsoluteWorkspacePath, projectNameFromPath } from "@/lib/workspace";
import { useClient } from "@/providers/ClientProvider";
import { cn } from "@/lib/utils";

interface FolderForm {
  path: string;
  label: string;
  purpose: string;
  agentMap: boolean;
  access: "inherit" | "read_only" | "read_write";
  original: RepositoryRecord | null;
  editing: boolean;
  kind: "project" | "knowledge";
  links: string;
  autoSync: boolean;
  syncPaths: string;
}

export function FoldersProjectsManager({ workspaces, onRefreshWorkspaces, onPickFolder, className }: {
  workspaces: WorkspacesPayload | null;
  onRefreshWorkspaces: () => Promise<void>;
  onPickFolder?: () => Promise<string | null>;
  className?: string;
}) {
  const { t } = useTranslation();
  const text = (key: string, fallback: string) => t(`workspace.folders.${key}`, { defaultValue: fallback });
  const { client } = useClient();
  const [form, setForm] = useState<FolderForm | null>(null);
  const [removing, setRemoving] = useState<FolderLocationRow | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const retry = useRef<FolderLocationEdit | null>(null);
  const management = workspaces?.management;
  const rows = useMemo(() => management ? folderLocationRows(management) : [], [management]);
  const canManage = workspaces?.controls.can_manage_folders && workspaces?.controls.can_manage_catalog;
  let picker = onPickFolder;
  if (!picker) {
    try { picker = getRuntimeHost().pickFolder; } catch { /* No local runtime picker. */ }
  }

  const change = (update: Partial<FolderForm>) => {
    retry.current = null;
    setForm((current) => current ? { ...current, ...update } : null);
  };
  const openForm = (row?: FolderLocationRow) => {
    const entry = row?.entry;
    retry.current = null;
    setError(null); setNotice(null); setRemoving(null);
    setForm({ path: row?.path ?? "", label: entry?.label || entry?.name || "", purpose: entry?.purpose ?? "",
      agentMap: entry ? entry.agent_map !== false : !row, access: row?.grant?.file_access ?? "inherit",
      original: entry ?? null, editing: !!row, kind: entry?.kind ?? "project",
      links: entry?.knowledge_repos.join(", ") ?? "", autoSync: entry?.auto_sync ?? false,
      syncPaths: entry?.sync_paths.join("\n") ?? "" });
  };
  const apply = async (edit: FolderLocationEdit) => {
    setPending(true); setError(null); setNotice(null);
    try {
      await editFolderLocation(client, edit);
      await onRefreshWorkspaces();
      retry.current = null; setForm(null); setRemoving(null);
      setNotice(text("saved", "Folder saved. The map and Sandbox rules are up to date."));
    } catch (reason) {
      if (reason instanceof ApiError && ["grants_changed", "catalog_changed"].includes(reason.message)) {
        await onRefreshWorkspaces();
        retry.current = null;
      }
      setError(folderAccessErrorMessage(reason, t));
    } finally { setPending(false); }
  };
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (!form || !management) return;
    const path = form.path.trim();
    if (!isAbsoluteWorkspacePath(path)) {
      setError(t("workspace.dialog.absolutePathRequired")); return;
    }
    const label = form.label.trim() || projectNameFromPath(path);
    const base = label.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 56) || "folder";
    let name = form.original?.name ?? base;
    let suffix = 2;
    while (!form.original && management.repositories.some((row) => row.name === name && row.path !== path)) name = `${base}-${suffix++}`;
    const entry: RepositoryRecord = {
      name, label, path, purpose: form.purpose.trim(), agent_map: form.agentMap,
      kind: form.kind, knowledge_repos: form.links.split(",").map((link) => link.trim()).filter(Boolean),
      auto_sync: form.autoSync, sync_paths: form.syncPaths.split("\n").map((pattern) => pattern.trim()).filter(Boolean),
    };
    const edit: FolderLocationEdit = retry.current ?? {
      operation_id: folderOperationId(), expected_grants_revision: management.grants_revision,
      expected_catalog_revision: management.catalog_revision ?? 0, path, entry, access: form.access,
    };
    retry.current = edit;
    await apply(edit);
  };
  const remove = async () => {
    if (!management || !removing) return;
    const edit: FolderLocationEdit = retry.current ?? {
      operation_id: folderOperationId(), expected_grants_revision: management.grants_revision,
      expected_catalog_revision: management.catalog_revision ?? 0, path: removing.path, entry: null, access: "inherit",
    };
    retry.current = edit;
    await apply(edit);
  };
  const inherited = (path: string) => {
    const parent = management?.grants.filter((grant) => grant.path !== path && pathContains(grant.path, path))
      .sort((a, b) => b.path.length - a.path.length)[0];
    const agentHome = workspaces?.agent_home ?? workspaces?.default_scope.project_path;
    const inWorkspace = !!agentHome && pathContains(agentHome, path);
    const parentAtLeastAsSpecific = !!parent && !!agentHome && pathContains(agentHome, parent.path);
    if (parent && (!inWorkspace || parentAtLeastAsSpecific)) {
      const label = parent.effective_access.write ? text("readWrite", "Read and write") : text("readOnly", "Read-only");
      return `${label} · ${parent.path}`;
    }
    if (inWorkspace) {
      return text("agentWorkspaceWrite", "Read and write · Agent workspace");
    }
    return text("defaultReadOnly", "Read-only by default");
  };

  return <div className={cn("space-y-4", className)}>
    <div className="flex items-start justify-between gap-3">
      <div><h3 className="text-sm font-semibold">{text("title", "Folders")}</h3>
        <p className="mt-1 max-w-2xl text-xs leading-5 text-muted-foreground">{text("description", "Stored Sandbox rules control writes. Agent map shortcuts help navigation. Full access uses normal OS-account access.")}</p>
      </div>
      <div className="flex shrink-0 gap-2">
        <Button variant="outline" size="sm" disabled={pending} onClick={() => void onRefreshWorkspaces()} aria-label={text("refresh", "Refresh folders")}><RefreshCw className="size-3.5" /></Button>
        <Button size="sm" disabled={!canManage || pending} onClick={() => openForm()}><FolderPlus className="mr-1.5 size-3.5" />{text("add", "Add folder")}</Button>
      </div>
    </div>
    <p className="text-xs text-muted-foreground">{text("policy", "Sandbox can read ordinary files and run commands. Writes stay within approved folders and private temporary/cache storage. Running commands keep their launch access until they finish.")}</p>
    {!canManage && <p role="status" className="text-sm text-muted-foreground">{text("unavailable", "Folder management is available only in the local app.")}</p>}
    {error && <p role="alert" className="rounded-control border border-destructive/30 p-3 text-xs text-destructive">{error}</p>}
    {notice && <p role="status" className="text-xs text-emerald-700 dark:text-emerald-300">{notice}</p>}
    {form && <form onSubmit={(event) => void save(event)} className="space-y-3 rounded-xl border bg-muted/20 p-4">
      <h4 className="text-sm font-medium">{form.editing ? text("edit", "Edit folder") : text("add", "Add folder")}</h4>
      <div className="block space-y-1 text-xs"><label htmlFor="settings-folder-path">{text("path", "Folder path")}</label>
        <div className="flex gap-2"><Input id="settings-folder-path" value={form.path} disabled={form.editing || pending} onChange={(event) => change({ path: event.target.value })} required aria-label={text("path", "Folder path")} />
          {picker && !form.editing && <Button type="button" variant="outline" disabled={pending} onClick={() => void picker!().then((path) => { if (path) change({ path }); }).catch((reason: unknown) => setError(folderAccessErrorMessage(reason, t)))}>{t("workspace.dialog.browse", { defaultValue: "Browse…" })}</Button>}
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1 text-xs">{text("name", "Name")}<Input value={form.label} maxLength={80} disabled={pending} onChange={(event) => change({ label: event.target.value })} /></label>
        <label className="space-y-1 text-xs">{text("purpose", "Purpose")}<Input value={form.purpose} maxLength={240} disabled={pending} onChange={(event) => change({ purpose: event.target.value })} /></label>
      </div>
      <label className="block space-y-1 text-xs">{text("access", "Sandbox access")}
        <select className="h-9 w-full rounded-md border bg-background px-3 text-sm" value={form.access} disabled={pending} onChange={(event) => change({ access: event.target.value as FolderForm["access"] })}>
          <option value="inherit">{text("inherit", "Inherit/default")}</option><option value="read_only">{text("readOnly", "Read-only")}</option><option value="read_write">{text("readWrite", "Read and write")}</option>
        </select>
      </label>
      <p className="break-all text-xs text-muted-foreground">{text("fallback", "Inherited/default access")}: {inherited(form.path)}</p>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={form.agentMap} disabled={pending} onChange={(event) => change({ agentMap: event.target.checked })} />{text("map", "Include in Agent map")}</label>
      <p className="text-xs text-muted-foreground">{text("mapHelp", "The map shares only the name, path, purpose and effective access. Files and project instructions are read when needed.")}</p>
      <details><summary className="cursor-pointer text-xs text-muted-foreground">{text("knowledge", "Knowledge and Git sync")}</summary><div className="mt-3 space-y-3">
        <label className="block text-xs">{text("kind", "Folder role")}<select className="ml-3 rounded border bg-background p-1" value={form.kind} disabled={pending} onChange={(event) => change({ kind: event.target.value as FolderForm["kind"] })}><option value="project">{text("project", "Project")}</option><option value="knowledge">{text("knowledgeFolder", "Knowledge")}</option></select></label>
        <label className="block space-y-1 text-xs">{text("links", "Knowledge folder IDs (comma-separated)")}<Input value={form.links} disabled={pending} onChange={(event) => change({ links: event.target.value })} /></label>
        <label className="flex gap-2 text-xs"><input type="checkbox" checked={form.autoSync} disabled={pending} onChange={(event) => change({ autoSync: event.target.checked })} />{text("sync", "Automatically sync selected paths in this Git knowledge folder")}</label>
        {form.autoSync && <label className="block space-y-1 text-xs">{text("syncPaths", "Sync paths (one per line)")}<textarea className="w-full rounded-md border bg-background p-2" value={form.syncPaths} disabled={pending} onChange={(event) => change({ syncPaths: event.target.value })} /></label>}
      </div></details>
      <div className="flex gap-2"><Button type="submit" size="sm" disabled={pending}>{pending && <Loader2 className="mr-2 size-3 animate-spin" />}{text("save", "Save folder")}</Button><Button type="button" variant="ghost" size="sm" disabled={pending} onClick={() => { setForm(null); retry.current = null; }}>{t("common.cancel", { defaultValue: "Cancel" })}</Button></div>
    </form>}
    {removing && <div className="space-y-3 rounded-xl border border-destructive/30 p-4">
      <p className="break-all text-sm">{text("removeExplain", "Remove this folder's saved metadata and its own explicit Sandbox rule? Parent rules remain.")} {removing.path}</p>
      <p className="break-all text-xs text-muted-foreground">{text("afterRemoval", "Access after removal")}: {inherited(removing.path)}</p>
      <Button variant="destructive" size="sm" disabled={pending} onClick={() => void remove()}>{text("confirmRemove", "Remove folder")}</Button> <Button variant="ghost" size="sm" disabled={pending} onClick={() => setRemoving(null)}>{t("common.cancel", { defaultValue: "Cancel" })}</Button>
    </div>}
    <div className="divide-y overflow-hidden rounded-xl border">
      {rows.length === 0 && <p className="p-5 text-sm text-muted-foreground">{text("empty", "No saved folders yet. Add a folder for access, navigation, or both.")}</p>}
      {rows.map((row) => {
        const access = row.entry?.effective_access ?? row.grant?.effective_access;
        const mapped = !!row.entry && row.entry.agent_map !== false;
        const name = row.entry?.label || row.entry?.name || projectNameFromPath(row.path);
        const source = rows.find((candidate) => candidate.path === access?.grant_root);
        const sourceLabel = source?.entry?.label || source?.entry?.name || (source ? projectNameFromPath(source.path) : access?.grant_root);
        return <div key={row.path} className="flex min-w-0 items-start gap-3 py-4 pr-3" style={{ paddingLeft: `${16 + Math.min(row.depth, 8) * 20}px` }} data-folder-path={row.path} data-depth={row.depth}>
          <Folder className="mt-1 size-4 shrink-0 text-muted-foreground" />
          <div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><span className="text-sm font-medium">{name}</span>
            {row.entry?.git && <span className="rounded border px-1.5 text-[10px] text-muted-foreground">Git</span>}
            {mapped && <span className="rounded border px-1.5 text-[10px] text-muted-foreground">{text("mapBadge", "Agent map")}</span>}
            {!access?.exists && <span className="text-xs text-amber-700">{text("missing", "Folder missing or moved")}</span>}
          </div><p className="mt-1 break-all font-mono text-[11px] text-muted-foreground">{row.path}</p>
            {row.entry?.purpose && <p className="mt-1 text-xs">{row.entry.purpose}</p>}
            <p title={access?.grant_root ?? undefined} className="mt-2 break-all text-xs text-muted-foreground">{access?.write ? text("readWrite", "Read and write") : text("readOnly", "Read-only")} · {row.grant && access?.source === "grant" ? text("ownRule", "Own Sandbox rule") : access?.source === "agent_home" ? text("agentWorkspaceDefault", "Agent workspace default") : access?.grant_root ? `${text("inheritedFrom", "Inherited from")} ${sourceLabel}` : text("defaultReadOnly", "Read-only by default")}</p>
            {access?.commands.requested && !access.commands.available && <p className="mt-1 text-xs text-amber-700">{t(`workspace.management.commandReasons.${access.commands.reason}`, { defaultValue: "Sandbox commands unavailable" })}</p>}
          </div>
          <Button size="sm" variant="ghost" disabled={!canManage || pending} aria-label={`${text("edit", "Edit folder")} ${name}`} onClick={() => openForm(row)}>{text("editShort", "Edit")}</Button>
          <Button size="sm" variant="ghost" disabled={!canManage || pending} aria-label={`${text("remove", "Remove")} ${name}`} onClick={() => { setRemoving(row); setForm(null); retry.current = null; setError(null); }}><Trash2 className="size-3.5" /></Button>
        </div>;
      })}
    </div>
  </div>;
}
