import { useEffect, useMemo, useState } from "react";
import { ChevronRight, Folder, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { fetchWithTimeout } from "@/lib/http";
import { useClient } from "@/providers/ClientProvider";

export interface ServerFolderRoot {
  id: string;
  label: string;
  display_path: string;
}

export interface ServerFolderEntry {
  name: string;
  directory_id: string;
  symlink: boolean;
  accessible: boolean;
}

export interface ServerFolderSelection {
  root_id: string;
  relative_path: string;
  display_path: string;
  policy_revision: string;
  commit_token: string;
}

interface RootsPayload {
  roots: ServerFolderRoot[];
  policy_revision: string;
  empty: boolean;
}

interface ListingPayload {
  root_id: string;
  relative_path: string;
  entries: ServerFolderEntry[];
  next_cursor: string | null;
  policy_revision: string;
}

export function parseRoots(value: unknown): RootsPayload | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (!Array.isArray(record.roots) || typeof record.policy_revision !== "string") return null;
  const roots = record.roots.filter((item): item is ServerFolderRoot => {
    if (!item || typeof item !== "object") return false;
    const row = item as Record<string, unknown>;
    return typeof row.id === "string" && typeof row.label === "string"
      && typeof row.display_path === "string";
  });
  return { roots, policy_revision: record.policy_revision, empty: record.empty === true || roots.length === 0 };
}

export function parseListing(value: unknown): ListingPayload | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (typeof record.root_id !== "string" || typeof record.relative_path !== "string"
    || !Array.isArray(record.entries) || typeof record.policy_revision !== "string") return null;
  const entries = record.entries.filter((item): item is ServerFolderEntry => {
    if (!item || typeof item !== "object") return false;
    const row = item as Record<string, unknown>;
    return typeof row.name === "string" && typeof row.directory_id === "string"
      && typeof row.symlink === "boolean" && typeof row.accessible === "boolean";
  });
  return {
    root_id: record.root_id,
    relative_path: record.relative_path,
    entries,
    next_cursor: typeof record.next_cursor === "string" ? record.next_cursor : null,
    policy_revision: record.policy_revision,
  };
}

export function RemoteFolderBrowserDialog({
  open,
  onOpenChange,
  onSelected,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelected: (selection: ServerFolderSelection) => Promise<void> | void;
}) {
  const { t } = useTranslation();
  const { client } = useClient();
  const [roots, setRoots] = useState<RootsPayload | null>(null);
  const [rootId, setRootId] = useState<string>("");
  const [path, setPath] = useState("");
  const [listing, setListing] = useState<ListingPayload | null>(null);
  const [manualPath, setManualPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const crumbs = useMemo(() => path ? path.split("/").filter(Boolean) : [], [path]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setRoots(null);
    setListing(null);
    setRootId("");
    setPath("");
    void fetchWithTimeout("/api/workspace-directories/roots")
      .then(async (response) => {
        if (!response.ok) throw new Error(`roots failed: HTTP ${response.status}`);
        const parsed = parseRoots(await response.json());
        if (!parsed) throw new Error("invalid roots response");
        if (cancelled) return;
        setRoots(parsed);
        if (parsed.roots.length === 1) {
          const only = parsed.roots[0].id;
          setRootId(only);
          void load(only, "");
        }
      })
      .catch((reason) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const load = async (nextRoot: string, nextPath: string) => {
    setLoading(true);
    setError(null);
    try {
      const query = new URLSearchParams({ root: nextRoot, path: nextPath, limit: "100" });
      const response = await fetchWithTimeout(`/api/workspace-directories/list?${query.toString()}`);
      if (!response.ok) throw new Error(`list failed: HTTP ${response.status}`);
      const parsed = parseListing(await response.json());
      if (!parsed) throw new Error("invalid listing response");
      setListing(parsed);
      setPath(parsed.relative_path);
      setManualPath(parsed.relative_path);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  };

  const loadMore = async () => {
    if (!listing?.next_cursor) return;
    setLoading(true);
    setError(null);
    try {
      const query = new URLSearchParams({
        root: rootId,
        path: listing.relative_path,
        limit: "100",
        cursor: listing.next_cursor,
      });
      const response = await fetchWithTimeout(`/api/workspace-directories/list?${query.toString()}`);
      if (!response.ok) throw new Error(`list failed: HTTP ${response.status}`);
      const parsed = parseListing(await response.json());
      if (!parsed) throw new Error("invalid listing response");
      setListing({
        ...parsed,
        entries: [...listing.entries, ...parsed.entries],
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  };

  const selectRoot = (nextRoot: string) => {
    setRootId(nextRoot);
    setPath("");
    void load(nextRoot, "");
  };

  const openEntry = (entry: ServerFolderEntry) => {
    if (!entry.accessible) return;
    const next = path ? `${path}/${entry.name}` : entry.name;
    void load(rootId, next);
  };

  const selectFolder = async () => {
    if (!listing) return;
    setBusy(true);
    setError(null);
    try {
      const response = await client.requestMutation<Record<string, unknown>>(
        "workspace.directories.resolve",
        { root: rootId, path: listing.relative_path, policy_revision: listing.policy_revision },
      );
      const token = response.commit_token;
      const display = response.display_path;
      if (typeof token !== "string" || typeof display !== "string") {
        throw new Error("invalid resolve response");
      }
      await onSelected({
        root_id: rootId,
        relative_path: listing.relative_path,
        display_path: display,
        policy_revision: listing.policy_revision,
        commit_token: token,
      });
      onOpenChange(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[min(88dvh,40rem)] w-[min(calc(100vw - 1rem),32rem)] overflow-y-auto p-4 sm:p-6">
        <DialogHeader className="pr-8 text-left">
          <DialogTitle>{t("workspace.remoteBrowser.title", { defaultValue: "Browse server folders" })}</DialogTitle>
          <DialogDescription>
            {t("workspace.remoteBrowser.description", {
              defaultValue: "Only folders exposed by the server's configured roots are visible.",
            })}
          </DialogDescription>
        </DialogHeader>

        {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
        {loading && !roots ? <Loader2 className="mx-auto size-5 animate-spin text-muted-foreground" /> : null}
        {roots?.empty ? (
          <p className="rounded-lg border p-3 text-sm text-muted-foreground">
            {t("workspace.remoteBrowser.empty", {
              defaultValue: "No server folder roots are configured. Add remote_access.directory_browser.roots on the server.",
            })}
          </p>
        ) : null}

        {roots && roots.roots.length > 1 && !listing ? (
          <div className="space-y-1">
            {roots.roots.map((root) => (
              <button
                key={root.id}
                type="button"
                onClick={() => selectRoot(root.id)}
                className="flex min-h-11 w-full items-center gap-2 rounded-md border px-3 text-left text-sm"
              >
                <Folder className="size-4 shrink-0" aria-hidden />
                <span className="min-w-0 flex-1 truncate">{root.label}</span>
                <ChevronRight className="size-4 shrink-0 text-muted-foreground" aria-hidden />
              </button>
            ))}
          </div>
        ) : null}

        {listing ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <input
                aria-label="Relative folder path"
                value={manualPath}
                onChange={(event) => setManualPath(event.target.value)}
                className="min-w-0 flex-1 rounded-md border bg-transparent px-2 py-1.5 text-xs"
                placeholder="relative/path"
              />
              <Button
                type="button"
                variant="outline"
                disabled={loading}
                onClick={() => void load(rootId, manualPath.replace(/^\/+|\/+$/g, ""))}
              >
                {t("workspace.remoteBrowser.open", { defaultValue: "Open" })}
              </Button>
            </div>
            <div className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
              <button type="button" className="underline-offset-2 hover:underline" onClick={() => selectRoot(rootId)}>
                {roots?.roots.find((root) => root.id === rootId)?.label ?? rootId}
              </button>
              {crumbs.map((crumb, index) => (
                <span key={`${crumb}-${index}`} className="flex items-center gap-1">
                  <ChevronRight className="size-3" aria-hidden />
                  <button
                    type="button"
                    className="underline-offset-2 hover:underline"
                    onClick={() => void load(rootId, crumbs.slice(0, index + 1).join("/"))}
                  >
                    {crumb}
                  </button>
                </span>
              ))}
            </div>
            <ul className="divide-y rounded-lg border">
              {listing.entries.map((entry) => (
                <li key={entry.directory_id}>
                  <button
                    type="button"
                    disabled={!entry.accessible}
                    onClick={() => openEntry(entry)}
                    className="flex min-h-11 w-full items-center gap-2 px-3 text-left text-sm disabled:opacity-50"
                  >
                    <Folder className="size-4 shrink-0" aria-hidden />
                    <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                    {entry.symlink ? <span className="text-[11px] text-muted-foreground">symlink</span> : null}
                    <ChevronRight className="size-4 shrink-0 text-muted-foreground" aria-hidden />
                  </button>
                </li>
              ))}
              {listing.entries.length === 0 ? (
                <li className="px-3 py-4 text-center text-xs text-muted-foreground">
                  {t("workspace.remoteBrowser.noSubfolders", { defaultValue: "No subfolders" })}
                </li>
              ) : null}
            </ul>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  disabled={loading}
                  onClick={() => void load(rootId, path)}
                >
                  {t("workspace.remoteBrowser.refresh", { defaultValue: "Refresh" })}
                </Button>
                {listing.next_cursor ? (
                  <Button
                    type="button"
                    variant="outline"
                    disabled={loading}
                    onClick={() => void loadMore()}
                  >
                    {t("workspace.remoteBrowser.more", { defaultValue: "Load more" })}
                  </Button>
                ) : null}
              </div>
              <Button type="button" disabled={busy} onClick={() => void selectFolder()}>
                {busy ? <Loader2 className="mr-1.5 size-4 animate-spin" /> : null}
                {t("workspace.remoteBrowser.select", { defaultValue: "Use this folder" })}
              </Button>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
