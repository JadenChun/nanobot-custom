import type { FolderGrantRow, RepositoryRow, WorkspaceManagementPayload } from "@/lib/types";
import { normalizeWorkspacePath } from "@/lib/workspace";

export interface FolderLocationRow {
  path: string;
  grant?: FolderGrantRow;
  entry?: RepositoryRow;
  depth: number;
}

/** Paths are canonicalized by the server. Group saved entries, never scan disks. */
export function folderLocationRows(management: WorkspaceManagementPayload): FolderLocationRow[] {
  const byPath = new Map<string, FolderLocationRow>();
  for (const grant of management.grants) {
    const path = normalizeWorkspacePath(grant.path);
    byPath.set(path, { path, grant, depth: 0 });
  }
  for (const entry of management.repositories) {
    const path = normalizeWorkspacePath(entry.path);
    byPath.set(path, { ...byPath.get(path), path, entry, depth: 0 });
  }
  const rows = [...byPath.values()].sort((a, b) => a.path.localeCompare(b.path));
  const children = new Map<string | null, FolderLocationRow[]>();
  for (const row of rows) {
    const parent = rows.filter((candidate) => candidate.path !== row.path && pathContains(candidate.path, row.path))
      .sort((a, b) => b.path.length - a.path.length)[0];
    const key = parent?.path ?? null;
    children.set(key, [...(children.get(key) ?? []), row]);
  }
  const result: FolderLocationRow[] = [];
  function visit(parent: string | null, depth: number) {
    for (const row of children.get(parent) ?? []) {
      result.push({ ...row, depth });
      visit(row.path, depth + 1);
    }
  }
  visit(null, 0);
  return result;
}

export function pathContains(parent: string, child: string): boolean {
  const root = normalizeWorkspacePath(parent);
  const target = normalizeWorkspacePath(child);
  return target === root || target.startsWith(root === "/" ? "/" : `${root}/`);
}

export function folderOperationId(): string {
  return crypto.randomUUID().replaceAll("-", "");
}
