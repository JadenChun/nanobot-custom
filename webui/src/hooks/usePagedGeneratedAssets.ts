import { useCallback, useEffect, useRef, useState } from "react";

import type { GeneratedAssetGroup } from "@/lib/types";

export const GENERATED_ASSET_PAGE_SIZE = 50;

export interface GeneratedAssetPage {
  groups: GeneratedAssetGroup[];
  total: number;
  totalScope?: "page";
  next?: string | null;
}

export type GeneratedAssetPageFetcher = (
  token: string,
  options: { cursor?: string; limit: number },
) => Promise<GeneratedAssetPage>;

/**
 * Shared paginated loader for generated-asset views.
 *
 * Every request belongs to a generation. A generation ends when the view
 * closes, the credentials/scope change, or a refresh starts, so late pages from
 * an older view can never mutate newer state. In-flight flags are refs so two
 * clicks in one event tick cannot issue duplicate requests.
 */
export function usePagedGeneratedAssets(
  enabled: boolean,
  token: string,
  scopeKey: string,
  refreshKey: string | number,
  fetchPage: GeneratedAssetPageFetcher,
) {
  const [groups, setGroups] = useState<GeneratedAssetGroup[]>([]);
  const [total, setTotal] = useState(0);
  const [next, setNext] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);
  const [manualRefresh, setManualRefresh] = useState(0);

  const tokenRef = useRef(token);
  tokenRef.current = token;
  const fetchRef = useRef(fetchPage);
  fetchRef.current = fetchPage;
  const generationRef = useRef(0);
  const nextRef = useRef<string | null>(null);
  const loadingRef = useRef(false);
  const loadingMoreRef = useRef(false);

  // Drop data that belongs to a different scope or a closed view.
  useEffect(() => {
    generationRef.current += 1;
    nextRef.current = null;
    loadingRef.current = false;
    loadingMoreRef.current = false;
    setGroups([]);
    setTotal(0);
    setNext(null);
    setLoading(false);
    setLoadingMore(false);
    setLoadFailed(false);
  }, [enabled, scopeKey, token]);

  useEffect(() => {
    if (!enabled) return;
    const generation = ++generationRef.current;
    loadingMoreRef.current = false;
    loadingRef.current = true;
    setLoadingMore(false);
    setLoading(true);
    setLoadFailed(false);

    void fetchRef
      .current(tokenRef.current, { limit: GENERATED_ASSET_PAGE_SIZE })
      .then((payload) => {
        if (generation !== generationRef.current) return;
        nextRef.current = payload.next ?? null;
        setGroups(payload.groups);
        setTotal(payload.totalScope === "page"
          ? groupsTotal(payload.groups)
          : payload.total);
        setNext(payload.next ?? null);
      })
      .catch(() => {
        if (generation === generationRef.current) setLoadFailed(true);
      })
      .finally(() => {
        if (generation !== generationRef.current) return;
        loadingRef.current = false;
        setLoading(false);
      });

    return () => {
      // Invalidate in-flight requests when the dependencies change or unmount.
      if (generation === generationRef.current) generationRef.current += 1;
    };
  }, [enabled, scopeKey, token, refreshKey, manualRefresh]);

  const refresh = useCallback(() => {
    setManualRefresh((value) => value + 1);
  }, []);

  const loadMore = useCallback(() => {
    const cursor = nextRef.current;
    if (!cursor || loadingRef.current || loadingMoreRef.current) return;
    const generation = generationRef.current;
    loadingMoreRef.current = true;
    setLoadingMore(true);
    void fetchRef
      .current(tokenRef.current, { cursor, limit: GENERATED_ASSET_PAGE_SIZE })
      .then((payload) => {
        if (generation !== generationRef.current) return;
        nextRef.current = payload.next ?? null;
        setGroups((current) => [...current, ...payload.groups]);
        setTotal((current) => payload.totalScope === "page"
          ? current + groupsTotal(payload.groups)
          : payload.total);
        setNext(payload.next ?? null);
      })
      .catch(() => {
        // Keep the current page and cursor; the user can retry.
      })
      .finally(() => {
        if (generation !== generationRef.current) return;
        loadingMoreRef.current = false;
        setLoadingMore(false);
      });
  }, []);

  return { groups, total, next, loading, loadingMore, loadFailed, refresh, loadMore };
}

function groupsTotal(groups: GeneratedAssetGroup[]) {
  return groups.reduce((count, group) => count + group.artifacts.length, 0);
}
