import { useCallback, useEffect, useRef, useState } from "react";

import { fetchAllGeneratedAssets } from "@/lib/api";
import type { GeneratedAssetGroup } from "@/lib/types";

const PAGE_SIZE = 50;

export function useGeneratedAssetLibrary(
  active: boolean,
  token: string,
  refreshKey: string | number = 0,
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
  const nextRef = useRef(next);
  nextRef.current = next;
  const loadingMoreRef = useRef(loadingMore);
  loadingMoreRef.current = loadingMore;

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoading(true);
    setLoadFailed(false);
    void fetchAllGeneratedAssets(tokenRef.current, "", { limit: PAGE_SIZE })
      .then((payload) => {
        if (cancelled) return;
        setGroups(payload.groups);
        setTotal(payload.total);
        setNext(payload.next ?? null);
      })
      .catch(() => {
        if (!cancelled) setLoadFailed(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [active, token, refreshKey, manualRefresh]);

  const refresh = useCallback(() => {
    setManualRefresh((value) => value + 1);
  }, []);

  const loadMore = useCallback(() => {
    const cursor = nextRef.current;
    if (!cursor || loadingMoreRef.current) return;
    setLoadingMore(true);
    void fetchAllGeneratedAssets(tokenRef.current, "", {
      cursor,
      limit: PAGE_SIZE,
    })
      .then((payload) => {
        setGroups((current) => [...current, ...payload.groups]);
        setTotal(payload.total);
        setNext(payload.next ?? null);
      })
      .catch(() => {
        // Keep the current page; the user can retry with the same cursor.
      })
      .finally(() => {
        setLoadingMore(false);
      });
  }, []);

  return {
    groups,
    total,
    next,
    loading,
    loadingMore,
    loadFailed,
    refresh,
    loadMore,
  };
}
