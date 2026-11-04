import { useCallback, useEffect, useRef, useState } from "react";

import { fetchSessionGeneratedAssets } from "@/lib/api";
import type { GeneratedAssetGroup } from "@/lib/types";

const PAGE_SIZE = 50;

export function useSessionGeneratedAssets(
  open: boolean,
  token: string,
  sessionKey: string,
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
    setGroups([]);
    setTotal(0);
    setNext(null);
    setLoadFailed(false);
  }, [sessionKey]);

  useEffect(() => {
    if (!open || !sessionKey) return;
    let cancelled = false;
    setLoading(true);
    setLoadFailed(false);

    void fetchSessionGeneratedAssets(tokenRef.current, sessionKey, "", {
      limit: PAGE_SIZE,
    })
      .then((payload) => {
        if (cancelled) return;
        setGroups(payload.groups);
        setTotal(payload.total);
        setNext(payload.next ?? null);
        setLoadFailed(false);
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
  }, [open, sessionKey, token, refreshKey, manualRefresh]);

  const refresh = useCallback(() => {
    setManualRefresh((value) => value + 1);
  }, []);

  const loadMore = useCallback(() => {
    const cursor = nextRef.current;
    if (!cursor || loadingMoreRef.current) return;
    setLoadingMore(true);
    void fetchSessionGeneratedAssets(tokenRef.current, sessionKey, "", {
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
  }, [sessionKey]);

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
