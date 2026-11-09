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
  const sessionKeyRef = useRef(sessionKey);
  sessionKeyRef.current = sessionKey;

  useEffect(() => {
    setGroups([]);
    setTotal(0);
    setNext(null);
    setLoadFailed(false);
    setLoadingMore(false);
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
    const requestedSession = sessionKey;
    setLoadingMore(true);
    void fetchSessionGeneratedAssets(tokenRef.current, requestedSession, "", {
      cursor,
      limit: PAGE_SIZE,
    })
      .then((payload) => {
        // A late page may belong to a previous session or cursor. Never append
        // it to the current selection.
        if (
          sessionKeyRef.current !== requestedSession
          || nextRef.current !== cursor
        ) return;
        setGroups((current) => [...current, ...payload.groups]);
        setTotal(payload.total);
        setNext(payload.next ?? null);
      })
      .catch(() => {
        // Keep the current page; the user can retry with the same cursor.
      })
      .finally(() => {
        if (sessionKeyRef.current === requestedSession) {
          setLoadingMore(false);
        }
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
