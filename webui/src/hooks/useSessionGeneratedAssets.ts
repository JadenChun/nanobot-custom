import { useCallback } from "react";

import { fetchSessionGeneratedAssets } from "@/lib/api";
import { usePagedGeneratedAssets } from "@/hooks/usePagedGeneratedAssets";

export function useSessionGeneratedAssets(
  open: boolean,
  token: string,
  sessionKey: string,
  refreshKey: string | number = 0,
) {
  const fetchPage = useCallback(
    (authToken: string, options: { cursor?: string; limit: number }) =>
      fetchSessionGeneratedAssets(authToken, sessionKey, "", options),
    [sessionKey],
  );
  return usePagedGeneratedAssets(
    open && !!sessionKey,
    token,
    sessionKey,
    refreshKey,
    fetchPage,
  );
}
