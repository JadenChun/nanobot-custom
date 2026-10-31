import { useEffect, useState } from "react";

import { getMobileRuntime } from "./mobile-runtime";

function decodeBase64(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

/**
 * Resolve signed WebUI media through the mobile broker and expose a local blob
 * URL to the WebView. Browser/desktop URLs pass through unchanged.
 */
export function useMobileMediaUrl(url: string | undefined): string | undefined {
  const [resolved, setResolved] = useState<string | undefined>(url);
  useEffect(() => {
    const runtime = getMobileRuntime();
    const match = url?.match(/^\/api\/media\/([A-Za-z0-9_-]+)\/([A-Za-z0-9_-]+)$/);
    if (!runtime || !match) {
      setResolved(url);
      return;
    }
    let cancelled = false;
    let objectUrl: string | null = null;
    void runtime.request({
      operation: "mobile_media",
      method: "GET",
      path: `/mobile/v1/media/${match[1]}/${match[2]}`,
    }).then((response) => {
      if (cancelled || response.status < 200 || response.status >= 300) return;
      const blob = new Blob([decodeBase64(response.bodyBase64).buffer as ArrayBuffer], {
        type: response.headers["content-type"] ?? "application/octet-stream",
      });
      objectUrl = URL.createObjectURL(blob);
      setResolved(objectUrl);
    }).catch(() => {
      if (!cancelled) setResolved(undefined);
    });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);
  return resolved;
}
