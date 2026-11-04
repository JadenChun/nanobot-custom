import { useEffect, useRef } from "react";

import { registerBackHandler } from "@/lib/back-stack";

/**
 * Android WebView Back and Escape close the top mobile surface first. Return
 * true from the handler when the event was consumed.
 *
 * Handlers are registered only while their surface is `active`, ordered by
 * mount/open time, so a preview or lightbox opened above the shell closes
 * before the shell-level navigation does.
 */
export function useMobileBack(handler: () => boolean, active = true): void {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    if (!active) return;
    return registerBackHandler(() => handlerRef.current());
  }, [active]);
}
