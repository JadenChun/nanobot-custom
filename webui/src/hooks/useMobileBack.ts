import { useEffect } from "react";

/**
 * Android WebView Back and Escape close the top mobile surface first. Return
 * true from the handler when the event was consumed.
 */
export function useMobileBack(handler: () => boolean): void {
  useEffect(() => {
    const onPopState = () => {
      handler();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (handler()) event.preventDefault();
    };
    window.addEventListener("popstate", onPopState);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("popstate", onPopState);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [handler]);
}
