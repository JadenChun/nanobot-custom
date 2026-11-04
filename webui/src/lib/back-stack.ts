/**
 * Ordered stack of in-app back handlers.
 *
 * The Android shell calls `window.__nanobotHandleBack()` for a hardware back
 * press. Handlers are registered by mounted surfaces (sidebars, previews,
 * lightboxes) and run from the most recently mounted to the oldest, so the
 * topmost surface closes first. `Escape` and `popstate` use the same stack.
 */
type BackHandler = () => boolean;

const handlers: BackHandler[] = [];
let installed = false;

function runBackHandlers(): boolean {
  for (let index = handlers.length - 1; index >= 0; index -= 1) {
    if (handlers[index]()) return true;
  }
  return false;
}

function installBackListeners(): void {
  if (installed || typeof window === "undefined") return;
  installed = true;
  window.addEventListener("popstate", () => {
    runBackHandlers();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (runBackHandlers()) event.preventDefault();
  });
  (window as Window & { __nanobotHandleBack?: () => boolean }).__nanobotHandleBack =
    handleAppBack;
}

/** Run the topmost handler; true means an in-app surface consumed the press. */
export function handleAppBack(): boolean {
  if (typeof window === "undefined") return false;
  installBackListeners();
  return runBackHandlers();
}

/** Register a handler; returns the function that removes it again. */
export function registerBackHandler(handler: BackHandler): () => void {
  installBackListeners();
  handlers.push(handler);
  return () => {
    const index = handlers.lastIndexOf(handler);
    if (index >= 0) handlers.splice(index, 1);
  };
}
