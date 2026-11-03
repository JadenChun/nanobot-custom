import { createContext, useContext } from "react";

/** True while the composer's secondary chrome should stay out of the way. */
const ThreadComposerChromeHiddenContext = createContext(false);

export const ThreadComposerChromeHiddenProvider = ThreadComposerChromeHiddenContext.Provider;

export function useThreadComposerChromeHidden(): boolean {
  return useContext(ThreadComposerChromeHiddenContext);
}
