import ReactDOM from "react-dom/client";

import App from "./App";
import "./globals.css";
import { initializeI18n } from "./i18n";
import { initializeLoopbackRuntimeHost } from "./lib/runtime";

// `crypto.randomUUID` is only defined in secure contexts (HTTPS or localhost).
// LAN access over plain HTTP leaves it undefined, which crashes components that
// generate client-side message IDs. Shim a v4-ish fallback so call sites stay
// uniform across secure and non-secure contexts.
if (typeof globalThis.crypto !== "undefined" && !("randomUUID" in globalThis.crypto)) {
  Object.defineProperty(globalThis.crypto, "randomUUID", {
    value: () =>
      "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
      }),
    configurable: true,
  });
}

const root = document.getElementById("root");
if (!root) throw new Error("root element missing");

// WebKitGTK (Tauri's Linux webview) renders filter/backdrop animations on the
// main thread. Mark the runtime so CSS can keep the layout while dropping the
// paint-heavy motion that makes streaming and scrolling stutter.
const isWebKitRuntime = /AppleWebKit/i.test(navigator.userAgent)
  && /Linux/i.test(navigator.userAgent)
  && !/Chrome|Chromium|Edg\/|OPR\//i.test(navigator.userAgent);
document.documentElement.classList.toggle("webkit-runtime", isWebKitRuntime);

initializeLoopbackRuntimeHost();

async function renderWebui(container: HTMLElement) {
  await initializeI18n();
  /* StrictMode disabled: dev double-invokes state updaters; delta accumulation must stay pure — see useNanobotStream. */
  ReactDOM.createRoot(container).render(<App />);
}

void renderWebui(root);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js", {
        updateViaCache: "none",
      })
      .catch(() => {
        // Service workers are progressive enhancement; registration failures
        // (unsupported proxies, blocked storage) must not break the app.
      });
  });
}
