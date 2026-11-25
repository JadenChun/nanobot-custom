import { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./globals.css";
import "./styles/mobile-shell.css";
import { initializeI18n } from "./i18n";
import type { MobilePairingBridge } from "./lib/mobile-pairing-bridge";
import { initializeLoopbackRuntimeHost, isLinuxWebKitRuntime } from "./lib/runtime";
import {
  createMobileRuntime,
  installMobileRuntime,
  recoverMobileRuntime,
  isMobileRuntimeActive,
  shouldRegisterServiceWorker,
  type MobileTransport,
} from "./lib/mobile-runtime";

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
const appRoot: HTMLElement = root;
const mobilePairing: MobilePairingBridge | undefined = window.kohakuMobilePairing;
let appReady = false;

const injectedMobileTransport = (window as Window & { nanobotMobileTransport?: MobileTransport })
  .nanobotMobileTransport;
if (injectedMobileTransport) {
  installMobileRuntime(createMobileRuntime(injectedMobileTransport));
}

// WebKitGTK (Tauri's Linux webview) renders filter/backdrop animations on the
// main thread. Mark the runtime so CSS can keep the layout while dropping the
// paint-heavy motion that makes streaming and scrolling stutter.
document.documentElement.classList.toggle("webkit-runtime", isLinuxWebKitRuntime());

initializeLoopbackRuntimeHost();
if (isMobileRuntimeActive()) document.documentElement.classList.add("mobile-runtime");

if (isMobileRuntimeActive()) {
  const recover = () => {
    if (appReady && document.visibilityState === "visible") void recoverMobileRuntime();
  };
  document.addEventListener("visibilitychange", recover);
  window.addEventListener("pageshow", recover);
  window.addEventListener("online", recover);
  window.addEventListener("kohaku-native-resume", () => {
    // App and session listeners also use the legacy event.
    if (appReady) window.dispatchEvent(new Event("nanobot-native-resume"));
  });
  window.addEventListener("nanobot-native-resume", recover);
}

async function startWebui(): Promise<void> {
  await initializeI18n();
  /* StrictMode disabled: dev double-invokes state updaters; delta accumulation must stay pure — see useKohakuStream. */
  const reactRoot = ReactDOM.createRoot(appRoot);
  const showApp = () => {
    appReady = true;
    reactRoot.render(<ErrorBoundary><App /></ErrorBoundary>);
  };

  if (mobilePairing) {
    const MobileAuthGate = lazy(() => import("./components/mobile/MobileAuthGate"));
    reactRoot.render(
      <ErrorBoundary>
        <Suspense fallback={null}>
          <MobileAuthGate pairing={mobilePairing} transport={injectedMobileTransport} onReady={showApp} />
        </Suspense>
      </ErrorBoundary>,
    );
  } else {
    showApp();
  }
}

void startWebui();

if ("serviceWorker" in navigator && shouldRegisterServiceWorker()) {
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
