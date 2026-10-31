import { useTranslation } from "react-i18next";

import { isMobileRuntimeActive } from "@/lib/mobile-runtime";
import type { ConnectionStatus } from "@/lib/types";

export function MobileConnectionNotice({ status }: { status: ConnectionStatus }) {
  const { t } = useTranslation();
  if (!isMobileRuntimeActive()) return null;
  if (status === "open" || status === "idle") return null;
  const message = status === "error"
    ? t("connection.error")
    : status === "reconnecting"
      ? t("connection.reconnecting")
      : status === "connecting"
        ? t("connection.connecting")
        : t("connection.closed");
  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 top-0 z-50 flex justify-center px-3 pt-[calc(env(safe-area-inset-top,0px)+0.5rem)]"
    >
      <div className="rounded-full border bg-background/95 px-3 py-1 text-xs font-medium shadow-sm">
        {message}
      </div>
    </div>
  );
}
