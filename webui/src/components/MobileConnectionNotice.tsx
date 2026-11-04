import { useTranslation } from "react-i18next";

import { isMobileRuntimeActive } from "@/lib/mobile-runtime";
import type { ConnectionStatus } from "@/lib/types";

export function MobileConnectionNotice({
  status,
  onRetry,
}: {
  status: ConnectionStatus;
  onRetry?: () => void;
}) {
  const { t } = useTranslation();
  if (!isMobileRuntimeActive()) return null;
  if (status === "open" || status === "idle") return null;
  const actionable = status === "error" || status === "closed" || status === "reconnecting";
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
      <div className="pointer-events-auto flex items-center gap-2 rounded-full border bg-background/95 px-3 py-1 text-xs font-medium shadow-sm">
        <span>{message}</span>
        {actionable && onRetry ? (
          <button
            type="button"
            className="rounded-full border border-border/60 px-2 py-0.5 font-semibold"
            onClick={onRetry}
          >
            {t("common.retry", { defaultValue: "Retry" })}
          </button>
        ) : null}
      </div>
    </div>
  );
}
