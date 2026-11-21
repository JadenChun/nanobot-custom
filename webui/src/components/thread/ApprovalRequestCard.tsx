import { useState } from "react";
import { Loader2, ShieldAlert } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";

export interface ApprovalRequestCardProps {
  summary: string;
  approvalId?: string | null;
  contextLabel?: string;
  requestLabel?: string;
  description?: string;
  busy?: boolean;
  onApprove: () => void;
  onCancel: () => void;
}

/**
 * Inline approval card for the risky-action policy. The buttons send a typed
 * decision for the server-owned approval ID. Chat text alone cannot grant it.
 */
export function ApprovalRequestCard({
  summary,
  approvalId = null,
  contextLabel,
  requestLabel,
  description,
  busy = false,
  onApprove,
  onCancel,
}: ApprovalRequestCardProps) {
  const { t } = useTranslation();
  const [pending, setPending] = useState<"approve" | "cancel" | null>(null);
  const disabled = busy || pending !== null || !approvalId;
  const approvalDescription = description ?? (approvalId
    ? t("approval.description", {
        defaultValue:
          "Nanobot paused before a risky action. Review the planned action before continuing.",
      })
    : t("approval.updateRequired", {
        defaultValue:
          "This approval request needs an updated client. Chat text cannot grant it.",
      }));

  const submit = (action: "approve" | "cancel") => {
    if (disabled) return;
    setPending(action);
    if (action === "approve") onApprove();
    else onCancel();
  };

  return (
    <section
      className="mx-4 space-y-3 rounded-xl border bg-card p-4"
      aria-label={requestLabel ?? t("approval.title", { defaultValue: "Approval required" })}
    >
      <div className="flex items-center gap-2 text-sm font-medium">
        <ShieldAlert className="size-4" aria-hidden />
        {t("approval.title", { defaultValue: "Approval required" })}
      </div>
      <p className="text-xs leading-relaxed text-muted-foreground">
        {approvalDescription}
      </p>
      {contextLabel ? (
        <p className="text-xs font-medium text-foreground">{contextLabel}</p>
      ) : null}
      <div className="rounded-control bg-muted/35 px-3 py-2 text-[12px] leading-relaxed">
        <span className="font-medium">
          {t("approval.plannedAction", { defaultValue: "Planned action" })}:{" "}
        </span>
        <span className="break-words">{summary}</span>
      </div>
      <div className="flex flex-wrap justify-end gap-2">
        <Button
          type="button"
          variant="ghost"
          disabled={disabled}
          onClick={() => submit("cancel")}
        >
          {pending === "cancel" ? (
            <Loader2 className="mr-2 size-4 animate-spin" aria-hidden />
          ) : null}
          {t("approval.cancel", { defaultValue: "Cancel" })}
        </Button>
        <Button type="button" disabled={disabled} onClick={() => submit("approve")}>
          {pending === "approve" ? (
            <Loader2 className="mr-2 size-4 animate-spin" aria-hidden />
          ) : null}
          {t("approval.approve", { defaultValue: "Approve and continue" })}
        </Button>
      </div>
    </section>
  );
}
