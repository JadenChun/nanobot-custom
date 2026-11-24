import { useEffect, useState } from "react";
import { Loader2, ShieldAlert } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import type { ApprovalDecisionPayload } from "@/lib/types";

export interface ApprovalRequestCardProps {
  summary: string;
  approvalId?: string | null;
  contextLabel?: string;
  requestLabel?: string;
  description?: string;
  busy?: boolean;
  /** Increment to clear the local submitting spinner after a rejected send. */
  resetSignal?: number;
  saveExclusionCommand?: string;
  onApprove: () => void;
  onApproveAndSave?: () => void;
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
  resetSignal = 0,
  saveExclusionCommand,
  onApprove,
  onApproveAndSave,
  onCancel,
}: ApprovalRequestCardProps) {
  const { t } = useTranslation();
  const [pending, setPending] = useState<ApprovalDecisionPayload["decision"] | null>(null);
  useEffect(() => {
    setPending(null);
  }, [approvalId, resetSignal]);
  const disabled = busy || pending !== null || !approvalId;
  const canSave = Boolean(saveExclusionCommand && onApproveAndSave);
  const approvalDescription = description ?? (approvalId
    ? t("approval.description", {
        defaultValue:
          "Kohaku paused before a risky action. Review the planned action before continuing.",
      })
    : t("approval.updateRequired", {
        defaultValue:
          "This approval request needs an updated client. Chat text cannot grant it.",
      }));

  const submit = (action: ApprovalDecisionPayload["decision"]) => {
    if (disabled) return;
    setPending(action);
    if (action === "approve") onApprove();
    else if (action === "approve_and_save") onApproveAndSave?.();
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
          onClick={() => submit("deny")}
        >
          {pending === "deny" ? (
            <Loader2 className="mr-2 size-4 animate-spin" aria-hidden />
          ) : null}
          {canSave ? t("approval.deny", { defaultValue: "Deny" }) : t("approval.cancel", { defaultValue: "Cancel" })}
        </Button>
        <Button type="button" disabled={disabled} onClick={() => submit("approve")}>
          {pending === "approve" ? (
            <Loader2 className="mr-2 size-4 animate-spin" aria-hidden />
          ) : null}
          {canSave ? t("approval.allowOnce", { defaultValue: "Allow once" }) : t("approval.approve", { defaultValue: "Approve and continue" })}
        </Button>
        {canSave ? (
          <Button type="button" disabled={disabled} onClick={() => submit("approve_and_save")}>
            {pending === "approve_and_save" ? (
              <Loader2 className="mr-2 size-4 animate-spin" aria-hidden />
            ) : null}
            {t("approval.allowAndSave", {
              defaultValue: "Allow & save {{command}}",
              command: saveExclusionCommand,
            })}
          </Button>
        ) : null}
      </div>
    </section>
  );
}
