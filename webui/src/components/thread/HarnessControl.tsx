import { useId } from "react";
import { useTranslation } from "react-i18next";
import { Check, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import type { ChatHarness } from "@/lib/types";

function KohakuIcon({ className }: { className?: string }) {
  return (
    <img
      src="/brand/kohaku_icon_transparent.png"
      alt=""
      aria-hidden
      draggable={false}
      className={cn("select-none object-contain", className)}
    />
  );
}

function ClaudeIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
      className={cn("text-[#D97757]", className)}
    >
      <path
        fill="currentColor"
        d="M16.92 15.17 13.73 13l4.64-.55a.84.84 0 0 0 .74-.93.85.85 0 0 0-.93-.74l-4.53.54 3.4-2.8a.84.84 0 1 0-1.07-1.3l-3.13 2.57 1.72-4.4a.84.84 0 1 0-1.57-.61l-1.56 3.99-.64-4.23a.84.84 0 1 0-1.66.25l.65 4.26-2.87-3.42a.84.84 0 0 0-1.29 1.08l2.7 3.22-4.38-1.63a.84.84 0 0 0-.59 1.58l4.02 1.5-4.39.52a.84.84 0 1 0 .2 1.67l4.38-.52-3.45 2.83a.84.84 0 1 0 1.07 1.3l3.27-2.68-1.66 4.26a.84.84 0 1 0 1.57.61l1.56-4.01.66 4.35a.84.84 0 1 0 1.66-.25l-.63-4.14 2.9 3.46a.84.84 0 0 0 1.29-1.08l-2.46-2.93 3.74 2.55a.84.84 0 1 0 .95-1.39Z"
      />
    </svg>
  );
}

export interface HarnessControlProps {
  value: ChatHarness;
  claudeAvailable: boolean;
  temporary?: boolean;
  disabled?: boolean;
  onChange: (harness: ChatHarness) => void;
}

export function HarnessControl({
  value,
  claudeAvailable,
  temporary = false,
  disabled = false,
  onChange,
}: HarnessControlProps) {
  const { t } = useTranslation();
  const reasonId = useId();
  const isClaude = value === "claude_sdk";
  const claudeDisabled = temporary || !claudeAvailable;
  const label = isClaude
    ? t("thread.harness.claude", { defaultValue: "Claude" })
    : t("thread.harness.kohaku", { defaultValue: "Kohaku" });
  const unavailableReason = temporary
    ? t("thread.harness.temporary", { defaultValue: "Claude requires a saved chat." })
    : !claudeAvailable
      ? t("thread.harness.unavailable", { defaultValue: "Claude requires the Claude Agent SDK." })
      : null;
  const ariaLabel = `${t("thread.harness.label", { defaultValue: "Harness" })}: ${label}`;
  const title = unavailableReason && !isClaude ? `${label}. ${unavailableReason}` : label;

  return (
    <div className="flex min-w-0 items-center">
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild disabled={disabled}>
          <Button
            type="button"
            variant="ghost"
            aria-label={ariaLabel}
            aria-describedby={unavailableReason ? reasonId : undefined}
            title={title}
            disabled={disabled}
            className={cn(
              "touch-target h-7 min-w-0 max-w-[min(15rem,45vw)] whitespace-nowrap rounded-control border border-transparent px-1.5 text-[12px] font-semibold shadow-none",
              isClaude
                ? "bg-transparent text-amber-700 hover:bg-amber-500/10 dark:text-amber-300 dark:hover:bg-amber-400/10"
                : "bg-transparent text-muted-foreground hover:bg-foreground/[0.045] hover:text-foreground dark:text-foreground/70 dark:hover:bg-white/[0.06]",
            )}
          >
            {isClaude ? (
              <ClaudeIcon className="mr-1.5 h-3.5 w-3.5 shrink-0" />
            ) : (
              <KohakuIcon className="mr-1.5 h-4 w-4 shrink-0" />
            )}
            <span className="min-w-0 truncate">{label}</span>
            <ChevronDown className="ml-1.5 h-3 w-3 shrink-0" aria-hidden />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-56">
          <div className="space-y-1 p-1.5">
            <DropdownMenuItem
              disabled={disabled}
              onSelect={() => onChange("native")}
              className="flex h-10 items-center gap-3 px-3 text-[13.5px] font-semibold"
            >
              <span className="grid h-5 w-5 shrink-0 place-items-center" aria-hidden>
                <KohakuIcon className="h-4.5 w-4.5" />
              </span>
              <span className="min-w-0 flex-1 truncate">
                {t("thread.harness.kohaku", { defaultValue: "Kohaku" })}
              </span>
              {!isClaude ? <Check className="h-4 w-4 shrink-0" aria-hidden /> : null}
            </DropdownMenuItem>
            <DropdownMenuItem
              disabled={disabled || claudeDisabled}
              onSelect={() => {
                if (!claudeDisabled) onChange("claude_sdk");
              }}
              className={cn(
                "flex h-10 items-center gap-3 px-3 text-[13.5px] font-semibold",
                "text-amber-700 focus:text-amber-700 dark:text-amber-300 dark:focus:text-amber-300",
              )}
            >
              <span className="grid h-5 w-5 shrink-0 place-items-center" aria-hidden>
                <ClaudeIcon className="h-4 w-4" />
              </span>
              <span className="min-w-0 flex-1 truncate">
                {t("thread.harness.claude", { defaultValue: "Claude" })}
              </span>
              {isClaude ? <Check className="h-4 w-4 shrink-0" aria-hidden /> : null}
            </DropdownMenuItem>
            {claudeDisabled && unavailableReason ? (
              <p className="px-2 py-1 text-[11px] leading-relaxed text-muted-foreground">
                {unavailableReason}
              </p>
            ) : null}
          </div>
        </DropdownMenuContent>
      </DropdownMenu>
      {unavailableReason ? (
        <span id={reasonId} className="sr-only">
          {unavailableReason}
        </span>
      ) : null}
    </div>
  );
}
