import { useEffect, useMemo, useState } from "react";
import { ListTree, Search } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { formControlFocusClassName } from "@/components/ui/form-control";
import {
  Sheet,
  SheetContent,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  type PromptAnchor,
  userPromptAnchors,
} from "@/components/thread/promptNavigation";
import type { RevealPromptTarget } from "@/hooks/useSessions";
import { fmtDateTime } from "@/lib/format";
import type {
  SessionPromptSummary,
  SessionTurnSummary,
  UIMessage,
} from "@/lib/types";
import { cn } from "@/lib/utils";

interface PromptNavigatorProps {
  messages: UIMessage[];
  onJumpToPrompt: (promptId: string) => void;
  sessionPrompts?: SessionPromptSummary[];
  sessionTurns?: SessionTurnSummary[];
  totalPromptCount?: number;
  hasMoreBefore?: boolean;
  hasMorePromptsBefore?: boolean;
  loadingMorePrompts?: boolean;
  onNeedSessionPrompts?: () => void;
  onLoadMorePrompts?: () => void;
  onRevealPrompt?: (target: RevealPromptTarget) => void;
}

interface NavigatorPromptItem {
  key: string;
  label: string;
  preview: string;
  createdAt: number | null;
  renderedId?: string;
  promptId: string;
  locator?: string | null;
  cursor?: string | null;
}

export function PromptNavigator({
  messages,
  onJumpToPrompt,
  sessionPrompts = [],
  sessionTurns = [],
  totalPromptCount = 0,
  hasMoreBefore = false,
  hasMorePromptsBefore = false,
  loadingMorePrompts = false,
  onNeedSessionPrompts,
  onLoadMorePrompts,
  onRevealPrompt,
}: PromptNavigatorProps) {
  const { i18n, t } = useTranslation();
  const localAnchors = useMemo(() => userPromptAnchors(messages), [messages]);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!open || !onNeedSessionPrompts) return;
    if (sessionPrompts.length > 0 || sessionTurns.length > 0) return;
    if (totalPromptCount > localAnchors.length || hasMoreBefore) {
      onNeedSessionPrompts();
    }
  }, [
    hasMoreBefore,
    localAnchors.length,
    onNeedSessionPrompts,
    open,
    sessionPrompts.length,
    sessionTurns.length,
    totalPromptCount,
  ]);

  const prompts = useMemo<NavigatorPromptItem[]>(() => {
    const byPromptId = new Map<string, PromptAnchor>();
    const byOrdinal = new Map<number, PromptAnchor[]>();
    for (const anchor of localAnchors) {
      if (anchor.promptId) byPromptId.set(anchor.promptId, anchor);
      byPromptId.set(anchor.id, anchor);
      if (anchor.ordinal !== undefined) {
        const list = byOrdinal.get(anchor.ordinal) ?? [];
        list.push(anchor);
        byOrdinal.set(anchor.ordinal, list);
      }
    }
    if (sessionPrompts.length > 0) {
      const countsByTurn = new Map<number, number>();
      for (const entry of sessionPrompts) {
        countsByTurn.set(entry.turn_ordinal, (countsByTurn.get(entry.turn_ordinal) ?? 0) + 1);
      }
      const matchedAnchorIds = new Set<string>();
      const items: NavigatorPromptItem[] = sessionPrompts.map((entry) => {
        let rendered = byPromptId.get(entry.prompt_id);
        if (!rendered && (countsByTurn.get(entry.turn_ordinal) ?? 0) === 1) {
          const candidates = byOrdinal.get(entry.turn_ordinal);
          if (candidates && candidates.length === 1 && !candidates[0].promptId) {
            rendered = candidates[0];
          }
        }
        if (rendered) matchedAnchorIds.add(rendered.id);
        return {
          key: `prompt:${entry.prompt_id}`,
          label: rendered?.label ?? entry.label,
          preview: rendered?.preview ?? entry.label,
          createdAt: rendered?.createdAt ?? entry.created_at,
          renderedId: rendered?.id,
          promptId: entry.prompt_id,
          locator: entry.locator || null,
          cursor: null,
        };
      });
      for (const anchor of localAnchors) {
        if (matchedAnchorIds.has(anchor.id)) continue;
        items.push({
          key: `local:${anchor.promptId ?? anchor.id}`,
          label: anchor.label,
          preview: anchor.preview,
          createdAt: anchor.createdAt,
          renderedId: anchor.id,
          promptId: anchor.promptId ?? anchor.id,
          locator: null,
          cursor: null,
        });
      }
      return items;
    }
    if (sessionTurns.length > 0) {
      const matchedAnchorIds = new Set<string>();
      const items: NavigatorPromptItem[] = sessionTurns.map((turn) => {
        const candidates = byOrdinal.get(turn.ordinal);
        const rendered = candidates?.[0];
        if (rendered) matchedAnchorIds.add(rendered.id);
        return {
          key: `turn:${turn.ordinal}:${turn.cursor}`,
          label: rendered?.label ?? turn.label,
          preview: rendered?.preview ?? turn.label,
          createdAt: rendered?.createdAt ?? turn.created_at,
          renderedId: rendered?.id,
          promptId: rendered?.promptId ?? rendered?.id ?? `turn:${turn.ordinal}`,
          locator: null,
          cursor: turn.cursor,
        };
      });
      for (const anchor of localAnchors) {
        if (matchedAnchorIds.has(anchor.id)) continue;
        items.push({
          key: `local:${anchor.promptId ?? anchor.id}`,
          label: anchor.label,
          preview: anchor.preview,
          createdAt: anchor.createdAt,
          renderedId: anchor.id,
          promptId: anchor.promptId ?? anchor.id,
          locator: null,
          cursor: null,
        });
      }
      return items;
    }
    return localAnchors.map((anchor) => ({
      key: `local:${anchor.promptId ?? anchor.id}`,
      label: anchor.label,
      preview: anchor.preview,
      createdAt: anchor.createdAt,
      renderedId: anchor.id,
      promptId: anchor.promptId ?? anchor.id,
      locator: null,
      cursor: null,
    }));
  }, [localAnchors, sessionPrompts, sessionTurns]);

  const filteredPrompts = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return prompts;
    return prompts.filter((prompt) =>
      `${prompt.label}\n${prompt.preview}`.toLocaleLowerCase().includes(needle),
    );
  }, [prompts, query]);

  if (prompts.length === 0 && totalPromptCount <= 0 && !hasMoreBefore) return null;

  const jump = (item: NavigatorPromptItem) => {
    setOpen(false);
    if (item.renderedId) {
      onJumpToPrompt(item.renderedId);
      return;
    }
    if (onRevealPrompt) {
      onRevealPrompt({
        promptId: item.promptId,
        locator: item.locator ?? null,
        cursor: item.cursor ?? null,
      });
      return;
    }
    onJumpToPrompt(item.promptId);
  };

  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={cn(
          "host-no-drag h-8 w-8 rounded-full text-muted-foreground/80",
          "hover:bg-accent/40 hover:text-foreground",
        )}
        aria-label={t("thread.promptNavigator.open")}
        onClick={() => setOpen(true)}
      >
        <ListTree className="h-4 w-4" />
      </Button>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent
          side="right"
          aria-describedby={undefined}
          className="w-[min(92vw,24rem)] gap-0 p-0 sm:max-w-[24rem]"
        >
          <div className="border-b px-5 pb-4 pt-5">
            <SheetTitle className="text-base font-medium">
              {t("thread.promptNavigator.title")}
            </SheetTitle>
            <div className="relative mt-4">
              <Search
                aria-hidden
                className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
              />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                aria-label={t("thread.promptNavigator.search")}
                placeholder={t("thread.promptNavigator.search")}
                className={cn(
                  "h-10 w-full rounded-full border border-border bg-background pl-9 pr-3 text-sm",
                  "transition-colors",
                  formControlFocusClassName,
                )}
              />
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
            {hasMorePromptsBefore && onLoadMorePrompts ? (
              <div className="mb-2 px-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={loadingMorePrompts}
                  data-testid="prompt-navigator-load-older"
                  className="w-full justify-center text-xs text-muted-foreground"
                  onClick={onLoadMorePrompts}
                >
                  {loadingMorePrompts
                    ? t("thread.generatedAssets.loadingMore")
                    : t("thread.promptNavigator.loadOlder", {
                        defaultValue: "Load older prompts",
                      })}
                </Button>
              </div>
            ) : null}
            {filteredPrompts.length > 0 ? (
              <div className="space-y-1">
                {filteredPrompts.map((prompt) => (
                  <PromptNavigatorRow
                    key={prompt.key}
                    locale={i18n.resolvedLanguage || i18n.language}
                    prompt={prompt}
                    onJump={jump}
                  />
                ))}
              </div>
            ) : (
              <div className="px-3 py-10 text-center text-sm text-muted-foreground">
                {t("thread.promptNavigator.noResults")}
              </div>
            )}
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}

interface PromptNavigatorRowProps {
  locale: string;
  onJump: (prompt: NavigatorPromptItem) => void;
  prompt: NavigatorPromptItem;
}

function PromptNavigatorRow({
  locale,
  onJump,
  prompt,
}: PromptNavigatorRowProps) {
  const { t } = useTranslation();
  const timestamp = prompt.createdAt ? fmtDateTime(prompt.createdAt, locale) : "";
  return (
    <button
      type="button"
      className={cn(
        "w-full rounded-xl px-3 py-3 text-left transition",
        "hover:bg-accent focus-visible:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30",
      )}
      aria-label={t("thread.promptNavigator.jumpTo", { label: prompt.label })}
      onClick={() => onJump(prompt)}
    >
      <div className="max-h-20 overflow-hidden whitespace-pre-wrap break-words text-sm leading-5 text-foreground">
        {prompt.preview}
      </div>
      {timestamp ? (
        <div className="mt-1 text-[10px] leading-4 text-muted-foreground/75">
          {timestamp}
        </div>
      ) : null}
    </button>
  );
}
