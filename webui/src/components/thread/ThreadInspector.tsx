import * as Dialog from "@radix-ui/react-dialog";
import { X, type LucideIcon } from "lucide-react";
import { useId, useRef, useState, type ReactNode, type RefObject } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const MIN_INSPECTOR_WIDTH = 360;
const MIN_CHAT_WIDTH = 420;
const MAX_INSPECTOR_WIDTH = 960;
export const MIN_INSPECTOR_DOCK_WIDTH = MIN_CHAT_WIDTH + MIN_INSPECTOR_WIDTH;

export type InspectorTabId = "subagents" | "file";

export interface InspectorTab {
  id: InspectorTabId;
  label: string;
  closeLabel: string;
  icon: LucideIcon;
  content: ReactNode;
}

interface ThreadInspectorProps {
  tabs: InspectorTab[];
  selected: InspectorTabId | null;
  containerWidth: number;
  visible: boolean;
  onSelect: (id: InspectorTabId) => void;
  onClose: (id: InspectorTabId) => void;
}

export function ThreadInspector({
  tabs,
  selected,
  containerWidth,
  visible,
  onSelect,
  onClose,
}: ThreadInspectorProps) {
  const { t } = useTranslation();
  const id = useId();
  const [preferredWidth, setPreferredWidth] = useState<number | null>(null);
  const tabRefs = useRef<Partial<Record<InspectorTabId, HTMLButtonElement | null>>>({});
  const dragRef = useRef<{ pointer: number; startX: number; width: number; direction: number } | null>(null);
  const docked = containerWidth >= MIN_INSPECTOR_DOCK_WIDTH;
  const maxWidth = Math.max(MIN_INSPECTOR_WIDTH, Math.min(MAX_INSPECTOR_WIDTH, containerWidth - MIN_CHAT_WIDTH));
  const clampWidth = (width: number) => Math.min(maxWidth, Math.max(MIN_INSPECTOR_WIDTH, Math.round(width)));
  const width = clampWidth(preferredWidth ?? containerWidth * 0.44);
  if (!selected) return null;
  const closeTab = (id: InspectorTabId) => {
    const remaining = tabs.find((tab) => tab.id !== id);
    onClose(id);
    if (remaining) tabRefs.current[remaining.id]?.focus({ preventScroll: true });
  };

  return (
    <aside
      aria-label={docked ? t("thread.inspector.label", "Chat side panel") : undefined}
      className={!visible ? "hidden" : docked
          ? "relative flex min-h-0 min-w-0 shrink-0 flex-col border-s border-border/70 bg-background"
          : "contents"}
      style={docked ? { width } : undefined}
    >
      {docked ? (
        <>
          <button
            type="button"
            role="separator"
            aria-orientation="vertical"
            aria-label={t("thread.inspector.resize", "Resize side panel")}
            aria-valuemin={MIN_INSPECTOR_WIDTH}
            aria-valuemax={maxWidth}
            aria-valuenow={width}
            className="absolute inset-y-0 start-0 z-20 w-2 -translate-x-1/2 cursor-col-resize touch-none hover:bg-foreground/10 focus-visible:bg-foreground/15 focus-visible:outline-none rtl:translate-x-1/2"
            onPointerDown={(event) => {
              if (event.button !== 0) return;
              event.preventDefault();
              dragRef.current = {
                pointer: event.pointerId,
                startX: event.clientX,
                width,
                direction: getComputedStyle(event.currentTarget).direction === "rtl" ? 1 : -1,
              };
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={(event) => {
              const drag = dragRef.current;
              if (drag?.pointer !== event.pointerId) return;
              setPreferredWidth(clampWidth(drag.width + (event.clientX - drag.startX) * drag.direction));
            }}
            onLostPointerCapture={() => { dragRef.current = null; }}
            onKeyDown={(event) => {
              const direction = getComputedStyle(event.currentTarget).direction === "rtl" ? 1 : -1;
              switch (event.key) {
                case "ArrowLeft": setPreferredWidth(clampWidth(width - 24 * direction)); break;
                case "ArrowRight": setPreferredWidth(clampWidth(width + 24 * direction)); break;
                case "Home": setPreferredWidth(MIN_INSPECTOR_WIDTH); break;
                case "End": setPreferredWidth(maxWidth); break;
                default: return;
              }
              event.preventDefault();
            }}
          />
          <div className="flex h-10 shrink-0 items-center border-b border-border/60 px-2">
            <div role="tablist" aria-label={t("thread.inspector.tabs", "Open previews")} className="flex min-w-0 flex-1 items-center gap-1">
              {tabs.map((tab, index) => (
                <div key={tab.id} role="presentation" className={cn(
                  "group/tab flex h-8 min-w-0 max-w-60 flex-1 items-center rounded-md border border-transparent text-sm",
                  tab.id === selected ? "border-border/70 bg-muted/60 text-foreground" : "text-muted-foreground hover:bg-muted/40 hover:text-foreground",
                )}>
                  <button
                    ref={(element) => { tabRefs.current[tab.id] = element; }}
                    type="button"
                    role="tab"
                    id={`${id}-${tab.id}`}
                    aria-selected={tab.id === selected}
                    aria-controls={`${id}-${tab.id}-panel`}
                    tabIndex={tab.id === selected ? 0 : -1}
                    title={tab.label}
                    className="flex h-full min-w-0 flex-1 items-center gap-2 rounded-md px-2.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
                    onClick={() => onSelect(tab.id)}
                    onKeyDown={(event) => {
                      const direction = getComputedStyle(event.currentTarget).direction === "rtl" ? -1 : 1;
                      let nextIndex: number;
                      switch (event.key) {
                        case "ArrowLeft": nextIndex = (index - direction + tabs.length) % tabs.length; break;
                        case "ArrowRight": nextIndex = (index + direction + tabs.length) % tabs.length; break;
                        case "Home": nextIndex = 0; break;
                        case "End": nextIndex = tabs.length - 1; break;
                        case "Delete": event.preventDefault(); closeTab(tab.id); return;
                        default: return;
                      }
                      event.preventDefault();
                      const next = tabs[nextIndex].id;
                      onSelect(next);
                      tabRefs.current[next]?.focus();
                    }}
                  >
                    <tab.icon className="size-4 shrink-0" aria-hidden />
                    <span className="truncate">{tab.label}</span>
                  </button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className={cn("me-0.5 size-7 shrink-0 text-muted-foreground", tab.id !== selected && "invisible group-hover/tab:visible group-focus-within/tab:visible")}
                    aria-label={tab.closeLabel}
                    title={tab.closeLabel}
                    onClick={() => closeTab(tab.id)}
                  >
                    <X className="size-3.5" aria-hidden />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : null}
      {tabs.map((tab) => (
        <div
          key={tab.id}
          id={docked ? `${id}-${tab.id}-panel` : undefined}
          role={docked ? "tabpanel" : undefined}
          aria-labelledby={docked ? `${id}-${tab.id}` : undefined}
          tabIndex={docked && tab.id === selected ? 0 : undefined}
          hidden={tab.id !== selected}
          className={tab.id !== selected ? "hidden" : docked ? "flex min-h-0 flex-1" : "contents"}
        >
          {tab.content}
        </div>
      ))}
    </aside>
  );
}

export function ThreadInspectorDialog({
  children,
  focusRef,
  onClose,
  testId,
}: {
  children: ReactNode;
  focusRef: RefObject<HTMLButtonElement>;
  onClose: () => void;
  testId?: string;
}) {
  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Content
          aria-modal="true"
          className="fixed inset-0 z-50 flex min-h-0 flex-col overflow-hidden bg-background pb-[env(safe-area-inset-bottom)] pt-[env(safe-area-inset-top)]"
          data-testid={testId}
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            focusRef.current?.focus({ preventScroll: true });
          }}
          onCloseAutoFocus={(event) => event.preventDefault()}
        >
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
