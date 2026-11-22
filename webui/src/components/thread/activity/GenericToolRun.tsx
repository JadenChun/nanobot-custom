import {
  AlertCircle,
  FileSearch,
  FolderOpen,
  ListTree,
  MemoryStick,
  Play,
  type LucideIcon,
} from "lucide-react";
import { useMemo } from "react";

import {
  ActivityDetailPanel,
  ActivityDisclosureStep,
  hasActivityDetailContent,
  type ActivityDetailContent,
  type ActivityDetailField,
  type ActivityDetailItem,
} from "@/components/thread/activity/ActivityDisclosureStep";
import { compactActivityPath, redactActivityText } from "@/components/thread/activity/activity-text";
import {
  describeGenericToolRun,
  type GenericToolRunItem,
  type GenericToolStatus,
  type ToolFamily,
} from "@/components/thread/activity/generic-tool-model";

interface GenericToolRunModel {
  status: GenericToolStatus;
  label: string;
  detail: string;
  aside: string;
  icon: LucideIcon;
}

export function GenericToolRun({ items }: { items: GenericToolRunItem[] }) {
  const model = useMemo(() => buildModel(items), [items]);
  const action = [model.label, model.detail].filter(Boolean).join(" ");
  const label = model.aside ? `${action} · ${model.aside}` : action;
  const detail = useMemo(() => buildGenericToolDetail(items), [items]);
  const hasDetails = hasActivityDetailContent(detail);

  return (
    <ActivityDisclosureStep
      icon={model.status === "error" ? AlertCircle : model.icon}
      active={model.status === "running"}
      tone={model.status === "error" ? "error" : model.status === "done" ? "success" : "active"}
      label={label}
      hasDetails={hasDetails}
    >
      {hasDetails ? <ActivityDetailPanel detail={detail} scrollable={false} /> : null}
    </ActivityDisclosureStep>
  );
}

function buildModel(items: GenericToolRunItem[]): GenericToolRunModel {
  const family = items[0]?.trace.family ?? "generic";
  const presentation = describeGenericToolRun(items);
  return {
    ...presentation,
    icon: activityIcon(family),
  };
}

function buildGenericToolDetail(items: GenericToolRunItem[]): ActivityDetailContent {
  if (items.length > 1) {
    const detailItems: ActivityDetailItem[] = items.map((item) => {
      const primary = item.trace.fields[0]?.value
        ? compactActivityPath(redactActivityText(item.trace.fields[0].value))
        : item.trace.name;
      const extra = item.trace.fields
        .slice(1)
        .map((field) => `${field.key.replace(/_/g, " ")}: ${compactActivityPath(redactActivityText(field.value))}`)
        .join(" · ");
      return {
        title: primary,
        meta: extra || (item.status === "error" ? "failed" : undefined),
        error: item.error,
      };
    });
    return { items: detailItems };
  }

  const single = items[0];
  if (!single) return {};
  const fields: ActivityDetailField[] = single.trace.fields.map((field) => ({
    label: field.key.replace(/_/g, " "),
    value: compactActivityPath(redactActivityText(field.value)),
    mono: /path|file|url|pattern|glob|key/i.test(field.key),
  }));
  return {
    ...(fields.length > 0 ? { fields } : {}),
    ...(single.error ? { error: single.error } : {}),
  };
}

function activityIcon(family: ToolFamily): LucideIcon {
  if (family === "content-search" || family === "file-search") return FileSearch;
  if (family === "list") return ListTree;
  if (family === "read") return FolderOpen;
  if (family === "memory") return MemoryStick;
  return Play;
}
