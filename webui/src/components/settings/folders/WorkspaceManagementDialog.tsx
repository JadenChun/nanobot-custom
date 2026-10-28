import { useTranslation } from "react-i18next";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { WorkspacesPayload } from "@/lib/types";
import { FoldersProjectsManager } from "./FoldersProjectsManager";

export function WorkspaceManagementDialog({
  open,
  onOpenChange,
  workspaces,
  onRefreshWorkspaces,
  onPickFolder,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workspaces: WorkspacesPayload | null;
  onRefreshWorkspaces: () => Promise<void>;
  onPickFolder?: () => Promise<string | null>;
}) {
  const { t } = useTranslation();
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[min(88dvh,52rem)] w-[min(calc(100vw-1rem),52rem)] max-w-3xl overflow-y-auto p-4 sm:p-6">
        <DialogHeader className="pr-8 text-left">
          <DialogTitle>{t("settings.nav.folders", { defaultValue: "Folders & projects" })}</DialogTitle>
          <DialogDescription>
            {t("workspace.management.dialogDescription", { defaultValue: "Choose folders Nanobot may use and the projects it remembers." })}
          </DialogDescription>
        </DialogHeader>
        <FoldersProjectsManager
          workspaces={workspaces}
          onRefreshWorkspaces={onRefreshWorkspaces}
          onPickFolder={onPickFolder}
        />
      </DialogContent>
    </Dialog>
  );
}
