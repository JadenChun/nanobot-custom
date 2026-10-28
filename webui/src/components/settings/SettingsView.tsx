import { SettingsPage } from "@/components/settings/SettingsPage";
import type { SettingsSectionKey } from "@/components/settings/contracts";
import { useSettingsController } from "@/components/settings/useSettingsController";
import type { SettingsPayload, SkillSummary, WorkspacesPayload } from "@/lib/types";

export type { SettingsSectionKey } from "@/components/settings/contracts";

interface SettingsViewProps {
  theme: "light" | "dark";
  initialSection?: SettingsSectionKey;
  initialSettings?: SettingsPayload | null;
  showSidebar?: boolean;
  onToggleTheme: () => void;
  onBackToChat: () => void;
  onModelNameChange: (modelName: string | null) => void;
  onSettingsChange?: (payload: SettingsPayload) => void;
  skills?: SkillSummary[];
  onSectionChange?: (section: SettingsSectionKey) => void;
  onLogout?: () => void;
  onRestart?: () => void;
  onNativeEngineRestart?: () => Promise<string>;
  isRestarting?: boolean;
  hostChromeInset?: boolean;
  workspaces?: WorkspacesPayload | null;
  onRefreshWorkspaces?: () => Promise<void>;
  onPickWorkspaceFolder?: () => Promise<string | null>;
}

export function SettingsView({
  theme,
  initialSection = "overview",
  initialSettings = null,
  showSidebar = true,
  onToggleTheme,
  onBackToChat,
  onModelNameChange,
  onSettingsChange,
  skills = [],
  onSectionChange,
  onLogout,
  onRestart,
  onNativeEngineRestart,
  isRestarting = false,
  hostChromeInset = false,
  workspaces = null,
  onRefreshWorkspaces = async () => {},
  onPickWorkspaceFolder,
}: SettingsViewProps) {
  const controller = useSettingsController({
    initialSection,
    initialSettings,
    onModelNameChange,
    onSettingsChange,
    onSectionChange,
    onRestart,
    onNativeEngineRestart,
  });

  return (
    <SettingsPage
      controller={controller}
      theme={theme}
      showSidebar={showSidebar}
      onToggleTheme={onToggleTheme}
      onBackToChat={onBackToChat}
      skills={skills}
      onLogout={onLogout}
      isRestarting={isRestarting}
      hostChromeInset={hostChromeInset}
      workspaces={workspaces}
      onRefreshWorkspaces={onRefreshWorkspaces}
      onPickWorkspaceFolder={onPickWorkspaceFolder}
    />
  );
}
