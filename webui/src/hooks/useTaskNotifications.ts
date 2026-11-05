import { useEffect } from "react";
import { useTranslation } from "react-i18next";

import { readLocalPreferences } from "@/lib/local-preferences";
import {
  buildTaskNotificationBody,
  showTaskNotification,
} from "@/lib/task-notifications";
import { useClient } from "@/providers/ClientProvider";

/**
 * Global completion/failure notification subscriber.
 *
 * Living above the chat view means a run still notifies after the user leaves
 * its thread or navigates elsewhere in the app. Focus and preference checks
 * keep the existing "notify only while the app is not focused" contract.
 */
export function useTaskNotifications(): void {
  const { client } = useClient();
  const { t } = useTranslation();

  useEffect(() => client.onTaskNotification((event) => {
    if (!readLocalPreferences().browserNotifications) return;
    if (
      typeof document !== "undefined"
      && document.visibilityState === "visible"
      && document.hasFocus()
    ) return;
    showTaskNotification({
      body: buildTaskNotificationBody(event, t),
      tag: `nanobot-turn-${event.chatId}`,
    });
  }), [client, t]);
}
