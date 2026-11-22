import { useEffect, useLayoutEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { readLocalPreferences } from "@/lib/local-preferences";
import { showTaskNotification } from "@/lib/task-notifications";
import { useClient } from "@/providers/ClientProvider";

/**
 * Global completion/failure notification subscriber.
 *
 * Living above the chat view means a run still notifies after the user leaves
 * its thread or navigates elsewhere in the app. Focus and preference checks
 * keep the existing "notify only while the app is not focused" contract.
 */
export function useTaskNotifications(thread?: (id: string) => { threadTitle?: string; temporary?: boolean }): void {
  const { client } = useClient();
  const { t } = useTranslation();

  const threadRef = useRef(thread);
  useLayoutEffect(() => { threadRef.current = thread; }, [thread]);

  useEffect(() => client.onTaskNotification((event) => {
    const prefs = readLocalPreferences();
    if (!prefs.browserNotifications) return;
    if (
      typeof document !== "undefined"
      && document.visibilityState === "visible"
      && document.hasFocus()
    ) return;
    void showTaskNotification({
      ...threadRef.current?.(event.chatId),
      sound: prefs.notificationSound,
      event,
      t,
      requestMutation: (action, payload) => client.requestMutation(action, payload),
    });
  }), [client, t]);
}
