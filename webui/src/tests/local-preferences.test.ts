import { beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_LOCAL_PREFS,
  LOCAL_PREFS_STORAGE_KEY,
  readLocalPreferences,
  writeLocalPreferences,
} from "@/lib/local-preferences";

describe("local preferences", () => {
  beforeEach(() => localStorage.clear());

  it("defaults existing notification settings to audible and persists mute", () => {
    localStorage.setItem(LOCAL_PREFS_STORAGE_KEY, JSON.stringify({ browserNotifications: true }));
    expect(readLocalPreferences().notificationSound).toBe(true);
    writeLocalPreferences({ ...readLocalPreferences(), notificationSound: false });
    expect(readLocalPreferences().notificationSound).toBe(false);
  });

  it("enables third-party brand logos by default and migrates the previous default", () => {
    expect(DEFAULT_LOCAL_PREFS.brandLogos).toBe(true);
    expect(readLocalPreferences().brandLogos).toBe(true);

    localStorage.setItem(
      LOCAL_PREFS_STORAGE_KEY,
      JSON.stringify({ brandLogos: false }),
    );
    expect(readLocalPreferences().brandLogos).toBe(true);
  });

  it("preserves a brand logo opt-out after the default-on migration", () => {
    writeLocalPreferences({ ...DEFAULT_LOCAL_PREFS, brandLogos: false });
    expect(readLocalPreferences().brandLogos).toBe(false);
  });

  it("keeps browser notifications opt-in", () => {
    expect(DEFAULT_LOCAL_PREFS.browserNotifications).toBe(false);
    expect(readLocalPreferences().browserNotifications).toBe(false);

    writeLocalPreferences({ ...DEFAULT_LOCAL_PREFS, browserNotifications: true });
    expect(readLocalPreferences().browserNotifications).toBe(true);
  });
});
