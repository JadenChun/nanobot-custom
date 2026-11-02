import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.nanobot.mobile',
  appName: 'Nanobot Mobile',
  webDir: process.env.NANOBOT_MOBILE_WEB_DIR || 'www',
  android: {
    allowMixedContent: false,
  },
};

export default config;
