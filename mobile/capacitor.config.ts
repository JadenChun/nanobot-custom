import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.nanobot.mobile',
  appName: 'Nanobot Mobile',
  webDir: 'www',
  android: {
    allowMixedContent: false,
  },
};

export default config;
