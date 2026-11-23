import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.nanobot.mobile',
  appName: 'Kohaku',
  webDir: process.env.NANOBOT_MOBILE_WEB_DIR || 'www',
  plugins: {
    LocalNotifications: {
      smallIcon: 'ic_stat_nanobot',
      iconColor: '#EF8E30',
    },
  },
  android: {
    allowMixedContent: false,
  },
};

export default config;
