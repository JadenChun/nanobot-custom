import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.kohaku.mobile',
  appName: 'Kohaku',
  webDir: process.env.KOHAKU_MOBILE_WEB_DIR || 'www',
  plugins: {
    LocalNotifications: {
      smallIcon: 'ic_stat_kohaku',
      iconColor: '#EF8E30',
    },
  },
  android: {
    allowMixedContent: false,
  },
};

export default config;
