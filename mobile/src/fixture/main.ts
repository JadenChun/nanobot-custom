import { Capacitor } from '@capacitor/core';

type NativeProbe = {
  abi: string;
  architecture: string;
  androidApi: number;
  pageSize: number;
  machine: string;
  supportedAbis: string[];
};

declare global {
  interface Window {
    NanobotNativeProbe?: {
      probe(): string;
    };
  }
}

const status = document.querySelector<HTMLParagraphElement>('#status');
const shell = document.querySelector<HTMLElement>('#shell');
const probe = document.querySelector<HTMLElement>('#probe');

if (status && shell && probe) {
  shell.textContent = Capacitor.getPlatform();
  status.textContent = 'Fixture shell ready.';

  if (Capacitor.isNativePlatform()) {
    const nativeProbe = window.NanobotNativeProbe;
    if (!nativeProbe) {
      probe.textContent = 'native probe bridge missing';
      status.textContent = 'Fixture shell loaded without the native probe bridge.';
    } else {
      try {
        const result = JSON.parse(nativeProbe.probe()) as NativeProbe;
        probe.textContent = `${result.abi} / ${result.architecture} / API ${result.androidApi} / page ${result.pageSize}`;
        status.textContent = `Native probe passed: ${result.supportedAbis.join(', ')}`;
        console.log('NANOBOT_MOBILE_PROBE_OK', JSON.stringify(result));
      } catch (error) {
        probe.textContent = 'native probe failed';
        status.textContent = `Native probe failed: ${String(error)}`;
        console.error('NANOBOT_MOBILE_PROBE_FAIL', error);
      }
    }
  } else {
    probe.textContent = 'browser shell (no native probe)';
    console.log('NANOBOT_MOBILE_PROBE_SKIP browser');
  }
}
