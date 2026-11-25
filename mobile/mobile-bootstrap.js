/* Installed by build-shared-webui.sh before the shared React bundle. */
(function () {
  const capacitor = window.Capacitor;
  const plugin = capacitor?.Plugins?.NativeNetwork;
  if (!plugin) return;

  const subscribe = async (name, listener) => {
    const handle = await plugin.addListener(name, listener);
    return () => { void Promise.resolve(handle.remove()).catch(() => {}); };
  };

  // Android's JSON bridge can omit fields with no value.
  function publicSnapshot({
    stage, attemptId = null, method = null, origin = null, installationId = null,
    pairingId = null, tailscaleState, tailscaleHasLoginUrl, comparisonCode = null,
    expiresAt = null, canCancel, canRetry, error = null,
  }) {
    return {
      stage, attemptId, method, origin, installationId, pairingId, tailscaleState,
      tailscaleHasLoginUrl, comparisonCode, expiresAt, canCancel, canRetry, error,
    };
  }
  const pairingCall = async (method, ...args) => publicSnapshot(await plugin[method](...args));

  window.kohakuMobilePairing = {
    async target() {
      try {
        const target = await plugin.target();
        if (!target) return null;
        const { origin, installationId, hasAccess, hasRefresh } = target;
        return { origin, installationId, hasAccess, ...(hasRefresh === undefined ? {} : { hasRefresh }) };
      } catch (error) {
        if ((error?.message ?? error) === "no paired target") return null;
        throw error;
      }
    },
    pairingState: () => pairingCall("pairingState"),
    onPairingStateChange: (listener) => subscribe("pairingStateChanged", (event) => listener(publicSnapshot(event))),
    pairWithLink: (rawLink) => pairingCall("pairWithLink", { rawLink }),
    pairWithCode: (origin, code) => pairingCall("pairWithCode", { origin, code }),
    pairRetry: () => pairingCall("pairRetry"),
    pairCancel: () => pairingCall("pairCancel"),
    pairResetToIdle: () => pairingCall("pairResetToIdle"),
    pairCheckResume: () => pairingCall("pairCheckResume"),
    scanPairingQr: () => plugin.scanPairingQr(),
    openLogin: async () => { await plugin.openLogin(); },
    status: () => plugin.status(),
    refresh: async () => { await plugin.refresh(); },
    logout: async (serverReachable) => { await plugin.logout({ serverReachable }); },
  };

  const sendChains = new Map();
  const chunkBytes = 256 * 1024;
  function encode(bytes) {
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary);
  }

  window.nanobotMobileTransport = {
    start: async (origin, installationId, hostname) => {
      await plugin.start({ origin, installationId, hostname });
    },
    status: () => plugin.status(),
    remoteStatus: () => plugin.remoteStatus(),
    remoteDevices: () => plugin.remoteDevices(),
    remoteRevoke: (installationId) => plugin.remoteRevoke({ installationId }),
    http: async (request) => (await plugin.http(request)).requestId,
    cancel: async (requestId) => { await plugin.cancel({ requestId }); },
    onEvent: (listener) => subscribe("networkEvent", listener),
    openLogin: async () => { await plugin.openLogin(); },
    pickCamera: () => plugin.pickCamera(),
    async refresh() {
      try {
        await plugin.refresh();
        return true;
      } catch (_) {
        return false;
      }
    },
    async logout() {
      try {
        await plugin.logout({ serverReachable: true });
      } catch (_) {
        try { await plugin.logout({ serverReachable: false }); } catch (_) {}
      }
      window.location.reload();
    },
    wsOpen: async (path, headers) =>
      (await plugin.wsOpen({ operation: "mobile_ws", path, headers })).wsId,
    wsSend(wsId, data, kind = "text") {
      if (data.length <= chunkBytes) {
        return plugin.wsSend({ wsId, kind, dataBase64: encode(data) });
      }
      const previous = sendChains.get(wsId) ?? Promise.resolve();
      const next = previous.catch(() => {}).then(async () => {
        await plugin.wsSendBegin({ wsId, kind });
        try {
          for (let offset = 0; offset < data.length; offset += chunkBytes) {
            const chunk = data.subarray(offset, offset + chunkBytes);
            await plugin.wsSendChunk({ wsId, dataBase64: encode(chunk) });
          }
          await plugin.wsSendEnd({ wsId });
        } catch (error) {
          await plugin.wsSendEnd({ wsId, discard: true }).catch(() => {});
          throw error;
        }
      });
      sendChains.set(wsId, next);
      return next.finally(() => {
        if (sendChains.get(wsId) === next) sendChains.delete(wsId);
      });
    },
    wsClose: async (wsId) => { await plugin.wsClose({ wsId }); },
    wsPing: async (wsId) => Boolean((await plugin.wsPing({ wsId }))?.ok),
    wsRefreshAuth: async (wsId) => Boolean((await plugin.wsRefreshAuth({ wsId }))?.ok),
    stop: async () => { await plugin.stop(); },
  };

  window.nanobotMobileRunKeepAlive = {
    start: async (chatId) => { await plugin.startRunKeepAlive({ chatId }); },
    stop: async (chatId) => { await plugin.stopRunKeepAlive({ chatId }); },
  };

  // Mirror console output to logcat for packaged WebViews without devtools.
  const probe = window.KohakuNativeProbe;
  if (typeof probe?.log === "function") {
    for (const level of ["log", "warn", "error"]) {
      const original = console[level];
      console[level] = function (...values) {
        try {
          const parts = values.map((value) => {
            if (typeof value === "string") return value;
            try { return JSON.stringify(value); } catch (_) { return String(value); }
          });
          probe.log("[" + level + "] " + parts.join(" ").slice(0, 400));
        } catch (_) {}
        original.apply(console, values);
      };
    }
  }

  function notificationId(key) {
    let hash = 2166136261;
    for (let i = 0; i < key.length; i++) hash = Math.imul(hash ^ key.charCodeAt(i), 16777619);
    // Reserve low IDs for native services, including the run keepalive.
    return 65536 + (hash >>> 0) % (2147483648 - 65536);
  }

  try {
    const notifications = capacitor.Plugins.LocalNotifications
      || capacitor.registerPlugin?.("LocalNotifications");
    if (!notifications) return;

    const alertChannel = "kohaku_task_alerts";
    const silentChannel = "kohaku_task_alerts_silent";
    const channelReady = Promise.resolve().then(() => notifications.createChannel({
      id: alertChannel, name: "Task updates", importance: 4, vibration: true,
    })).then(() => true, (error) => error?.code === "UNAVAILABLE");

    void Promise.resolve().then(() => notifications.addListener?.(
      "localNotificationActionPerformed", (action) => {
        const extra = action.notification?.extra;
        if (typeof extra?.chatId !== "string" || !extra.chatId) return;
        window.location.hash = extra.temporary
          ? "#/temporary/" + encodeURIComponent(extra.chatId)
          : "#/chat/" + encodeURIComponent("websocket:" + extra.chatId);
      },
    )).catch(() => {});

    window.nanobotMobileNotifications = {
      async requestPermission() {
        try {
          return (await notifications.requestPermissions())?.display === "granted";
        } catch (_) {
          return false;
        }
      },
      async notify(options) {
        if (!await channelReady) return false;
        try {
          const audible = options.sound !== false;
          await notifications.schedule({ notifications: [{
            id: notificationId(options.tag || options.chatId || "kohaku-test"),
            title: options.title || "Kohaku",
            body: options.body || "",
            channelId: audible ? alertChannel : silentChannel,
            // Android 7 ignores channel sound settings.
            sound: audible ? undefined : "kohaku_silent.wav",
            extra: options.chatId ? {
              chatId: options.chatId,
              ...(options.temporary ? { temporary: true } : {}),
            } : {},
            isExactNotification: false,
            foreground: true,
            smallIcon: "ic_stat_kohaku",
            largeIcon: "ic_kohaku_notification",
          }] });
          return true;
        } catch (_) {
          return false;
        }
      },
    };
  } catch (_) {
    // Notification setup is optional.
  }
})();
