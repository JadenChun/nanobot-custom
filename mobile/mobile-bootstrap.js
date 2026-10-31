/* Installed by build-shared-webui.sh before the shared React bundle. */
(function () {
  var plugin = window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.NativeNetwork;
  if (!plugin) return;

  var transport = {
    start: async function (origin, installationId, hostname) {
      await plugin.start({ origin: origin, installationId: installationId, hostname: hostname });
    },
    status: async function () { return await plugin.status(); },
    http: async function (request) {
      var result = await plugin.http(request);
      return result.requestId;
    },
    cancel: async function (requestId) { await plugin.cancel({ requestId: requestId }); },
    onEvent: async function (listener) {
      var handle = await plugin.addListener("networkEvent", function (event) { listener(event); });
      return function () { void handle.remove(); };
    },
    openLogin: async function () { await plugin.openLogin(); },
    wsOpen: async function (path, headers) {
      var result = await plugin.wsOpen({ operation: "fixture_ws", path: path, headers: headers });
      return result.wsId;
    },
    wsSend: async function (wsId, data, kind) {
      var binary = "";
      for (var i = 0; i < data.length; i += 1) binary += String.fromCharCode(data[i]);
      await plugin.wsSend({ wsId: wsId, kind: kind || "text", dataBase64: btoa(binary) });
    },
    wsClose: async function (wsId) { await plugin.wsClose({ wsId: wsId }); },
    stop: async function () { await plugin.stop(); }
  };
  window.nanobotMobileTransport = transport;

  void (async function () {
    try {
      var target = await plugin.target();
      await transport.start(target.origin, target.installationId, "nanobot-android");
    } catch (_) {
      /* No paired target yet: the shared UI renders its bootstrap/auth state. */
    }
  })();
})();
