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
    pickCamera: async function () { return await plugin.pickCamera(); },
    refresh: async function () {
      try {
        await plugin.refresh();
        return true;
      } catch (_) {
        return false;
      }
    },
    wsOpen: async function (path, headers) {
      var result = await plugin.wsOpen({ operation: "mobile_ws", path: path, headers: headers });
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

  var resolveReady;
  var rejectReady;
  window.nanobotMobileReady = new Promise(function (resolve, reject) {
    resolveReady = resolve;
    rejectReady = reject;
  });

  function textElement(tag, className, value) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (value !== undefined) element.textContent = value;
    return element;
  }

  function copyText(value, button) {
    var done = function () {
      var previous = button.textContent;
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = previous; }, 1500);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(value).then(done).catch(function () {});
      return;
    }
    var area = textElement("textarea");
    area.value = value;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); done(); } catch (_) {}
    area.remove();
  }

  function createPanel() {
    var splash = document.querySelector(".boot-splash");
    if (splash) splash.remove();
    var panel = textElement("section", "mobile-pairing");
    panel.setAttribute("role", "main");
    panel.setAttribute("aria-labelledby", "mobile-pairing-title");
    var card = textElement("div", "mobile-pairing-card");
    panel.appendChild(card);

    var title = textElement("h1", "mobile-pairing-title", "Pair this phone");
    title.id = "mobile-pairing-title";
    card.appendChild(title);
    card.appendChild(textElement("p", "mobile-pairing-copy", "This phone is not paired yet. Open the pairing link created in Nanobot Remote Access settings."));

    var status = textElement("p", "mobile-pairing-status", "Waiting for a pairing link.");
    status.setAttribute("aria-live", "polite");
    card.appendChild(status);

    var originRow = textElement("div", "mobile-pairing-row");
    originRow.appendChild(textElement("span", "mobile-pairing-label", "Server"));
    var originValue = textElement("code", "mobile-pairing-value", "not selected");
    originRow.appendChild(originValue);
    card.appendChild(originRow);

    var installationRow = textElement("div", "mobile-pairing-row");
    installationRow.appendChild(textElement("span", "mobile-pairing-label", "Installation ID"));
    var installationValue = textElement("code", "mobile-pairing-value", "loading");
    installationRow.appendChild(installationValue);
    var copyInstallation = textElement("button", "mobile-pairing-button", "Copy");
    copyInstallation.type = "button";
    copyInstallation.addEventListener("click", function () {
      copyText(installationValue.textContent || "", copyInstallation);
    });
    installationRow.appendChild(copyInstallation);
    card.appendChild(installationRow);

    var code = textElement("div", "mobile-pairing-code", "------");
    code.setAttribute("aria-label", "Comparison code");
    card.appendChild(code);
    var codeHint = textElement("p", "mobile-pairing-copy", "Compare this code with the server, then approve the phone in Remote Access settings.");
    card.appendChild(codeHint);

    var actions = textElement("div", "mobile-pairing-actions");
    var login = textElement("button", "mobile-pairing-button", "Open Tailscale login");
    login.type = "button";
    login.hidden = true;
    login.addEventListener("click", function () {
      void plugin.openLogin().catch(function (error) {
        status.textContent = String(error && error.message ? error.message : error);
      });
    });
    actions.appendChild(login);
    var retry = textElement("button", "mobile-pairing-button", "Retry");
    retry.type = "button";
    retry.hidden = true;
    actions.appendChild(retry);
    card.appendChild(actions);

    document.body.appendChild(panel);
    return {
      panel: panel,
      status: status,
      origin: originValue,
      installation: installationValue,
      code: code,
      codeHint: codeHint,
      login: login,
      retry: retry
    };
  }

  var pairingPresented = false;

  function presentPairing() {
    if (pairingPresented) return;
    pairingPresented = true;
    var stale = document.querySelector(".mobile-pairing");
    if (stale) stale.remove();
    var ui = createPanel();
    var pairingLinkTimer = null;
    var pollTimer = null;
    var stopped = false;
    var lastLink = null;

    function stopTimers() {
      if (pairingLinkTimer) clearInterval(pairingLinkTimer);
      if (pollTimer) clearTimeout(pollTimer);
      pairingLinkTimer = null;
      pollTimer = null;
    }

    function finish() {
      if (stopped) return;
      stopped = true;
      stopTimers();
      ui.panel.remove();
      resolveReady();
    }

    function fail(message) {
      if (stopped) return;
      stopTimers();
      ui.status.textContent = message;
      ui.status.classList.add("is-error");
      ui.retry.hidden = false;
    }

    async function poll(pairingId) {
      if (stopped) return;
      try {
        var result = await plugin.pairPoll({ pairingId: pairingId });
        if (result && result.status === "approved") {
          ui.status.textContent = "Paired. Loading Nanobot...";
          finish();
          return;
        }
        if (result && result.comparison_code) {
          ui.code.textContent = result.comparison_code;
        }
        ui.status.textContent = "Waiting for approval on the server.";
        ui.status.classList.remove("is-error");
        pollTimer = setTimeout(function () { void poll(pairingId); }, 2000);
      } catch (error) {
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0 || message.indexOf("needs_login") >= 0) {
          ui.status.textContent = "Tailscale login is required before pairing can continue.";
          ui.login.hidden = false;
          ui.retry.hidden = false;
          return;
        }
        fail(message);
      }
    }

    async function begin(link) {
      stopTimers();
      lastLink = link;
      ui.retry.hidden = true;
      ui.login.hidden = true;
      ui.status.classList.remove("is-error");
      ui.origin.textContent = link.origin || "unknown";
      ui.installation.textContent = link.installationId || "unknown";
      ui.status.textContent = "Connecting to the server...";
      try {
        var collected = await plugin.pairStart({
          origin: link.origin,
          installationId: link.installationId,
          label: "android"
        });
        ui.installation.textContent = collected.installationId || link.installationId;
        ui.code.textContent = collected.comparison_code || "------";
        ui.status.textContent = "Waiting for approval on the server.";
        void poll(collected.pairing_id);
      } catch (error) {
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0 || message.indexOf("needs_login") >= 0) {
          ui.status.textContent = "Tailscale login is required before pairing can continue.";
          ui.login.hidden = false;
          ui.retry.hidden = false;
          return;
        }
        fail(message);
      }
    }

    function watchForLink() {
      pairingLinkTimer = setInterval(async function () {
        try {
          var link = await plugin.pendingPairingLink();
          if (!link || !link.origin) return;
          if (pairingLinkTimer) clearInterval(pairingLinkTimer);
          pairingLinkTimer = null;
          await begin(link);
        } catch (_) {
          /* A malformed pending link is handled by pairStart when retried. */
        }
      }, 1000);
    }

    ui.retry.addEventListener("click", function () {
      ui.retry.hidden = true;
      ui.login.hidden = true;
      ui.status.classList.remove("is-error");
      ui.status.textContent = "Waiting for a pairing link.";
      if (lastLink) {
        void begin(lastLink);
      } else {
        watchForLink();
      }
    });

    void (async function () {
      try {
        var identity = await plugin.pairingIdentity();
        ui.installation.textContent = identity.installationId;
        if (identity.origin) ui.origin.textContent = identity.origin;
        console.log("NANOBOT_MOBILE_PAIRING_ID " + identity.installationId);
        console.log("NANOBOT_PAIRING_PANELS " + document.querySelectorAll(".mobile-pairing").length);
      } catch (error) {
        fail(String(error && error.message ? error.message : error));
        return;
      }
      watchForLink();
    })();
  }

  void (async function () {
    var target = null;
    try {
      target = await plugin.target();
    } catch (_) {
      target = null;
    }
    if (target && target.hasAccess) {
      try {
        await transport.start(target.origin, target.installationId, "nanobot-" + String(target.installationId).slice(-6));
        for (var i = 0; i < 180; i += 1) {
          var status = await plugin.status();
          if (status && status.state === "ready") {
            resolveReady();
            return;
          }
          if (status && status.state === "error") throw new Error("native transport failed");
          await new Promise(function (resolve) { setTimeout(resolve, 500); });
        }
        throw new Error("native transport readiness timeout");
      } catch (error) {
        rejectReady(error);
        return;
      }
    }
    presentPairing();
  })();
})();
