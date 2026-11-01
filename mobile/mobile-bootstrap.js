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

    var title = textElement("h1", "mobile-pairing-title", "Connect this phone");
    title.id = "mobile-pairing-title";
    card.appendChild(title);
    card.appendChild(textElement("p", "mobile-pairing-copy", "Pair with the Nanobot desktop app."));

    var statusList = textElement("div", "mobile-pairing-status-list");
    var serverRow = textElement("div", "mobile-pairing-status-row");
    serverRow.appendChild(textElement("span", "mobile-pairing-label", "Server"));
    var serverPill = textElement("span", "mobile-pairing-pill", "Not selected");
    serverRow.appendChild(serverPill);
    statusList.appendChild(serverRow);
    var nodeRow = textElement("div", "mobile-pairing-status-row");
    nodeRow.appendChild(textElement("span", "mobile-pairing-label", "Tailscale"));
    var nodePill = textElement("span", "mobile-pairing-pill", "Connects when pairing");
    nodeRow.appendChild(nodePill);
    statusList.appendChild(nodeRow);
    card.appendChild(statusList);

    var form = textElement("form", "mobile-pairing-form");
    var inputLabel = textElement("label", "mobile-pairing-label", "Pairing code or link");
    inputLabel.setAttribute("for", "mobile-pairing-input");
    form.appendChild(inputLabel);
    var inputRow = textElement("div", "mobile-pairing-input-row");
    var input = document.createElement("input");
    input.id = "mobile-pairing-input";
    input.className = "mobile-pairing-input";
    input.type = "text";
    input.autocomplete = "off";
    input.autocapitalize = "off";
    input.spellcheck = false;
    input.placeholder = "Paste the pairing link";
    inputRow.appendChild(input);
    var pairButton = textElement("button", "mobile-pairing-primary", "Pair");
    pairButton.type = "submit";
    inputRow.appendChild(pairButton);
    form.appendChild(inputRow);
    card.appendChild(form);

    var divider = textElement("div", "mobile-pairing-divider");
    divider.appendChild(textElement("span", "", "or"));
    card.appendChild(divider);

    var scan = textElement("button", "mobile-pairing-scan", "Scan QR code");
    scan.type = "button";
    card.appendChild(scan);

    var codeBlock = textElement("div", "mobile-pairing-code-block");
    codeBlock.hidden = true;
    codeBlock.appendChild(textElement("p", "mobile-pairing-copy", "Confirm this code on the desktop, then approve the phone."));
    var code = textElement("div", "mobile-pairing-code", "------");
    code.setAttribute("aria-label", "Comparison code");
    codeBlock.appendChild(code);
    card.appendChild(codeBlock);

    var status = textElement("p", "mobile-pairing-status", "Enter the pairing link shown on the desktop or scan its QR code.");
    status.setAttribute("aria-live", "polite");
    card.appendChild(status);

    var actions = textElement("div", "mobile-pairing-actions");
    var login = textElement("button", "mobile-pairing-button", "Open Tailscale login");
    login.type = "button";
    login.hidden = true;
    actions.appendChild(login);
    var retry = textElement("button", "mobile-pairing-button", "Retry");
    retry.type = "button";
    retry.hidden = true;
    actions.appendChild(retry);
    card.appendChild(actions);

    var details = document.createElement("details");
    details.className = "mobile-pairing-details";
    details.appendChild(textElement("summary", "", "Device details"));
    var installationRow = textElement("div", "mobile-pairing-installation");
    var installationValue = textElement("code", "mobile-pairing-value", "loading");
    installationRow.appendChild(installationValue);
    var copyInstallation = textElement("button", "mobile-pairing-button", "Copy");
    copyInstallation.type = "button";
    copyInstallation.addEventListener("click", function () {
      copyText(installationValue.textContent || "", copyInstallation);
    });
    installationRow.appendChild(copyInstallation);
    details.appendChild(installationRow);
    card.appendChild(details);

    document.body.appendChild(panel);
    return {
      panel: panel,
      status: status,
      server: serverPill,
      node: nodePill,
      installation: installationValue,
      input: input,
      form: form,
      code: code,
      codeBlock: codeBlock,
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
    var identityRef = null;

    function hostLabel(origin) {
      try {
        return new URL(origin).host;
      } catch (_) {
        return origin || "Not selected";
      }
    }

    function setPill(element, text, tone) {
      element.textContent = text;
      element.classList.remove("is-ok", "is-warn");
      if (tone) element.classList.add(tone);
    }

    function parseLink(value) {
      var url;
      try {
        url = new URL(value);
      } catch (_) {
        return null;
      }
      if (url.protocol !== "nanobot:" || url.host !== "pair") return null;
      var origin = url.searchParams.get("origin") || "";
      var installationId = url.searchParams.get("installation") || "";
      var challenge = url.searchParams.get("challenge") || "";
      if (!origin || !installationId || !challenge) return null;
      return { origin: origin, installationId: installationId, challenge: challenge };
    }

    function linkFromInput(raw) {
      var value = String(raw || "").trim();
      if (!value) return null;
      if (value.indexOf("nanobot://") === 0) return value;
      if (value.indexOf("origin=") >= 0) return "nanobot://pair?" + value.replace(/^\?/, "");
      if (identityRef && identityRef.origin && /^[A-Za-z0-9._-]{16,256}$/.test(value)) {
        return "nanobot://pair?" + [
          "origin=" + encodeURIComponent(identityRef.origin),
          "installation=" + encodeURIComponent(identityRef.installationId),
          "challenge=" + encodeURIComponent(value)
        ].join("&");
      }
      return null;
    }

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
          setPill(ui.node, "Connected", "is-ok");
          ui.status.textContent = "Paired. Loading Nanobot...";
          finish();
          return;
        }
        if (result && result.comparison_code) {
          ui.code.textContent = result.comparison_code;
        }
        ui.status.textContent = "Waiting for approval on the desktop.";
        ui.status.classList.remove("is-error");
        pollTimer = setTimeout(function () { void poll(pairingId); }, 2000);
      } catch (error) {
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0 || message.indexOf("needs_login") >= 0) {
          setPill(ui.node, "Login required", "is-warn");
          ui.status.textContent = "Sign in to Tailscale, then retry.";
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
      setPill(ui.server, hostLabel(link.origin), "is-ok");
      ui.server.title = link.origin || "";
      setPill(ui.node, "Connecting", "is-warn");
      ui.installation.textContent = link.installationId || "unknown";
      ui.codeBlock.hidden = false;
      ui.code.textContent = "------";
      ui.status.textContent = "Connecting to the server...";
      try {
        var collected = await plugin.pairStart({
          origin: link.origin,
          installationId: link.installationId,
          challenge: link.challenge,
          label: "android"
        });
        ui.installation.textContent = collected.installationId || link.installationId;
        ui.code.textContent = collected.comparison_code || "------";
        setPill(ui.node, "Connected", "is-ok");
        ui.status.textContent = "Confirm the code on the desktop, then approve.";
        void poll(collected.pairing_id);
      } catch (error) {
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0 || message.indexOf("needs_login") >= 0) {
          setPill(ui.node, "Login required", "is-warn");
          ui.status.textContent = "Sign in to Tailscale, then retry.";
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

    ui.form.addEventListener("submit", function (event) {
      event.preventDefault();
      var link = linkFromInput(ui.input.value);
      var parsed = link ? parseLink(link) : null;
      if (!parsed) {
        ui.status.textContent = "Paste the full pairing link shown on the desktop, or scan its QR code.";
        ui.status.classList.add("is-error");
        return;
      }
      void begin(parsed);
    });

    ui.retry.addEventListener("click", function () {
      ui.retry.hidden = true;
      ui.login.hidden = true;
      ui.status.classList.remove("is-error");
      setPill(ui.node, "Connecting", "is-warn");
      if (lastLink) {
        void begin(lastLink);
      } else {
        ui.status.textContent = "Enter the pairing link or scan the QR code.";
        watchForLink();
      }
    });

    ui.login.addEventListener("click", function () {
      void plugin.openLogin().catch(function (error) {
        ui.status.textContent = String(error && error.message ? error.message : error);
      });
    });

    void (async function () {
      try {
        var identity = await plugin.pairingIdentity();
        identityRef = identity;
        ui.installation.textContent = identity.installationId;
        if (identity.origin) setPill(ui.server, hostLabel(identity.origin), "is-ok");
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
    async function connectTarget(link) {
      await transport.start(link.origin, link.installationId, "nanobot-" + String(link.installationId).slice(-6));
      for (var i = 0; i < 180; i += 1) {
        var status = await plugin.status();
        if (status && status.state === "ready") return;
        if (status && status.state === "error") throw new Error("native transport failed");
        await new Promise(function (resolve) { setTimeout(resolve, 500); });
      }
      throw new Error("native transport readiness timeout");
    }

    function presentReauth(link, initialMessage) {
      var stale = document.querySelector(".mobile-reauth");
      if (stale) stale.remove();
      var panel = textElement("section", "mobile-pairing mobile-reauth");
      panel.setAttribute("role", "main");
      var card = textElement("div", "mobile-pairing-card");
      panel.appendChild(card);
      var title = textElement("h1", "mobile-pairing-title", "Reconnect this phone");
      card.appendChild(title);
      card.appendChild(textElement("p", "mobile-pairing-copy", "Your Nanobot pairing is saved. Tailscale needs a fresh sign-in before this phone can reach the server."));
      var status = textElement("p", "mobile-pairing-status", initialMessage || "Tailscale sign-in is required.");
      status.setAttribute("aria-live", "polite");
      card.appendChild(status);
      var actions = textElement("div", "mobile-pairing-actions");
      var login = textElement("button", "mobile-pairing-button", "Open Tailscale login");
      login.type = "button";
      login.addEventListener("click", function () {
        status.classList.remove("is-error");
        status.textContent = "Opening Tailscale sign-in...";
        void plugin.openLogin().catch(function (error) {
          status.textContent = String(error && error.message ? error.message : error);
        });
      });
      actions.appendChild(login);
      var retry = textElement("button", "mobile-pairing-button", "Retry");
      retry.type = "button";
      retry.addEventListener("click", function () {
        status.classList.remove("is-error");
        status.textContent = "Connecting to the server...";
        void connectTarget(link).then(function () {
          panel.remove();
          resolveReady();
        }).catch(function (error) {
          status.textContent = String(error && error.message ? error.message : error);
          status.classList.add("is-error");
        });
      });
      actions.appendChild(retry);
      card.appendChild(actions);
      document.body.appendChild(panel);
    }

    if (target && target.hasAccess) {
      try {
        await connectTarget(target);
        resolveReady();
      } catch (error) {
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0 || message.indexOf("needs_login") >= 0) {
          presentReauth(target, message);
        } else {
          rejectReady(error);
        }
      }
      return;
    }
    presentPairing();
  })();
})();
