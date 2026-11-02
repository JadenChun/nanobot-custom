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
  window.nanobotMobileReady = new Promise(function (resolve) {
    resolveReady = resolve;
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
    codeBlock.appendChild(textElement("p", "mobile-pairing-copy", "Confirm this code matches the desktop, then approve the phone there."));
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
    var cancel = textElement("button", "mobile-pairing-button", "Cancel");
    cancel.type = "button";
    cancel.hidden = true;
    actions.appendChild(cancel);
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
      scan: scan,
      code: code,
      codeBlock: codeBlock,
      login: login,
      retry: retry,
      cancel: cancel
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
    var identityPromise = null;
    var generation = 0;
    var state = "idle";

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

    function setState(next) {
      state = next;
      ui.panel.dataset.state = next;
      ui.codeBlock.hidden = next !== "waiting";
      ui.login.hidden = next !== "login";
      ui.retry.hidden = next !== "error" && next !== "login";
      ui.cancel.hidden = next === "idle";
      ui.cancel.textContent = next === "waiting" ? "Cancel" : "Use a different link";
      if (next === "idle") {
        ui.form.reset();
        ui.code.textContent = "------";
        setPill(ui.node, "Connects when pairing");
        setPill(ui.server, identityRef && identityRef.origin ? hostLabel(identityRef.origin) : "Not selected", identityRef && identityRef.origin ? "is-ok" : null);
        ui.status.textContent = "Scan the QR code on the desktop or paste its pairing link.";
        ui.status.classList.remove("is-error");
      }
    }

    function fail(message) {
      if (stopped) return;
      stopTimers();
      setState("error");
      setPill(ui.node, "Not connected", "is-warn");
      ui.status.textContent = message;
      ui.status.classList.add("is-error");
    }

    function needLogin(run) {
      if (stopped || (run !== undefined && run !== generation)) return;
      stopTimers();
      setState("login");
      setPill(ui.node, "Sign in required", "is-warn");
      ui.status.textContent = "Sign in to Tailscale, then retry.";
      ui.status.classList.remove("is-error");
    }

    async function poll(pairingId, run) {
      if (stopped || run !== generation || state !== "waiting") return;
      try {
        var result = await plugin.pairPoll({ pairingId: pairingId });
        if (stopped || run !== generation || state !== "waiting") return;
        if (result && result.status === "approved") {
          setPill(ui.node, "Connected", "is-ok");
          ui.status.textContent = "Paired. Loading Nanobot...";
          finish();
          return;
        }
        if (result && result.comparison_code) ui.code.textContent = result.comparison_code;
        ui.status.textContent = "Waiting for approval on the desktop.";
        ui.status.classList.remove("is-error");
        pollTimer = setTimeout(function () { void poll(pairingId, run); }, 2000);
      } catch (error) {
        if (stopped || run !== generation || state !== "waiting") return;
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0 || message.indexOf("needs_login") >= 0) {
          needLogin(run);
          return;
        }
        fail(message);
      }
    }

    function loadIdentity() {
      if (identityRef) return Promise.resolve(identityRef);
      if (identityPromise) return identityPromise;
      identityPromise = plugin.pairingIdentity().then(function (identity) {
        if (!identity || !identity.installationId) {
          throw new Error("pairing identity is unavailable");
        }
        identityRef = identity;
        console.log("NANOBOT_MOBILE_PAIRING_ID " + identity.installationId);
        console.log("NANOBOT_PAIRING_PANELS " + document.querySelectorAll(".mobile-pairing").length);
        return identity;
      }).catch(function (error) {
        identityPromise = null;
        throw error;
      });
      return identityPromise;
    }

    async function begin(link) {
      var run = ++generation;
      stopTimers();
      lastLink = link;
      setState("connecting");
      setPill(ui.server, hostLabel(link.origin), "is-ok");
      ui.server.title = link.origin || "";
      setPill(ui.node, "Starting", "is-warn");
      var identity;
      try {
        identity = await loadIdentity();
      } catch (error) {
        if (run !== generation || lastLink !== link || state !== "connecting") return;
        fail(String(error && error.message ? error.message : error));
        return;
      }
      if (run !== generation || lastLink !== link || state !== "connecting") return;
      var localInstallation = identity.installationId;
      ui.installation.textContent = localInstallation;
      ui.status.textContent = "Connecting to the server...";
      ui.status.classList.remove("is-error");
      try {
        var collected = await plugin.pairStart({
          origin: link.origin,
          installationId: localInstallation,
          challenge: link.challenge,
          label: "android"
        });
        if (run !== generation || lastLink !== link || state !== "connecting") return;
        ui.installation.textContent = collected.installationId || localInstallation;
        ui.code.textContent = collected.comparison_code || "------";
        setState("waiting");
        setPill(ui.node, "Connected", "is-ok");
        ui.status.textContent = "Confirm the code matches the desktop, then approve the phone there.";
        void poll(collected.pairing_id, run);
      } catch (error) {
        if (run !== generation || lastLink !== link || state !== "connecting") return;
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0 || message.indexOf("needs_login") >= 0) {
          needLogin(run);
          return;
        }
        fail(message);
      }
    }

    function watchForLink() {
      if (pairingLinkTimer) return;
      pairingLinkTimer = setInterval(async function () {
        if (stopped || state !== "idle") return;
        try {
          var link = await plugin.pendingPairingLink();
          if (!link || !link.origin) return;
          await begin(link);
        } catch (_) {
          /* A malformed pending link is ignored until the user retries. */
        }
      }, 1000);
    }

    ui.form.addEventListener("submit", function (event) {
      event.preventDefault();
      var raw = linkFromInput(ui.input.value);
      var parsed = raw ? parseLink(raw) : null;
      if (!parsed) {
        ui.status.textContent = "That pairing link is not valid. Paste the full link from the desktop.";
        ui.status.classList.add("is-error");
        return;
      }
      void begin(parsed);
    });

    ui.scan.addEventListener("click", function () {
      if (typeof plugin.scanPairingQr !== "function") {
        ui.status.textContent = "QR scanning is unavailable in this build.";
        return;
      }
      ui.status.textContent = "Opening QR scanner...";
      ui.status.classList.remove("is-error");
      void plugin.scanPairingQr().catch(function (error) {
        ui.status.textContent = String(error && error.message ? error.message : error);
      });
    });

    ui.login.addEventListener("click", function () {
      ui.status.textContent = "Complete the Tailscale sign-in, then retry.";
      void plugin.openLogin().catch(function (error) {
        ui.status.textContent = String(error && error.message ? error.message : error);
      });
    });

    ui.retry.addEventListener("click", function () {
      if (lastLink) {
        void begin(lastLink);
      } else {
        setState("idle");
        watchForLink();
      }
    });

    ui.cancel.addEventListener("click", function () {
      generation += 1;
      stopTimers();
      lastLink = null;
      setState("idle");
      watchForLink();
    });

    setState("idle");
    void (async function () {
      var startupGeneration = generation;
      var identity;
      try {
        identity = await loadIdentity();
      } catch (error) {
        if (generation !== startupGeneration || state !== "idle") return;
        fail(String(error && error.message ? error.message : error));
        return;
      }
      if (generation !== startupGeneration || state !== "idle") return;
      ui.installation.textContent = identity.installationId;
      if (identity.origin) {
        setPill(ui.server, hostLabel(identity.origin), "is-ok");
        ui.server.title = identity.origin;
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
        if (status && status.state === "needs_login") throw new Error("tailnet login required");
        if (status && status.state === "error") throw new Error("native transport failed");
        await new Promise(function (resolve) { setTimeout(resolve, 500); });
      }
      throw new Error("native transport readiness timeout");
    }

    function probeHandshake() {
      return new Promise(async function (resolve) {
        var requestId = 0;
        var done = false;
        var unsubscribe = null;
        var timer = setTimeout(function () { finish({ error: "timeout" }); }, 8000);
        function finish(result) {
          if (done) return;
          done = true;
          clearTimeout(timer);
          if (unsubscribe) unsubscribe();
          resolve(result);
        }
        try {
          unsubscribe = await transport.onEvent(function (event) {
            if (requestId === 0 || event.request_id !== requestId) return;
            if (event.type === "http_response") {
              finish({ status: event.status, body: event.body_base64 });
            } else if (event.type === "http_error") {
              finish({ error: event.code });
            }
          });
          requestId = await transport.http({
            operation: "mobile_handshake",
            method: "GET",
            path: "/mobile/v1/handshake"
          });
        } catch (error) {
          finish({ error: String(error && error.message ? error.message : error) });
        }
      });
    }

    async function requireValidAccess() {
      var probe = await probeHandshake();
      if (probe.error) return "unreachable";
      if (probe.status !== 401) return probe.status === 200 ? "ok" : "unreachable";
      var refreshed = false;
      try {
        refreshed = await plugin.refresh();
      } catch (error) {
        var message = String(error && error.message ? error.message : error);
        if (
          message.indexOf("pairing was revoked") < 0 &&
          message.indexOf("no refresh credential") < 0
        ) {
          return "unreachable";
        }
      }
      if (refreshed) {
        probe = await probeHandshake();
        if (probe.error) return "unreachable";
        if (probe.status !== 401) return probe.status === 200 ? "ok" : "unreachable";
      }
      try {
        await plugin.logout({ serverReachable: true });
      } catch (_) {
        /* The next launch still starts from the stored pairing state. */
      }
      window.location.reload();
      return "revoked";
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
          return requireValidAccess();
        }).then(function (access) {
          if (access === "revoked") return;
          if (access === "unreachable") {
            status.textContent = "Still can't reach this desktop. Check Tailscale and retry.";
            status.classList.add("is-error");
            return;
          }
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
        var access = await requireValidAccess();
        if (access === "revoked") return;
        if (access === "unreachable") {
          presentReauth(target, "Can't reach this desktop right now. Check Tailscale and retry.");
          return;
        }
        resolveReady();
      } catch (error) {
        var message = String(error && error.message ? error.message : error);
        presentReauth(target, message || "Can't reach this desktop right now. Check Tailscale and retry.");
      }
      return;
    }
    presentPairing();
  })();
})();
