/* Installed by build-shared-webui.sh before the shared React bundle. */
(function () {
  var plugin = window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.NativeNetwork;
  if (!plugin) return;

  var capacitor = window.Capacitor;
  if (capacitor) {
    try {
      var localNotifications = capacitor.Plugins && capacitor.Plugins.LocalNotifications;
      if (!localNotifications && capacitor.registerPlugin) {
        localNotifications = capacitor.registerPlugin("LocalNotifications");
      }
      if (localNotifications) {
        var alertChannel = "kohaku_task_alerts";
        var silentChannel = "kohaku_task_alerts_silent";
        // Android 7 has no channels. Its notifications use per-alert sounds.
        var channelReady = Promise.resolve()
          .then(function () {
            return localNotifications.createChannel({
              id: alertChannel, name: "Task updates", importance: 4, vibration: true
            });
          })
          .then(function () { return true; }, function (error) {
            return error && error.code === "UNAVAILABLE";
          });

        if (typeof localNotifications.addListener === "function") {
          Promise.resolve()
            .then(function () {
              return localNotifications.addListener("localNotificationActionPerformed", function (action) {
                var extra = action.notification && action.notification.extra;
                if (!extra || typeof extra.chatId !== "string" || !extra.chatId) return;
                window.location.hash = extra.temporary
                  ? "#/temporary/" + encodeURIComponent(extra.chatId)
                  : "#/chat/" + encodeURIComponent("websocket:" + extra.chatId);
              });
            })
            .catch(function () {});
        }

        function notificationId(key) {
          var hash = 2166136261;
          for (var i = 0; i < key.length; i++) {
            hash = Math.imul(hash ^ key.charCodeAt(i), 16777619);
          }
          // Reserve low IDs for native services, including the run keepalive.
          return 65536 + (hash >>> 0) % (2147483648 - 65536);
        }

        window.nanobotMobileNotifications = {
          requestPermission: async function () {
            try {
              var result = await localNotifications.requestPermissions();
              return !!result && result.display === "granted";
            } catch (_) {
              return false;
            }
          },
          notify: async function (options) {
            if (!await channelReady) return false;
            try {
              var audible = options.sound !== false;
              await localNotifications.schedule({
                notifications: [{
                  id: notificationId(options.tag || options.chatId || "kohaku-test"),
                  title: options.title || "Kohaku",
                  body: options.body || "",
                  channelId: audible ? alertChannel : silentChannel,
                  // Android 7 ignores channel sound settings.
                  sound: audible ? undefined : "kohaku_silent.wav",
                  extra: options.chatId ? {
                    chatId: options.chatId,
                    ...(options.temporary ? { temporary: true } : {})
                  } : {},
                  isExactNotification: false,
                  foreground: true,
                  smallIcon: "ic_stat_kohaku",
                  largeIcon: "ic_kohaku_notification"
                }]
              });
              return true;
            } catch (_) {
              return false;
            }
          }
        };
      }
    } catch (_) {
      // Optional notification setup must never block mobile transport boot.
    }
  }

  window.nanobotMobileRunKeepAlive = {
    start: async function (chatId) {
      await plugin.startRunKeepAlive({ chatId: chatId });
    },
    stop: async function (chatId) {
      await plugin.stopRunKeepAlive({ chatId: chatId });
    }
  };

  var transport = {
    start: async function (origin, installationId, hostname) {
      await plugin.start({ origin: origin, installationId: installationId, hostname: hostname });
    },
    status: async function () { return await plugin.status(); },
    remoteStatus: async function () { return await plugin.remoteStatus(); },
    remoteDevices: async function () { return await plugin.remoteDevices(); },
    remoteRevoke: async function (installationId) {
      return await plugin.remoteRevoke({ installationId: installationId });
    },
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
    logout: async function () {
      try {
        await plugin.logout({ serverReachable: true });
      } catch (_) {
        try {
          await plugin.logout({ serverReachable: false });
        } catch (_) {}
      }
      window.location.reload();
    },
    wsOpen: async function (path, headers) {
      var result = await plugin.wsOpen({ operation: "mobile_ws", path: path, headers: headers });
      return result.wsId;
    },
    wsSend: (function () {
      var sendChains = Object.create(null);
      return function (wsId, data, kind) {
        var chunkBytes = 256 * 1024;
        var frameKind = kind || "text";
        var encode = function (bytes) {
          var binary = "";
          for (var i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
          return btoa(binary);
        };
        if (data.length <= chunkBytes) {
          return plugin.wsSend({ wsId: wsId, kind: frameKind, dataBase64: encode(data) });
        }
        var previous = sendChains[wsId] || Promise.resolve();
        var next = previous.catch(function () {}).then(async function () {
          await plugin.wsSendBegin({ wsId: wsId, kind: frameKind });
          try {
            for (var offset = 0; offset < data.length; offset += chunkBytes) {
              var chunk = data.subarray(offset, Math.min(offset + chunkBytes, data.length));
              await plugin.wsSendChunk({ wsId: wsId, dataBase64: encode(chunk) });
            }
            await plugin.wsSendEnd({ wsId: wsId });
          } catch (error) {
            await plugin.wsSendEnd({ wsId: wsId, discard: true }).catch(function () {});
            throw error;
          }
        });
        sendChains[wsId] = next;
        return next.finally(function () {
          if (sendChains[wsId] === next) delete sendChains[wsId];
        });
      };
    })(),
    wsClose: async function (wsId) { await plugin.wsClose({ wsId: wsId }); },
    wsPing: async function (wsId) {
      var result = await plugin.wsPing({ wsId: wsId });
      return Boolean(result && result.ok);
    },
    wsRefreshAuth: async function (wsId) {
      var result = await plugin.wsRefreshAuth({ wsId: wsId });
      return Boolean(result && result.ok);
    },
    stop: async function () { await plugin.stop(); }
  };
  window.nanobotMobileTransport = transport;

  // Mirror WebUI console output to logcat: the packaged client has no devtools
  // bridge, so mobile-only connection faults are otherwise invisible.
  (function () {
    var probe = window.KohakuNativeProbe;
    if (!probe || typeof probe.log !== "function") return;
    ["log", "warn", "error"].forEach(function (level) {
      var original = console[level];
      console[level] = function () {
        try {
          var parts = Array.prototype.map.call(arguments, function (value) {
            if (typeof value === "string") return value;
            try { return JSON.stringify(value); } catch (_) { return String(value); }
          });
          probe.log("[" + level + "] " + parts.join(" ").slice(0, 400));
        } catch (_) {}
        original.apply(console, arguments);
      };
    });
  })();

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
    var copy = textElement("p", "mobile-pairing-copy", "Pair with the Kohaku desktop app.");
    card.appendChild(copy);

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

    var scan = textElement("button", "mobile-pairing-scan-primary", "Scan QR code");
    scan.type = "button";
    card.appendChild(scan);

    var linkDetails = document.createElement("details");
    linkDetails.className = "mobile-pairing-link";
    linkDetails.appendChild(textElement("summary", "", "Paste a pairing link instead"));
    var linkBody = textElement("div", "mobile-pairing-details-body");
    var form = textElement("form", "mobile-pairing-form");
    var inputLabel = textElement("label", "mobile-pairing-label", "Pairing link");
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
    input.placeholder = "kohaku://pair?…";
    inputRow.appendChild(input);
    var pairButton = textElement("button", "mobile-pairing-primary", "Pair with link");
    pairButton.type = "submit";
    inputRow.appendChild(pairButton);
    form.appendChild(inputRow);
    linkBody.appendChild(form);
    linkDetails.appendChild(linkBody);
    card.appendChild(linkDetails);

    var manual = document.createElement("details");
    manual.className = "mobile-pairing-manual";
    manual.appendChild(textElement("summary", "", "Use an 8-digit code instead"));
    var manualBody = textElement("div", "mobile-pairing-details-body");
    var manualOriginLabel = textElement("label", "mobile-pairing-label", "Desktop HTTPS address");
    manualOriginLabel.setAttribute("for", "mobile-pairing-origin");
    manualBody.appendChild(manualOriginLabel);
    var manualOrigin = document.createElement("input");
    manualOrigin.id = "mobile-pairing-origin";
    manualOrigin.className = "mobile-pairing-input";
    manualOrigin.type = "url";
    manualOrigin.placeholder = "https://your-tailnet-host.ts.net";
    manualOrigin.autocomplete = "off";
    manualBody.appendChild(manualOrigin);
    var manualCodeLabel = textElement("label", "mobile-pairing-label", "8-digit code");
    manualCodeLabel.setAttribute("for", "mobile-pairing-code-input");
    manualBody.appendChild(manualCodeLabel);
    var manualCode = document.createElement("input");
    manualCode.id = "mobile-pairing-code-input";
    manualCode.className = "mobile-pairing-input";
    manualCode.type = "text";
    manualCode.inputMode = "numeric";
    manualCode.maxLength = 8;
    manualCode.placeholder = "12345678";
    manualCode.autocomplete = "one-time-code";
    manualBody.appendChild(manualCode);
    var manualButton = textElement("button", "mobile-pairing-button", "Pair with code");
    manualButton.type = "button";
    manualBody.appendChild(manualButton);
    manual.appendChild(manualBody);
    card.appendChild(manual);

    var status = textElement("p", "mobile-pairing-status", "Scan the QR code on the desktop, paste its link, or enter the 8-digit code. Tailscale starts automatically; sign in if prompted.");
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


    document.body.appendChild(panel);
    return {
      panel: panel,
      title: title,
      copy: copy,
      status: status,
      server: serverPill,
      node: nodePill,
      input: input,
      form: form,
      linkDetails: linkDetails,
      scan: scan,
      manual: manual,
      manualOrigin: manualOrigin,
      manualCode: manualCode,
      manualButton: manualButton,
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
    var loginTimer = null;
    var stopped = false;
    var lastLink = null;
    var lastManual = null;
    var pendingSetup = null;
    var expiredRetries = 0;
    var identityRef = null;
    var identityPromise = null;
    var generation = 0;
    var state = "idle";
    var cancellationPending = false;

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
      if (url.protocol !== "kohaku:" || url.host !== "pair") return null;
      var origin = url.searchParams.get("origin") || "";
      var installationId = url.searchParams.get("installation") || "";
      var challenge = url.searchParams.get("challenge") || "";
      if (!origin || !challenge) return null;
      return { origin: origin, installationId: installationId, challenge: challenge };
    }

    function linkFromInput(raw) {
      var value = String(raw || "").trim();
      if (!value) return null;
      if (value.indexOf("kohaku://") === 0) return value;
      if (value.indexOf("origin=") >= 0) return "kohaku://pair?" + value.replace(/^\?/, "");
      if (identityRef && identityRef.origin && /^[A-Za-z0-9._-]{16,256}$/.test(value)) {
        return "kohaku://pair?" + [
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
      if (loginTimer) clearInterval(loginTimer);
      pairingLinkTimer = null;
      pollTimer = null;
      loginTimer = null;
    }

    function finish() {
      if (stopped) return;
      stopped = true;
      pendingSetup = null;
      stopTimers();
      ui.panel.remove();
      resolveReady();
    }

    function setState(next) {
      state = next;
      ui.panel.dataset.state = next;
      ui.login.hidden = next !== "login";
      ui.retry.hidden = next !== "error" && next !== "login";
      ui.cancel.hidden = next === "idle" || next === "accepting" || next === "cancelling";
      ui.cancel.textContent = next === "waiting" ? "Cancel" : "Use a different link";
      ui.form.hidden = next !== "idle";
      ui.linkDetails.hidden = next !== "idle";
      ui.scan.hidden = next !== "idle";
      ui.manual.hidden = next !== "idle";
      if (next === "idle") {
        ui.title.textContent = "Connect this phone";
        ui.copy.textContent = "Pair with the Kohaku desktop app.";
        ui.form.reset();
        ui.manualOrigin.value = "";
        ui.manualCode.value = "";
        setPill(ui.node, "Connects when pairing");
        setPill(ui.server, identityRef && identityRef.origin ? hostLabel(identityRef.origin) : "Not selected", identityRef && identityRef.origin ? "is-ok" : null);
        ui.status.textContent = "Scan the QR code on the desktop, paste its link, or enter the 8-digit code. Tailscale starts automatically; sign in if prompted.";
        ui.status.classList.remove("is-error");
      }
    }

    function fail(message) {
      if (stopped) return;
      stopTimers();
      setState("error");
      var httpError = /^HTTP [0-9]+/.test(message);
      setPill(ui.node, httpError ? "Service error" : "Not connected", "is-warn");
      if (message.indexOf("HTTP 502") >= 0) {
        message = "Desktop returned 502. Check Tailscale Serve and the gateway.";
      }
      ui.status.textContent = message;
      ui.status.classList.add("is-error");
    }

    function needLogin(run) {
      if (stopped || (run !== undefined && run !== generation)) return;
      stopTimers();
      setState("login");
      ui.title.textContent = "Connect Tailscale";
      ui.copy.textContent = "Sign in before pairing this phone with the desktop.";
      setPill(ui.node, "Sign in required", "is-warn");
      ui.status.textContent =
        "Tailscale sign-in is required before pairing. Complete it and this screen resumes automatically.";
      ui.status.classList.remove("is-error");
      if (!loginTimer) {
        loginTimer = setInterval(async function () {
          if (stopped || state !== "login") return;
          try {
            var current = await plugin.status();
            if (current && current.state === "ready") {
              retryPending();
            } else if (current && current.state === "error") {
              fail(current.message || "Tailscale connection failed");
            }
          } catch (_) {
            /* Keep the login step visible while the node is still starting. */
          }
        }, 2000);
      }
    }

    async function poll(pairingId, run) {
      if (stopped || run !== generation || state !== "waiting") return;
      try {
        var result = await plugin.pairPoll({ pairingId: pairingId });
        if (stopped || run !== generation || state !== "waiting") return;
        if (result && result.status === "approved") {
          setState("accepting");
          await plugin.pairAccept({ pairingId: pairingId });
          if (stopped || run !== generation || state !== "accepting") return;
          setPill(ui.node, "Connected", "is-ok");
          ui.status.textContent = "Paired. Kohaku is loading…";
          finish();
          return;
        }
        expiredRetries = 0;
        ui.status.textContent = "Waiting for approval on the desktop.";
        ui.status.classList.remove("is-error");
        pollTimer = setTimeout(function () { void poll(pairingId, run); }, 2000);
      } catch (error) {
        if (stopped || run !== generation || (state !== "waiting" && state !== "accepting")) return;
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0 || message.indexOf("needs_login") >= 0) {
          needLogin(run);
          return;
        }
        if (message.indexOf("expired") >= 0 || message.indexOf("pairing_expired") >= 0) {
          ui.status.textContent = "Continue setup on the desktop: renew the pairing code, then this phone resumes automatically.";
          ui.status.classList.add("is-error");
          if (expiredRetries < 60) {
            expiredRetries += 1;
            pollTimer = setTimeout(function () { void poll(pairingId, run); }, 5000);
          } else {
            fail("Pairing setup expired. Create a new pairing code on the desktop.");
          }
          return;
        }
        if (
          /HTTP (403|404|410)\b/.test(message) &&
          /device_revoked|pairing_rejected|pairing_not_found|pairing_consumed/.test(message)
        ) {
          await discardStalePairing();
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
        console.log("KOHAKU_MOBILE_PAIRING_ID " + identity.installationId);
        console.log("KOHAKU_PAIRING_PANELS " + document.querySelectorAll(".mobile-pairing").length);
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
      lastManual = null;
      setState("connecting");
      if (pendingSetup && pendingSetup.kind === "link" && pendingSetup.challenge === link.challenge) {
        setState("waiting");
        setPill(ui.server, hostLabel(link.origin), "is-ok");
        void poll(pendingSetup.pairingId, run);
        return;
      }
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
        pendingSetup = { kind: "link", challenge: link.challenge, pairingId: collected.pairing_id };
        expiredRetries = 0;
        setState("waiting");
        setPill(ui.node, "Connected", "is-ok");
        ui.status.textContent = "Waiting for the desktop owner to approve this phone.";
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

    async function beginCode(origin, code) {
      var run = ++generation;
      stopTimers();
      lastLink = null;
      lastManual = { origin: origin, code: code };
      setState("connecting");
      if (pendingSetup && pendingSetup.kind === "code" && pendingSetup.origin === origin && pendingSetup.code === code) {
        setState("waiting");
        setPill(ui.server, hostLabel(origin), "is-ok");
        void poll(pendingSetup.pairingId, run);
        return;
      }
      setPill(ui.server, hostLabel(origin), "is-ok");
      setPill(ui.node, "Starting", "is-warn");
      var identity;
      try {
        identity = await loadIdentity();
      } catch (error) {
        if (run !== generation || state !== "connecting") return;
        fail(String(error && error.message ? error.message : error));
        return;
      }
      if (run !== generation || state !== "connecting") return;
      var localInstallation = identity.installationId;
      ui.status.textContent = "Connecting to the server...";
      ui.status.classList.remove("is-error");
      try {
        var collected = await plugin.pairStartCode({
          origin: origin,
          installationId: localInstallation,
          code: code,
          label: "android"
        });
        if (run !== generation || state !== "connecting") return;
        pendingSetup = { kind: "code", origin: origin, code: code, pairingId: collected.pairing_id };
        expiredRetries = 0;
        setState("waiting");
        setPill(ui.node, "Connected", "is-ok");
        ui.status.textContent = "Waiting for the desktop owner to approve this phone.";
        void poll(collected.pairing_id, run);
      } catch (error) {
        if (run !== generation || state !== "connecting") return;
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
        if (stopped || state === "waiting" || state === "accepting" || state === "cancelling") return;
        try {
          var link = await plugin.pendingPairingLink();
          if (!link || !link.origin) return;
          await begin(link);
        } catch (error) {
          var linkError = String(error && error.message ? error.message : error);
          if (linkError && state === "idle") {
            ui.status.textContent = "Invalid pairing link: " + linkError;
            ui.status.classList.add("is-error");
          }
        }
      }, 1000);
    }

    ui.manualButton.addEventListener("click", function () {
      var manualOrigin = String(ui.manualOrigin.value || "").trim();
      var manualCode = String(ui.manualCode.value || "").trim();
      if (!/^https:\/\//.test(manualOrigin) || !/^\d{8}$/.test(manualCode)) {
        ui.status.textContent = "Enter an HTTPS server address and the 8-digit code shown on the desktop.";
        ui.status.classList.add("is-error");
        return;
      }
      void beginCode(manualOrigin, manualCode);
    });

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
      ui.status.textContent = "Opening Tailscale sign-in…";
      void plugin.openLogin().then(function () {
        if (state === "login") {
          ui.status.textContent =
            "Complete the Tailscale sign-in, then return here. This screen resumes automatically.";
        }
      }).catch(function (error) {
        ui.status.textContent = String(error && error.message ? error.message : error);
      });
    });

    async function restorePendingPairing(run) {
      if (typeof plugin.pendingPairing !== "function") return false;
      var pending = null;
      try {
        pending = await plugin.pendingPairing();
      } catch (_) {
        return false;
      }
      if (stopped || run !== generation || state !== "idle") return false;
      if (pending && pending.cancellationPending === true) {
        void cancelPairing();
        return true;
      }
      if (!pending || pending.pending !== true || !pending.pairingId) return false;
      pendingSetup = {
        kind: "link",
        challenge: pending.challenge,
        pairingId: pending.pairingId
      };
      lastLink = { origin: pending.origin, challenge: pending.challenge };
      setState("waiting");
      setPill(ui.server, hostLabel(pending.origin), "is-ok");
      setPill(ui.node, "Connected", "is-ok");
      ui.status.textContent = "Waiting for the desktop owner to approve this phone.";
      void poll(pending.pairingId, run);
      return true;
    }

    function retryPending() {
      if (cancellationPending) {
        void cancelPairing();
      } else if (pendingSetup) {
        var run = ++generation;
        stopTimers();
        setState("waiting");
        setPill(ui.node, "Connected", "is-ok");
        void poll(pendingSetup.pairingId, run);
      } else if (lastLink) {
        void begin(lastLink);
      } else if (lastManual) {
        void beginCode(lastManual.origin, lastManual.code);
      } else {
        setState("idle");
        void restorePendingPairing(generation).then(function (restored) {
          if (!restored && state === "idle") watchForLink();
        });
      }
    }

    ui.retry.addEventListener("click", retryPending);

    function finishCancellation() {
      cancellationPending = false;
      lastLink = null;
      lastManual = null;
      pendingSetup = null;
      setState("idle");
      watchForLink();
    }

    async function discardStalePairing() {
      try {
        if (typeof plugin.logout === "function") {
          await plugin.logout({ serverReachable: true });
        }
      } catch (_) {}
      if (stopped) return;
      finishCancellation();
    }

    async function cancelPairing() {
      if (stopped || state === "accepting" || state === "cancelling") return;
      var run = ++generation;
      cancellationPending = true;
      stopTimers();
      setState("cancelling");
      ui.status.textContent = "Cancelling pairing...";
      try {
        if (typeof plugin.pairCancel !== "function") {
          throw new Error("native pairing cancellation is unavailable");
        }
        await plugin.pairCancel();
        if (stopped || run !== generation) return;
        finishCancellation();
      } catch (error) {
        if (stopped || run !== generation) return;
        var message = String(error && error.message ? error.message : error);
        if (message.indexOf("tailnet login required") >= 0) {
          needLogin(run);
          return;
        }
        // The owner already dropped this attempt, or the local challenge has no
        // collector proof left to cancel remotely. Clear the local record and
        // return to the pairing page instead of trapping the phone on Retry.
        if (
          /^HTTP 4\d\d\b/.test(message) ||
          message.indexOf("pairing challenge is") >= 0 ||
          message.indexOf("pairing secret is") >= 0 ||
          message.indexOf("pairing id is") >= 0 ||
          message.indexOf("no pending pairing challenge") >= 0
        ) {
          try {
            await plugin.logout({ serverReachable: true });
          } catch (_) {}
          if (stopped || run !== generation) return;
          finishCancellation();
          return;
        }
        if (/keystore|corrupt|invalidated|mismatch/i.test(message)) {
          try {
            await plugin.logout({ serverReachable: false });
          } catch (_) {}
          if (stopped || run !== generation) return;
          finishCancellation();
          return;
        }
        fail("Pairing cancellation failed: " + message);
      }
    }

    ui.cancel.addEventListener("click", cancelPairing);

    setState("idle");
    void (async function () {
      var startupGeneration = generation;
      var identity;
      try {
        identity = await loadIdentity();
      } catch (error) {
        if (generation !== startupGeneration || state !== "idle") return;
        var idError = String(error && error.message ? error.message : error);
        // Recover from corrupted/invalidated local Keystore custody by clearing
        // local state once before falling back to an error screen.
        if (/keystore|corrupt|invalidated|mismatch/i.test(idError)) {
          try {
            await plugin.logout({ serverReachable: false });
            identity = await loadIdentity();
          } catch (retryError) {
            if (generation !== startupGeneration || state !== "idle") return;
            fail(String(retryError && retryError.message ? retryError.message : retryError || error));
            return;
          }
        } else {
          fail(idError);
          return;
        }
      }
      if (generation !== startupGeneration || state !== "idle") return;
      if (identity.origin) {
        setPill(ui.server, hostLabel(identity.origin), "is-ok");
        ui.server.title = identity.origin;
      }
      if (
        identity.origin &&
        identity.hasAccess !== true &&
        typeof plugin.connectTailscale === "function"
      ) {
        setPill(ui.node, "Checking", "is-warn");
        var tailscaleResolved = false;
        for (var attempt = 0; attempt < 8; attempt += 1) {
          if (generation !== startupGeneration || state !== "idle") return;
          try {
            var readiness = await plugin.connectTailscale({
              origin: identity.origin,
              installationId: identity.installationId
            });
            if (generation !== startupGeneration || state !== "idle") return;
            if (readiness && readiness.state === "needs_login") {
              needLogin(startupGeneration);
              return;
            }
            if (readiness && readiness.state === "ready") {
              setPill(ui.node, "Connected", "is-ok");
              tailscaleResolved = true;
              break;
            }
            if (readiness && readiness.state === "error") {
              setPill(ui.node, "Not connected", "is-warn");
              tailscaleResolved = true;
              break;
            }
          } catch (_) {
            /* The node may still be starting; keep checking. */
          }
          await new Promise(function (resolve) { setTimeout(resolve, 1000); });
        }
        if (!tailscaleResolved && generation === startupGeneration && state === "idle") {
          setPill(ui.node, "Connects when pairing");
        }
      }
      var startupLink = null;
      try {
        startupLink = await plugin.pendingPairingLink();
      } catch (_) {
        startupLink = null;
      }
      if (startupLink && startupLink.origin) {
        await begin(startupLink);
        return;
      }
      if (!(await restorePendingPairing(startupGeneration))) {
        watchForLink();
      }
    })();
  }

  void (async function () {
    var target = null;
    try {
      target = await plugin.target();
    } catch (_) {
      target = null;
    }
    function setSplashStatus(message) {
      var note = document.querySelector(".boot-splash-status");
      if (!note) return;
      note.hidden = false;
      note.textContent = message;
    }

    async function connectTarget(link) {
      setSplashStatus("Connecting to Tailscale…");
      await transport.start(link.origin, link.installationId, "kohaku-" + String(link.installationId).slice(-6));
      for (var i = 0; i < 40; i += 1) {
        var status = await plugin.status();
        if (status && status.state === "ready") {
          setSplashStatus("Verifying server…");
          return;
        }
        if (status && status.state === "needs_login") throw new Error("tailnet login required");
        if (status && status.state === "error") {
          throw new Error("native transport failed" + (status.message ? ": " + status.message : ""));
        }
        if (i === 10) setSplashStatus("Still connecting to Tailscale…");
        await new Promise(function (resolve) { setTimeout(resolve, 500); });
      }
      throw new Error("Timed out connecting to Tailscale. Check your connection and retry.");
    }

    var activeHandshakeProbe = null;

    function probeHandshake() {
      if (activeHandshakeProbe) return activeHandshakeProbe.promise;
      var probe = {
        id: 0,
        done: false,
        launchSettled: false,
        cancelRequested: false,
        unsubscribe: null,
        early: new Map(),
        promise: null,
        resolve: null,
        timer: null
      };
      activeHandshakeProbe = probe;
      probe.promise = new Promise(function (resolve) { probe.resolve = resolve; });

      var maxEarlyEvents = 64;
      function releaseIfSettled() {
        if (probe.done && probe.launchSettled && activeHandshakeProbe === probe) {
          activeHandshakeProbe = null;
        }
      }
      function clearEarly() {
        probe.early.clear();
      }
      function cancel(id) {
        if (!id || probe.cancelRequested) return;
        probe.cancelRequested = true;
        try {
          void Promise.resolve(transport.cancel(id)).catch(function () {});
        } catch (_) {}
      }
      function unsubscribe() {
        if (!probe.unsubscribe) return;
        var remove = probe.unsubscribe;
        probe.unsubscribe = null;
        try { remove(); } catch (_) {}
      }
      function finish(result, terminal) {
        if (probe.done) return;
        probe.done = true;
        clearTimeout(probe.timer);
        unsubscribe();
        clearEarly();
        if (!terminal && probe.id) cancel(probe.id);
        probe.resolve(result);
        releaseIfSettled();
      }
      function terminalResult(event) {
        return event.type === "http_response"
          ? { status: typeof event.status === "number" && Number.isFinite(event.status) ? event.status : 0 }
          : { error: String(event.code || "native_request_failed").slice(0, 128) };
      }
      function handleEvent(event) {
        if (probe.done || (event.type !== "http_response" && event.type !== "http_error")) return;
        if (typeof event.request_id !== "number" || !Number.isSafeInteger(event.request_id) || event.request_id <= 0) return;
        if (probe.id) {
          if (event.request_id !== probe.id) return;
          finish(terminalResult(event), true);
          return;
        }
        if (probe.early.has(event.request_id)) return;
        if (probe.early.size >= maxEarlyEvents) {
          finish({ error: "handshake completion queue is full" }, false);
          return;
        }
        probe.early.set(event.request_id, terminalResult(event));
      }

      probe.timer = setTimeout(function () { finish({ error: "timeout" }, false); }, 8000);
      void (async function () {
        try {
          var remove = await transport.onEvent(handleEvent);
          if (probe.done) {
            try { remove(); } catch (_) {}
            probe.launchSettled = true;
            releaseIfSettled();
            return;
          }
          probe.unsubscribe = remove;
          var id = await transport.http({
            operation: "mobile_handshake",
            method: "GET",
            path: "/mobile/v1/handshake"
          });
          probe.launchSettled = true;
          if (typeof id !== "number" || !Number.isSafeInteger(id) || id <= 0) {
            finish({ error: "handshake request was not admitted" }, false);
            releaseIfSettled();
            return;
          }
          probe.id = id;
          if (probe.done) {
            cancel(id);
            releaseIfSettled();
            return;
          }
          var early = probe.early.get(id);
          clearEarly();
          if (early) finish(early, true);
          releaseIfSettled();
        } catch (error) {
          probe.launchSettled = true;
          finish({ error: String(error && error.message ? error.message : error) }, false);
          releaseIfSettled();
        }
      })();
      return probe.promise;
    }

    function refreshNeedsPairing(message) {
      if (message === "pairing was revoked" || message === "pairing upgrade required") return true;
      return /^HTTP (400|401|403|410)\b/.test(message);
    }

    async function requireValidAccess() {
      var probe = await probeHandshake();
      if (probe.error) return "unreachable";
      if (probe.status !== 401) return probe.status === 200 ? "ok" : "unreachable";
      var needsPairing = false;
      try {
        await plugin.refresh();
      } catch (error) {
        var message = String(error && error.message ? error.message : error);
        if (message === "pairing was revoked" || message === "pairing upgrade required") return "revoked";
        if (!refreshNeedsPairing(message)) return "unreachable";
        needsPairing = true;
      }
      probe = await probeHandshake();
      if (probe.status === 200) return "ok";
      return needsPairing ? "repair" : "unreachable";
    }

    async function discardInvalidPairing() {
      try {
        await plugin.logout({ serverReachable: true });
      } catch (_) {}
    }

    function presentReauth(link, initialMessage) {
      var splash = document.querySelector(".boot-splash");
      if (splash) splash.remove();
      var stale = document.querySelector(".mobile-reauth");
      if (stale) stale.remove();
      var panel = textElement("section", "mobile-pairing mobile-reauth");
      panel.setAttribute("role", "main");
      var card = textElement("div", "mobile-pairing-card");
      panel.appendChild(card);
      var title = textElement("h1", "mobile-pairing-title", "Reconnect this phone");
      card.appendChild(title);
      card.appendChild(textElement("p", "mobile-pairing-copy", "Your Kohaku pairing is saved. This phone can't reach the desktop over Tailscale right now."));
      var status = textElement("p", "mobile-pairing-status", initialMessage || "Tailscale sign-in is required.");
      status.setAttribute("aria-live", "polite");
      card.appendChild(status);
      var actions = textElement("div", "mobile-pairing-actions");
      var disposed = false;
      var retrying = false;
      var pollTimer = null;

      function cleanup() {
        disposed = true;
        if (pollTimer !== null) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        window.removeEventListener("kohaku-native-resume", onAutoResume);
        window.removeEventListener("online", onAutoResume);
        panel.remove();
      }

      async function attemptReconnect() {
        if (disposed || retrying) return;
        retrying = true;
        retry.disabled = true;
        status.classList.remove("is-error");
        status.textContent = "Connecting to the server...";
        try {
          await connectTarget(link);
          var access = await requireValidAccess();
          if (disposed) return;
          if (access === "revoked") return;
          if (access === "repair") {
            cleanup();
            await discardInvalidPairing();
            presentPairing();
            return;
          }
          if (access === "unreachable") {
            status.textContent = "Still can't reach this desktop. Check Tailscale and retry.";
            status.classList.add("is-error");
            return;
          }
          cleanup();
          resolveReady();
        } catch (error) {
          if (disposed) return;
          status.textContent = String(error && error.message ? error.message : error);
          status.classList.add("is-error");
        } finally {
          retrying = false;
          retry.disabled = false;
        }
      }

      function onAutoResume() {
        if (disposed || retrying) return;
        void plugin.status().then(function (current) {
          if (disposed || retrying) return;
          if (current && current.state === "ready") {
            void attemptReconnect();
          }
        }).catch(function () {});
      }

      function startLoginWatch() {
        if (pollTimer !== null) return;
        pollTimer = setInterval(onAutoResume, 2000);
      }

      window.addEventListener("kohaku-native-resume", onAutoResume);
      window.addEventListener("online", onAutoResume);

      var login = textElement("button", "mobile-pairing-button", "Open Tailscale login");
      login.type = "button";
      login.addEventListener("click", function () {
        status.classList.remove("is-error");
        status.textContent = "Opening Tailscale sign-in...";
        startLoginWatch();
        void plugin.openLogin().catch(function (error) {
          var message = String(error && error.message ? error.message : error);
          if (message.indexOf("no approved login URL") === 0) {
            status.textContent =
              "Tailscale is already signed in on this phone. Check that the desktop is running, then retry.";
            status.classList.add("is-error");
            return;
          }
          status.textContent = message;
        });
      });
      actions.appendChild(login);
      var retry = textElement("button", "mobile-pairing-button", "Retry");
      retry.type = "button";
      retry.addEventListener("click", function () {
        void attemptReconnect();
      });
      actions.appendChild(retry);
      var unpair = textElement("button", "mobile-pairing-button mobile-pairing-button-secondary", "Unpair / Switch server");
      unpair.type = "button";
      unpair.addEventListener("click", function () {
        unpair.disabled = true;
        retry.disabled = true;
        login.disabled = true;
        status.classList.remove("is-error");
        status.textContent = "Clearing saved pairing...";
        cleanup();
        void (async function () {
          try {
            await plugin.logout({ serverReachable: false });
          } catch (_) {}
          presentPairing();
        })();
      });
      actions.appendChild(unpair);
      card.appendChild(actions);
      document.body.appendChild(panel);
    }

    if (target && target.hasAccess) {
      try {
        await connectTarget(target);
        var access = await requireValidAccess();
        if (access === "revoked") return;
        if (access === "repair") {
          await discardInvalidPairing();
          presentPairing();
          return;
        }
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
