import { useCallback, type Dispatch, type SetStateAction } from "react";
import type { TFunction } from "i18next";

import {
  webSearchProviderAcceptsApiKey,
  webSearchProviderRequiresApiKey,
} from "@/components/settings/capabilities/WebSettings";
import type { CapabilitySettingsState } from "@/components/settings/capabilities/useCapabilitySettingsState";
import type {
  ApplySettingsPayload,
  MaybeRestartHostEngine,
  PendingRestartSections,
} from "@/components/settings/contracts";
import {
  updateImageGenerationSettings,
  updateNetworkSafetySettings,
  updateTranscriptionSettings,
  updateWebSearchSettings,
} from "@/lib/api";
import type { NanobotClient } from "@/lib/nanobot-client";
import type { SettingsPayload, WebSearchSettingsUpdate } from "@/lib/types";

interface CapabilitySettingsActionsOptions {
  state: CapabilitySettingsState;
  settings: SettingsPayload | null;
  client: NanobotClient;
  t: TFunction;
  applyPayload: ApplySettingsPayload;
  maybeRestartHostEngine: MaybeRestartHostEngine;
  setPendingRestartSections: Dispatch<SetStateAction<PendingRestartSections>>;
  setError: Dispatch<SetStateAction<string | null>>;
  installCapabilities: (names: string[]) => Promise<boolean>;
  imageGenerationDirty: boolean;
  transcriptionDirty: boolean;
  networkSafetyDirty: boolean;
}

export function useCapabilitySettingsActions({
  state,
  settings,
  client,
  t,
  applyPayload,
  maybeRestartHostEngine,
  setPendingRestartSections,
  setError,
  installCapabilities,
  imageGenerationDirty,
  transcriptionDirty,
  networkSafetyDirty,
}: CapabilitySettingsActionsOptions) {
  const {
    imageGenerationForm,
    imageGenerationSaving,
    networkSafetyForm,
    networkSafetySaving,
    setImageGenerationSaving,
    setNetworkSafetySaving,
    setTranscriptionSaving,
    setWebSearchForm,
    setWebSearchKeyEditing,
    setWebSearchKeyVisible,
    setWebSearchSaving,
    transcriptionForm,
    transcriptionSaving,
    webSearchForm,
    webSearchKeyEditing,
    webSearchSaving,
  } = state;

  const saveImageGenerationSettings = async () => {
    if (!settings || !imageGenerationDirty || imageGenerationSaving) return;
    setImageGenerationSaving(true);
    try {
      const payload = await updateImageGenerationSettings(client, imageGenerationForm);
      applyPayload(payload);
      if (payload.requires_restart) {
        setPendingRestartSections((prev) => ({ ...prev, image: true }));
      }
      await maybeRestartHostEngine(payload);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setImageGenerationSaving(false);
    }
  };

  const saveTranscriptionSettings = async () => {
    if (!settings || !transcriptionDirty || transcriptionSaving) return;
    setTranscriptionSaving(true);
    try {
      const payload = await updateTranscriptionSettings(client, transcriptionForm);
      applyPayload(payload);
      if (payload.requires_restart) {
        setPendingRestartSections((prev) => ({ ...prev, browser: true }));
      }
      await maybeRestartHostEngine(payload);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setTranscriptionSaving(false);
    }
  };

  const saveNetworkSafetySettings = async () => {
    if (!settings || !networkSafetyDirty || networkSafetySaving) return;
    setNetworkSafetySaving(true);
    try {
      const payload = await updateNetworkSafetySettings(client, networkSafetyForm);
      applyPayload(payload);
      if (payload.requires_restart) {
        setPendingRestartSections((prev) => ({ ...prev, runtime: true }));
      }
      await maybeRestartHostEngine(payload);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setNetworkSafetySaving(false);
    }
  };

  const saveWebSearch = async () => {
    if (!settings || webSearchSaving) return;
    const provider = settings.web_search.providers.find((item) => item.name === webSearchForm.provider);
    if (!provider) return;
    const apiKey = webSearchForm.apiKey?.trim() ?? "";
    const baseUrl = webSearchForm.baseUrl?.trim() ?? "";
    const hasExistingSecret =
      webSearchProviderAcceptsApiKey(provider) &&
      webSearchForm.provider === settings.web_search.provider &&
      !!settings.web_search.api_key_hint;

    if (webSearchProviderRequiresApiKey(provider) && !apiKey && !hasExistingSecret) {
      setError(t("settings.byok.webSearch.apiKeyRequired"));
      return;
    }
    if (provider.credential === "base_url" && !baseUrl) {
      setError(t("settings.byok.webSearch.baseUrlRequired"));
      return;
    }

    setWebSearchSaving(true);
    try {
      if (provider.name === "olostep" && !(await installCapabilities(["olostep"]))) return;
      const webFetchRestartRequired =
        (webSearchForm.useJinaReader ?? settings.web.fetch.use_jina_reader) !==
          settings.web.fetch.use_jina_reader ||
        (webSearchForm.fetchProvider ?? settings.web.fetch.provider) !==
          settings.web.fetch.provider ||
        (webSearchForm.browserCdpUrl ?? settings.web.browser.cdp_url) !==
          settings.web.browser.cdp_url ||
        (webSearchForm.browserPinTab ?? settings.web.browser.pin_tab) !==
          settings.web.browser.pin_tab ||
        (webSearchForm.browserSession ?? settings.web.browser.session) !==
          settings.web.browser.session ||
        (webSearchForm.browserLaunchCommand ?? settings.web.browser.launch_command) !==
          settings.web.browser.launch_command ||
        (webSearchForm.browserLaunchWaitSeconds ?? settings.web.browser.launch_wait_seconds) !==
          settings.web.browser.launch_wait_seconds;
      const update: WebSearchSettingsUpdate = {
        provider: webSearchForm.provider,
        maxResults: webSearchForm.maxResults,
        timeout: webSearchForm.timeout,
        useJinaReader: webSearchForm.useJinaReader,
        fetchProvider: webSearchForm.fetchProvider,
        browserCdpUrl: webSearchForm.browserCdpUrl,
        browserPinTab: webSearchForm.browserPinTab,
        browserSession: webSearchForm.browserSession,
        browserLaunchCommand: webSearchForm.browserLaunchCommand,
        browserLaunchWaitSeconds: webSearchForm.browserLaunchWaitSeconds,
      };
      if (
        webSearchProviderAcceptsApiKey(provider) &&
        (apiKey || (provider.credential === "optional_api_key" && webSearchKeyEditing))
      ) {
        update.apiKey = apiKey;
      }
      if (provider.credential === "base_url") update.baseUrl = baseUrl;
      const payload = await updateWebSearchSettings(client, update);
      applyPayload(payload);
      if (payload.requires_restart || webFetchRestartRequired) {
        setPendingRestartSections((prev) => ({ ...prev, browser: true }));
      }
      await maybeRestartHostEngine(payload);
      setWebSearchForm((prev) => ({
        provider: payload.web_search.provider,
        apiKey: "",
        baseUrl: payload.web_search.base_url ?? prev.baseUrl ?? "",
        maxResults: payload.web_search.max_results,
        timeout: payload.web_search.timeout,
        useJinaReader: payload.web.fetch.use_jina_reader,
        fetchProvider: payload.web.fetch.provider,
        browserCdpUrl: payload.web.browser.cdp_url,
        browserPinTab: payload.web.browser.pin_tab,
        browserSession: payload.web.browser.session,
        browserLaunchCommand: payload.web.browser.launch_command,
        browserLaunchWaitSeconds: payload.web.browser.launch_wait_seconds,
      }));
      setWebSearchKeyVisible(false);
      setWebSearchKeyEditing(false);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setWebSearchSaving(false);
    }
  };

  const resetWebSearchDraft = useCallback(() => {
    if (!settings) return;
    setWebSearchForm({
      provider: settings.web_search.provider,
      apiKey: "",
      baseUrl: settings.web_search.base_url ?? "",
      maxResults: settings.web_search.max_results,
      timeout: settings.web_search.timeout,
      useJinaReader: settings.web.fetch.use_jina_reader,
      fetchProvider: settings.web.fetch.provider,
      browserCdpUrl: settings.web.browser.cdp_url,
      browserPinTab: settings.web.browser.pin_tab,
      browserSession: settings.web.browser.session,
      browserLaunchCommand: settings.web.browser.launch_command,
      browserLaunchWaitSeconds: settings.web.browser.launch_wait_seconds,
    });
    setWebSearchKeyVisible(false);
    setWebSearchKeyEditing(false);
  }, [settings]);

  const handleWebSearchProviderChange = useCallback((provider: string) => {
    if (!settings) return;
    setWebSearchForm((prev) => ({
      provider,
      apiKey: "",
      baseUrl: provider === settings.web_search.provider ? settings.web_search.base_url ?? "" : "",
      maxResults: prev.maxResults ?? settings.web_search.max_results,
      timeout: prev.timeout ?? settings.web_search.timeout,
      useJinaReader: prev.useJinaReader ?? settings.web.fetch.use_jina_reader,
      fetchProvider: prev.fetchProvider ?? settings.web.fetch.provider,
      browserCdpUrl: prev.browserCdpUrl ?? settings.web.browser.cdp_url,
      browserPinTab: prev.browserPinTab ?? settings.web.browser.pin_tab,
      browserSession: prev.browserSession ?? settings.web.browser.session,
      browserLaunchCommand: prev.browserLaunchCommand ?? settings.web.browser.launch_command,
      browserLaunchWaitSeconds:
        prev.browserLaunchWaitSeconds ?? settings.web.browser.launch_wait_seconds,
    }));
    setWebSearchKeyVisible(false);
    setWebSearchKeyEditing(false);
  }, [settings]);

  return {
    handleWebSearchProviderChange,
    resetWebSearchDraft,
    saveImageGenerationSettings,
    saveNetworkSafetySettings,
    saveTranscriptionSettings,
    saveWebSearch,
  };
}
