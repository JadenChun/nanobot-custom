import type { Dispatch, SetStateAction } from "react";
import { Eye, EyeOff, Pencil } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ProviderPicker } from "@/components/settings/shared/ModelControls";
import {
  CapabilityInstallNotice,
  NumberInput,
  RestartSettingsFooter,
  SettingsGroup,
  SettingsRow,
  SettingsSectionTitle,
  StatusPill,
} from "@/components/settings/shared/SettingsControls";
import { ToggleButton } from "@/components/settings/ToggleButton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type {
  NanobotFeatureInfo,
  SettingsPayload,
  WebSearchSettingsUpdate,
} from "@/lib/types";

export const DEFAULT_WEB_SEARCH_FORM: WebSearchSettingsUpdate = {
  provider: "duckduckgo",
  apiKey: "",
  baseUrl: "",
  maxResults: 5,
  timeout: 30,
  useJinaReader: true,
  fetchProvider: "auto",
  browserCdpUrl: "",
  browserPinTab: true,
  browserSession: "default",
  browserLaunchCommand: "",
  browserLaunchWaitSeconds: 20,
};

export function webSearchFormFromPayload(
  payload: SettingsPayload,
  previous?: WebSearchSettingsUpdate,
): WebSearchSettingsUpdate {
  return {
    provider: payload.web_search.provider,
    apiKey: previous?.provider === payload.web_search.provider ? previous.apiKey ?? "" : "",
    baseUrl: payload.web_search.base_url ?? "",
    maxResults: payload.web_search.max_results,
    timeout: payload.web_search.timeout,
    useJinaReader: payload.web.fetch.use_jina_reader,
    fetchProvider: payload.web.fetch.provider,
    browserCdpUrl: payload.web.browser.cdp_url,
    browserPinTab: payload.web.browser.pin_tab,
    browserSession: payload.web.browser.session,
    browserLaunchCommand: payload.web.browser.launch_command,
    browserLaunchWaitSeconds: payload.web.browser.launch_wait_seconds,
  };
}

type WebSearchProviderOption = SettingsPayload["web_search"]["providers"][number];

export function webSearchProviderAcceptsApiKey(provider?: WebSearchProviderOption): boolean {
  return provider?.credential === "api_key" || provider?.credential === "optional_api_key";
}

export function webSearchProviderRequiresApiKey(provider?: WebSearchProviderOption): boolean {
  return provider?.credential === "api_key";
}

export function WebSettings({
  settings,
  form,
  keyVisible,
  keyEditing,
  saving,
  onChangeForm,
  onChangeProvider,
  onToggleKey,
  onToggleKeyEditing,
  onReset,
  onSave,
  showBrandLogos,
  onRestart,
  isRestarting,
  requiresRestartPending,
  olostepFeature,
  olostepInstalling,
  capabilityError,
}: {
  settings: SettingsPayload;
  form: WebSearchSettingsUpdate;
  keyVisible: boolean;
  keyEditing: boolean;
  saving: boolean;
  onChangeForm: Dispatch<SetStateAction<WebSearchSettingsUpdate>>;
  onChangeProvider: (provider: string) => void;
  onToggleKey: () => void;
  onToggleKeyEditing: () => void;
  onReset: () => void;
  onSave: () => void;
  showBrandLogos: boolean;
  onRestart?: () => void;
  isRestarting?: boolean;
  requiresRestartPending: boolean;
  olostepFeature?: NanobotFeatureInfo;
  olostepInstalling: boolean;
  capabilityError: string | null;
}) {
  const { t } = useTranslation();
  const tx = (key: string, fallback: string) => t(key, { defaultValue: fallback });
  const selectedProvider =
    settings.web_search.providers.find((provider) => provider.name === form.provider) ??
    settings.web_search.providers[0];
  const hasExistingSecret =
    webSearchProviderAcceptsApiKey(selectedProvider) &&
    form.provider === settings.web_search.provider &&
    !!settings.web_search.api_key_hint;
  const showKeyInput = webSearchProviderAcceptsApiKey(selectedProvider) && (!hasExistingSecret || keyEditing);
  const apiKey = form.apiKey?.trim() ?? "";
  const baseUrl = form.baseUrl?.trim() ?? "";
  const effectiveJinaReader = form.useJinaReader ?? settings.web.fetch.use_jina_reader;
  const effectiveFetchProvider = form.fetchProvider ?? settings.web.fetch.provider;
  const browser = settings.web.browser;
  const effectiveCdpUrl = form.browserCdpUrl ?? browser.cdp_url;
  const effectivePinTab = form.browserPinTab ?? browser.pin_tab;
  const effectiveSession = form.browserSession ?? browser.session;
  const effectiveLaunchCommand = form.browserLaunchCommand ?? browser.launch_command;
  const effectiveLaunchWait = form.browserLaunchWaitSeconds ?? browser.launch_wait_seconds;
  const fetchOptions = [
    { name: "auto", label: tx("settings.values.auto", "Auto") },
    { name: "tinyfish", label: "TinyFish" },
    { name: "jina", label: "Jina" },
    { name: "local", label: tx("settings.values.local", "Local") },
  ];
  const browserDirty =
    effectiveCdpUrl !== browser.cdp_url ||
    effectivePinTab !== browser.pin_tab ||
    effectiveSession !== browser.session ||
    effectiveLaunchCommand !== browser.launch_command ||
    effectiveLaunchWait !== browser.launch_wait_seconds;
  const dirty =
    form.provider !== settings.web_search.provider ||
    apiKey.length > 0 ||
    baseUrl !== (settings.web_search.base_url ?? "") ||
    form.maxResults !== settings.web_search.max_results ||
    form.timeout !== settings.web_search.timeout ||
    effectiveJinaReader !== settings.web.fetch.use_jina_reader ||
    effectiveFetchProvider !== settings.web.fetch.provider ||
    browserDirty;
  const restartDirty =
    effectiveJinaReader !== settings.web.fetch.use_jina_reader ||
    effectiveFetchProvider !== settings.web.fetch.provider ||
    browserDirty;
  const missingCredential =
    webSearchProviderRequiresApiKey(selectedProvider)
      ? !apiKey && !hasExistingSecret
      : selectedProvider?.credential === "base_url"
        ? !baseUrl
        : false;

  return (
    <div className="space-y-7">
      <section>
        <SettingsSectionTitle>{tx("settings.sections.webSearch", "Web search")}</SettingsSectionTitle>
        {form.provider === "olostep" && olostepFeature && !olostepFeature.installed ? (
          <div className="mb-3">
            <CapabilityInstallNotice
              title={tx("settings.capabilities.searchSupport", "Search provider support")}
              description={tx(
                "settings.capabilities.searchInstallOnSave",
                "Olostep support will be installed automatically when you save.",
              )}
              installing={olostepInstalling}
            />
          </div>
        ) : null}
        {capabilityError ? (
          <p className="mb-3 text-[12px] text-destructive">{capabilityError}</p>
        ) : null}
        <SettingsGroup>
          <SettingsRow title={t("settings.byok.webSearch.provider")}>
            <ProviderPicker
              providers={settings.web_search.providers}
              value={form.provider}
              emptyLabel={t("settings.byok.webSearch.selectProvider")}
              showProviderLogos={showBrandLogos}
              onChange={onChangeProvider}
            />
          </SettingsRow>

          {selectedProvider?.credential === "none" ? (
            <SettingsRow title={t("settings.byok.webSearch.credentials")}>
              <StatusPill tone="success">{t("settings.byok.webSearch.noCredentialRequired")}</StatusPill>
            </SettingsRow>
          ) : null}

          {webSearchProviderAcceptsApiKey(selectedProvider) ? (
            <SettingsRow
              title={t("settings.byok.apiKey")}
              description={t("settings.byok.webSearch.apiKeyHelp")}
            >
              <div className="relative w-[280px] max-w-full">
                {showKeyInput ? (
                  <>
                    <Input
                      type={keyVisible ? "text" : "password"}
                      value={form.apiKey ?? ""}
                      onChange={(event) =>
                        onChangeForm((prev) => ({ ...prev, apiKey: event.target.value }))
                      }
                      placeholder={
                        hasExistingSecret
                          ? t("settings.byok.apiKeyConfiguredPlaceholder")
                          : t("settings.byok.apiKeyPlaceholder")
                      }
                      className="h-9 rounded-full pr-11 text-[13px]"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={onToggleKey}
                      aria-label={
                        keyVisible ? t("settings.byok.hideApiKey") : t("settings.byok.showApiKey")
                      }
                      className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
                    >
                      {keyVisible ? (
                        <EyeOff className="h-3.5 w-3.5" aria-hidden />
                      ) : (
                        <Eye className="h-3.5 w-3.5" aria-hidden />
                      )}
                    </Button>
                  </>
                ) : (
                  <>
                    <div className="flex h-9 items-center rounded-full border border-input bg-background px-3 pr-11 text-[13px] text-muted-foreground">
                      {settings.web_search.api_key_hint ?? t("settings.byok.configuredKeyHint")}
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={onToggleKeyEditing}
                      aria-label={t("settings.actions.edit")}
                      className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
                    >
                      <Pencil className="h-3.5 w-3.5" aria-hidden />
                    </Button>
                  </>
                )}
              </div>
            </SettingsRow>
          ) : null}

          {selectedProvider?.credential === "base_url" ? (
            <SettingsRow
              title={t("settings.byok.webSearch.baseUrl")}
              description={t("settings.byok.webSearch.baseUrlHelp")}
            >
              <Input
                value={form.baseUrl ?? ""}
                onChange={(event) =>
                  onChangeForm((prev) => ({ ...prev, baseUrl: event.target.value }))
                }
                placeholder={t("settings.byok.webSearch.baseUrlPlaceholder")}
                className="h-9 w-[280px] rounded-full text-[13px]"
              />
            </SettingsRow>
          ) : null}
        </SettingsGroup>
      </section>

      <section>
        <SettingsSectionTitle>{tx("settings.sections.webBehavior", "Behavior")}</SettingsSectionTitle>
        <SettingsGroup>
          <SettingsRow title={tx("settings.rows.maxResults", "Max results")}>
            <NumberInput
              value={form.maxResults ?? settings.web_search.max_results}
              min={1}
              max={10}
              onChange={(maxResults) => onChangeForm((prev) => ({ ...prev, maxResults }))}
            />
          </SettingsRow>
          <SettingsRow title={tx("settings.rows.timeout", "Timeout")}>
            <NumberInput
              value={form.timeout ?? settings.web_search.timeout}
              min={1}
              max={120}
              onChange={(timeout) => onChangeForm((prev) => ({ ...prev, timeout }))}
              suffix="s"
            />
          </SettingsRow>
          <SettingsRow
            title={tx("settings.rows.fetchExtractor", "Fetch extractor")}
            description={tx(
              "settings.help.fetchExtractor",
              "Auto uses TinyFish when a key is set, then Jina, then the local extractor.",
            )}
          >
            <ProviderPicker
              providers={fetchOptions}
              value={effectiveFetchProvider}
              emptyLabel={tx("settings.values.auto", "Auto")}
              showProviderLogos={false}
              onChange={(fetchProvider) => onChangeForm((prev) => ({ ...prev, fetchProvider }))}
            />
          </SettingsRow>
          {effectiveFetchProvider === "auto" || effectiveFetchProvider === "jina" ? (
            <SettingsRow
              title={tx("settings.rows.jinaReader", "Jina reader")}
              description={tx("settings.help.jinaReader", "Allow Jina Reader as the remote extraction fallback.")}
            >
              <ToggleButton
                checked={effectiveJinaReader}
                onChange={(useJinaReader) => onChangeForm((prev) => ({ ...prev, useJinaReader }))}
                ariaLabel={tx("settings.rows.jinaReader", "Jina reader")}
                label={effectiveJinaReader ? tx("settings.values.on", "On") : tx("settings.values.off", "Off")}
              />
            </SettingsRow>
          ) : null}
        </SettingsGroup>
      </section>

      <section>
        <SettingsSectionTitle>{tx("settings.sections.browserSession", "Browser session")}</SettingsSectionTitle>
        <SettingsGroup>
          <SettingsRow
            title={tx("settings.rows.browserAttach", "Attach URL")}
            description={tx(
              "settings.help.browserAttach",
              "Chrome DevTools endpoint. When it answers, the agent drives this running browser and shares its sign-ins instead of launching its own profile.",
            )}
          >
            <div className="flex items-center gap-2">
              <StatusPill tone={browser.attached ? "success" : "neutral"}>
                {browser.attached
                  ? tx("settings.values.connected", "Connected")
                  : tx("settings.values.notConnected", "Not connected")}
              </StatusPill>
              <Input
                value={effectiveCdpUrl}
                onChange={(event) =>
                  onChangeForm((prev) => ({ ...prev, browserCdpUrl: event.target.value }))
                }
                placeholder="http://127.0.0.1:9222"
                className="h-9 w-[260px] rounded-full text-[13px]"
              />
            </div>
          </SettingsRow>
          <SettingsRow
            title={tx("settings.rows.browserPinTab", "Pin tab")}
            description={tx(
              "settings.help.browserPinTab",
              "Keep agent actions on one tab so the user's active tab is not taken over.",
            )}
          >
            <ToggleButton
              checked={effectivePinTab}
              onChange={(browserPinTab) => onChangeForm((prev) => ({ ...prev, browserPinTab }))}
              ariaLabel={tx("settings.rows.browserPinTab", "Pin tab")}
              label={effectivePinTab ? tx("settings.values.on", "On") : tx("settings.values.off", "Off")}
            />
          </SettingsRow>
          <SettingsRow
            title={tx("settings.rows.browserSessionName", "Session name")}
            description={tx("settings.help.browserSessionName", "agent-browser session used for the shared browser.")}
          >
            <Input
              value={effectiveSession}
              onChange={(event) =>
                onChangeForm((prev) => ({ ...prev, browserSession: event.target.value }))
              }
              className="h-9 w-[200px] rounded-full text-[13px]"
            />
          </SettingsRow>
          <SettingsRow
            title={tx("settings.rows.browserLaunchCommand", "Launch command")}
            description={tx(
              "settings.help.browserLaunchCommand",
              "Started when the attach URL is not listening yet. Leave empty to never launch a browser.",
            )}
          >
            <Input
              value={effectiveLaunchCommand}
              onChange={(event) =>
                onChangeForm((prev) => ({ ...prev, browserLaunchCommand: event.target.value }))
              }
              placeholder="/home/jaden/.local/bin/chrome-agent"
              className="h-9 w-[280px] rounded-full text-[13px]"
            />
          </SettingsRow>
          <SettingsRow
            title={tx("settings.rows.browserLaunchWait", "Launch wait")}
            description={tx("settings.help.browserLaunchWait", "How long to wait for the shared browser to expose its endpoint.")}
          >
            <NumberInput
              value={effectiveLaunchWait}
              min={1}
              max={120}
              onChange={(browserLaunchWaitSeconds) =>
                onChangeForm((prev) => ({ ...prev, browserLaunchWaitSeconds }))
              }
              suffix="s"
            />
          </SettingsRow>
        </SettingsGroup>
      </section>

      <section>
        <SettingsGroup>
          <RestartSettingsFooter
            dirty={dirty}
            saving={saving}
            pendingRestart={requiresRestartPending}
            disabled={missingCredential}
            message={
              missingCredential
                ? t("settings.byok.webSearch.missingCredential")
                : requiresRestartPending && !dirty
                  ? tx("settings.status.savedRestartApply", "Saved. Restart when ready.")
                  : restartDirty
                    ? tx("settings.status.restartAfterSaving", "Save changes, then restart when ready.")
                    : dirty
                      ? t("settings.byok.webSearch.saveHint")
                      : undefined
            }
            onSave={onSave}
            onRestart={onRestart}
            onReset={onReset}
            isRestarting={isRestarting}
          />
        </SettingsGroup>
      </section>
    </div>
  );
}
