import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2, RefreshCw, Search } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ToggleButton } from "@/components/settings/ToggleButton";
import {
  DEFERRED_MODEL_LIST_PROVIDERS,
  formatContextWindow,
  ProviderPickerIcon,
  providerModelKey,
} from "@/components/settings/shared/ModelControls";
import { SettingsGroup, SettingsSectionTitle } from "@/components/settings/shared/SettingsControls";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { fetchProviderModels } from "@/lib/api";
import type { ProviderModelsPayload, ProviderModelToggle, SettingsPayload } from "@/lib/types";

type Provider = SettingsPayload["providers"][number];
type CatalogModel = ProviderModelsPayload["models"][number];

interface CatalogRequest {
  revision: string;
  controller: AbortController | null;
}

interface CatalogResult {
  revision: string;
  loading: boolean;
  payload?: ProviderModelsPayload;
  error?: string;
}

const PAGE_SIZE = 50;

function supportsCatalog(provider: Provider): boolean {
  return provider.configured &&
    provider.model_selectable !== false &&
    provider.model_catalog !== "unsupported" &&
    (provider.auth_type !== "oauth" ||
      provider.model_catalog === "builtin" ||
      provider.model_catalog === "hybrid");
}

function defersCatalog(provider: Provider): boolean {
  return provider.model_catalog === "catalog" ||
    (provider.model_catalog !== "local" && DEFERRED_MODEL_LIST_PROVIDERS.has(provider.name));
}

function catalogRevision(provider: Provider, token: string, refreshKey: number): string {
  return JSON.stringify([token, refreshKey, provider]);
}

function matchesSearch(model: CatalogModel, provider: Provider, search: string): boolean {
  return [model.label, model.id, model.description, model.owned_by, provider.label]
    .some((value) => value?.toLowerCase().includes(search));
}

export function ProviderModelsCatalog({
  token,
  settings,
  modelCallOrder,
  providerCatalogRefreshKey,
  showBrandLogos,
  pendingToggles,
  orderSaving,
  onToggleProviderModel,
}: {
  token: string;
  settings: SettingsPayload;
  modelCallOrder: string[];
  providerCatalogRefreshKey: number;
  showBrandLogos: boolean;
  pendingToggles: ReadonlySet<string>;
  orderSaving: boolean;
  onToggleProviderModel: (update: ProviderModelToggle) => void;
}) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, CatalogResult>>({});
  const [rowLimits, setRowLimits] = useState<Record<string, number>>({});
  const requests = useRef(new Map<string, CatalogRequest>());
  const providers = useMemo(() => settings.providers.filter(supportsCatalog), [settings.providers]);
  const filter = providers.some((provider) => provider.name === selectedProvider)
    ? selectedProvider
    : null;
  const search = query.trim().toLowerCase();

  const load = useCallback((provider: Provider, refresh = false) => {
    const name = provider.name;
    const revision = catalogRevision(provider, token, providerCatalogRefreshKey);
    const current = requests.current.get(name);
    if (current?.revision === revision && (current.controller || !refresh)) return;
    current?.controller?.abort();

    const controller = new AbortController();
    const request: CatalogRequest = { revision, controller };
    requests.current.set(name, request);
    setResults((previous) => ({
      ...previous,
      [name]: {
        revision,
        loading: true,
        payload: previous[name]?.revision === revision ? previous[name].payload : undefined,
      },
    }));

    const settle = (result: Pick<CatalogResult, "payload" | "error">) => {
      if (requests.current.get(name) !== request || controller.signal.aborted) return;
      request.controller = null;
      setResults((previous) => ({
        ...previous,
        [name]: { revision, loading: false, ...result },
      }));
    };

    void fetchProviderModels(token, name, "", controller.signal)
      .then((payload) => settle({ payload }))
      .catch((error: unknown) => settle({ error: error instanceof Error ? error.message : String(error) }));
  }, [providerCatalogRefreshKey, token]);

  useEffect(() => {
    const activeRequests = requests.current;
    return () => {
      for (const request of activeRequests.values()) request.controller?.abort();
      activeRequests.clear();
    };
  }, []);

  useEffect(() => {
    const eligible = new Set(providers.map((provider) => provider.name));
    for (const [name, request] of requests.current) {
      if (eligible.has(name)) continue;
      request.controller?.abort();
      requests.current.delete(name);
      setResults((previous) => {
        const next = { ...previous };
        delete next[name];
        return next;
      });
    }

    for (const provider of providers) {
      if (
        !defersCatalog(provider) ||
        filter === provider.name ||
        (!filter && search.length >= 2) ||
        requests.current.has(provider.name)
      ) {
        load(provider);
      }
    }
  }, [filter, load, providers, search]);

  const enabledNames = new Set(modelCallOrder);
  const enabledByModel = new Map<string, Set<string>>();
  for (const preset of settings.model_presets) {
    if (preset.is_default || !enabledNames.has(preset.name)) continue;
    for (const provider of new Set([preset.provider, preset.resolved_provider])) {
      if (!provider) continue;
      const key = providerModelKey(provider, preset.model);
      const names = enabledByModel.get(key) ?? new Set();
      names.add(preset.name);
      enabledByModel.set(key, names);
    }
  }

  const groups = providers.flatMap((provider) => {
    if (filter && filter !== provider.name) return [];
    const cached = results[provider.name];
    const revision = catalogRevision(provider, token, providerCatalogRefreshKey);
    const result = cached?.revision === revision ? cached : undefined;
    const rows = result?.payload?.models ?? [];
    const matches = search ? rows.filter((model) => matchesSearch(model, provider, search)) : rows;
    if (search && result?.payload?.status === "available" && !result.loading && !matches.length) {
      return [];
    }
    return [{ provider, result, matches }];
  });

  const refreshable = providers.filter((provider) =>
    !defersCatalog(provider) || results[provider.name] !== undefined,
  );

  return (
    <section aria-label={t("settings.models.catalog.title", { defaultValue: "Available models" })}>
      <div className="flex items-center justify-between gap-3">
        <SettingsSectionTitle>
          {t("settings.models.catalog.title", { defaultValue: "Available models" })}
        </SettingsSectionTitle>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!refreshable.length || refreshable.every((provider) => results[provider.name]?.loading)}
          onClick={() => refreshable.forEach((provider) => load(provider, true))}
        >
          <RefreshCw className="mr-2 h-3.5 w-3.5" aria-hidden />
          {t("settings.models.catalog.refresh", { defaultValue: "Refresh" })}
        </Button>
      </div>
      <p className="mb-3 px-1 text-xs text-muted-foreground">
        {t("settings.models.catalog.description", { defaultValue: "Enabled models appear in the chat model selector." })}
      </p>
      {providers.length > 0 ? (
        <>
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
            <Input
              type="search"
              className="pl-9"
              aria-label={t("settings.models.catalog.searchLabel", { defaultValue: "Search available models" })}
              placeholder={t("settings.models.catalog.searchPlaceholder", { defaultValue: "Search models across providers..." })}
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setRowLimits({});
              }}
            />
          </div>
          {providers.length > 1 ? (
            <div className="my-3 flex flex-wrap gap-2">
              <Button type="button" size="sm" variant={filter ? "outline" : "secondary"} aria-pressed={!filter} onClick={() => setSelectedProvider(null)}>
                {t("settings.models.catalog.all", { defaultValue: "All" })}
              </Button>
              {providers.map((provider) => (
                <Button
                  key={provider.name}
                  type="button"
                  size="sm"
                  variant={filter === provider.name ? "secondary" : "outline"}
                  aria-pressed={filter === provider.name}
                  className="gap-1.5"
                  onClick={() => setSelectedProvider(provider.name)}
                >
                  <ProviderPickerIcon provider={provider.name} showBrandLogos={showBrandLogos} />
                  {provider.label}
                </Button>
              ))}
            </div>
          ) : null}
        </>
      ) : null}
      <div className="mt-3 space-y-3">
        {groups.map(({ provider, result, matches }) => {
          const { name, label } = provider;
          const limit = rowLimits[name] ?? PAGE_SIZE;
          const count = result?.payload?.model_count;
          const message = result?.error || result?.payload?.message;
          const failed = Boolean(result?.error || (result?.payload && result.payload.status !== "available"));
          return (
            <SettingsGroup key={name}>
              <div className="flex min-h-12 items-center justify-between gap-2 px-4 py-2">
                <div className="flex min-w-0 items-center gap-2">
                  <ProviderPickerIcon provider={name} showBrandLogos={showBrandLogos} />
                  <h3 className="min-w-0 truncate text-sm font-medium">{label}</h3>
                  {count !== undefined && !failed ? (
                    <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                      {search && count !== matches.length ? `${matches.length} / ${count}` : count}
                    </span>
                  ) : null}
                </div>
                {result ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    aria-label={t("settings.models.catalog.refreshProvider", { defaultValue: "Refresh {{provider}} models", provider: label })}
                    disabled={result.loading}
                    onClick={() => load(provider, true)}
                  >
                    {result.loading
                      ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                      : <RefreshCw className="h-4 w-4" aria-hidden />}
                  </Button>
                ) : defersCatalog(provider) ? (
                  <Button type="button" variant="outline" size="sm" onClick={() => load(provider)}>
                    {t("settings.models.catalog.load", { defaultValue: "Load models" })}
                  </Button>
                ) : null}
              </div>
              {result?.loading || message || (result?.payload?.status === "available" && !matches.length) ? (
                <p role="status" className="px-4 py-2 text-xs text-muted-foreground">
                  {result?.loading
                    ? t("settings.models.catalog.loading", { defaultValue: "Loading models…" })
                    : message || t("settings.models.catalog.empty", { defaultValue: "No models available." })}
                </p>
              ) : null}
              {matches.slice(0, limit).map((model) => {
                const key = providerModelKey(name, model.id);
                const enabledCount = enabledByModel.get(key)?.size ?? 0;
                const enabled = enabledCount > 0;
                const displayName = model.label || model.id;
                return (
                  <div key={model.id} className="flex items-center gap-2 px-4 py-2.5 sm:gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="break-words text-sm font-medium">{displayName}</div>
                      {displayName !== model.id || model.description ? (
                        <p className="mt-0.5 break-words text-xs text-muted-foreground">
                          {[displayName !== model.id ? model.id : null, model.description].filter(Boolean).join(" · ")}
                        </p>
                      ) : null}
                    </div>
                    {model.context_window ? (
                      <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        {formatContextWindow(model.context_window)}
                      </span>
                    ) : null}
                    <ToggleButton
                      label={t("settings.models.catalog.enable", { defaultValue: "Enable {{model}}", model: displayName })}
                      checked={enabled}
                      disabled={orderSaving || pendingToggles.has(key) || (enabled && enabledCount === enabledNames.size)}
                      onChange={(checked) => onToggleProviderModel({
                        provider: name,
                        model: model.id,
                        enabled: checked,
                        label: model.label ?? undefined,
                        contextWindowTokens: model.context_window ?? undefined,
                      })}
                    />
                  </div>
                );
              })}
              {matches.length > limit ? (
                <div className="px-4 py-2">
                  <Button type="button" variant="ghost" size="sm" onClick={() => setRowLimits((previous) => ({
                    ...previous,
                    [name]: limit + PAGE_SIZE,
                  }))}>
                    {t("settings.models.catalog.more", { defaultValue: "Show more" })}
                  </Button>
                </div>
              ) : null}
            </SettingsGroup>
          );
        })}
        {groups.length === 0 ? (
          <p className="px-1 text-xs text-muted-foreground">
            {providers.length
              ? t("settings.models.catalog.noMatches", { defaultValue: "No models match your search." })
              : t("settings.models.catalog.noProviders", { defaultValue: "Configure a provider to see its models." })}
          </p>
        ) : null}
      </div>
    </section>
  );
}
