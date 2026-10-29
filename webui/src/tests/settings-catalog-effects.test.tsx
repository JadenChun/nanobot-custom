import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SkillsCatalogSettings } from "@/components/settings/SkillsCatalogSettings";
import {
  AppsCatalogSettings,
  CLI_APPS_REFRESH_RETRY_MS,
  DEFAULT_CUSTOM_MCP_FORM,
  type AppsKindFilter,
} from "@/components/settings/system/AppsSettings";
import { useSystemSettingsEffects } from "@/components/settings/system/useSystemSettingsEffects";
import { useSystemSettingsState } from "@/components/settings/system/useSystemSettingsState";
import type {
  CliAppInfo,
  CliAppsPayload,
  McpPresetInfo,
  McpPresetsPayload,
  SkillSummary,
} from "@/lib/types";
import type { NanobotClient } from "@/lib/nanobot-client";
import { ClientProvider } from "@/providers/ClientProvider";

const getToken = () => "test-token";

function jsonResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
  } as Response;
}

function cliPayload(names: string[]): CliAppsPayload {
  return {
    apps: names.map((name) => ({
      name,
      display_name: name,
      category: "tools",
      description: name,
      requires: "",
      source: "registry",
      entry_point: name,
      install_supported: true,
      installed: true,
      available: true,
      status: "installed",
      logo_url: null,
      brand_color: null,
      skill_installed: false,
    })),
    installed_count: names.length,
    catalog_updated_at: null,
  };
}

const emptyMcpPayload: McpPresetsPayload = { presets: [], installed_count: 0 };

function mcpPreset(overrides: Partial<McpPresetInfo>): McpPresetInfo {
  return {
    name: "comfy",
    display_name: "ComfyUI",
    category: "media",
    description: "Local image generation tools.",
    docs_url: "",
    transport: "streamableHttp",
    requires: "",
    note: "",
    install_supported: false,
    installed: true,
    configured: true,
    available: true,
    status: "configured",
    required_fields: [],
    connection_summary: "",
    source: "custom",
    ...overrides,
  };
}

function cliApp(overrides: Partial<CliAppInfo>): CliAppInfo {
  return {
    name: "installed-cli",
    display_name: "Installed CLI",
    category: "tools",
    description: "Installed local command.",
    requires: "",
    source: "registry",
    entry_point: "installed-cli",
    install_supported: true,
    installed: true,
    available: true,
    status: "installed",
    skill_installed: false,
    ...overrides,
  };
}

function renderAppsCatalog(
  mcpPresets: McpPresetInfo[],
  options: { filter?: AppsKindFilter; query?: string; cliApps?: CliAppsPayload } = {},
) {
  const cliApps: CliAppsPayload = options.cliApps ?? {
    apps: [
      cliApp({}),
      cliApp({
        name: "catalog-only-cli",
        display_name: "Online catalog CLI",
        description: "Not installed.",
        installed: false,
        status: "available",
      }),
    ],
    installed_count: 1,
  };
  return render(
    <AppsCatalogSettings
      cliApps={cliApps}
      mcpPresets={{ presets: mcpPresets, installed_count: mcpPresets.length }}
      cliAppsLoading={false}
      mcpPresetsLoading={false}
      query={options.query ?? ""}
      filter={options.filter ?? "configured"}
      cliActionKey={null}
      mcpActionKey={null}
      mcpOAuthFlow={null}
      mcpOAuthPopupBlocked={false}
      mcpOAuthCallbackUrl=""
      mcpOAuthCompleting={false}
      mcpOAuthCallbackError={null}
      cliMessage={null}
      cliError={null}
      cliFocusName={null}
      mcpMessage={null}
      mcpError={null}
      mcpFieldValues={{}}
      customMcpForm={DEFAULT_CUSTOM_MCP_FORM}
      mcpConfigImport=""
      showBrandLogos={false}
      requiresRestartPending={false}
      onQueryChange={() => undefined}
      onFilterChange={() => undefined}
      onCliAction={() => undefined}
      onMcpAction={() => undefined}
      onMcpOAuthConnect={() => undefined}
      onMcpOAuthCancel={() => undefined}
      onMcpOAuthOpen={() => undefined}
      onMcpOAuthCallbackUrlChange={() => undefined}
      onMcpOAuthComplete={() => undefined}
      onDismissStatus={() => undefined}
      onBackToChat={() => undefined}
      onMcpFieldChange={() => undefined}
      onCustomMcpFormChange={() => undefined}
      onMcpConfigImportChange={() => undefined}
      onSaveCustomMcp={() => undefined}
      onImportMcpConfig={() => undefined}
      onMcpToolsChange={() => undefined}
    />,
  );
}

function EffectsHarness({ pageVisible = true }: { pageVisible?: boolean }) {
  const state = useSystemSettingsState();
  useSystemSettingsEffects({
    state,
    activeSection: "apps",
    getToken,
    pageVisible,
  });

  return (
    <>
      <output data-testid="filter">{state.appsKindFilter}</output>
      <output data-testid="apps">{state.cliApps?.apps.map((app) => app.name).join(",")}</output>
      <button type="button" onClick={() => state.setAppsKindFilter("cli")}>
        Browse apps
      </button>
      <button type="button" onClick={() => state.setAppsKindFilter("configured")}>
        Configured
      </button>
      <button type="button" onClick={() => state.setAppsKindFilter("mcp")}>
        MCP
      </button>
    </>
  );
}

describe("Apps settings catalog loading", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/settings/mcp-presets")) return jsonResponse(emptyMcpPayload);
        if (url.includes("/api/settings/cli-apps?installed_only=1")) {
          return jsonResponse(cliPayload(["local-installed"]));
        }
        if (url.includes("/api/settings/cli-apps")) {
          return jsonResponse(cliPayload(["catalog-entry"]));
        }
        if (url.includes("/api/webui/skills/trending")) {
          return jsonResponse({
            skills: [],
            period: "mixed",
            provider: "all",
            install_supported: false,
          });
        }
        if (url.includes("/api/webui/skills/trends")) return jsonResponse({ trends: {} });
        return jsonResponse({});
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("starts in configured view and loads only local installed CLI entries", async () => {
    render(<EffectsHarness />);

    expect(screen.getByTestId("filter")).toHaveTextContent("configured");
    await waitFor(() => expect(screen.getByTestId("apps")).toHaveTextContent("local-installed"));
    const urls = vi.mocked(fetch).mock.calls.map(([input]) => String(input));
    expect(urls).toContain("/api/settings/cli-apps?installed_only=1");
    expect(urls).not.toContain("/api/settings/cli-apps");
  });

  it("loads the online CLI catalog only after opening Browse apps", async () => {
    render(<EffectsHarness />);
    await waitFor(() => expect(screen.getByTestId("apps")).toHaveTextContent("local-installed"));

    fireEvent.click(screen.getByRole("button", { name: "Browse apps" }));

    await waitFor(() => expect(screen.getByTestId("apps")).toHaveTextContent("catalog-entry"));
    const urls = vi.mocked(fetch).mock.calls.map(([input]) => String(input));
    expect(urls).toContain("/api/settings/cli-apps");
  });

  it("does not request app catalogs while the settings page is hidden", () => {
    render(<EffectsHarness pageVisible={false} />);

    expect(fetch).not.toHaveBeenCalled();
  });

  it("shows configured custom MCP, failed presets, disabled plugins, and installed CLIs only", () => {
    renderAppsCatalog([
      mcpPreset({ runtime_status: "connecting" }),
      mcpPreset({
        name: "failed-preset",
        display_name: "Failed preset",
        source: "preset",
        runtime_status: "failed",
      }),
      mcpPreset({
        name: "disabled-plugin",
        display_name: "Disabled plugin",
        source: "agent-plugin",
        enabled: false,
        configured: false,
        status: "missing_dependency",
      }),
      mcpPreset({
        name: "configured-custom",
        display_name: "Configured custom MCP",
        installed: false,
        configured: true,
      }),
      mcpPreset({
        name: "catalog-preset",
        display_name: "Uninstalled preset",
        source: "preset",
        installed: false,
        configured: false,
      }),
    ]);

    expect(screen.getByText("ComfyUI")).toBeInTheDocument();
    expect(screen.getByText("Connecting…")).toBeInTheDocument();
    expect(screen.getAllByText("Custom MCP")).toHaveLength(2);
    expect(screen.getByText("Failed preset")).toBeInTheDocument();
    expect(screen.getByText("Connection failed.")).toBeInTheDocument();
    expect(screen.getByText("Built-in MCP preset")).toBeInTheDocument();
    expect(screen.getByText("Disabled plugin")).toBeInTheDocument();
    expect(screen.getByText("Configured custom MCP")).toBeInTheDocument();
    expect(screen.getByText("Agent plugin")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enable" })).toBeInTheDocument();
    expect(screen.getByText("Installed CLI", { selector: "h3" })).toBeInTheDocument();
    expect(screen.getByText("Installed CLI", { selector: "span" })).toBeInTheDocument();
    expect(screen.queryByText("Online catalog CLI")).not.toBeInTheDocument();
    expect(screen.queryByText("Uninstalled preset")).not.toBeInTheDocument();
  });

  it("applies the search query inside the Configured and MCP views", () => {
    const browserCli = cliApp({ name: "browser-cli", display_name: "Browser CLI" });
    const browserMcp = mcpPreset({ name: "browser-mcp", display_name: "Browser MCP" });
    const browserPlugin = mcpPreset({
      name: "browser-plugin",
      display_name: "Browser plugin",
      source: "agent-plugin",
      enabled: false,
      configured: false,
    });
    const browserCatalogCli = cliApp({
      name: "browser-catalog",
      display_name: "Browser catalog CLI",
      installed: false,
    });
    const cliApps: CliAppsPayload = {
      apps: [browserCli, browserCatalogCli],
      installed_count: 1,
    };

    renderAppsCatalog([browserMcp, browserPlugin], {
      cliApps,
      filter: "configured",
      query: "browser",
    });

    expect(screen.getByText("Browser CLI", { selector: "h3" })).toBeInTheDocument();
    expect(screen.getByText("Browser MCP", { selector: "h3" })).toBeInTheDocument();
    expect(screen.getByText("Browser plugin", { selector: "h3" })).toBeInTheDocument();
    expect(screen.queryByText("Browser catalog CLI")).not.toBeInTheDocument();
    cleanup();

    renderAppsCatalog([browserMcp, browserPlugin], {
      cliApps,
      filter: "mcp",
      query: "browser",
    });

    expect(screen.getByText("Browser MCP", { selector: "h3" })).toBeInTheDocument();
    expect(screen.queryByText("Browser CLI")).not.toBeInTheDocument();
    expect(screen.queryByText("Browser plugin")).not.toBeInTheDocument();
  });

  it("ignores a late configured response after switching to the online catalog", async () => {
    let resolveConfigured: ((response: Response) => void) | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/settings/mcp-presets")) {
          return Promise.resolve(jsonResponse(emptyMcpPayload));
        }
        if (url.includes("/api/settings/cli-apps?installed_only=1")) {
          return new Promise<Response>((resolve) => {
            resolveConfigured = resolve;
          });
        }
        if (url.includes("/api/settings/cli-apps")) {
          return Promise.resolve(jsonResponse(cliPayload(["catalog-entry"])));
        }
        if (url.includes("/api/webui/skills/trending")) {
          return Promise.resolve(jsonResponse({
            skills: [],
            period: "mixed",
            provider: "all",
            install_supported: false,
          }));
        }
        if (url.includes("/api/webui/skills/trends")) return Promise.resolve(jsonResponse({ trends: {} }));
        return Promise.resolve(jsonResponse({}));
      }),
    );
    render(<EffectsHarness />);
    await waitFor(() => expect(resolveConfigured).toBeTypeOf("function"));

    fireEvent.click(screen.getByRole("button", { name: "Browse apps" }));
    await waitFor(() => expect(screen.getByTestId("apps")).toHaveTextContent("catalog-entry"));

    await act(async () => {
      resolveConfigured?.(jsonResponse(cliPayload(["stale-local-entry"])));
    });
    expect(screen.getByTestId("apps")).toHaveTextContent("catalog-entry");
    expect(screen.getByTestId("apps")).not.toHaveTextContent("stale-local-entry");
  });

  it("stops a pending online refresh and ignores its delayed result after returning to Configured", async () => {
    let installedRequest = 0;
    let onlineRequest = 0;
    let resolveDelayedRefresh: ((response: Response) => void) | undefined;
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/settings/mcp-presets")) {
        return Promise.resolve(jsonResponse(emptyMcpPayload));
      }
      if (url.includes("/api/settings/cli-apps?installed_only=1")) {
        installedRequest += 1;
        return Promise.resolve(jsonResponse(cliPayload([
          installedRequest === 1 ? "local-before-browse" : "local-after-browse",
        ])));
      }
      if (url.includes("/api/settings/cli-apps")) {
        onlineRequest += 1;
        if (onlineRequest === 1) {
          return Promise.resolve(jsonResponse({
            ...cliPayload(["catalog-entry"]),
            catalog_refresh_pending: true,
          }));
        }
        return new Promise<Response>((resolve) => {
          resolveDelayedRefresh = resolve;
        });
      }
      return Promise.resolve(jsonResponse({}));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<EffectsHarness />);
    await waitFor(() => expect(screen.getByTestId("apps")).toHaveTextContent("local-before-browse"));

    fireEvent.click(screen.getByRole("button", { name: "Browse apps" }));
    await waitFor(() => expect(screen.getByTestId("apps")).toHaveTextContent("catalog-entry"));
    await new Promise((resolve) => setTimeout(resolve, CLI_APPS_REFRESH_RETRY_MS + 50));
    expect(resolveDelayedRefresh).toBeTypeOf("function");
    expect(onlineRequest).toBe(2);

    const timerSpy = vi.spyOn(window, "setTimeout");
    fireEvent.click(screen.getByRole("button", { name: "Configured" }));
    await waitFor(() => expect(screen.getByTestId("apps")).toHaveTextContent("local-after-browse"));
    expect(installedRequest).toBe(2);

    await act(async () => {
      resolveDelayedRefresh?.(jsonResponse({
        ...cliPayload(["stale-catalog-entry"]),
        catalog_refresh_pending: true,
      }));
    });
    expect(screen.getByTestId("apps")).toHaveTextContent("local-after-browse");
    expect(screen.getByTestId("apps")).not.toHaveTextContent("stale-catalog-entry");
    expect(timerSpy).not.toHaveBeenCalledWith(expect.any(Function), CLI_APPS_REFRESH_RETRY_MS);
    expect(onlineRequest).toBe(2);
    timerSpy.mockRestore();
  });

  it("labels skill origins clearly and opens online skill discovery only on request", async () => {
    const skills: SkillSummary[] = [
      { name: "workspace-skill", description: "Workspace description", source: "workspace", available: true },
      { name: "shared-skill", description: "Shared description", source: "library", available: true },
      { name: "bundled-skill", description: "Bundled description", source: "builtin", available: true },
      { name: "plugin-skill", description: "Plugin description", source: "plugin", available: true },
    ];
    const client = {} as NanobotClient;

    render(
      <ClientProvider client={client} token="test-token">
        <SkillsCatalogSettings skills={skills} />
      </ClientProvider>,
    );

    expect(screen.getByRole("heading", { name: "Workspace" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Shared custom" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Built-in" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Plugin" })).toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("tab", { name: "Browse online" }));
    await waitFor(() => {
      expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).includes("/api/webui/skills/trending")))
        .toBe(true);
    });
  });
});
