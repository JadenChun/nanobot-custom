import { act, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SkillsMarketplace } from "@/components/settings/SkillsMarketplace";
import { SkillsCatalogSettings } from "@/components/settings/SkillsCatalogSettings";
import {
  fetchSkills,
  fetchTrendingMarketplaceSkills,
  installMarketplaceSkill,
  searchMarketplaceSkills,
} from "@/lib/api";
import type { NanobotClient } from "@/lib/nanobot-client";
import { notifySkillsChanged } from "@/lib/skill-events";
import { ClientProvider } from "@/providers/ClientProvider";
import { useSkills, useSkillsCatalog } from "@/hooks/useSkills";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchSkills: vi.fn(),
    fetchTrendingMarketplaceSkills: vi.fn(),
    installMarketplaceSkill: vi.fn(),
    searchMarketplaceSkills: vi.fn(),
  };
});

const client = {} as NanobotClient;

function marketplace(
  token: string,
  props: {
    catalogComplete?: boolean;
    installedSkills?: Array<{
      name: string;
      description: string;
      source: string;
      available: boolean;
      verification?: "verified" | "unverified";
    }>;
  } = {},
) {
  return (
    <ClientProvider client={client} token={token}>
      <SkillsMarketplace
        installedSkills={props.installedSkills ?? []}
        catalogComplete={props.catalogComplete}
        installing=""
        onInstallingChange={() => {}}
      />
    </ClientProvider>
  );
}

describe("useSkills", () => {
  it("distinguishes a failed initial load from an empty catalog and retries", async () => {
    vi.mocked(fetchSkills).mockReset().mockRejectedValueOnce(new Error("offline"));
    const { result } = renderHook(() => useSkillsCatalog(() => "tok"));
    expect(result.current.loading).toBe(true);
    expect(result.current.catalog_complete).toBe(false);
    await waitFor(() => expect(result.current.error).toContain("Unable to load"));
    expect(result.current.skills).toEqual([]);
    vi.mocked(fetchSkills).mockResolvedValueOnce({ skills: [], catalog_complete: true });
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.catalog_complete).toBe(true));
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("retains the last successful catalog on refresh failure", async () => {
    const skill = { name: "retained", description: "Known skill", source: "workspace", available: true };
    vi.mocked(fetchSkills).mockReset().mockResolvedValueOnce({
      skills: [skill], catalog_complete: false, catalog_notice: "Restricted library",
    });
    const { result } = renderHook(() => useSkillsCatalog(() => "tok"));
    await waitFor(() => expect(result.current.skills).toEqual([skill]));
    vi.mocked(fetchSkills).mockRejectedValueOnce(new Error("offline"));
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(result.current.skills).toEqual([skill]);
    expect(result.current.catalog_complete).toBe(false);
    expect(result.current.catalog_notice).toBe("Restricted library");
  });

  it("clears the prior credentials' catalog and ignores their pending response", async () => {
    const skill = { name: "private", description: "Private skill", source: "workspace", available: true };
    vi.mocked(fetchSkills).mockReset().mockResolvedValueOnce({ skills: [skill] });
    let token = "old";
    const getToken = () => token;
    const { result, rerender } = renderHook(() => useSkillsCatalog(getToken));
    await waitFor(() => expect(result.current.skills).toEqual([skill]));
    let resolveOld!: (payload: Awaited<ReturnType<typeof fetchSkills>>) => void;
    vi.mocked(fetchSkills).mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }));
    act(() => result.current.refresh());
    vi.mocked(fetchSkills).mockRejectedValueOnce(new Error("new token offline"));
    token = "new";
    rerender();
    expect(result.current.skills).toEqual([]);
    await act(async () => resolveOld({ skills: [skill], catalog_complete: true }));
    await waitFor(() => expect(result.current.loading).toBe(false));
    act(() => notifySkillsChanged({ skills: [skill], catalog_complete: true }, "old"));
    expect(result.current.skills).toEqual([]);
    expect(result.current.catalog_complete).toBe(false);
    expect(result.current.error).toBeTruthy();
  });

  it("shows retry instead of an empty or restricted catalog on transport failure", () => {
    const retry = vi.fn();
    render(
      <ClientProvider client={client} token="tok">
        <SkillsCatalogSettings skills={[]} catalogComplete={false}
          error="Unable to load skills." onRetry={retry} />
      </ClientProvider>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Unable to load skills.");
    expect(screen.queryByText("No matching skills.")).toBeNull();
    expect(screen.queryByText(/Some configured skill sources/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it("does not let an older request overwrite a newer skill event", async () => {
    let resolveSkills!: (value: Awaited<ReturnType<typeof fetchSkills>>) => void;
    vi.mocked(fetchSkills).mockReset().mockImplementationOnce(
      () => new Promise((resolve) => {
        resolveSkills = resolve;
      }),
    );
    const installed = {
      name: "react-testing",
      description: "Test React apps.",
      source: "workspace",
      available: true,
    };
    const getToken = () => "tok";
    const { result } = renderHook(() => useSkills(getToken));

    expect(fetchSkills).toHaveBeenCalledTimes(1);
    act(() => {
      notifySkillsChanged({ skills: [installed] }, "tok");
    });
    expect(result.current).toEqual([installed]);

    await act(async () => {
      resolveSkills({ skills: [] });
    });

    expect(result.current).toEqual([installed]);
  });

  it("keeps incomplete metadata and marks legacy shared rows unverified", async () => {
    vi.mocked(fetchSkills).mockReset().mockResolvedValue({
      skills: [],
      catalog_complete: true,
    });
    const shared = {
      name: "shared-skill",
      description: "A shared skill.",
      source: "library",
      available: true,
    };
    const getToken = () => "tok";
    const { result } = renderHook(() => useSkillsCatalog(getToken));
    await act(async () => {});

    act(() => {
      notifySkillsChanged({
          skills: [shared],
          catalog_complete: false,
          catalog_notice: "A configured shared skill library is restricted by policy.",
      }, "tok");
    });
    expect(result.current.catalog_complete).toBe(false);
    expect(result.current.skills[0]).toMatchObject({
      verification: "unverified",
      available: false,
    });

    act(() => {
      notifySkillsChanged({ skills: [shared] }, "tok");
    });
    expect(result.current.catalog_notice).toBe(
      "A configured shared skill library is restricted by policy.",
    );
    expect(result.current.skills[0]).toMatchObject({
      verification: "unverified",
      available: false,
    });
  });
});

describe("SkillsMarketplace", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(fetchTrendingMarketplaceSkills).mockReset().mockResolvedValue({
      period: "mixed",
      provider: "all",
      install_supported: true,
      skills: [],
    });
    vi.mocked(searchMarketplaceSkills).mockReset().mockImplementation(
      async (_token, query) => ({
        query,
        provider: "all",
        install_supported: true,
        skills: [],
      }),
    );
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps loaded marketplace data stable when the auth token rotates", async () => {
    const { rerender } = render(marketplace("tok-old"));

    await act(async () => {});
    expect(fetchTrendingMarketplaceSkills).toHaveBeenCalledTimes(1);
    expect(fetchTrendingMarketplaceSkills).toHaveBeenCalledWith("tok-old");

    fireEvent.change(screen.getByRole("textbox", { name: "Search skills" }), {
      target: { value: "React" },
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(searchMarketplaceSkills).toHaveBeenCalledTimes(1);
    expect(searchMarketplaceSkills).toHaveBeenLastCalledWith("tok-old", "React");

    rerender(marketplace("tok-new"));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });

    expect(fetchTrendingMarketplaceSkills).toHaveBeenCalledTimes(1);
    expect(searchMarketplaceSkills).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByRole("textbox", { name: "Search skills" }), {
      target: { value: "Vue" },
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(searchMarketplaceSkills).toHaveBeenCalledTimes(2);
    expect(searchMarketplaceSkills).toHaveBeenLastCalledWith("tok-new", "Vue");
  });

  it("disables cached marketplace installs when the local catalog becomes incomplete", async () => {
    const remoteSkill = {
      id: "skills_sh:demo-skill",
      skill_id: "demo-skill",
      name: "Demo Skill",
      source: "example/demo-skill",
      provider: "skills_sh" as const,
      installs: 20,
      url: "https://skills.sh/example/demo-skill",
      installed: false,
      install_supported: true,
      metric: "installs_total" as const,
    };
    vi.mocked(fetchTrendingMarketplaceSkills).mockResolvedValue({
      period: "mixed",
      provider: "all",
      install_supported: true,
      skills: [remoteSkill],
    });
    vi.mocked(installMarketplaceSkill).mockReset();
    const { rerender } = render(marketplace("tok", { catalogComplete: true }));
    await act(async () => {});

    fireEvent.click(screen.getByRole("button", { name: "Install Demo Skill" }));
    expect(screen.getByRole("button", { name: "Install skill" })).toBeEnabled();

    rerender(marketplace("tok", { catalogComplete: false }));
    expect(screen.getByRole("button", { name: "Install skill" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Install skill" }));
    expect(installMarketplaceSkill).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    const rowAction = screen.getByRole("button", { name: "Install Demo Skill" });
    expect(rowAction.parentElement).toHaveTextContent("Local install status unknown");
    expect(rowAction).toBeDisabled();
  });
});

describe("SkillsCatalogSettings", () => {
  it("does not label a legacy shared row Enabled while the catalog is incomplete", () => {
    const shared = {
      name: "shared-skill",
      description: "A shared skill.",
      source: "library",
      available: true,
    };
    render(
      <ClientProvider client={client} token="tok">
        <SkillsCatalogSettings
          skills={[shared]}
          catalogComplete={false}
          catalogNotice="A configured shared skill library is restricted by policy."
        />
      </ClientProvider>,
    );

    const row = screen.getByRole("button", { name: "Open details for shared-skill" });
    expect(within(row).getByText("Unverified")).toBeInTheDocument();
    expect(within(row).queryByText("Enabled")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(
      "A configured shared skill library is restricted by policy.",
    );
  });
});
