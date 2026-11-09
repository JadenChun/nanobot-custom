import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SkillsCatalogSettings } from "@/components/settings/SkillsCatalogSettings";
import { deleteSkill, fetchSkillDetail, updateSkillEnabled } from "@/lib/api";
import type { NanobotClient } from "@/lib/nanobot-client";
import { notifySkillsChanged } from "@/lib/skill-events";
import type { SkillActionPayload, SkillDetail, SkillSummary } from "@/lib/types";
import { ClientProvider } from "@/providers/ClientProvider";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchSkillDetail: vi.fn(),
    updateSkillEnabled: vi.fn(),
    deleteSkill: vi.fn(),
  };
});

vi.mock("@/lib/skill-events", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/skill-events")>();
  return {
    ...actual,
    notifySkillsChanged: vi.fn(),
  };
});

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: Deferred<T>["resolve"];
  let reject!: Deferred<T>["reject"];
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

const client = {} as NanobotClient;

function skill(name: string, overrides: Partial<SkillSummary> = {}): SkillSummary {
  return {
    name,
    description: `Instructions for ${name}.`,
    source: "workspace",
    enabled: true,
    deletable: true,
    available: true,
    ...overrides,
  };
}

function skillDetail(skillSummary: SkillSummary): SkillDetail {
  return {
    ...skillSummary,
    requirements: {
      bins: [],
      env: [],
      missing_bins: [],
      missing_env: [],
    },
    raw_markdown: `---\nname: ${skillSummary.name}\n---\nInstructions.`,
  };
}

function settings(skills: SkillSummary[]) {
  return (
    <ClientProvider client={client} token="tok">
      <SkillsCatalogSettings skills={skills} />
    </ClientProvider>
  );
}

async function openSkill(name: string) {
  fireEvent.click(screen.getByRole("button", { name: `Open details for ${name}` }));
  return screen.findByRole("dialog");
}

async function closeSkillSheet() {
  fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Close" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
}

const skillA = skill("skill-a");
const skillB = skill("skill-b");

describe("SkillsCatalogSettings async skill fences", () => {
  beforeEach(() => {
    vi.mocked(fetchSkillDetail).mockReset();
    vi.mocked(updateSkillEnabled).mockReset();
    vi.mocked(deleteSkill).mockReset();
    vi.mocked(notifySkillsChanged).mockClear();
    vi.mocked(fetchSkillDetail).mockImplementation(async (_token, name) => {
      const matched = [skillA, skillB].find((item) => item.name === name);
      return skillDetail(matched ?? skill(name));
    });
  });

  it("does not apply a pending toggle completion to the newly selected skill", async () => {
    const pendingToggle = deferred<SkillActionPayload>();
    vi.mocked(updateSkillEnabled).mockReturnValue(pendingToggle.promise);

    render(settings([skillA, skillB]));

    const firstSheet = await openSkill("skill-a");
    const toggleA = await within(firstSheet).findByRole("switch", { name: "Disable skill-a" });
    fireEvent.click(toggleA);

    await waitFor(() =>
      expect(updateSkillEnabled).toHaveBeenCalledWith(expect.anything(), "skill-a", false),
    );

    await closeSkillSheet();
    const secondSheet = await openSkill("skill-b");
    expect(
      within(secondSheet).getByRole("switch", { name: "Disable skill-b" }),
    ).toHaveAttribute("aria-checked", "true");

    await act(async () => {
      pendingToggle.resolve({
        skills: [
          skill("skill-a", { enabled: false }),
          skill("skill-b", { enabled: false }),
        ],
        last_action: { name: "skill-a", enabled: false, deleted: false },
      });
      await Promise.resolve();
    });

    expect(notifySkillsChanged).not.toHaveBeenCalled();
    const currentSheet = screen.getByRole("dialog");
    expect(
      within(currentSheet).getByRole("heading", { name: "skill-b" }),
    ).toBeInTheDocument();
    expect(
      within(currentSheet).getByRole("switch", { name: "Disable skill-b" }),
    ).toHaveAttribute("aria-checked", "true");
  });

  it("does not close the newly selected skill when a pending delete completes", async () => {
    const pendingDelete = deferred<SkillActionPayload>();
    vi.mocked(deleteSkill).mockReturnValue(pendingDelete.promise);

    render(settings([skillA, skillB]));

    const firstSheet = await openSkill("skill-a");
    await within(firstSheet).findByRole("switch", { name: "Disable skill-a" });
    fireEvent.click(within(firstSheet).getByRole("button", { name: "Delete" }));
    fireEvent.click(await screen.findByRole("button", { name: "Delete skill" }));

    await waitFor(() =>
      expect(deleteSkill).toHaveBeenCalledWith(expect.anything(), "skill-a"),
    );

    await closeSkillSheet();
    const secondSheet = await openSkill("skill-b");
    expect(
      within(secondSheet).getByRole("switch", { name: "Disable skill-b" }),
    ).toBeInTheDocument();

    await act(async () => {
      pendingDelete.resolve({
        skills: [skill("skill-b")],
        last_action: { name: "skill-a", enabled: false, deleted: true },
      });
      await Promise.resolve();
    });

    expect(notifySkillsChanged).not.toHaveBeenCalled();
    const currentSheet = screen.getByRole("dialog");
    expect(
      within(currentSheet).getByRole("heading", { name: "skill-b" }),
    ).toBeInTheDocument();
  });
});
