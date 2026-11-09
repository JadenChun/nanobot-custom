import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ChannelSetupPanel } from "@/components/settings/channels/ChannelSetupPanel";
import { configureChannel, validateChannel } from "@/lib/api";
import type { NanobotClient } from "@/lib/nanobot-client";
import type {
  ChannelConfigurePayload,
  ChannelValidationPayload,
  NanobotFeatureInfo,
  NanobotFeaturesPayload,
} from "@/lib/types";
import { ClientProvider } from "@/providers/ClientProvider";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    validateChannel: vi.fn(),
    configureChannel: vi.fn(),
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
const validateDeferreds = new Map<string, Deferred<ChannelValidationPayload>>();
const configureDeferreds = new Map<string, Deferred<ChannelConfigurePayload>>();

function channelFeature(name: string, displayName: string): NanobotFeatureInfo {
  return {
    name,
    display_name: displayName,
    type: "channel",
    enabled: false,
    configured: false,
    installed: true,
    ready: false,
    status: "not_enabled",
    install_supported: true,
    requires_restart: false,
    setup: {
      fields: [{
        key: `channels.${name}.token`,
        field: "token",
        kind: "secret",
        choices: [],
        required: true,
      }],
    },
  };
}

function validationPayload(name: string): ChannelValidationPayload {
  return {
    name,
    status: "connected",
    checks: [],
    missing_fields: [],
    can_enable: true,
    requires_restart: false,
  };
}

function panel(feature: NanobotFeatureInfo, onFeaturesUpdate: (payload: NanobotFeaturesPayload) => void) {
  return (
    <ClientProvider client={client} token="tok">
      <ChannelSetupPanel
        token="tok"
        feature={feature}
        actionKey={null}
        showBrandLogos={false}
        onAction={() => {}}
        onFeaturesUpdate={onFeaturesUpdate}
      />
    </ClientProvider>
  );
}

describe("ChannelSetupPanel async feature fences", () => {
  beforeEach(() => {
    vi.mocked(validateChannel).mockReset();
    vi.mocked(configureChannel).mockReset();
    validateDeferreds.clear();
    configureDeferreds.clear();
    vi.mocked(validateChannel).mockImplementation((_transport, name) => {
      const pending = deferred<ChannelValidationPayload>();
      validateDeferreds.set(name, pending);
      return pending.promise;
    });
    vi.mocked(configureChannel).mockImplementation((_transport, name) => {
      const pending = deferred<ChannelConfigurePayload>();
      configureDeferreds.set(name, pending);
      return pending.promise;
    });
  });

  it("drops a pending validation when the feature changes and keeps the new panel usable", async () => {
    const featureA = channelFeature("alpha-channel", "Alpha Channel");
    const featureB = channelFeature("beta-channel", "Beta Channel");
    const onFeaturesUpdate = vi.fn();
    const { rerender } = render(panel(featureA, onFeaturesUpdate));

    fireEvent.change(screen.getByLabelText("Token"), {
      target: { value: "alpha-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Check and enable" }));

    expect(await screen.findByText("Checking...")).toBeInTheDocument();
    expect(validateDeferreds.has("alpha-channel")).toBe(true);

    rerender(panel(featureB, onFeaturesUpdate));

    expect(screen.getByRole("heading", { name: "Beta Channel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check and enable" })).toBeEnabled();
    expect(screen.queryByText("Checking...")).not.toBeInTheDocument();

    await act(async () => {
      validateDeferreds.get("alpha-channel")!.resolve(validationPayload("alpha-channel"));
      await Promise.resolve();
    });

    expect(onFeaturesUpdate).not.toHaveBeenCalled();
    expect(configureChannel).not.toHaveBeenCalled();
    expect(screen.queryByText("Checked and enabled.")).not.toBeInTheDocument();
    expect(screen.queryByText("Checking...")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check and enable" })).toBeEnabled();

    fireEvent.change(screen.getByLabelText("Token"), {
      target: { value: "beta-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Check and enable" }));

    await waitFor(() => expect(validateDeferreds.has("beta-channel")).toBe(true));
    await act(async () => {
      validateDeferreds.get("beta-channel")!.resolve(validationPayload("beta-channel"));
      await Promise.resolve();
    });
    await waitFor(() => expect(configureDeferreds.has("beta-channel")).toBe(true));

    const featuresPayload: NanobotFeaturesPayload = {
      features: [{ ...featureB, enabled: true }],
      enabled_count: 1,
    };
    await act(async () => {
      configureDeferreds.get("beta-channel")!.resolve({
        name: "beta-channel",
        saved: true,
        nanobot_features: featuresPayload,
      });
      await Promise.resolve();
    });

    await waitFor(() => expect(onFeaturesUpdate).toHaveBeenCalledWith(featuresPayload));
    expect(await screen.findByText("Checked and enabled.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check and enable" })).toBeEnabled();
  });

  it("drops a pending configure when the feature changes", async () => {
    const featureA = channelFeature("alpha-channel", "Alpha Channel");
    const featureB = channelFeature("beta-channel", "Beta Channel");
    const onFeaturesUpdate = vi.fn();
    const { rerender } = render(panel(featureA, onFeaturesUpdate));

    fireEvent.change(screen.getByLabelText("Token"), {
      target: { value: "alpha-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Check and enable" }));

    await waitFor(() => expect(validateDeferreds.has("alpha-channel")).toBe(true));
    await act(async () => {
      validateDeferreds.get("alpha-channel")!.resolve(validationPayload("alpha-channel"));
      await Promise.resolve();
    });
    await waitFor(() => expect(configureDeferreds.has("alpha-channel")).toBe(true));

    rerender(panel(featureB, onFeaturesUpdate));

    expect(screen.getByRole("heading", { name: "Beta Channel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check and enable" })).toBeEnabled();
    expect(screen.queryByText("Checking...")).not.toBeInTheDocument();

    await act(async () => {
      configureDeferreds.get("alpha-channel")!.resolve({
        name: "alpha-channel",
        saved: true,
        nanobot_features: {
          features: [{ ...featureA, enabled: true }],
          enabled_count: 1,
        },
      });
      await Promise.resolve();
    });

    expect(onFeaturesUpdate).not.toHaveBeenCalled();
    expect(screen.queryByText("Checked and enabled.")).not.toBeInTheDocument();
    expect(screen.queryByText("Checking...")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check and enable" })).toBeEnabled();
  });
});
