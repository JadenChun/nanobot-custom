import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ChatList } from "@/components/ChatList";
import { sessionHandleColor } from "@/lib/session-handle";
import type { ChatSummary } from "@/lib/types";

function session(overrides: Partial<ChatSummary>): ChatSummary {
  const chatId = overrides.chatId ?? "chat";
  return {
    key: `websocket:${chatId}`,
    channel: "websocket",
    chatId,
    createdAt: "2026-05-20T10:00:00Z",
    updatedAt: "2026-05-20T10:00:00Z",
    preview: "",
    ...overrides,
  };
}

function rect(top: number): DOMRect {
  return {
    x: 0,
    y: top,
    width: 240,
    height: 32,
    top,
    right: 240,
    bottom: top + 32,
    left: 0,
    toJSON: () => ({}),
  };
}

describe("ChatList", () => {
  const originalAnimate = HTMLElement.prototype.animate;
  const originalGetBoundingClientRect = HTMLElement.prototype.getBoundingClientRect;

  afterEach(() => {
    HTMLElement.prototype.animate = originalAnimate;
    HTMLElement.prototype.getBoundingClientRect = originalGetBoundingClientRect;
    localStorage.removeItem("nanobot-webui.collapsed-pane-groups.v1");
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("opens a conversation's existing actions from the row context menu", async () => {
    const onTogglePin = vi.fn();
    render(
      <ChatList
        sessions={[session({ chatId: "review", title: "Review the patch" })]}
        activeKey="websocket:review"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={onTogglePin}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
      />,
    );

    const row = screen.getByRole("button", { name: "Review the patch" })
      .closest("[data-chat-row]")!;
    fireEvent.contextMenu(row);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Pin" }));

    expect(onTogglePin).toHaveBeenCalledWith("websocket:review");
  });

  it("restores the colored handle underline and animated active track", () => {
    render(
      <ChatList
        sessions={[session({
          chatId: "review",
          title: "Review the patch",
          handle: { id: "handle_1234", name: "mira" },
        })]}
        activeKey="websocket:review"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
      />,
    );

    const conversation = screen.getByRole("button", {
      name: "@mira Review the patch",
    });
    const handle = conversation.querySelector("[data-sidebar-session-handle]");
    expect(handle).toHaveClass("max-w-20", "shrink-0");
    const underline = handle?.querySelector("[data-sidebar-session-handle-underline]");
    expect(underline).toHaveClass("border-b-2", "text-foreground");
    expect(underline?.getAttribute("style"))
      .toContain(sessionHandleColor("handle_1234"));

    const activeTrack = conversation.querySelector("[data-sidebar-selection-track]");
    expect(activeTrack).toHaveClass(
      "origin-left",
      "scale-x-100",
      "transition-transform",
      "motion-reduce:transition-none",
    );
  });

  it("marks a conversation that needs recovery attention with a warning indicator", () => {
    render(
      <ChatList
        sessions={[session({ chatId: "recovery", title: "Interrupted task" })]}
        recoveryChatIds={["recovery"]}
        activeKey="websocket:other"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
      />,
    );

    expect(screen.getByRole("img", { name: "This conversation needs your attention" }))
      .toBeInTheDocument();
  });

  it("shows the running indicator while a recovery continuation is active", () => {
    render(
      <ChatList
        sessions={[session({ chatId: "recovery", title: "Interrupted task" })]}
        runningChatIds={["recovery"]}
        recoveryChatIds={["recovery"]}
        activeKey="websocket:other"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
      />,
    );

    expect(screen.getByRole("img", { name: "Agent running" })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "This conversation needs your attention" }))
      .not.toBeInTheDocument();
  });

  it("selects a contiguous session range with Shift-click", async () => {
    render(
      <ChatList
        sessions={[
          session({ chatId: "first", title: "First topic" }),
          session({ chatId: "second", title: "Second topic" }),
          session({ chatId: "third", title: "Third topic" }),
          session({ chatId: "fourth", title: "Fourth topic" }),
        ]}
        activeKey="websocket:first"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
      />,
    );

    fireEvent.pointerDown(screen.getByRole("button", {
      name: "Topic actions for First topic",
    }), { button: 0, ctrlKey: false });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Select" }));

    fireEvent.click(screen.getByRole("button", { name: "Fourth topic" }), {
      shiftKey: true,
    });

    expect(screen.getByText("4 selected")).toBeInTheDocument();
    for (const title of ["First topic", "Second topic", "Third topic", "Fourth topic"]) {
      expect(screen.getByRole("button", { name: title })).toHaveAttribute(
        "aria-pressed",
        "true",
      );
    }
  });

  it("shows temporary chats separately and lets the user reopen or close them", async () => {
    const temporarySession = session({
      key: "temporary:temporary-one",
      chatId: "temporary-one",
      preview: "hi",
    });
    const onSelect = vi.fn();
    const onClose = vi.fn();

    render(
      <ChatList
        sessions={[]}
        temporarySessions={[temporarySession]}
        activeKey={null}
        onSelect={onSelect}
        onCloseTemporaryChat={onClose}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
      />,
    );

    const section = screen.getByRole("region", { name: "Temporary chats" });
    fireEvent.click(within(section).getByRole("button", { name: "hi" }));
    expect(onSelect).toHaveBeenCalledWith("temporary:temporary-one");

    fireEvent.click(within(section).getByRole("button", {
      name: "Close temporary chat: hi",
    }));
    expect(onClose).toHaveBeenCalledWith("temporary:temporary-one");
  });

  it("orders chats by latest session activity by default", () => {
    const sessions = [
      session({
        chatId: "older",
        title: "Older chat",
        preview: "/model fast",
        updatedAt: "2026-05-21T10:00:00Z",
      }),
      session({
        chatId: "newest",
        title: "Newest chat",
        updatedAt: "2026-05-21T12:00:00Z",
      }),
      session({
        chatId: "middle",
        title: "Middle chat",
        updatedAt: "2026-05-21T11:00:00Z",
      }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey={null}
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        showPreviews
      />,
    );

    const chatsSection = screen.getAllByRole("region")[0];
    const text = chatsSection.textContent ?? "";

    expect(text.indexOf("Newest chat")).toBeLessThan(text.indexOf("Middle chat"));
    expect(text.indexOf("Middle chat")).toBeLessThan(text.indexOf("Older chat"));
    expect(screen.queryByText("/model fast")).not.toBeInTheDocument();
  });

  it("shows a pin indicator for pinned chats", () => {
    const sessions = [
      session({ chatId: "pinned", title: "Pinned chat" }),
      session({ chatId: "normal", title: "Normal chat" }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey={null}
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        pinnedKeys={["websocket:pinned"]}
      />,
    );

    const pinnedSection = screen.getByRole("region", { name: "Pinned" });
    expect(
      within(pinnedSection)
        .getByText("Pinned chat")
        .closest("[data-chat-row]")
        ?.querySelector("[data-sidebar-pinned-indicator]"),
    ).toBeInTheDocument();
    expect(document.querySelectorAll("[data-sidebar-pinned-indicator]")).toHaveLength(1);
  });

  it("groups WebUI chats by workspace project while preserving in-project sorting and activity", () => {
    const sessions = [
      session({
        chatId: "zeta",
        title: "Zeta task",
        updatedAt: "2026-05-20T12:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/nanobot",
          project_name: "nanobot",
          access_mode: "restricted",
        },
      }),
      session({
        chatId: "alpha",
        title: "Alpha task",
        handle: { id: "handle_alpha", name: "mira" },
        updatedAt: "2026-05-20T11:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/nanobot",
          project_name: "nanobot",
          access_mode: "restricted",
        },
      }),
      session({
        chatId: "bench",
        title: "Bench task",
        updatedAt: "2026-05-21T09:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/nanobot-bench",
          project_name: "nanobot-bench",
          access_mode: "full",
        },
      }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey="websocket:alpha"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        sort="title_asc"
        showTimestamps
        runningChatIds={["zeta"]}
      />,
    );

    const nanobotSection = screen.getByRole("region", { name: "nanobot" });
    const nanobotText = nanobotSection.textContent ?? "";
    const projectSurface = nanobotSection.querySelector(
      "[data-sidebar-project-surface]",
    );

    expect(screen.getByRole("region", { name: "nanobot-bench" })).toBeInTheDocument();
    expect(projectSurface).toHaveClass(
      "rounded-es-[16px]",
      "border-s-2",
      "border-sidebar-foreground/10",
    );
    expect(within(nanobotSection).getByText("Alpha task")).toBeInTheDocument();
    expect(
      within(nanobotSection)
        .getByText("@mira")
        .closest("[data-sidebar-session-handle]")
        ?.parentElement,
    ).toHaveClass("items-center");
    expect(within(nanobotSection).getByText("Zeta task")).toBeInTheDocument();
    expect(nanobotText.indexOf("Alpha task")).toBeLessThan(nanobotText.indexOf("Zeta task"));
    expect(within(nanobotSection).getByLabelText("Agent running")).toBeInTheDocument();
    expect(screen.queryByText("Today")).not.toBeInTheDocument();
  });

  it("keeps default workspace topics in the Topics section instead of a project folder", () => {
    const sessions = [
      session({
        chatId: "default",
        title: "Default workspace chat",
        updatedAt: "2026-05-21T10:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/.nanobot/workspace",
          project_name: "workspace",
          access_mode: "restricted",
        },
      }),
      session({
        chatId: "project",
        title: "Project chat",
        updatedAt: "2026-05-21T11:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/nanobot",
          project_name: "nanobot",
          access_mode: "restricted",
        },
      }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey="websocket:default"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        defaultWorkspacePath="/Users/me/.nanobot/workspace"
        showTimestamps
      />,
    );

    expect(screen.getByText("Projects")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "nanobot" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "workspace" })).not.toBeInTheDocument();

    const chatsSection = screen.getByRole("region", { name: "Topics" });
    expect(within(chatsSection).getByText("Default workspace chat")).toBeInTheDocument();
    expect(within(chatsSection).queryByText("Project chat")).not.toBeInTheDocument();
  });

  it("grows and retracts the row-owned selection track", () => {
    const props = {
      sessions: [
        session({ chatId: "active", title: "Active topic" }),
        session({ chatId: "inactive", title: "Inactive topic" }),
      ],
      onSelect: vi.fn(),
      onRequestDelete: vi.fn(),
      onTogglePin: vi.fn(),
      onRequestRename: vi.fn(),
      onToggleArchive: vi.fn(),
    };

    const { rerender } = render(
      <ChatList
        {...props}
        activeKey="websocket:active"
      />,
    );

    const activeButton = screen.getByRole("button", { name: "Active topic" });
    expect(activeButton).toHaveAttribute("aria-current", "page");
    expect(activeButton.querySelector("[data-sidebar-selection-track]"))
      .toHaveClass("origin-left", "scale-x-100", "transition-transform");
    expect(activeButton.querySelector("[data-sidebar-selection-track]"))
      .toHaveStyle({ backgroundColor: "currentColor" });

    rerender(
      <ChatList
        {...props}
        activeKey="websocket:inactive"
      />,
    );

    expect(screen.getByRole("button", { name: "Active topic" }))
      .not.toHaveAttribute("aria-current");
    expect(screen.getByRole("button", { name: "Inactive topic" }))
      .toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Active topic" })
      .querySelector("[data-sidebar-selection-track]"))
      .toHaveClass("scale-x-0");
    expect(screen.getByRole("button", { name: "Inactive topic" })
      .querySelector("[data-sidebar-selection-track]"))
      .toHaveClass("scale-x-100");
  });

  it("can collapse a project group and keeps project rename separate from chat titles", async () => {
    const onToggleGroup = vi.fn();
    const onRequestRenameProject = vi.fn();
    const onNewChatInProject = vi.fn();
    const sessions = [
      session({
        chatId: "alpha",
        title: "Alpha task",
        workspaceScope: {
          project_path: "/Users/me/nanobot",
          project_name: "nanobot",
          access_mode: "restricted",
        },
      }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey="websocket:alpha"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        onToggleGroup={onToggleGroup}
        onRequestRenameProject={onRequestRenameProject}
        onNewChatInProject={onNewChatInProject}
        projectNameOverrides={{ "/Users/me/nanobot": "Photos" }}
        collapsedGroups={{ "project:/Users/me/nanobot": true }}
      />,
    );

    const projectSection = screen.getByRole("region", { name: "Photos" });
    fireEvent.click(within(projectSection).getByRole("button", { name: "Photos" }));

    expect(onToggleGroup).toHaveBeenCalledWith("project:/Users/me/nanobot");
    expect(within(projectSection).queryByText("Alpha task")).not.toBeInTheDocument();

    const projectButton = within(projectSection).getByRole("button", { name: "Photos" });
    fireEvent.contextMenu(projectButton);
    fireEvent.click(await screen.findByRole("menuitem", { name: "New topic" }));
    expect(onNewChatInProject).toHaveBeenCalledWith("/Users/me/nanobot", "Photos");
    expect(onToggleGroup).toHaveBeenCalledTimes(1);

    fireEvent.contextMenu(projectButton);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Rename" }));

    expect(onRequestRenameProject).toHaveBeenCalledWith("/Users/me/nanobot", "Photos");
  });

  it("animates project disclosure and surrounding layout like tab groups", () => {
    let collapsed = false;
    const onToggleGroup = vi.fn();
    const animate = vi.fn(() => ({
      addEventListener: vi.fn(),
      cancel: vi.fn(),
    }) as unknown as Animation);
    HTMLElement.prototype.animate = animate;
    HTMLElement.prototype.getBoundingClientRect = function getBoundingClientRect() {
      const followsCollapsedProject = this.textContent?.includes("Beta") ?? false;
      return rect(followsCollapsedProject ? (collapsed ? 64 : 160) : 0);
    };
    const sessions = [
      session({
        chatId: "alpha",
        title: "Alpha task",
        workspaceScope: {
          project_path: "/Users/me/alpha",
          project_name: "Alpha project",
          access_mode: "restricted",
        },
      }),
      session({
        chatId: "beta",
        title: "Beta task",
        workspaceScope: {
          project_path: "/Users/me/beta",
          project_name: "Beta project",
          access_mode: "restricted",
        },
      }),
    ];
    const props = {
      sessions,
      activeKey: "websocket:alpha",
      onSelect: vi.fn(),
      onRequestDelete: vi.fn(),
      onTogglePin: vi.fn(),
      onRequestRename: vi.fn(),
      onRequestRenameProject: vi.fn(),
      onToggleArchive: vi.fn(),
      onToggleGroup,
    };

    const { rerender } = render(
      <ChatList {...props} collapsedGroups={{ "project:/Users/me/alpha": false }} />,
    );

    const projectButton = screen.getByRole("button", { name: "Alpha project" });
    const disclosureButton = screen.getByRole("button", {
      name: "Projects: Alpha project",
    });
    expect(projectButton).toHaveAttribute("aria-expanded", "true");
    expect(disclosureButton).toHaveAttribute("aria-expanded", "true");
    const expandedIcon = disclosureButton
      .querySelector("[data-sidebar-project-disclosure-icon]");
    expect(expandedIcon).toHaveClass(
      "transition-transform",
      "duration-200",
      "ease-out",
      "motion-reduce:transition-none",
    );
    expect(expandedIcon).not.toHaveClass("rotate-90");
    expect(screen.getByRole("button", { name: "Topic actions for Alpha project" })
      .compareDocumentPosition(disclosureButton) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();

    fireEvent.click(disclosureButton);
    expect(onToggleGroup).toHaveBeenCalledWith("project:/Users/me/alpha");
    collapsed = true;
    rerender(
      <ChatList {...props} collapsedGroups={{ "project:/Users/me/alpha": true }} />,
    );

    expect(screen.getByRole("button", { name: "Projects: Alpha project" })
      .querySelector("[data-sidebar-project-disclosure-icon]"))
      .toHaveClass("rotate-90");
    expect(projectButton).toHaveAttribute("aria-expanded", "false");
    expect(animate).toHaveBeenCalledWith(
      [
        { transform: "translateY(96px)" },
        { transform: "translateY(0)" },
      ],
      {
        duration: 180,
        easing: "cubic-bezier(0.2, 0, 0, 1)",
      },
    );
    expect(screen.getByRole("button", { name: "Beta project" })
      .closest("[data-sidebar-group-header]"))
      .toHaveAttribute("data-sidebar-group-header", "project:/Users/me/beta");
  });

  it("hides the updated dot for the active chat", () => {
    const sessions = [
      session({
        chatId: "active",
        title: "Active task",
      }),
      session({
        chatId: "done",
        title: "Done task",
      }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey="websocket:active"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        updatedChatIds={["active", "done"]}
      />,
    );

    const updated = screen.getAllByLabelText("New activity");
    expect(updated).toHaveLength(1);
    expect(updated[0].firstElementChild).toHaveClass("h-2", "w-2");
  });

  it("folds long default workspace chats and can show all", () => {
    const sessions = Array.from({ length: 10 }, (_, index) =>
      session({
        chatId: `chat-${index}`,
        title: `Chat ${index}`,
        updatedAt: `2026-05-21T10:${String(index).padStart(2, "0")}:00Z`,
        workspaceScope: {
          project_path: "/Users/me/.nanobot/workspace",
          project_name: "workspace",
          access_mode: "restricted",
        },
      }),
    );
    const onToggleGroup = vi.fn();
    const baseProps = {
      sessions,
      activeKey: null,
      onSelect: vi.fn(),
      onRequestDelete: vi.fn(),
      onTogglePin: vi.fn(),
      onRequestRename: vi.fn(),
      onToggleArchive: vi.fn(),
      onToggleGroup,
      defaultWorkspacePath: "/Users/me/.nanobot/workspace",
    };

    const { rerender } = render(<ChatList {...baseProps} />);
    const chatsSection = screen.getByRole("region", { name: "Topics" });

    expect(within(chatsSection).getByText("Chat 9")).toBeInTheDocument();
    expect(within(chatsSection).getByText("Chat 2")).toBeInTheDocument();
    expect(within(chatsSection).queryByText("Chat 1")).not.toBeInTheDocument();
    expect(within(chatsSection).queryByRole("button", { name: "Show all" })).not.toBeInTheDocument();
    fireEvent.click(within(chatsSection).getByRole("button", { name: "2 hidden topics" }));

    expect(onToggleGroup).toHaveBeenCalledWith("workspace:chats");

    rerender(
      <ChatList
        {...baseProps}
        collapsedGroups={{ "workspace:chats": false }}
      />,
    );

    expect(within(chatsSection).getByText("Chat 0")).toBeInTheDocument();
    expect(within(chatsSection).getByRole("button", { name: "Show less" })).toBeInTheDocument();
  });

  it("sorts Topics section among project groups by recency, not always last", () => {
    const sessions = [
      session({
        chatId: "recent-chat",
        title: "Recent chat",
        updatedAt: "2026-05-21T12:00:00Z",
      }),
      session({
        chatId: "project-a",
        title: "Project A task",
        updatedAt: "2026-05-21T10:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/project-a",
          project_name: "project-a",
          access_mode: "restricted",
        },
      }),
      session({
        chatId: "project-b",
        title: "Project B task",
        updatedAt: "2026-05-21T11:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/project-b",
          project_name: "project-b",
          access_mode: "restricted",
        },
      }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey="websocket:recent-chat"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        showTimestamps
      />,
    );

    const allRegions = screen.getAllByRole("region");
    const regionNames = allRegions.map((r) => r.getAttribute("aria-label") ?? r.textContent);

    // The most recently updated conversation ("Recent chat" at 12:00) must be
    // in the first group — Topics should come before both projects.
    const chatsIdx = regionNames.findIndex((n) => n?.includes("Topics"));
    const projAIdx = regionNames.findIndex((n) => n?.includes("project-a"));
    const projBIdx = regionNames.findIndex((n) => n?.includes("project-b"));

    expect(chatsIdx).toBeLessThan(projAIdx);
    expect(chatsIdx).toBeLessThan(projBIdx);
    expect(within(allRegions[chatsIdx]).getByText("Recent chat")).toBeInTheDocument();
  });

  it("keeps one Projects heading when Topics sorts between project groups", () => {
    const sessions = [
      session({
        chatId: "project-a",
        title: "Project A task",
        updatedAt: "2026-05-21T12:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/project-a",
          project_name: "project-a",
          access_mode: "restricted",
        },
      }),
      session({
        chatId: "middle-chat",
        title: "Middle chat",
        updatedAt: "2026-05-21T11:00:00Z",
      }),
      session({
        chatId: "project-b",
        title: "Project B task",
        updatedAt: "2026-05-21T10:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/project-b",
          project_name: "project-b",
          access_mode: "restricted",
        },
      }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey="websocket:middle-chat"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        showTimestamps
      />,
    );

    const regionNames = screen
      .getAllByRole("region")
      .map((r) => r.getAttribute("aria-label") ?? "");

    expect(regionNames).toEqual(["project-a", "Topics", "project-b"]);
    expect(screen.getAllByText("Projects")).toHaveLength(1);
  });

  it("keeps Topics last when its latest conversation is older than all projects", () => {
    const sessions = [
      session({
        chatId: "project-a",
        title: "Project A task",
        updatedAt: "2026-05-21T12:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/project-a",
          project_name: "project-a",
          access_mode: "restricted",
        },
      }),
      session({
        chatId: "project-b",
        title: "Project B task",
        updatedAt: "2026-05-21T11:00:00Z",
        workspaceScope: {
          project_path: "/Users/me/project-b",
          project_name: "project-b",
          access_mode: "restricted",
        },
      }),
      session({
        chatId: "old-chat",
        title: "Old chat",
        updatedAt: "2026-05-21T10:00:00Z",
      }),
    ];

    render(
      <ChatList
        sessions={sessions}
        activeKey="websocket:old-chat"
        onSelect={vi.fn()}
        onRequestDelete={vi.fn()}
        onTogglePin={vi.fn()}
        onRequestRename={vi.fn()}
        onToggleArchive={vi.fn()}
        showTimestamps
      />,
    );

    const regionNames = screen
      .getAllByRole("region")
      .map((r) => r.getAttribute("aria-label") ?? "");

    expect(regionNames).toEqual(["project-a", "project-b", "Topics"]);
    expect(screen.getAllByText("Projects")).toHaveLength(1);
  });
});
