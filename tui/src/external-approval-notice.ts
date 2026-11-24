import { BoxRenderable, TextRenderable, type CliRenderer } from "@opentui/core"

import type { ApprovalDecision, ExternalApproval } from "./protocol"
import type { PickerMenuTheme } from "./picker-menu"

const ACTIONS = [
  { decision: "approve", label: "Allow once" },
  { decision: "approve_and_save", label: "Allow & save" },
  { decision: "deny", label: "Deny" },
] as const

/** Shows one live request at a time and leaves the composer available. */
export class ExternalApprovalNotice {
  readonly root: BoxRenderable
  private readonly summary: TextRenderable
  private readonly command: TextRenderable
  private readonly actions: BoxRenderable
  private readonly detail: TextRenderable
  private readonly buttons: TextRenderable[]
  private approvalId: string | null = null

  constructor(
    renderer: CliRenderer,
    theme: PickerMenuTheme,
    onDecision: (id: string, decision: ApprovalDecision["decision"]) => void,
  ) {
    this.root = new BoxRenderable(renderer, {
      id: "kohaku-tui-external-approval",
      width: "100%",
      flexShrink: 0,
      flexDirection: "column",
      border: true,
      borderStyle: "rounded",
      borderColor: theme.border,
      paddingLeft: 1,
      paddingRight: 1,
      visible: false,
    })
    this.summary = new TextRenderable(renderer, { height: 1, truncate: true })
    this.command = new TextRenderable(renderer, { height: 1, truncate: true })
    this.detail = new TextRenderable(renderer, { height: 1, truncate: true })
    this.actions = new BoxRenderable(renderer, { height: 1, flexDirection: "row", gap: 2 })
    this.buttons = ACTIONS.map(({ decision, label }) => {
      const button = new TextRenderable(renderer, {
        content: label,
        height: 1,
        selectable: false,
        onMouseDown: (event) => {
          if (event.button !== 0 || !this.approvalId) return
          event.preventDefault()
          event.stopPropagation()
          onDecision(this.approvalId, decision)
        },
      })
      this.actions.add(button)
      return button
    })
    this.root.add(this.summary)
    this.root.add(this.command)
    this.root.add(this.detail)
    this.root.add(this.actions)
    this.setTheme(theme)
  }

  show(
    id: string,
    approval: ExternalApproval,
    owner: string,
    pending: boolean,
    remaining: number,
  ): void {
    this.approvalId = pending ? null : id
    this.root.visible = true
    const summary = approval.summary.replace(/\s+/gu, " ")
    this.summary.content = `${owner} approval${remaining ? ` (+${remaining} more)` : ""} · ${summary}`
    this.command.content = (approval.command ?? "").replace(/[\r\n]/gu, " ↵ ")
    this.command.visible = Boolean(approval.command)
    const saveCommand = approval.save_exclusion_command
    const help = "/approval shows the full request and keyboard choices."
    this.detail.content = pending
      ? "Waiting for the gateway to resolve the approval."
      : saveCommand ? `Save exclusion for ${saveCommand} · ${help}` : help
    this.buttons[1]!.visible = Boolean(saveCommand)
    this.actions.visible = !pending
  }

  hide(): void {
    this.approvalId = null
    this.root.visible = false
  }

  setTheme(theme: PickerMenuTheme): void {
    this.root.borderColor = theme.border
    this.summary.fg = theme.warning || theme.text
    this.command.fg = theme.text
    this.detail.fg = theme.muted
    for (const button of this.buttons) button.fg = theme.accent || theme.text
  }
}
