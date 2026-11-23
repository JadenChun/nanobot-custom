"""Outer sandbox confinement attestation for reduced-protection nested agent launches."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# A reduced-protection nested agent launch is refused, not approved, when the
# launch path cannot prove outer confinement and process ownership.
LAUNCH_CONFINEMENT_UNPROVEN = "launch_confinement_unproven"

# External agent launches that lower their own sandbox or approval protection.
# A launch that keeps the tool's default protection is not listed and stays an
# ordinary command. These patterns match the lowercased command, so every
# alternative is written in lowercase, including values such as
# ``bypassPermissions`` and ``danger-full-access``.
_REDUCED_PROTECTION_LAUNCHES = (
    re.compile(
        r"\bcodex\b[^|;&\n]*?"
        r"(?:--yolo\b|--dangerously-bypass-approvals-and-sandbox\b|--full-auto\b)"
    ),
    re.compile(
        r"\bcodex\b[^|;&\n]*?"
        r"(?:--sandbox[= ]+[\\'\"]*danger-full-access\b"
        r"|--ask-for-approval[= ]+[\\'\"]*never\b)"
    ),
    # Codex config overrides that select a full-access sandbox or disable approval.
    re.compile(
        r"\bcodex\b[^|;&\n]*?(?:-c\s+|--config\s+|--config=)[\\'\"]*"
        r"(?:sandbox_mode|approval_policy)\s*=\s*[\\'\"]*"
        r"(?:danger-full-access|never)\b"
    ),
    re.compile(
        r"\bclaude\b[^|;&\n]*?"
        r"(?:--dangerously-skip-permissions\b"
        r"|--permission-mode[= ]+[\\'\"]*bypasspermissions\b)"
    ),
    re.compile(r"\bgemini\b[^|;&\n]*?(?:--yolo\b|\s-y\b)"),
)

# A tmux command starts or types into a process the shell executor does not own.
_TMUX_LAUNCH_COMMANDS = re.compile(
    r"\btmux\b[^|;&\n]*\b(?:send-keys|new-session|new-window|respawn-pane|split-window)\b"
)


def nested_launch_path(command: str) -> str | None:
    """Classify a reduced-protection nested launch and its launch path.

    Returns ``"shell"`` for a direct launch the shell executor starts itself,
    ``"tmux"`` when the launch is typed into a tmux session, and ``None`` when
    the command does not lower a nested agent's protection.
    """
    lower = command.strip().lower()
    if not lower:
        return None
    if not any(pattern.search(lower) for pattern in _REDUCED_PROTECTION_LAUNCHES):
        return None
    return "tmux" if _TMUX_LAUNCH_COMMANDS.search(lower) else "shell"


def nested_launch_id(
    command: str, working_dir: str | None = None, workspace: Path | str | None = None,
) -> str:
    """Identity of one nested launch: the exact command and effective directory.

    The policy and the executor resolve the same directory from the same
    workspace, so one launch has one identity on both sides of the boundary.
    """
    base = Path(str(workspace if workspace is not None else Path.cwd())).expanduser()
    target = Path(working_dir).expanduser() if working_dir else base
    if not target.is_absolute():
        target = base / target
    payload = json.dumps(
        {"command": " ".join(command.split()), "cwd": str(target.resolve(strict=False))},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LaunchConfinementAttestation:
    """Executor statement that confinement and process ownership hold for one launch."""

    launch_id: str
    path: str
    sandbox: str
    process_owner: str

    def covers(self, launch_id: str, path: str) -> bool:
        """True only for the attested launch, path, and non-empty providers."""
        return (
            bool(launch_id)
            and launch_id == self.launch_id
            and path == self.path
            and bool(self.sandbox)
            and bool(self.process_owner)
        )


LaunchConfinementAttestor = Callable[[str, str | None], LaunchConfinementAttestation | None]

# Executors register the attestor they can honestly issue for their workspace.
_LAUNCH_CONFINEMENT_ATTESTORS: dict[str, LaunchConfinementAttestor] = {}


def _workspace_key(workspace: Path | str | None) -> str | None:
    if workspace is None:
        return None
    raw = str(workspace)
    if not raw:
        return None
    try:
        return str(Path(raw).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return raw


def register_launch_confinement_attestor(
    workspace: Path | str, attestor: LaunchConfinementAttestor,
) -> None:
    key = _workspace_key(workspace)
    if key is not None:
        _LAUNCH_CONFINEMENT_ATTESTORS[key] = attestor


def unregister_launch_confinement_attestor(workspace: Path | str) -> None:
    key = _workspace_key(workspace)
    if key is not None:
        _LAUNCH_CONFINEMENT_ATTESTORS.pop(key, None)


def resolve_launch_confinement_attestor(
    workspace: Path | None,
) -> LaunchConfinementAttestor | None:
    key = _workspace_key(workspace)
    return None if key is None else _LAUNCH_CONFINEMENT_ATTESTORS.get(key)


def launch_confinement_covers(
    *,
    workspace: Path,
    command: str,
    working_dir: str | None,
    launch_path: str,
    attestor: LaunchConfinementAttestor | None = None,
) -> bool:
    """True only when the executor attests this launch's confinement."""
    if launch_path != "shell":
        # A tmux command types into a process the shell executor neither
        # confines for that launch nor owns; it cannot attest either.
        return False
    resolved = attestor or resolve_launch_confinement_attestor(workspace)
    if resolved is None:
        return False
    try:
        attestation = resolved(command, working_dir)
    except (OSError, ValueError):
        return False
    if attestation is None:
        return False
    return attestation.covers(
        nested_launch_id(command, working_dir, workspace), launch_path,
    )
