#!/usr/bin/env bash
# Read-only audit of a deployed nanobot instance.
#
# Collects the facts needed to answer: which build is running, what the client
# depends on, and whether the candidate harness would preserve that behaviour.
# Prints nothing secret: tokens, keys and chat ids are redacted.
#
# Usage:
#   scripts/audit_deployed_instance.sh                 # default /opt/marketing-agent
#   scripts/audit_deployed_instance.sh /srv/instance   # explicit instance root
#
# Run it on the server, or over ssh:
#   ssh <node> 'bash -s' < scripts/audit_deployed_instance.sh

set -uo pipefail

ROOT="${1:-/opt/marketing-agent}"
CONFIG="${NANOBOT_CONFIG:-}"

say() { printf '\n===== %s =====\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "host"
hostname
date -Is
uname -sr
echo "instance_root=$ROOT"
echo "user=$(id -un) uid=$(id -u)"

# --- locate the running code ------------------------------------------------
say "process"
ps -eo pid,etime,cmd 2>/dev/null | grep -E "[n]anobot" | head -5 || echo "no nanobot process visible"

say "service"
if have systemctl; then
  systemctl list-units --type=service --all --no-legend 2>/dev/null \
    | grep -i nanobot | head -5 || echo "no nanobot systemd unit"
fi

# --- source checkout --------------------------------------------------------
REPO="${NANOBOT_REPO:-}"
if [ -z "$REPO" ]; then
  for candidate in "$ROOT" "$ROOT/nanobot" "$ROOT/nanobot-custom" "$ROOT/repo" "$ROOT/app" "$ROOT/src"; do
    if [ -d "$candidate/.git" ]; then REPO="$candidate"; break; fi
  done
fi

say "checkout"
if [ -n "$REPO" ]; then
  echo "repo=$REPO"
  git -C "$REPO" rev-parse HEAD 2>/dev/null
  git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null
  git -C "$REPO" log -1 --format='%h %ad %s' --date=iso-strict 2>/dev/null
  echo "dirty_files=$(git -C "$REPO" status --porcelain=v1 2>/dev/null | wc -l)"
  echo "remotes:"
  git -C "$REPO" remote -v 2>/dev/null | sed 's/^/  /'
else
  echo "no git checkout found under $ROOT"
  echo "python_package=$(python3 -c 'import nanobot,os;print(os.path.dirname(nanobot.__file__))' 2>/dev/null || echo unknown)"
fi

say "version"
python3 -c 'import importlib.metadata as m; print("nanobot", m.version("nanobot"))' 2>/dev/null \
  || echo "nanobot not importable by python3"
for py in "$ROOT/.venv/bin/python" "$ROOT/venv/bin/python"; do
  [ -x "$py" ] && "$py" -c 'import importlib.metadata as m,sys; print(sys.executable); print("nanobot", m.version("nanobot"))' 2>/dev/null
done

# --- config (redacted) ------------------------------------------------------
if [ -z "$CONFIG" ]; then
  for candidate in "$ROOT/.nanobot/config.json" "$HOME/.nanobot/config.json"; do
    [ -f "$candidate" ] && CONFIG="$candidate" && break
  done
fi

say "config_shape"
if [ -f "$CONFIG" ]; then
  echo "config=$CONFIG"
  python3 - "$CONFIG" <<'PY'
import json, sys
path = sys.argv[1]
try:
    data = json.load(open(path))
except Exception as exc:
    print("unreadable:", exc)
    raise SystemExit

SECRET = ("key", "token", "secret", "password", "cookie", "authorization")

def redact(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if any(s in k.lower() for s in SECRET) and isinstance(v, str) and v:
                out[k] = f"<set:{len(v)}>"
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value

d = redact(data).get("agents", {}).get("defaults", {})
print("model:", d.get("model"), "| provider:", d.get("provider"))
print("maxTokens:", json.dumps(d.get("maxTokens")))
print("workspace:", d.get("workspace"))
print("sessionTtlMinutes:", d.get("sessionTtlMinutes"), "| idleCompaction:", json.dumps(d.get("idleCompaction")))
repos = d.get("contextRepos") or d.get("context_repos") or []
print("contextRepos:", json.dumps(repos, indent=2))
ch = data.get("channels", {})
print("taskUpdateMode:", ch.get("taskUpdateMode"), "| sendToolHints:", ch.get("sendToolHints"))
tg = ch.get("telegram", {})
print("telegram.enabled:", tg.get("enabled"),
      "| allowFrom_count:", len(tg.get("allowFrom") or []),
      "| allowChats_count:", len(tg.get("allowChats") or []),
      "| groupPolicy:", tg.get("groupPolicy"),
      "| streaming:", tg.get("streaming"))
print("telegram.fastCommands:", sorted((tg.get("fastCommands") or {}).keys()))
hr = d.get("heartbeat") or data.get("heartbeat") or {}
print("heartbeat:", json.dumps(hr))
PY
else
  echo "no config found (looked under $ROOT/.nanobot and ~/.nanobot)"
fi

# --- cron store -------------------------------------------------------------
say "cron"
CRON_DIRS=("$ROOT/.nanobot/workspace/cron" "$ROOT/.nanobot/cron" "$HOME/.nanobot/workspace/cron" "$HOME/.nanobot/cron")
FOUND=0
for dir in "${CRON_DIRS[@]}"; do
  for store in "$dir/jobs.json" "$dir/cron/jobs.json"; do
    if [ -f "$store" ]; then
      FOUND=1
      echo "store=$store"
      python3 - "$store" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
jobs = data.get("jobs", data if isinstance(data, list) else [])
print("jobs:", len(jobs))
print("has_rate_limit_fields:", any(
    "rateLimitRetries" in (j.get("state") or {}) or "retryAtMs" in (j.get("state") or {})
    for j in jobs
))
for j in jobs:
    s = j.get("state") or {}
    sched = j.get("schedule") or {}
    print("- {name!r} schedule={kind}/{expr} enabled={en} last={st} next={nx} retries={ret}".format(
        name=j.get("name"),
        kind=sched.get("kind"),
        expr=sched.get("expr") or sched.get("at") or sched.get("everyMs"),
        en=j.get("enabled"),
        st=s.get("lastStatus"),
        nx=s.get("nextRunAtMs"),
        ret=s.get("rateLimitRetries"),
    ))
PY
    fi
  done
done
[ "$FOUND" -eq 0 ] && echo "no cron store found"

# --- feature probes ---------------------------------------------------------
say "pipeline_usage"
if [ -n "$REPO" ]; then
  echo "spawn_pipeline in source: $(grep -rl "spawn_pipeline" "$REPO/nanobot" 2>/dev/null | wc -l) file(s)"
  echo "workflow plan dir: $([ -d "$REPO/.nanobot/workspace/current" ] && echo present || echo absent)"
fi
for base in "$ROOT/.nanobot/workspace" "$HOME/.nanobot/workspace"; do
  [ -d "$base/current" ] && echo "found plan state at $base/current:" && ls -la "$base/current" | head -5
done

say "rate_limit_history"
for log in "$ROOT/logs/"*.log "$ROOT/.nanobot/logs/"*.log "$HOME/.nanobot/logs/"*.log; do
  [ -f "$log" ] || continue
  count=$(grep -ci "rate limit" "$log" 2>/dev/null || true)
  count=${count:-0}
  [ "$count" -gt 0 ] && echo "$log: $count mention(s)"
done
[ -d "$ROOT/logs" ] || [ -d "$ROOT/.nanobot/logs" ] || [ -d "$HOME/.nanobot/logs" ] || echo "no log dir found"

say "context_repo_state"
python3 - "$CONFIG" <<'PY' 2>/dev/null
import json, os, subprocess, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit
repos = (data.get("agents", {}).get("defaults", {}) or {}).get("contextRepos") or []
for repo in repos:
    path = repo.get("path") if isinstance(repo, dict) else repo
    print("repo:", path, "readOnly:", repo.get("readOnly"), "autoSync:", repo.get("autoSync"))
    if path and os.path.isdir(path):
        for key, cmd in (
            ("head", ["git", "-C", path, "rev-parse", "--short", "HEAD"]),
            ("branch", ["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"]),
            ("dirty", ["git", "-C", path, "status", "--porcelain"]),
            ("last_commit", ["git", "-C", path, "log", "-1", "--format=%ad %s", "--date=iso-strict"]),
        ):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception as exc:
                out = f"error: {exc}"
            print(f"  {key}: {out if key != 'dirty' else str(len(out.splitlines())) + ' file(s)'}")
        manifest = os.path.join(path, "nanobot.context.json")
        print("  manifest:", "present" if os.path.isfile(manifest) else "missing")
PY

say "done"
echo "Read-only audit finished. No files were written."
