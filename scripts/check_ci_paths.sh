#!/usr/bin/env bash
# Classify changed paths for the GitHub Actions change detector.
set -euo pipefail

python_required=false
webui_required=false
tui_required=false
docker_required=false

while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  case "$file" in
    .github/workflows/*)
      python_required=true
      webui_required=true
      tui_required=true
      docker_required=true
      ;;
    webui/*|nanobot/channels/*/webui/*)
      webui_required=true
      docker_required=true
      ;;
    nanobot/*)
      python_required=true
      docker_required=true
      ;;
    tests/test_docker.sh)
      docker_required=true
      ;;
    pyproject.toml|hatch_build.py|scripts/install_channel_dependencies.py)
      python_required=true
      docker_required=true
      ;;
    tests/*|conftest.py|uv.lock|scripts/*)
      python_required=true
      ;;
    Dockerfile|Dockerfile.*|.dockerignore|docker-compose*.yml|entrypoint.sh|render-config.json|README.md|LICENSE|THIRD_PARTY_NOTICES.md)
      docker_required=true
      ;;
    tui/*)
      tui_required=true
      ;;
    docs/migration/*|docs/migration/**/*)
      python_required=true
      ;;
    docs/*|.agent/*|.github/ISSUE_TEMPLATE/*|AGENTS.md|CLAUDE.md|COMMUNICATION.md|CONTRIBUTING.md|README.md|SECURITY.md|webui/README.md|render.yaml)
      ;;
    *)
      python_required=true
      webui_required=true
      tui_required=true
      docker_required=true
      ;;
  esac
done

echo "python_required=$python_required"
echo "webui_required=$webui_required"
echo "tui_required=$tui_required"
echo "docker_required=$docker_required"
