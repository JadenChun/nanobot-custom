from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PWSH = os.environ.get("NANOBOT_TEST_PWSH") or shutil.which("pwsh")

pytestmark = pytest.mark.skipif(PWSH is None, reason="PowerShell 7 is required for script tests")


def _ps_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_powershell(script: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [PWSH, "-NoLogo", "-NoProfile", "-Command", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "USERPROFILE": os.environ.get("USERPROFILE", str(Path.home()))},
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result


def _paths_block(tmp_path: Path, *, marker: bool) -> tuple[Path, Path, str]:
    profile = tmp_path / "profile with spaces"
    logs = tmp_path / "logs"
    profile.mkdir()
    logs.mkdir()
    marker_path = profile / ".nanobot-profile-ready"
    if marker:
        marker_path.touch()
    config = tmp_path / "config.json"
    config.write_text('{"agents":{"crawler":{"enabled":true}}}', encoding="utf-8")
    block = f"""
$testProfile = {_ps_string(profile)}
$testLogs = {_ps_string(logs)}
$testMarker = {_ps_string(marker_path)}
function Get-Command {{ param($Name, $CommandType, $ErrorAction) [pscustomobject]@{{ Source = "python" }} }}
function Get-LocalCrawl4AiPaths {{
    [pscustomobject]@{{ ProfilePath = $testProfile; LogDirectory = $testLogs; MarkerPath = $testMarker }}
}}
function Test-DedicatedCrawl4AiProfileIdle {{ return $true }}
function Test-LocalCrawl4AiPortIdle {{ return $true }}
"""
    return config, marker_path, block


def test_powershell_scripts_parse() -> None:
    scripts = [
        "deploy/scripts/local_crawl4ai_common.ps1",
        "deploy/scripts/start_local_marketing_agent.ps1",
        "deploy/scripts/setup_local_crawl4ai_profile.ps1",
    ]
    files = ", ".join(_ps_string(ROOT / path) for path in scripts)
    _run_powershell(
        f"""
foreach ($file in @({files})) {{
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$tokens, [ref]$errors)
    if ($errors.Count) {{ $errors | ForEach-Object {{ Write-Error $_ }}; exit 1 }}
}}
"""
    )


def test_launcher_rejects_missing_marker(tmp_path: Path) -> None:
    config, _, paths = _paths_block(tmp_path, marker=False)
    _run_powershell(
        f"""
. {_ps_string(ROOT / "deploy/scripts/start_local_marketing_agent.ps1")}
{paths}
try {{
    [void](Start-LocalMarketingAgent -ConfigPath {_ps_string(config)})
    throw "launcher unexpectedly started without a marker"
}}
catch {{
    if ($_.Exception.Message -notmatch "profile is not ready") {{ throw }}
}}
Write-Output "ok"
"""
    )


def test_launcher_rejects_disabled_crawler_and_occupied_port(tmp_path: Path) -> None:
    disabled = tmp_path / "disabled.json"
    disabled.write_text('{"agents":{"crawler":{"enabled":false}}}', encoding="utf-8")
    _, _, paths = _paths_block(tmp_path, marker=True)
    _run_powershell(
        f"""
. {_ps_string(ROOT / "deploy/scripts/start_local_marketing_agent.ps1")}
{paths}
try {{ [void](Start-LocalMarketingAgent -ConfigPath {_ps_string(disabled)}); throw "disabled crawler started" }}
catch {{ if ($_.Exception.Message -notmatch "agents.crawler.enabled") {{ throw }} }}
function Test-LocalCrawl4AiPortIdle {{ return $false }}
try {{ [void](Start-LocalMarketingAgent -ConfigPath {_ps_string(tmp_path / "config.json")}); throw "occupied port started" }}
catch {{ if ($_.Exception.Message -notmatch "already listening") {{ throw }} }}
Write-Output "ok"
"""
    )


def test_health_wait_accepts_idle_profile_and_rejects_early_exit_or_timeout() -> None:
    _run_powershell(
        f"""
. {_ps_string(ROOT / "deploy/scripts/local_crawl4ai_common.ps1")}
function Send-LocalCrawl4AiJsonLine {{
    [pscustomobject]@{{ ok = $true; status = "idle"; authenticated_profile = $true; visible_browser = $true; browser_channel = "msedge" }}
}}
$idle = [pscustomobject]@{{ HasExited = $false }}
$response = Wait-LocalCrawl4AiHealth -WorkerProcess $idle -Port 18791 -Attempts 1 -DelayMilliseconds 0
if ($response.status -ne "idle") {{ throw "idle health was not accepted" }}
$exited = [pscustomobject]@{{ HasExited = $true }}
try {{ [void](Wait-LocalCrawl4AiHealth -WorkerProcess $exited -Port 18791 -Attempts 1 -DelayMilliseconds 0); throw "early exit passed" }}
catch {{ if ($_.Exception.Message -notmatch "exited before") {{ throw }} }}
function Send-LocalCrawl4AiJsonLine {{ throw "not ready" }}
function Start-Sleep {{ param($Milliseconds) }}
try {{ [void](Wait-LocalCrawl4AiHealth -WorkerProcess $idle -Port 18791 -Attempts 2 -DelayMilliseconds 0); throw "timeout passed" }}
catch {{ if ($_.Exception.Message -notmatch "did not become healthy") {{ throw }} }}
Write-Output "ok"
"""
    )


def test_profile_idle_check_parses_the_dedicated_edge_argument(tmp_path: Path) -> None:
    profile = tmp_path / "profile with spaces"
    profile.mkdir()
    _run_powershell(
        f"""
. {_ps_string(ROOT / "deploy/scripts/local_crawl4ai_common.ps1")}
function Get-CimInstance {{
    @(
        [pscustomobject]@{{ CommandLine = "" }},
        [pscustomobject]@{{ CommandLine = '"C:\\Windows\\System32\\example.exe"' }},
        [pscustomobject]@{{ CommandLine = 'msedge.exe --new-window' }},
        [pscustomobject]@{{ CommandLine = 'msedge.exe --user-data-dir="{tmp_path / "other profile"}"' }}
    )
}}
if (-not (Test-DedicatedCrawl4AiProfileIdle -ProfilePath {_ps_string(profile)})) {{
    throw "unrelated process was treated as active"
}}
function Get-CimInstance {{
    [pscustomobject]@{{ CommandLine = 'msedge.exe --user-data-dir="{profile}" --new-window' }}
}}
if (Test-DedicatedCrawl4AiProfileIdle -ProfilePath {_ps_string(profile)}) {{
    throw "active dedicated profile was not detected"
}}
"""
    )


def test_json_health_client_writes_one_utf8_json_line_without_bom() -> None:
    received: list[bytes] = []
    ready = threading.Event()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def respond() -> None:
            ready.set()
            connection, _ = listener.accept()
            with connection:
                received.append(connection.recv(1024))
                connection.sendall(b'{"ok":true}\n')

        thread = threading.Thread(target=respond, daemon=True)
        thread.start()
        assert ready.wait(timeout=1)
        _run_powershell(
            f"""
. {_ps_string(ROOT / "deploy/scripts/local_crawl4ai_common.ps1")}
$response = Send-LocalCrawl4AiJsonLine -Request @{{ op = "health" }} -Port {port}
if ($response.ok -ne $true) {{ throw "health response was not decoded" }}
"""
        )
        thread.join(timeout=2)

    assert received == [b'{"op":"health"}\n']


def test_worker_logs_and_launcher_cleanup_restore_environment(tmp_path: Path) -> None:
    config, _, paths = _paths_block(tmp_path, marker=True)
    child_logs = tmp_path / "child logs"
    child_logs.mkdir()
    result = _run_powershell(
        f"""
. {_ps_string(ROOT / "deploy/scripts/start_local_marketing_agent.ps1")}
[Environment]::SetEnvironmentVariable("LOCAL_TEST_SECRET", "not-for-worker", "Process")
$child = Start-LocalCrawl4AiWorker -PythonExecutable {_ps_string(PWSH)} -Port 18791 -AllowedHosts @("facebook.com", "tiktok.com") -ProfilePath {_ps_string(tmp_path / "profile with spaces")} -LogDirectory {_ps_string(child_logs)} -WorkerArguments @("-NoLogo", "-NoProfile", "-Command", "[Console]::Out.WriteLine('child-out'); [Console]::Error.WriteLine('child-err'); if ($env:LOCAL_TEST_SECRET) {{ exit 9 }}")
[void]$child.WaitForExit(5000)
Stop-OwnedCrawl4AiWorker -WorkerProcess $child
if ((Get-Content -Raw {_ps_string(child_logs / "crawl4ai-worker.stdout.log")}) -notmatch "child-out") {{ throw "stdout was not redirected" }}
if ((Get-Content -Raw {_ps_string(child_logs / "crawl4ai-worker.stderr.log")}) -notmatch "child-err") {{ throw "stderr was not redirected" }}
{paths}
function Start-LocalCrawl4AiWorker {{ [pscustomobject]@{{ Id = 77; HasExited = $false }} }}
function Wait-LocalCrawl4AiHealth {{ [pscustomobject]@{{ ok = $true; status = "idle"; authenticated_profile = $true; visible_browser = $true; browser_channel = "msedge" }} }}
$script:resetCount = 0
$script:stopCount = 0
function Send-LocalCrawl4AiReset {{ $script:resetCount++ }}
function Stop-OwnedCrawl4AiWorker {{ $script:stopCount++ }}
function Invoke-LocalMarketingAgent {{ return 23 }}
[Environment]::SetEnvironmentVariable("CRAWL4AI_WORKER_ENABLED", "before", "Process")
Remove-Item -LiteralPath "Env:CRAWL4AI_AUTH_PROFILE_ENABLED" -ErrorAction SilentlyContinue
$result = Start-LocalMarketingAgent -ConfigPath {_ps_string(config)}
if ($result -ne 23) {{ throw "agent exit code was not preserved" }}
if ($script:resetCount -ne 1 -or $script:stopCount -ne 1) {{ throw "owned worker cleanup was skipped" }}
if ([Environment]::GetEnvironmentVariable("CRAWL4AI_WORKER_ENABLED", "Process") -ne "before") {{ throw "worker environment was not restored" }}
if ($null -ne [Environment]::GetEnvironmentVariable("CRAWL4AI_AUTH_PROFILE_ENABLED", "Process")) {{ throw "unset environment was not restored" }}
$script:resetCount = 0
$script:stopCount = 0
function Start-LocalCrawl4AiWorker {{ [pscustomobject]@{{ Id = 77; HasExited = $true }} }}
$result = Start-LocalMarketingAgent -ConfigPath {_ps_string(config)}
if ($result -ne 23) {{ throw "exited worker changed agent exit code" }}
if ($script:resetCount -ne 0 -or $script:stopCount -ne 1) {{ throw "exited worker was reset or not cleaned up" }}
function Start-LocalCrawl4AiWorker {{ [pscustomobject]@{{ Id = 77; HasExited = $false }} }}
function Send-LocalCrawl4AiReset {{ throw "injected cleanup failure" }}
$script:resetCount = 0
$script:stopCount = 0
[Environment]::SetEnvironmentVariable("CRAWL4AI_WORKER_ENABLED", "before-cleanup-failure", "Process")
Remove-Item -LiteralPath "Env:CRAWL4AI_AUTH_PROFILE_ENABLED" -ErrorAction SilentlyContinue
try {{
    [void](Start-LocalMarketingAgent -ConfigPath {_ps_string(config)})
    throw "cleanup failure was not surfaced"
}}
catch {{ if ($_.Exception.Message -notmatch "injected cleanup failure") {{ throw }} }}
if ($script:stopCount -ne 1) {{ throw "owned worker was not stopped after reset failure" }}
if ([Environment]::GetEnvironmentVariable("CRAWL4AI_WORKER_ENABLED", "Process") -ne "before-cleanup-failure") {{ throw "environment did not restore after cleanup failure" }}
if ($null -ne [Environment]::GetEnvironmentVariable("CRAWL4AI_AUTH_PROFILE_ENABLED", "Process")) {{ throw "unset environment did not restore after cleanup failure" }}
Write-Output "ok"
"""
    )
    assert result.returncode == 0


def test_profile_setup_cancellation_active_profile_and_ready_marker(tmp_path: Path) -> None:
    profile = tmp_path / "dedicated profile"
    logs = tmp_path / "logs"
    profile.mkdir()
    logs.mkdir()
    marker = profile / ".nanobot-profile-ready"
    marker.touch()
    path_block = f"""
$testProfile = {_ps_string(profile)}
$testLogs = {_ps_string(logs)}
$testMarker = {_ps_string(marker)}
function Get-LocalCrawl4AiPaths {{ [pscustomobject]@{{ ProfilePath = $testProfile; LogDirectory = $testLogs; MarkerPath = $testMarker }} }}
function Find-LocalMicrosoftEdge {{ return "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe" }}
"""
    _run_powershell(
        f"""
. {_ps_string(ROOT / "deploy/scripts/setup_local_crawl4ai_profile.ps1")}
{path_block}
function Test-DedicatedCrawl4AiProfileIdle {{ return $true }}
function Start-Process {{ param($FilePath, $ArgumentList) $script:argumentLine = $ArgumentList; [pscustomobject]@{{ Id = 1 }} }}
function Read-Host {{ param($Prompt) return "cancel" }}
if (Initialize-LocalCrawl4AiProfile) {{ throw "cancelled setup returned ready" }}
if (Test-Path -LiteralPath $testMarker) {{ throw "cancelled setup wrote a marker" }}
if ($script:argumentLine -notmatch "--user-data-dir.*dedicated profile") {{ throw "profile argument was not quoted separately" }}
"""
    )
    _run_powershell(
        f"""
. {_ps_string(ROOT / "deploy/scripts/setup_local_crawl4ai_profile.ps1")}
{path_block}
function Test-DedicatedCrawl4AiProfileIdle {{ return $false }}
function Find-LocalMicrosoftEdge {{ throw "active profile should stop before Edge lookup" }}
try {{ [void](Initialize-LocalCrawl4AiProfile); throw "active profile was accepted" }}
catch {{ if ($_.Exception.Message -notmatch "already in use") {{ throw }} }}
Write-Output "ok"
"""
    )
    _run_powershell(
        f"""
. {_ps_string(ROOT / "deploy/scripts/setup_local_crawl4ai_profile.ps1")}
{path_block}
function Test-DedicatedCrawl4AiProfileIdle {{ return $true }}
function Start-Process {{ param($FilePath, $ArgumentList) [pscustomobject]@{{ Id = 1 }} }}
function Read-Host {{ param($Prompt) return "READY" }}
if (-not (Initialize-LocalCrawl4AiProfile)) {{ throw "ready setup returned false" }}
if (-not (Test-Path -LiteralPath $testMarker -PathType Leaf)) {{ throw "ready setup did not write a marker" }}
"""
    )
