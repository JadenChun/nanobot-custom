[CmdletBinding()]
param(
    [string]$PythonExecutable = "python",
    [string]$ConfigPath = (Join-Path ([Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)) ".nanobot\config.json"),
    [ValidateRange(1, 65535)][int]$Port = 18791,
    [string[]]$AllowedHosts = @("facebook.com", "tiktok.com")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "PowerShell 7 or later is required for the local Crawl4AI launcher."
}

. (Join-Path $PSScriptRoot "local_crawl4ai_common.ps1")

function Assert-LocalMarketingPython {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$PythonExecutable)

    if ($null -eq (Get-Command -Name $PythonExecutable -CommandType Application -ErrorAction SilentlyContinue)) {
        throw "Python executable was not found: $PythonExecutable"
    }
}

function Get-LocalMarketingConfig {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$ConfigPath)

    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw "Nanobot config was not found: $ConfigPath"
    }
    try {
        return Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    }
    catch {
        throw "Nanobot config is not valid JSON: $ConfigPath"
    }
}

function Assert-LocalCrawlerEnabled {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Config)

    $agents = $Config.PSObject.Properties["agents"].Value
    $crawler = if ($null -ne $agents) { $agents.PSObject.Properties["crawler"].Value } else { $null }
    $enabled = if ($null -ne $crawler) { $crawler.PSObject.Properties["enabled"].Value } else { $null }
    if ($enabled -ne $true) {
        throw "agents.crawler.enabled must be true before starting the local Crawl4AI worker."
    }
}

function Get-LocalCrawl4AiWorkerEnvironment {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][int]$Port,
        [Parameter(Mandatory)][string[]]$AllowedHosts,
        [Parameter(Mandatory)][string]$ProfilePath
    )

    $workerEnvironment = @{}
    foreach ($name in @(
        "APPDATA", "COMSPEC", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT",
        "PROGRAMDATA", "PYTHONHOME", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "WINDIR"
    )) {
        $value = [Environment]::GetEnvironmentVariable($name, "Process")
        if ($null -ne $value) {
            $workerEnvironment[$name] = $value
        }
    }
    $workerEnvironment["CRAWL4AI_WORKER_ENABLED"] = "true"
    $workerEnvironment["CRAWL4AI_AUTH_PROFILE_ENABLED"] = "true"
    $workerEnvironment["CRAWL4AI_WORKER_TCP_HOST"] = "127.0.0.1"
    $workerEnvironment["CRAWL4AI_WORKER_TCP_PORT"] = $Port.ToString()
    $workerEnvironment["CRAWL4AI_ALLOWED_HOSTS"] = $AllowedHosts -join ","
    $workerEnvironment["CRAWL4AI_USER_DATA_DIR"] = $ProfilePath
    return $workerEnvironment
}

function Start-LocalCrawl4AiWorker {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$PythonExecutable,
        [Parameter(Mandatory)][int]$Port,
        [Parameter(Mandatory)][string[]]$AllowedHosts,
        [Parameter(Mandatory)][string]$ProfilePath,
        [Parameter(Mandatory)][string]$LogDirectory,
        [string[]]$WorkerArguments
    )

    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $stdoutPath = Join-Path $LogDirectory "crawl4ai-worker.stdout.log"
    $stderrPath = Join-Path $LogDirectory "crawl4ai-worker.stderr.log"
    New-Item -ItemType File -Path $stdoutPath -Force | Out-Null
    New-Item -ItemType File -Path $stderrPath -Force | Out-Null

    if ($null -eq $WorkerArguments) {
        $browserChannel = "msedge"
        $WorkerArguments = @(
            "-m", "nanobot.crawler.worker",
            "--tcp-host", "127.0.0.1",
            "--tcp-port", $Port.ToString(),
            "--allowed-hosts", ($AllowedHosts -join ","),
            "--user-data-dir", $ProfilePath,
            "--headed",
            "--browser-channel", $browserChannel
        )
    }
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $PythonExecutable
    $startInfo.WorkingDirectory = (Get-Location).Path
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.Environment.Clear()
    foreach ($argument in $WorkerArguments) {
        [void]$startInfo.ArgumentList.Add($argument)
    }
    foreach ($entry in (Get-LocalCrawl4AiWorkerEnvironment -Port $Port -AllowedHosts $AllowedHosts -ProfilePath $ProfilePath).GetEnumerator()) {
        $startInfo.Environment[$entry.Key] = $entry.Value
    }

    $workerProcess = [Diagnostics.Process]::new()
    $workerProcess.StartInfo = $startInfo
    if (-not $workerProcess.Start()) {
        throw "Could not start the Crawl4AI worker."
    }
    $stdoutStream = [IO.File]::Open($stdoutPath, [IO.FileMode]::Append, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $stderrStream = [IO.File]::Open($stderrPath, [IO.FileMode]::Append, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try {
        $stdoutTask = $workerProcess.StandardOutput.BaseStream.CopyToAsync($stdoutStream)
        $stderrTask = $workerProcess.StandardError.BaseStream.CopyToAsync($stderrStream)
        $workerProcess | Add-Member -NotePropertyName LocalCrawl4AiStdoutTask -NotePropertyValue $stdoutTask
        $workerProcess | Add-Member -NotePropertyName LocalCrawl4AiStderrTask -NotePropertyValue $stderrTask
        $workerProcess | Add-Member -NotePropertyName LocalCrawl4AiStdoutStream -NotePropertyValue $stdoutStream
        $workerProcess | Add-Member -NotePropertyName LocalCrawl4AiStderrStream -NotePropertyValue $stderrStream
    }
    catch {
        $stdoutStream.Dispose()
        $stderrStream.Dispose()
        if (-not $workerProcess.HasExited) {
            Stop-Process -InputObject $workerProcess -ErrorAction SilentlyContinue
        }
        throw
    }
    return $workerProcess
}

function Invoke-LocalMarketingAgent {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$PythonExecutable,
        [Parameter(Mandatory)][string]$ConfigPath,
        [Parameter(Mandatory)][string]$LogPath
    )

    & $PythonExecutable -m nanobot agent --config $ConfigPath --session "egocat-marketing" --no-logs --log-file $LogPath | Out-Host
    return [int]$LASTEXITCODE
}

function Start-LocalMarketingAgent {
    [CmdletBinding()]
    param(
        [string]$PythonExecutable = "python",
        [string]$ConfigPath = (Join-Path ([Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)) ".nanobot\config.json"),
        [ValidateRange(1, 65535)][int]$Port = 18791,
        [string[]]$AllowedHosts = @("facebook.com", "tiktok.com")
    )

    if ($AllowedHosts.Count -eq 0 -or @($AllowedHosts | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
        throw "AllowedHosts must contain explicit non-empty hosts."
    }
    Assert-LocalMarketingPython -PythonExecutable $PythonExecutable
    $config = Get-LocalMarketingConfig -ConfigPath $ConfigPath
    Assert-LocalCrawlerEnabled -Config $config

    $paths = Get-LocalCrawl4AiPaths
    if (-not (Test-Path -LiteralPath $paths.ProfilePath -PathType Container)) {
        throw "Dedicated Crawl4AI Edge profile is missing: $($paths.ProfilePath)"
    }
    if (-not (Test-Path -LiteralPath $paths.MarkerPath -PathType Leaf)) {
        throw "Dedicated Crawl4AI Edge profile is not ready. Run setup_local_crawl4ai_profile.ps1 first."
    }
    if (-not (Test-DedicatedCrawl4AiProfileIdle -ProfilePath $paths.ProfilePath)) {
        throw "The dedicated Crawl4AI Edge profile is already in use. Close every window using it before starting."
    }
    if (-not (Test-LocalCrawl4AiPortIdle -Port $Port)) {
        throw "TCP port $Port is already listening; the launcher will not reuse an unknown worker."
    }

    New-Item -ItemType Directory -Path $paths.LogDirectory -Force | Out-Null
    $agentLogPath = Join-Path $paths.LogDirectory "nanobot-agent.log"
    $environmentNames = @(
        "CRAWL4AI_WORKER_ENABLED", "CRAWL4AI_AUTH_PROFILE_ENABLED",
        "CRAWL4AI_WORKER_TCP_HOST", "CRAWL4AI_WORKER_TCP_PORT"
    )
    $previousEnvironment = @{}
    foreach ($name in $environmentNames) {
        $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    }

    $workerProcess = $null
    try {
        $env:CRAWL4AI_WORKER_ENABLED = "true"
        $env:CRAWL4AI_AUTH_PROFILE_ENABLED = "true"
        $env:CRAWL4AI_WORKER_TCP_HOST = "127.0.0.1"
        $env:CRAWL4AI_WORKER_TCP_PORT = $Port.ToString()

        $workerProcess = Start-LocalCrawl4AiWorker -PythonExecutable $PythonExecutable -Port $Port -AllowedHosts $AllowedHosts -ProfilePath $paths.ProfilePath -LogDirectory $paths.LogDirectory
        [void](Wait-LocalCrawl4AiHealth -WorkerProcess $workerProcess -Port $Port)
        return Invoke-LocalMarketingAgent -PythonExecutable $PythonExecutable -ConfigPath $ConfigPath -LogPath $agentLogPath
    }
    finally {
        try {
            if ($null -ne $workerProcess) {
                try {
                    if (-not $workerProcess.HasExited) {
                        Send-LocalCrawl4AiReset -Port $Port
                    }
                }
                finally {
                    Stop-OwnedCrawl4AiWorker -WorkerProcess $workerProcess
                }
            }
        }
        finally {
            foreach ($name in $environmentNames) {
                if ($null -eq $previousEnvironment[$name]) {
                    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
                }
                else {
                    [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], "Process")
                }
            }
        }
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    exit (Start-LocalMarketingAgent @PSBoundParameters)
}
