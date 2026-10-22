[CmdletBinding()]
param(
    [string[]]$LoginUrls = @("https://www.facebook.com/", "https://www.tiktok.com/")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "local_crawl4ai_common.ps1")

function Find-LocalMicrosoftEdge {
    [CmdletBinding()]
    param()

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($env:ProgramFiles)) {
        $candidates += Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"
    }
    if (-not [string]::IsNullOrWhiteSpace(${env:ProgramFiles(x86)})) {
        $candidates += Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"
    }
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $candidates += Join-Path $env:LOCALAPPDATA "Microsoft\Edge\Application\msedge.exe"
    }
    foreach ($chrome in $candidates) {
        if (Test-Path -LiteralPath $chrome -PathType Leaf) {
            return $chrome
        }
    }
    throw "Microsoft Edge was not found under Program Files or LOCALAPPDATA."
}

function Initialize-LocalCrawl4AiProfile {
    [CmdletBinding()]
    param([string[]]$LoginUrls = @("https://www.facebook.com/", "https://www.tiktok.com/"))

    if ($LoginUrls.Count -eq 0 -or @($LoginUrls | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
        throw "LoginUrls must contain explicit non-empty URLs."
    }
    $paths = Get-LocalCrawl4AiPaths
    if (-not (Test-DedicatedCrawl4AiProfileIdle -ProfilePath $paths.ProfilePath)) {
        throw "The dedicated Crawl4AI Edge profile is already in use. Close every window using it before setup."
    }
    $chrome = Find-LocalMicrosoftEdge

    # .nanobot-profile-ready is removed until the operator explicitly confirms closure.
    New-Item -ItemType Directory -Path $paths.ProfilePath -Force | Out-Null
    Remove-Item -LiteralPath $paths.MarkerPath -Force -ErrorAction SilentlyContinue
    $browserArguments = @("--user-data-dir", $paths.ProfilePath, "--no-first-run", "--new-window") + $LoginUrls
    Start-Process -FilePath $chrome -ArgumentList (Join-WindowsCommandLine $browserArguments) | Out-Null

    Write-Host "Log in only with the dedicated Crawl4AI Edge profile."
    $confirmation = Read-Host "After login, close every window using this profile, then type READY to mark it prepared"
    if ($confirmation -cne "READY") {
        Write-Host "Profile preparation cancelled. The readiness marker was not written."
        return $false
    }
    if (-not (Test-DedicatedCrawl4AiProfileIdle -ProfilePath $paths.ProfilePath)) {
        throw "The dedicated Crawl4AI Edge profile is still in use. Close every window before confirming readiness."
    }

    New-Item -ItemType File -Path $paths.MarkerPath -Force | Out-Null
    Write-Host "Dedicated profile marker created: $($paths.MarkerPath)"
    return $true
}

if ($MyInvocation.InvocationName -ne ".") {
    $result = Initialize-LocalCrawl4AiProfile @PSBoundParameters
    if (-not $result) {
        exit 1
    }
}
