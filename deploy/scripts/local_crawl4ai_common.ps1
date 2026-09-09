Set-StrictMode -Version Latest

function Get-LocalCrawl4AiPaths {
    [CmdletBinding()]
    param()

    $localAppData = $env:LOCALAPPDATA
    if ([string]::IsNullOrWhiteSpace($localAppData)) {
        $localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    }
    if ([string]::IsNullOrWhiteSpace($localAppData)) {
        throw "LOCALAPPDATA is not available."
    }

    $root = Join-Path $localAppData "nanobot\egocat-marketing"
    $profilePath = Join-Path $root "crawl4ai-edge-profile"
    [pscustomobject]@{
        RootPath = $root
        ProfilePath = $profilePath
        LogDirectory = Join-Path $root "logs"
        MarkerPath = Join-Path $profilePath ".nanobot-profile-ready"
    }
}

function Get-NormalizedLocalPath {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)

    [IO.Path]::GetFullPath($Path).TrimEnd([char[]]@("\", "/"))
}

function Split-WindowsCommandLine {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$CommandLine)

    $tokens = [Collections.Generic.List[string]]::new()
    $token = [Text.StringBuilder]::new()
    $insideQuotes = $false
    $backslashCount = 0
    foreach ($character in $CommandLine.ToCharArray()) {
        if ($character -eq [char]92) {
            $backslashCount++
            continue
        }
        if ($character -eq [char]34) {
            if ($backslashCount -gt 0) {
                [void]$token.Append([char]92, [int]($backslashCount / 2))
            }
            if ($backslashCount % 2 -eq 0) {
                $insideQuotes = -not $insideQuotes
            }
            else {
                [void]$token.Append([char]34)
            }
            $backslashCount = 0
            continue
        }
        if ($backslashCount -gt 0) {
            [void]$token.Append([char]92, $backslashCount)
            $backslashCount = 0
        }
        if ([char]::IsWhiteSpace($character) -and -not $insideQuotes) {
            if ($token.Length -gt 0) {
                [void]$tokens.Add($token.ToString())
                [void]$token.Clear()
            }
            continue
        }
        [void]$token.Append($character)
    }
    if ($backslashCount -gt 0) {
        [void]$token.Append([char]92, $backslashCount)
    }
    if ($token.Length -gt 0) {
        [void]$tokens.Add($token.ToString())
    }
    return $tokens.ToArray()
}

function Test-DedicatedCrawl4AiProfileIdle {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$ProfilePath)

    $targetPath = Get-NormalizedLocalPath $ProfilePath
    foreach ($process in @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)) {
        if ([string]::IsNullOrWhiteSpace($process.CommandLine)) {
            continue
        }

        $arguments = @(Split-WindowsCommandLine $process.CommandLine)
        for ($index = 0; $index -lt $arguments.Count; $index++) {
            $argument = $arguments[$index]
            $configuredPath = $null
            if ($argument -eq "--user-data-dir" -and $index + 1 -lt $arguments.Count) {
                $configuredPath = $arguments[$index + 1]
            }
            elseif ($argument.StartsWith("--user-data-dir=")) {
                $configuredPath = $argument.Substring("--user-data-dir=".Length)
            }

            if ($null -ne $configuredPath -and -not [string]::IsNullOrWhiteSpace($configuredPath)) {
                $candidatePath = Get-NormalizedLocalPath $configuredPath
                if ([string]::Equals($candidatePath, $targetPath, [StringComparison]::OrdinalIgnoreCase)) {
                    return $false
                }
            }
        }
    }
    return $true
}

function Test-LocalCrawl4AiPortIdle {
    [CmdletBinding()]
    param([Parameter(Mandatory)][int]$Port)

    return @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue).Count -eq 0
}

function Join-WindowsCommandLine {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$ArgumentList)

    $quoted = foreach ($argument in $ArgumentList) {
        if ($argument -match '[\s"]') {
            '"' + $argument.Replace('"', '\"') + '"'
        }
        elseif ($argument.Length -eq 0) {
            '""'
        }
        else {
            $argument
        }
    }
    return $quoted -join " "
}

function Send-LocalCrawl4AiJsonLine {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$Request,
        [Parameter(Mandatory)][int]$Port,
        [int]$TimeoutMilliseconds = 1000
    )

    $client = [Net.Sockets.TcpClient]::new()
    try {
        $connectTask = $client.ConnectAsync("127.0.0.1", $Port)
        if (-not $connectTask.Wait($TimeoutMilliseconds)) {
            throw "Timed out connecting to Crawl4AI worker on port $Port."
        }

        $stream = $client.GetStream()
        $stream.ReadTimeout = $TimeoutMilliseconds
        $stream.WriteTimeout = $TimeoutMilliseconds
        $utf8WithoutBom = [Text.UTF8Encoding]::new($false)
        $writer = [IO.StreamWriter]::new($stream, $utf8WithoutBom, 1024, $true)
        $reader = [IO.StreamReader]::new($stream, $utf8WithoutBom, $false, 1024, $true)
        try {
            $writer.NewLine = "`n"
            $writer.WriteLine(($Request | ConvertTo-Json -Compress))
            $writer.Flush()
            $line = $reader.ReadLine()
            if ([string]::IsNullOrWhiteSpace($line)) {
                throw "Crawl4AI worker returned an empty response."
            }
            return $line | ConvertFrom-Json
        }
        finally {
            $reader.Dispose()
            $writer.Dispose()
        }
    }
    finally {
        $client.Dispose()
    }
}

function Wait-LocalCrawl4AiHealth {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]$WorkerProcess,
        [Parameter(Mandatory)][int]$Port,
        [int]$Attempts = 20,
        [int]$DelayMilliseconds = 250
    )

    $lastError = $null
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        if ($WorkerProcess.HasExited) {
            throw "Crawl4AI worker exited before its health check succeeded."
        }
        try {
            $response = Send-LocalCrawl4AiJsonLine -Request @{ op = "health" } -Port $Port
            if (
                $response.ok -eq $true -and
                $response.status -eq "idle" -and
                $response.authenticated_profile -eq $true -and
                $response.visible_browser -eq $true -and
                $response.browser_channel -eq "msedge"
            ) {
                return $response
            }
            $lastError = "Crawl4AI worker health response did not confirm the dedicated idle Edge profile."
        }
        catch {
            $lastError = $_
        }
        if ($attempt -lt $Attempts) {
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }
    throw "Crawl4AI worker did not become healthy: $lastError"
}

function Send-LocalCrawl4AiReset {
    [CmdletBinding()]
    param([Parameter(Mandatory)][int]$Port)

    try {
        [void](Send-LocalCrawl4AiJsonLine -Request @{ op = "reset" } -Port $Port -TimeoutMilliseconds 1000)
    }
    catch {
        Write-Warning "Crawl4AI worker reset did not complete: $($_.Exception.Message)"
    }
}

function Stop-OwnedCrawl4AiWorker {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$WorkerProcess)

    if (-not $WorkerProcess.HasExited) {
        try {
            Stop-Process -InputObject $WorkerProcess -ErrorAction Stop
        }
        catch {
            Write-Warning "Could not stop the owned Crawl4AI worker: $($_.Exception.Message)"
        }
    }
    try {
        [void]$WorkerProcess.WaitForExit(5000)
    }
    catch {
        Write-Warning "Owned Crawl4AI worker did not exit within five seconds."
    }
    foreach ($taskName in @("LocalCrawl4AiStdoutTask", "LocalCrawl4AiStderrTask")) {
        $taskProperty = $WorkerProcess.PSObject.Properties[$taskName]
        if ($null -ne $taskProperty -and $null -ne $taskProperty.Value) {
            try {
                [void]$taskProperty.Value.Wait(2000)
            }
            catch {
                Write-Warning "Timed out copying owned Crawl4AI worker logs."
            }
        }
    }
    foreach ($streamName in @("LocalCrawl4AiStdoutStream", "LocalCrawl4AiStderrStream")) {
        $streamProperty = $WorkerProcess.PSObject.Properties[$streamName]
        if ($null -ne $streamProperty -and $null -ne $streamProperty.Value) {
            $streamProperty.Value.Dispose()
        }
    }
}
