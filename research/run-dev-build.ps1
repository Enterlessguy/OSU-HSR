# OSU Human Simulator - dev build launcher
# Prompts for skill/effort, shows an educated sweet-spot effort recommendation,
# rebuilds the research client + tools, updates the Python environment, then
# starts the guarded auto-run session. The window stays open after the runner
# exits so errors remain visible.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-RecommendedEffort {
    param([double]$Skill)
    if ($Skill -ge 100) { return 100 }
    $s = $Skill / 100.0
    # Sweet-spot effort keeps the aim-lapse probability near a typical human
    # rate (~0.2% per dense object): p = (1-s)^2 * (1-e) * 0.025.
    # Solving for e gives e = 1 - 0.08/(1-s)^2, clamped to [40, 100].
    $e = 1.0 - (0.08 / [Math]::Pow(1.0 - $s, 2))
    if ($e -gt 1.0) { $e = 1.0 }
    if ($e -lt 0.40) { $e = 0.40 }
    return [Math]::Round($e * 100.0)
}

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " OSU Human Simulator - research dev build" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# --- Skill selection ---
$skill = $null
while ($null -eq $skill) {
    $input = Read-Host "Skill level (0-100, or P for perfect)"
    if ($input -match '^\s*p\s*$') {
        $skill = 100.0
        break
    }
    if ($input -match '^\s*(\d{1,3}(?:\.\d+)?)\s*$') {
        $value = [double]$Matches[1]
        if ($value -ge 0 -and $value -le 100) { $skill = $value }
        else { Write-Host "Enter a value between 0 and 100." -ForegroundColor Yellow }
    }
    else { Write-Host "Enter a number 0-100, or P for perfect." -ForegroundColor Yellow }
}

# --- Effort selection with recommendation ---
if ($skill -ge 100) {
    $effort = 100.0
    Write-Host ""
    Write-Host "Perfect mode (skill 100): zero-error baseline, effort is ignored." -ForegroundColor Green
}
else {
    $recommended = Get-RecommendedEffort $skill
    Write-Host ""
    Write-Host "Recommended effort for skill $skill : $recommended" -ForegroundColor Green
    Write-Host "  (sweet spot: keeps aim-lapse rate near a typical human, clamped 40-100)" -ForegroundColor DarkGray
    $effort = $null
    while ($null -eq $effort) {
        $input = Read-Host "Effort level (0-100) [Enter = $recommended]"
        if ([string]::IsNullOrWhiteSpace($input)) {
            $effort = [double]$recommended
            break
        }
        if ($input -match '^\s*(\d{1,3}(?:\.\d+)?)\s*$') {
            $value = [double]$Matches[1]
            if ($value -ge 0 -and $value -le 100) { $effort = $value }
            else { Write-Host "Enter a value between 0 and 100." -ForegroundColor Yellow }
        }
        else { Write-Host "Enter a number 0-100." -ForegroundColor Yellow }
    }
}

$mode = if ($skill -ge 100) { "perfect" } else { "profile" }

Write-Host ""
Write-Host "Launch profile: mode=$mode skill=$skill effort=$effort" -ForegroundColor Cyan

$env:DOTNET_CLI_HOME = Join-Path $root ".dotnet-home"
$env:NUGET_PACKAGES = Join-Path $root ".nuget\packages"
$env:APPDATA = Join-Path $root ".dotnet-home\AppData\Roaming"
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"

$dotnet = Join-Path $root ".dotnet\dotnet.exe"

Write-Host "=== [1/4] Building research client (osu!lazer + HSR) ===" -ForegroundColor Cyan
& (Join-Path $root "research\build-research.ps1")

Write-Host "=== [2/4] Building research tools ===" -ForegroundColor Cyan
& $dotnet build (Join-Path $root "research\HumanSim.MapExporter\HumanSim.MapExporter.csproj") --configfile (Join-Path $root "NuGet.Config")
& $dotnet build (Join-Path $root "research\HumanSim.ReplayExtractor\HumanSim.ReplayExtractor.csproj") --configfile (Join-Path $root "NuGet.Config")
& $dotnet build (Join-Path $root "research\HumanSim.Runner\HumanSim.Runner.csproj") --configfile (Join-Path $root "NuGet.Config")

Write-Host "=== [3/4] Updating Python environment ===" -ForegroundColor Cyan
Push-Location (Join-Path $root "research\human-sim")
& ".\.venv\Scripts\python.exe" -m pip install -e ".[test]"
Pop-Location

Write-Host "=== [4/4] Launching auto-run ===" -ForegroundColor Green
if ($mode -eq "perfect") {
    & (Join-Path $root "research\human-sim\.venv\Scripts\human-sim.exe") auto-run (Join-Path $root "osu.Desktop\bin\Debug\net8.0\osu!.exe") --mode perfect
}
else {
    & (Join-Path $root "research\human-sim\.venv\Scripts\human-sim.exe") auto-run (Join-Path $root "osu.Desktop\bin\Debug\net8.0\osu!.exe") --mode profile --skill $skill --effort $effort
}

Write-Host ""
Write-Host "Runner exited with code $LASTEXITCODE. This window stays open; press Enter to close." -ForegroundColor Yellow
Read-Host "Press Enter to close"
