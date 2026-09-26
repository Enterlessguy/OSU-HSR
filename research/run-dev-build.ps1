param(
    [switch]$Update,
    [string]$VerifyTrace
)

# OSU Human Simulator - dev build launcher
# Prompts for skill/effort, verifies the current checkout, rebuilds the
# research client + tools, updates the editable Python package, then starts
# the guarded offline auto-run session. The terminal remains open after every
# outcome so diagnostics are not lost.

$ErrorActionPreference = "Stop"
$scriptRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$root = (Resolve-Path -LiteralPath (Join-Path $scriptRoot "..")).Path
Set-Location -LiteralPath $root

function Invoke-GitChecked {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $output = & git @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE. $($output -join ' ')"
    }
    return $output
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $false)][object[]]$Arguments = @()
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

function Get-PlannerVersion {
    $plannerPath = Join-Path $root "research\human-sim\src\human_sim\planner.py"
    $source = Get-Content -LiteralPath $plannerPath -Raw
    if ($source -notmatch 'PLANNER_VERSION\s*=\s*"([^"]+)"') {
        throw "PLANNER_VERSION was not found in $plannerPath."
    }
    return $Matches[1]
}

function Get-CheckoutIdentity {
    $branch = (@(Invoke-GitChecked @("branch", "--show-current")) -join "").Trim()
    if ([string]::IsNullOrWhiteSpace($branch)) {
        $branch = "(detached HEAD)"
    }
    $commit = ([string](Invoke-GitChecked @("rev-parse", "HEAD"))).Trim()
    $status = @(Invoke-GitChecked @("status", "--porcelain=v1"))
    return [pscustomobject]@{
        Branch = $branch
        Commit = $commit
        Dirty = $status.Count -gt 0
        Status = $status
        PlannerVersion = Get-PlannerVersion
    }
}

function Assert-TraceIdentity {
    param([Parameter(Mandatory = $true)][string]$TracePath, [Parameter(Mandatory = $true)]$Identity)
    $resolvedTrace = (Resolve-Path -LiteralPath $TracePath).Path
    $manifestPath = "$resolvedTrace.manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Trace manifest is missing: $manifestPath"
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    foreach ($property in @("planner_version", "git_commit", "build_identity")) {
        if ([string]::IsNullOrWhiteSpace([string]$manifest.$property)) {
            throw "Trace manifest lacks ${property}: $manifestPath"
        }
    }
    $expectedBuildIdentity = "human-sim-python:$($Identity.PlannerVersion):$($Identity.Commit)"
    if ([string]$manifest.planner_version -ne $Identity.PlannerVersion -or [string]$manifest.git_commit -ne $Identity.Commit -or [string]$manifest.build_identity -ne $expectedBuildIdentity) {
        throw "Trace identity mismatch. expected planner=$($Identity.PlannerVersion), git=$($Identity.Commit); found planner=$($manifest.planner_version), git=$($manifest.git_commit)."
    }
    $file = [IO.File]::OpenRead($resolvedTrace)
    try {
        $gzip = [IO.Compression.GzipStream]::new($file, [IO.Compression.CompressionMode]::Decompress)
        try {
            $reader = [IO.StreamReader]::new($gzip)
            try {
                $header = $reader.ReadLine() | ConvertFrom-Json
            }
            finally { $reader.Dispose() }
        }
        finally { $gzip.Dispose() }
    }
    finally { $file.Dispose() }
    if ([string]$header.planner_version -ne $Identity.PlannerVersion -or [string]$header.git_commit -ne $Identity.Commit -or [string]$header.build_identity -ne $expectedBuildIdentity) {
        throw "Trace header identity mismatch: $resolvedTrace"
    }
    Write-Host "Verified trace identity: planner=$($Identity.PlannerVersion), git=$($Identity.Commit), trace=$resolvedTrace" -ForegroundColor Green
}

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

function Get-CategoryTable {
    # Midpoints chosen as the center of each band from the phase-1 benchmark
    # ("Skill categories vs real players", estimates based on the oii+ dataset
    # and community PP tiers). Rough real-world equivalents are informational
    # only; manual % entry remains available.
    @(
        @{ Id = 1; Name = "beginner";       Mid = 10;  Range = "0-14";   Real = "<~500 pp; first weeks-months; struggles on Hard" },
        @{ Id = 2; Name = "beginner+";      Mid = 20;  Range = "15-29";  Real = "~500-1000 pp; passes Hard, low acc on Insane" },
        @{ Id = 3; Name = "intermediate";   Mid = 35;  Range = "30-44";  Real = "~1k-2k pp; comfortable Hard/Insane" },
        @{ Id = 4; Name = "intermediate+";  Mid = 50;  Range = "45-59";  Real = "~2k-4k pp; plays 6-7* with moderate acc" },
        @{ Id = 5; Name = "expert";         Mid = 65;  Range = "60-74";  Real = "~4k-6k pp; solid on 7*" },
        @{ Id = 6; Name = "expert+";        Mid = 78;  Range = "75-87";  Real = "~6k-8k pp; high acc on 7-8*" },
        @{ Id = 7; Name = "competitive";    Mid = 90;  Range = "88-95";  Real = "~8k-10k+ pp; top ~1-2%; near-FC on 7-8*" },
        @{ Id = 8; Name = "superhuman";     Mid = 97;  Range = "96-99.9"; Real = "10k+ pp; top ~0.1%; 99%+ consistency" },
        @{ Id = 9; Name = "max";            Mid = 100; Range = "100";    Real = "machine-perfect baseline (calibration, not human)" }
    )
}

function Show-CategoryMenu {
    Write-Host ""
    Write-Host "Preset player categories (skill bands are estimates):" -ForegroundColor Cyan
    foreach ($c in Get-CategoryTable) {
        $real = $c.Real
        if ($real.Length -gt 44) { $real = $real.Substring(0, 41) + "..." }
        Write-Host ("  [{0}] {1,-15} skill {2,-8} {3}" -f $c.Id, $c.Name, $c.Range, $real)
    }
    Write-Host "  [M] Manual %  (pick any skill 0-100 yourself)" -ForegroundColor Yellow
    Write-Host "  [P] Perfect   (skill 100, machine baseline)" -ForegroundColor Yellow
}

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " OSU Human Simulator - research dev build" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# --- Skill selection ---
$skill = $null
while ($null -eq $skill) {
    Show-CategoryMenu
    $input = Read-Host "Choose category [1-9], M for manual %, or P for perfect"
    $modeHint = ""
    if ($input -match '^\s*m\s*$') {
        $modeHint = "manual"
    }
    elseif ($input -match '^\s*p\s*$') {
        $skill = 100.0
        $modeHint = "perfect"
        break
    }
    elseif ($input -match '^\s*([1-9])\s*$') {
        $category = Get-CategoryTable | Where-Object { $_.Id -eq [int]$Matches[1] }
        if ($null -ne $category) {
            Write-Host ""
            Write-Host ("Preset: {0} -> skill {1} (band {2})." -f $category.Name, $category.Mid, $category.Range) -ForegroundColor Green
            $accept = Read-Host "Use this preset (Enter), or type a manual skill %"
            if ([string]::IsNullOrWhiteSpace($accept)) {
                $skill = [double]$category.Mid
                $modeHint = "preset $($category.Name)"
                break
            }
            if ($accept -match '^\s*(\d{1,3}(?:\.\d+)?)\s*$') {
                $value = [double]$Matches[1]
                if ($value -ge 0 -and $value -le 100) { $skill = $value; $modeHint = "manual" }
                else { Write-Host "Enter a value between 0 and 100." -ForegroundColor Yellow; continue }
                break
            }
            Write-Host "Not a number; staying in the category menu." -ForegroundColor Yellow
            continue
        }
    }
    elseif ($input -match '^\s*(\d{1,3}(?:\.\d+)?)\s*$') {
        $value = [double]$Matches[1]
        if ($value -ge 0 -and $value -le 100) { $skill = $value; $modeHint = "manual" }
        else { Write-Host "Enter a value between 0 and 100." -ForegroundColor Yellow }
    }
    else { Write-Host "Enter 1-9, M, P, or a manual number 0-100." -ForegroundColor Yellow }
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
$exitCode = 0

try {
    $identity = Get-CheckoutIdentity
    Write-Host ""
    Write-Host "Checkout: branch=$($identity.Branch) commit=$($identity.Commit) dirty=$($identity.Dirty) planner=$($identity.PlannerVersion)" -ForegroundColor Cyan

    if ($Update) {
        if ($identity.Dirty) {
            Write-Host "Update skipped: checkout has local changes. Building and launching the local checkout without overwriting them." -ForegroundColor Yellow
        }
        else {
            $upstream = ([string](Invoke-GitChecked @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"))).Trim()
            if ([string]::IsNullOrWhiteSpace($upstream)) {
                throw "-Update requires a configured upstream branch; none is available."
            }
            Write-Host "Updating only by fetch + fast-forward from $upstream..." -ForegroundColor Yellow
            Invoke-GitChecked @("fetch", "--prune") | Out-Host
            Invoke-GitChecked @("merge", "--ff-only", $upstream) | Out-Host
            $identity = Get-CheckoutIdentity
            if ($identity.Dirty) { throw "Checkout became dirty after fast-forward; refusing to launch." }
            Write-Host "Updated checkout: branch=$($identity.Branch) commit=$($identity.Commit) planner=$($identity.PlannerVersion)" -ForegroundColor Green
        }
    }
    elseif ($identity.Dirty) {
        Write-Host "Checkout is dirty; continuing without update because no -Update was requested." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Launch profile: $modeHint, mode=$mode skill=$skill effort=$effort" -ForegroundColor Cyan

    $env:DOTNET_CLI_HOME = Join-Path $root ".dotnet-home"
    $env:NUGET_PACKAGES = Join-Path $root ".nuget\packages"
    $env:APPDATA = Join-Path $root ".dotnet-home\AppData\Roaming"
    $env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"

    $dotnet = Join-Path $root ".dotnet\dotnet.exe"
    if (-not (Test-Path -LiteralPath $dotnet)) { $dotnet = (Get-Command dotnet -ErrorAction Stop).Source }
    $python = Join-Path $root "research\human-sim\.venv\Scripts\python.exe"
    $humanSim = Join-Path $root "research\human-sim\.venv\Scripts\human-sim.exe"
    $client = Join-Path $root "osu.Desktop\bin\Debug\net8.0\osu!.exe"
    $runnerProject = Join-Path $root "research\HumanSim.Runner\HumanSim.Runner.csproj"
    $runnerDll = Join-Path $root "research\HumanSim.Runner\bin\Debug\net8.0-windows\HumanSim.Runner.dll"
    $coherentModel = Join-Path $root "research\human-sim\models\experimental\seed101-math-residual-g100-v4.json"
    foreach ($required in @($dotnet, $python, $humanSim)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required local tool is missing: $required"
        }
    }

    Write-Host "=== [1/5] Building research client (osu!lazer + HSR) ===" -ForegroundColor Cyan
    Invoke-NativeChecked "powershell.exe" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $root "research\build-research.ps1"))

    Write-Host "=== [2/5] Building research tools ===" -ForegroundColor Cyan
    foreach ($project in @(
        (Join-Path $root "research\HumanSim.MapExporter\HumanSim.MapExporter.csproj"),
        (Join-Path $root "research\HumanSim.ReplayExtractor\HumanSim.ReplayExtractor.csproj"),
        $runnerProject
    )) {
        $buildArguments = @("build", $project, "--configfile", (Join-Path $root "NuGet.config"))
        if (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $project) "obj\project.assets.json") -PathType Leaf) {
            $buildArguments += "--no-restore"
        }
        Invoke-NativeChecked $dotnet $buildArguments
    }
    if (-not (Test-Path -LiteralPath $runnerDll -PathType Leaf)) { throw "Runner build output is missing: $runnerDll" }

    Write-Host "=== [3/5] Updating editable Python package ===" -ForegroundColor Cyan
    Push-Location (Join-Path $root "research\human-sim")
    try { Invoke-NativeChecked $python @("-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", ".[test]") }
    finally { Pop-Location }

    Write-Host "=== [4/5] Verifying runtime identity ===" -ForegroundColor Cyan
    $runtimeVersion = (& $python -c "from human_sim.planner import PLANNER_VERSION; print(PLANNER_VERSION)" | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $runtimeVersion -ne $identity.PlannerVersion) {
        throw "Python runtime planner mismatch: source=$($identity.PlannerVersion), runtime=$runtimeVersion"
    }
    $runnerIdentity = (& $dotnet $runnerDll "--print-identity" | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $runnerIdentity -ne "planner_version=$($identity.PlannerVersion)") {
        throw "Runner planner mismatch: expected planner_version=$($identity.PlannerVersion), found $runnerIdentity"
    }
    Write-Host "Verified source/Python/runner identity: planner=$($identity.PlannerVersion), git=$($identity.Commit)" -ForegroundColor Green
    if ($VerifyTrace) { Assert-TraceIdentity -TracePath $VerifyTrace -Identity $identity }
    else { Write-Host "No trace supplied for pre-launch identity verification; the runner will reject stale cache manifests." -ForegroundColor DarkYellow }

    if (-not (Test-Path -LiteralPath $client -PathType Leaf)) { throw "Built client is missing: $client" }
    Write-Host "=== [5/5] Launching guarded auto-run ===" -ForegroundColor Green
    if ($mode -eq "perfect") {
        & $humanSim auto-run $client --mode perfect --motion-mode perfect --execution-mode math-only --execution-blend 0
    }
    else {
        if (-not (Test-Path -LiteralPath $coherentModel -PathType Leaf)) {
            throw "Experimental coherent model is missing: $coherentModel"
        }
        $modelHash = (Get-FileHash -LiteralPath $coherentModel -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($modelHash -ne "3ba4d157124fa078b862062a38578561265bf03aaa2180a566440874b22bc7a2") {
            throw "Experimental coherent model hash mismatch: $modelHash"
        }
        Write-Host "Experimental gated hybrid: opened-validation circle component only; full OSI V2 unconfirmed." -ForegroundColor Yellow
        & $humanSim auto-run $client --mode profile --motion-mode profile --skill $skill --effort $effort `
            --execution-mode coherent --execution-model $coherentModel --execution-blend 1
    }
    $exitCode = $LASTEXITCODE
    Write-Host "Runner exited with code $exitCode." -ForegroundColor Yellow
}
catch {
    $exitCode = 1
    Write-Host "LAUNCH BLOCKED: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Write-Host ""
    Write-Host "Diagnostics are complete. This terminal stays open; press Enter to close." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
}

exit $exitCode
