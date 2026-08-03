param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:DOTNET_CLI_HOME = Join-Path $root ".dotnet-home"
$env:NUGET_PACKAGES = Join-Path $root ".nuget\packages"
$env:APPDATA = Join-Path $root ".dotnet-home\AppData\Roaming"
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"

& (Join-Path $root ".dotnet\dotnet.exe") build (Join-Path $root "osu.Desktop\osu.Desktop.csproj") `
    --configuration $Configuration `
    --configfile (Join-Path $root "NuGet.Config") `
    -p:HumanSimResearchBuild=true

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
