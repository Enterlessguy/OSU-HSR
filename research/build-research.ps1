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

$clientProject = Join-Path $root "osu.Desktop\osu.Desktop.csproj"
$buildArguments = @("build", $clientProject, "--configuration", $Configuration,
    "--configfile", (Join-Path $root "NuGet.Config"), "-p:HumanSimResearchBuild=true")
if (Test-Path -LiteralPath (Join-Path $root "osu.Desktop\obj\project.assets.json") -PathType Leaf) {
    $buildArguments += "--no-restore"
}
$dotnet = Join-Path $root ".dotnet\dotnet.exe"
if (-not (Test-Path -LiteralPath $dotnet)) { $dotnet = (Get-Command dotnet -ErrorAction Stop).Source }
& $dotnet @buildArguments

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
