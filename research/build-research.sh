#!/usr/bin/env bash
set -euo pipefail

configuration="${1:-Release}"
case "$configuration" in
    Debug|Release) ;;
    *) printf 'Usage: %s [Debug|Release]\n' "$0" >&2; exit 2 ;;
esac

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DOTNET_CLI_HOME="$root/.dotnet-home"
export NUGET_PACKAGES="$root/.nuget/packages"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
dotnet_cmd="${DOTNET:-dotnet}"
command -v "$dotnet_cmd" >/dev/null 2>&1 || { printf 'Install the .NET 8 SDK and make dotnet available on PATH.\n' >&2; exit 1; }

build_project() {
    local project="$1"
    local assets="$(dirname -- "$project")/obj/project.assets.json"
    local args=(build "$project" --configuration "$configuration" -p:HumanSimResearchBuild=true)
    if [[ -f "$assets" ]]; then
        args+=(--no-restore)
    else
        args+=(--configfile "$root/NuGet.Config")
    fi
    "$dotnet_cmd" "${args[@]}"
}

build_project "$root/osu.Desktop/osu.Desktop.csproj"
build_project "$root/research/HumanSim.MapExporter/HumanSim.MapExporter.csproj"
build_project "$root/research/HumanSim.ReplayExtractor/HumanSim.ReplayExtractor.csproj"
build_project "$root/research/HumanSim.Runner/HumanSim.Runner.csproj"
