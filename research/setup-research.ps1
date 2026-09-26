param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $root "research\human-sim"
$environmentPython = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $Python -m venv (Join-Path $project ".venv")
    if ($LASTEXITCODE) { throw "Python 3.12 or newer is required." }
}
& $environmentPython -m pip install --upgrade "pip>=26.2" setuptools
if ($LASTEXITCODE) { throw "Python tooling installation failed." }
& $environmentPython -m pip install -r (Join-Path $project "requirements-release.txt")
if ($LASTEXITCODE) { throw "Pinned dependency installation failed." }
& $environmentPython -m pip install --no-deps --no-build-isolation -e $project
if ($LASTEXITCODE) { throw "Editable HSR installation failed." }
Write-Host "Intelligence Database HSR environment ready. Launch research\run-dev-build.ps1."
