param([string]$OutputDirectory)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root "output\release" }
$dirty = & git -C $root status --porcelain
if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Commit and review the source before packaging." }
$commit = (& git -C $root rev-parse HEAD).Trim()
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$archive = Join-Path $OutputDirectory "Intelligence-Database-HSR-local-v2-$($commit.Substring(0,12))-source.zip"
& git -C $root archive --format=zip "--prefix=Intelligence-Database-HSR/" "--output=$archive" HEAD
if ($LASTEXITCODE) { throw "Git source archive failed." }
$model = Join-Path $root "research\human-sim\models\experimental\seed101-math-residual-g100-v4.json"
$snapshot = Join-Path $root "research\human-sim\benchmarks\LOCAL_V2_20260926.json"
$hashes = [ordered]@{ commit = $commit; files = @() }
foreach ($artifact in @($archive, $model, $snapshot)) {
    $hashes.files += [ordered]@{ name = (Split-Path -Leaf $artifact); sha256 = (Get-FileHash -LiteralPath $artifact -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$manifest = Join-Path $OutputDirectory "SHA256SUMS.json"
$hashes | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifest -Encoding UTF8
Write-Host "Reviewed source package: $archive"
Write-Host "Checksums: $manifest"
