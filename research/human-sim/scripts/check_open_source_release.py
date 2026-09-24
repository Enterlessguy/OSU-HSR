"""Fail closed on local HSR artifacts and machine paths before a public commit."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAX_SOURCE_BYTES = 5_000_000
LOCAL_PATH = re.compile(rb"[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\\\s]+", re.I)
PRIVATE_SUFFIXES = {".osr", ".npy", ".npz", ".joblib", ".zip", ".7z", ".pfx", ".pem", ".key"}
PRIVATE_NAMES = {"corpus.private.json", ".env", "credentials.json"}
SECRET_MARKERS = {
    "private key block": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})"),
    "OpenAI style secret": re.compile(rb"sk-[A-Za-z0-9]{32,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "Slack token": re.compile(rb"xox[baprs]-[A-Za-z0-9-]{24,}"),
    "literal credential assignment": re.compile(
        rb"(?:api[_-]?key|client[_-]?secret|password|access[_-]?token)\s*[:=]\s*['\"][^'\"\r\n]{12,}['\"]",
        re.I,
    ),
}


def candidate_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True, capture_output=True,
    )
    return sorted({Path(raw.decode("utf-8", errors="surrogateescape"))
                   for raw in result.stdout.split(b"\0") if raw})


def history_findings() -> tuple[int, list[tuple[Path, str]]]:
    """Scan local Git object history without printing matching secret bytes."""
    process = subprocess.Popen(
        ["git", "-C", str(ROOT), "cat-file", "--batch-all-objects", "--batch"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    findings: list[tuple[Path, str]] = []
    blobs = 0
    while header := process.stdout.readline():
        parts = header.strip().split()
        if len(parts) != 3:
            raise ValueError("Unexpected Git object header")
        object_id, kind, size_raw = parts
        remaining = int(size_raw)
        chunks = []
        while remaining:
            chunk = process.stdout.read(min(remaining, 1_048_576))
            if not chunk:
                raise ValueError("Truncated Git object stream")
            if kind == b"blob" and int(size_raw) <= MAX_SOURCE_BYTES:
                chunks.append(chunk)
            remaining -= len(chunk)
        if process.stdout.read(1) != b"\n":
            raise ValueError("Git object delimiter missing")
        if kind != b"blob":
            continue
        blobs += 1
        if int(size_raw) > MAX_SOURCE_BYTES:
            continue
        content = b"".join(chunks)
        for label, marker in SECRET_MARKERS.items():
            if marker.search(content):
                findings.append((Path(object_id.decode()), "history " + label))
    if process.wait() != 0:
        raise ValueError("Git history scan failed")
    return blobs, findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit nonzero when any release issue is found")
    parser.add_argument("--history", action="store_true", help="also scan local Git object history for secret signatures")
    args = parser.parse_args()
    findings = []
    paths = candidate_paths()
    for relative in paths:
        value = relative.as_posix().lower()
        source = ROOT / relative
        if not source.is_file():
            continue
        hsr_scope = value.startswith("research/") or value in {
            "readme.md", "release_readiness.md", ".gitignore", "licence",
        }
        if hsr_scope and value.startswith("research/human-sim/output/") and value not in {
            "research/human-sim/output/phase1-aim-benchmark.md",
            "research/human-sim/output/phase2-timing-analysis.md",
            "research/human-sim/output/analysis/phase2_analyze.py",
            "research/human-sim/output/analysis/statistical-regression.md",
        }:
            findings.append((relative, "generated research output is publishable"))
        if hsr_scope and ("/supervision/" in value or value.startswith("output/")):
            findings.append((relative, "local supervision or trace artifact is publishable"))
        if hsr_scope and (source.suffix.lower() in PRIVATE_SUFFIXES or source.name.lower() in PRIVATE_NAMES or source.name.lower().startswith(".env.")):
            findings.append((relative, "raw data or private material is publishable"))
        if hsr_scope and source.stat().st_size > MAX_SOURCE_BYTES:
            findings.append((relative, "file exceeds 5 MB source release limit"))
            continue
        if source.stat().st_size > MAX_SOURCE_BYTES:
            continue
        if source.suffix.lower() in {".py", ".ps1", ".cs", ".md", ".json", ".toml", ".yml", ".yaml", ".sh", ".config"}:
            content = source.read_bytes()
            if hsr_scope and LOCAL_PATH.search(content):
                findings.append((relative, "hardcoded Windows user directory"))
            for label, marker in SECRET_MARKERS.items():
                if marker.search(content):
                    findings.append((relative, label))
    for required in ("LICENCE", "README.md", "research/human-sim/OSI_REALISM_V2.md"):
        if not (ROOT / required).is_file():
            findings.append((Path(required), "required release document missing"))
    if args.history:
        blobs, old_findings = history_findings()
        findings.extend(old_findings)
        print(f"Scanned {blobs} local Git blobs")
    print(f"Inspected {len(paths)} tracked and unignored files; {len(findings)} issue(s)")
    for path, reason in findings:
        print(f"{path.as_posix()}: {reason}")
    if findings and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
