#!/usr/bin/env python3
"""Stage license metadata and texts for NuGet runtime packages in build outputs."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET


def collect(destination: Path, nuget_root: Path, output_roots: list[Path]) -> None:
    referenced: set[tuple[str, str]] = set()
    for output_root in output_roots:
        deps_files = sorted(output_root.glob("*.deps.json"))
        if not deps_files:
            raise FileNotFoundError(f"No dependency manifest in {output_root}")
        for deps_file in deps_files:
            data = json.loads(deps_file.read_text(encoding="utf-8"))
            for package_key, metadata in data.get("libraries", {}).items():
                if metadata.get("type") != "package":
                    continue
                package_id, separator, version = package_key.rpartition("/")
                if not separator or not package_id or not version:
                    raise ValueError(f"Malformed package identity {package_key!r} in {deps_file}")
                referenced.add((package_id, version))

    destination.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str, str, str]] = []
    for package_id, version in sorted(referenced, key=lambda item: (item[0].casefold(), item[1])):
        package_root = nuget_root / package_id.lower() / version.lower()
        nuspecs = sorted(package_root.glob("*.nuspec"))
        if not package_root.is_dir() or not nuspecs:
            raise FileNotFoundError(f"NuGet package metadata missing for {package_id} {version}: {package_root}")

        root = ET.parse(nuspecs[0]).getroot()
        metadata = next((element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "metadata"), None)
        if metadata is None:
            raise ValueError(f"NuGet nuspec has no metadata: {nuspecs[0]}")
        values = {
            element.tag.rsplit("}", 1)[-1]: element
            for element in metadata
        }
        license_element = values.get("license")
        license_text = (license_element.text or "").strip() if license_element is not None else ""
        license_type = license_element.attrib.get("type", "unknown") if license_element is not None else "unknown"
        license_url = (values.get("licenseUrl").text or "").strip() if values.get("licenseUrl") is not None else ""
        project_url = (values.get("projectUrl").text or "").strip() if values.get("projectUrl") is not None else ""

        safe_id = re.sub(r"[^A-Za-z0-9._+-]", "_", package_id)
        safe_version = re.sub(r"[^A-Za-z0-9._+-]", "_", version)
        package_destination = destination / f"{safe_id}-{safe_version}"
        copied: list[str] = []
        for license_file in package_root.rglob("*"):
            if not license_file.is_file():
                continue
            if not re.search(r"(licen[cs]e|copying|notice)", license_file.name, re.IGNORECASE):
                continue
            relative = license_file.relative_to(package_root)
            target = package_destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(license_file, target)
            copied.append(relative.as_posix())

        detail = f"{license_type}: {license_text}" if license_text else "not declared in nuspec"
        if license_url:
            detail += f" ({license_url})"
        if copied:
            detail += "; included files: " + ", ".join(copied)
        entries.append((package_id, version, detail, project_url))

    lines = [
        "# Third-party NuGet runtime dependency license inventory",
        "",
        "Generated during package(). This inventory records package metadata; it is not a legal determination.",
        "",
    ]
    for package_id, version, detail, project_url in entries:
        link = f"; project: {project_url}" if project_url else ""
        lines.append(f"- `{package_id} {version}` — {detail}{link}")
    (destination / "THIRD-PARTY-NUGET-LICENSES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("usage: collect-nuget-licenses.py DESTINATION NUGET_ROOT OUTPUT_DIR...", file=sys.stderr)
        return 2
    destination = Path(argv[1])
    nuget_root = Path(argv[2])
    collect(destination, nuget_root, [Path(value) for value in argv[3:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
