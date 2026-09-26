"""Bounded, provenance-neutral acquisition helpers for public replay archives.

This module is deliberately separate from the human-corpus loader.  A public
archive can be useful for discovery and exact-file pairing without being
admissible human training data.  The caller must still supply reviewed human
provenance, dated skill evidence, a stable player hash, and a tier label before
adding a row to ``training/corpus.json``.

The Kaggle helper uses the public download redirect, reads only HTTP byte
ranges, understands ZIP64 central-directory extra field 0x0001, and verifies
HTTP 206/Content-Range, decompressed size, and CRC32 for every extracted file.
It never downloads the complete archive unless a caller explicitly asks for
every entry (which this module does not expose).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import html
import io
import json
from pathlib import Path
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from typing import Any, Iterable, Iterator, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request
import zlib

from .network import open_https


ZIP_LOCAL = 0x04034B50
ZIP_CENTRAL = 0x02014B50
ZIP_EOCD = 0x06054B50
ZIP64_EOCD = 0x06064B50
ZIP64_LOCATOR = 0x07064B50
ZIP64_EXTRA = 0x0001
ZIP32_SENTINEL = 0xFFFFFFFF


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compression: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    local_header_offset: int


@dataclass(frozen=True)
class ArchiveLayout:
    archive_size: int
    entry_count: int
    central_offset: int
    central_size: int


AUTOMATION_MOD_MASK = 2048 | 128 | 8192  # Auto, Relax, Relax2.


def _read_exact(stream: io.BufferedIOBase, limit: int) -> bytes:
    if limit < 0:
        raise ValueError("read limit must be non-negative")
    chunks: list[bytes] = []
    remaining = limit
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) != limit:
        raise IOError(f"short ranged response: expected {limit} bytes, received {len(data)}")
    return data


def fetch_range(url: str, start: int, end: int, *, max_bytes: int) -> bytes:
    """Fetch one inclusive byte range and enforce a real 206 response."""
    if start < 0 or end < start:
        raise ValueError("invalid inclusive byte range")
    expected = end - start + 1
    if expected > max_bytes:
        raise ValueError(f"requested range {expected} exceeds max_bytes={max_bytes}")
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with open_https(request, timeout=90) as response:
        status = int(getattr(response, "status", response.getcode()))
        if status != 206:
            raise IOError(f"expected HTTP 206 for bytes={start}-{end}, got {status}")
        content_range = str(response.headers.get("Content-Range", ""))
        prefix = f"bytes {start}-{end}/"
        if not content_range.startswith(prefix):
            raise IOError(f"unexpected Content-Range: {content_range!r}")
        data = _read_exact(response, expected)
        if response.read(1):
            raise IOError("ranged response exceeded the requested byte limit")
    return data


def _archive_size(url: str) -> int:
    probe = fetch_range(url, 0, 0, max_bytes=1)
    if probe != b"" and len(probe) != 1:
        raise IOError("archive probe failed")
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with open_https(request, timeout=90) as response:
        content_range = str(response.headers.get("Content-Range", ""))
    try:
        total = int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError) as error:
        raise IOError(f"archive probe did not include total size: {content_range!r}") from error
    if total <= 0:
        raise IOError("archive size is not positive")
    return total


def kaggle_signed_url(owner: str, dataset: str, version: int) -> str:
    """Resolve a public Kaggle dataset to its signed archive URL."""
    quoted_owner = urllib.parse.quote(owner, safe="")
    quoted_dataset = urllib.parse.quote(dataset, safe="")
    endpoint = (
        f"https://www.kaggle.com/api/v1/datasets/download/{quoted_owner}/{quoted_dataset}"
        f"?datasetVersionNumber={int(version)}"
    )
    request = urllib.request.Request(endpoint)
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args: Any, **_kwargs: Any):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=90) as response:
            location = response.headers.get("Location")
    except urllib.error.HTTPError as error:
        if error.code != 302:
            raise
        location = error.headers.get("Location")
    if not location:
        raise IOError("Kaggle did not return a signed archive location")
    return str(location)


def _find_signature(data: bytes, signature: int) -> int:
    marker = struct.pack("<I", signature)
    position = data.rfind(marker)
    if position < 0:
        raise IOError(f"ZIP signature 0x{signature:08x} was not found")
    return position


def _zip64_layout(tail: bytes, archive_size: int) -> ArchiveLayout:
    eocd = _find_signature(tail, ZIP_EOCD)
    _, _, _, _, entries32, central_size32, central_offset32, _ = struct.unpack_from("<I4H2IH", tail, eocd)
    if entries32 != ZIP32_SENTINEL and central_size32 != ZIP32_SENTINEL and central_offset32 != ZIP32_SENTINEL:
        return ArchiveLayout(archive_size, entries32, central_offset32, central_size32)
    zip64 = _find_signature(tail, ZIP64_EOCD)
    _, _record_size, _made_by, _needed, _disk, _central_disk, _entries_disk, entries = struct.unpack_from(
        "<IQ2H2I2Q", tail, zip64
    )
    locator = tail.find(struct.pack("<I", ZIP64_LOCATOR), zip64 + 56)
    if locator < 0:
        raise IOError("ZIP64 locator is missing")
    _signature, _disk_number, zip64_offset, _disk_count = struct.unpack_from("<IIQI", tail, locator)
    if _signature != ZIP64_LOCATOR:
        raise IOError("ZIP64 locator is malformed")
    _central_size, _central_offset = struct.unpack_from("<QQ", tail, zip64 + 40)
    if zip64_offset != archive_size - (len(tail) - zip64):
        # The locator is only a consistency hint; some mirrors normalize the
        # tail response.  Keep the central values authoritative but reject
        # obviously impossible offsets.
        if not (0 <= _central_offset < archive_size):
            raise IOError("ZIP64 central-directory offset is outside the archive")
    return ArchiveLayout(archive_size, entries, _central_offset, _central_size)


def _zip64_values(
    extra: bytes,
    *,
    uncompressed: int,
    compressed: int,
    local_offset: int,
) -> tuple[int, int, int]:
    position = 0
    while position + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, position)
        payload_start = position + 4
        payload_end = payload_start + field_size
        if payload_end > len(extra):
            raise IOError("ZIP extra field exceeds central entry")
        if field_id == ZIP64_EXTRA:
            cursor = payload_start
            if uncompressed == ZIP32_SENTINEL:
                uncompressed = struct.unpack_from("<Q", extra, cursor)[0]
                cursor += 8
            if compressed == ZIP32_SENTINEL:
                compressed = struct.unpack_from("<Q", extra, cursor)[0]
                cursor += 8
            if local_offset == ZIP32_SENTINEL:
                local_offset = struct.unpack_from("<Q", extra, cursor)[0]
            return uncompressed, compressed, local_offset
        position = payload_end
    if ZIP32_SENTINEL in (uncompressed, compressed, local_offset):
        raise IOError("ZIP64 sentinel was present without a ZIP64 extra field")
    return uncompressed, compressed, local_offset


def iter_central_entries(central: bytes) -> Iterator[ZipEntry]:
    """Stream central-directory records without accumulating all 821k entries."""
    position = 0
    while position < len(central):
        if len(central) - position < 46:
            raise IOError("truncated ZIP central-directory record")
        values = struct.unpack_from("<I6H3I5H2I", central, position)
        signature = values[0]
        if signature != ZIP_CENTRAL:
            raise IOError(f"unexpected central-directory signature 0x{signature:08x}")
        compression = values[4]
        crc32 = values[7]
        compressed = values[8]
        uncompressed = values[9]
        name_length, extra_length, comment_length = values[10:13]
        local_offset = values[16]
        record_end = position + 46 + name_length + extra_length + comment_length
        if record_end > len(central):
            raise IOError("truncated ZIP central-directory name/extra data")
        name_start = position + 46
        name = central[name_start : name_start + name_length].decode("utf-8", errors="strict")
        extra_start = name_start + name_length
        extra = central[extra_start : extra_start + extra_length]
        uncompressed, compressed, local_offset = _zip64_values(
            extra,
            uncompressed=uncompressed,
            compressed=compressed,
            local_offset=local_offset,
        )
        yield ZipEntry(name, compression, compressed, uncompressed, crc32, local_offset)
        position = record_end


class BoundedZip64Archive:
    """Read selected files from a remote ZIP64 archive using bounded ranges."""

    def __init__(self, url: str, *, max_central_bytes: int = 512 * 1024 * 1024) -> None:
        self.url = url
        self.archive_size = _archive_size(url)
        tail_size = min(self.archive_size, 128 * 1024)
        tail = fetch_range(url, self.archive_size - tail_size, self.archive_size - 1, max_bytes=tail_size)
        self.layout = _zip64_layout(tail, self.archive_size)
        if self.layout.central_size > max_central_bytes:
            raise ValueError("central directory exceeds configured safety limit")
        central = fetch_range(
            url,
            self.layout.central_offset,
            self.layout.central_offset + self.layout.central_size - 1,
            max_bytes=max_central_bytes,
        )
        self._entries = central

    def iter_entries(self, *, prefix: str | None = None, suffix: str | None = None) -> Iterator[ZipEntry]:
        for entry in iter_central_entries(self._entries):
            if prefix is not None and not entry.name.startswith(prefix):
                continue
            if suffix is not None and not entry.name.endswith(suffix):
                continue
            yield entry

    def find(self, name: str) -> ZipEntry:
        for entry in self.iter_entries():
            if entry.name == name:
                return entry
        raise KeyError(name)

    def extract(self, entry: ZipEntry, destination: str | Path, *, max_uncompressed_bytes: int) -> dict[str, Any]:
        if entry.uncompressed_size > max_uncompressed_bytes:
            raise ValueError(f"{entry.name} exceeds max_uncompressed_bytes")
        header = fetch_range(
            self.url,
            entry.local_header_offset,
            entry.local_header_offset + 65535,
            max_bytes=65536,
        )
        if struct.unpack_from("<I", header, 0)[0] != ZIP_LOCAL:
            raise IOError(f"invalid local header for {entry.name}")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        data_start = entry.local_header_offset + 30 + name_length + extra_length
        compressed = fetch_range(
            self.url,
            data_start,
            data_start + entry.compressed_size - 1,
            max_bytes=max(entry.compressed_size, 1),
        )
        if entry.compression == 0:
            output = compressed
        elif entry.compression == 8:
            output = zlib.decompress(compressed, -15)
        else:
            raise ValueError(f"unsupported ZIP compression method {entry.compression}")
        if len(output) != entry.uncompressed_size:
            raise IOError(f"size mismatch for {entry.name}: {len(output)} != {entry.uncompressed_size}")
        actual_crc = zlib.crc32(output) & 0xFFFFFFFF
        if actual_crc != entry.crc32:
            raise IOError(f"CRC mismatch for {entry.name}: {actual_crc:08x} != {entry.crc32:08x}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(output)
        return {
            "name": entry.name,
            "path": str(destination),
            "compressed_bytes": entry.compressed_size,
            "uncompressed_bytes": entry.uncompressed_size,
            "crc32": f"{actual_crc:08x}",
            "sha256": hashlib.sha256(output).hexdigest(),
        }


def _osu_string(stream: io.BytesIO) -> str:
    marker = stream.read(1)
    if marker == b"\x00":
        return ""
    if marker != b"\x0b":
        raise ValueError("invalid osu! string marker in replay header")
    length = 0
    shift = 0
    while True:
        value = stream.read(1)
        if not value:
            raise ValueError("truncated osu! string length")
        byte = value[0]
        length |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
        if shift > 28:
            raise ValueError("oversized osu! string length")
    value = stream.read(length)
    if len(value) != length:
        raise ValueError("truncated osu! string payload")
    return value.decode("utf-8", errors="strict")


def read_osr_header(path: str | Path) -> dict[str, Any]:
    """Read identity/mod fields without decoding the replay frame stream."""
    source = Path(path)
    payload = source.read_bytes()
    stream = io.BytesIO(payload)
    mode_raw = stream.read(1)
    if len(mode_raw) != 1:
        raise ValueError(f"empty replay: {source}")
    mode = mode_raw[0]
    version_raw = stream.read(4)
    if len(version_raw) != 4:
        raise ValueError(f"truncated replay version: {source}")
    version = struct.unpack("<I", version_raw)[0]
    beatmap_md5 = _osu_string(stream)
    player_name = _osu_string(stream)
    replay_hash = _osu_string(stream)
    counts = struct.unpack("<6H", stream.read(12))
    score, max_combo = struct.unpack("<IH", stream.read(6))
    perfect_raw = stream.read(1)
    if len(perfect_raw) != 1:
        raise ValueError(f"truncated replay perfect flag: {source}")
    mods_raw = stream.read(4)
    if len(mods_raw) != 4:
        raise ValueError(f"truncated replay mods: {source}")
    mods = struct.unpack("<I", mods_raw)[0]
    _life_bar = _osu_string(stream)
    timestamp_raw = stream.read(8)
    length_raw = stream.read(4)
    if len(timestamp_raw) != 8 or len(length_raw) != 4:
        raise ValueError(f"truncated replay timing fields: {source}")
    timestamp_ticks = struct.unpack("<q", timestamp_raw)[0]
    replay_data_length = struct.unpack("<I", length_raw)[0]
    replay_data = stream.read(replay_data_length)
    if len(replay_data) != replay_data_length:
        raise ValueError(f"truncated replay frame payload: {source}")
    online_score_id = None
    trailing = stream.read(8)
    if len(trailing) == 8:
        online_score_id = struct.unpack("<q", trailing)[0]
    return {
        "mode": mode,
        "version": version,
        "beatmap_md5": beatmap_md5,
        "player_name": player_name,
        "replay_hash": replay_hash,
        "count_300": counts[0],
        "count_100": counts[1],
        "count_50": counts[2],
        "count_geki": counts[3],
        "count_katu": counts[4],
        "count_miss": counts[5],
        "score": score,
        "max_combo": max_combo,
        "perfect": bool(perfect_raw[0]),
        "mods": mods,
        "automation_mods": bool(mods & AUTOMATION_MOD_MASK),
        "timestamp_ticks": timestamp_ticks,
        "replay_data_length": replay_data_length,
        "online_score_id": online_score_id,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def summarize_index(path: str | Path) -> dict[str, Any]:
    """Summarize source metadata without assigning human skill tiers."""
    source = Path(path)
    players: set[str] = set()
    maps: set[str] = set()
    mods: dict[str, int] = {}
    automation_rows = 0
    dates: set[str] = set()
    rows = 0
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows += 1
            players.add(str(row.get("playerName", "")))
            maps.add(str(row.get("beatmapHash", "")))
            mod = str(row.get("modsReadable", ""))
            mods[mod] = mods.get(mod, 0) + 1
            dates.add(str(row.get("date", ""))[:10])
            lowered = mod.lower()
            if any(token in lowered for token in ("auto", "relax", "timewarp", "replay")):
                automation_rows += 1
    return {
        "schema_version": 1,
        "source_type": "public_replay_archive_candidate",
        "rows": rows,
        "unique_players": len(players),
        "unique_maps": len(maps),
        "dated_rows": len(dates - {""}),
        "automation_or_unsupported_mod_rows": automation_rows,
        "top_mods": sorted(mods.items(), key=lambda item: (-item[1], item[0]))[:20],
        "skill_tiers": {"beginner": 0, "intermediate": 0, "expert": 0, "competitive": 0},
        "skill_tier_status": "unavailable_without_dated_player_evidence",
        "human_training_admission": "blocked_until_provenance_review",
    }


def parse_public_score_page(page: str, score_id: int | str) -> dict[str, Any]:
    """Extract public, dated score evidence without treating it as human proof."""
    text = html.unescape(page)
    rank_match = re.search(r'"rank_global"\s*:\s*(\d+|null)', text)
    window = text[rank_match.end() : rank_match.end() + 8000] if rank_match else text
    user_match = re.search(
        r'"user"\s*:\s*\{(?P<body>.*?)\}',
        window,
        flags=re.DOTALL,
    )
    user_body = user_match.group("body") if user_match else ""

    def _field(pattern: str, source: str = user_body) -> str | None:
        match = re.search(pattern, source)
        return match.group(1) if match else None

    rank_value = rank_match.group(1) if rank_match else None
    return {
        "score_id": str(score_id),
        "source_url": f"https://osu.ppy.sh/scores/osu/{score_id}",
        "public_score_page_found": rank_match is not None,
        "rank_global": int(rank_value) if rank_value and rank_value != "null" else None,
        "player_id": int(_field(r'"id"\s*:\s*(\d+)') or 0) or None,
        "player_name": _field(r'"username"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'),
        "player_is_bot": (_field(r'"is_bot"\s*:\s*(true|false)') == "true"),
        "has_replay": '"has_replay":true' in text.replace(" ", ""),
        "beatmap_md5": _field(r'"checksum"\s*:\s*"([0-9a-f]{32})"', text),
        "evidence_status": "public_score_metadata_only",
        "human_provenance_status": "not_proven_by_score_page",
    }


def fetch_public_score_evidence(score_id: int | str) -> dict[str, Any]:
    """Fetch one public score page; API credentials are not used."""
    url = f"https://osu.ppy.sh/scores/osu/{urllib.parse.quote(str(score_id), safe='')}"
    request = urllib.request.Request(url, headers={"User-Agent": "human-sim-research/1.0"})
    with open_https(request, timeout=45) as response:
        payload = response.read(4 * 1024 * 1024 + 1)
        if len(payload) > 4 * 1024 * 1024:
            raise ValueError("score evidence exceeds 4 MiB")
        page = payload.decode("utf-8", errors="replace")
    return parse_public_score_page(page, score_id)


def fetch_public_score_evidence_batch(
    score_ids: Sequence[int | str],
    *,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Fetch independent public score pages concurrently, preserving order."""
    if not 1 <= int(max_workers) <= 8:
        raise ValueError("max_workers must be between 1 and 8")

    def _fetch(score_id: int | str) -> dict[str, Any]:
        try:
            return fetch_public_score_evidence(score_id)
        except Exception as error:  # evidence collection should retain failures for review
            return {
                "score_id": str(score_id),
                "public_score_page_found": False,
                "error": f"{type(error).__name__}: {error}",
            }

    with ThreadPoolExecutor(max_workers=min(int(max_workers), max(len(score_ids), 1))) as executor:
        return list(executor.map(_fetch, score_ids))


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_candidate_pilot(
    candidate_report: Mapping[str, Any],
    score_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Review selected archive candidates without admitting them to training."""
    records = list(candidate_report.get("records", []))
    evidence_by_score = {
        str(item.get("score_id")): item
        for item in (score_evidence or {}).get("records", [])
        if item.get("score_id") is not None
    }
    replay_hashes = [str(record.get("replay_header", {}).get("sha256", "")) for record in records]
    player_map_keys = [
        (
            str(record.get("replay_header", {}).get("player_name", "")),
            str(record.get("replay_header", {}).get("beatmap_md5", "")),
        )
        for record in records
    ]
    reviewed: list[dict[str, Any]] = []
    for record in records:
        header = dict(record.get("replay_header", {}))
        score_id = str(header.get("online_score_id") or "")
        evidence = evidence_by_score.get(score_id) or record.get("source_identity_evidence")
        reasons: list[str] = []
        if not record.get("map_md5_matches_header", False):
            reasons.append("map_md5_mismatch")
        if bool(header.get("automation_mods", False)):
            reasons.append("automation_or_unsupported_mod")
        if evidence is None and score_id:
            reasons.append("public_score_evidence_missing")
        if evidence is not None:
            if evidence.get("error"):
                reasons.append("public_score_page_unavailable")
            elif evidence.get("player_is_bot"):
                reasons.append("public_score_marks_bot")
            elif evidence.get("beatmap_md5") != header.get("beatmap_md5"):
                reasons.append("public_score_beatmap_mismatch")
            elif evidence.get("player_name_matches_archive") is False:
                reasons.append("public_username_mismatch")
        if not reasons:
            reasons.append("awaiting_reviewed_human_provenance_and_skill_evidence")
        reviewed.append(
            {
                "replay": record.get("replay"),
                "replay_sha256": header.get("sha256"),
                "player_name": header.get("player_name"),
                "beatmap_md5": header.get("beatmap_md5"),
                "online_score_id": header.get("online_score_id"),
                "mods": header.get("mods"),
                "automation_mods": bool(header.get("automation_mods", False)),
                "public_score_evidence": evidence,
                "admission": "quarantined",
                "quarantine_reasons": reasons,
                "skill_group": None,
                "decoded_windows": 0,
            }
        )
    duplicate_replays = len(replay_hashes) - len(set(replay_hashes))
    player_names = [str(record.get("replay_header", {}).get("player_name", "")) for record in records]
    duplicate_player_map = len(player_map_keys) - len(set(player_map_keys))
    score_ids = [
        str(record.get("replay_header", {}).get("online_score_id", ""))
        for record in records
        if record.get("replay_header", {}).get("online_score_id")
    ]
    public_player_ids = [str(item.get("player_id")) for item in evidence_by_score.values() if item.get("player_id")]
    return {
        "schema_version": 1,
        "source_type": "public_replay_archive_candidate_review",
        "source": candidate_report.get("source"),
        "source_license": candidate_report.get("source_license", "unrecorded"),
        "source_license_status": "recorded_public_archive_term_separate_from_human_provenance",
        "candidate_replays": len(records),
        "exact_map_pairs": sum(bool(record.get("map_md5_matches_header")) for record in records),
        "duplicate_replay_content_count": duplicate_replays,
        "duplicate_archive_player_count": len(player_names) - len(set(player_names)),
        "duplicate_player_map_count": duplicate_player_map,
        "duplicate_score_id_count": len(score_ids) - len(set(score_ids)),
        "public_score_evidence_records": len(evidence_by_score),
        "dated_official_rank_evidence_records": sum(
            item.get("dated_rank_evidence_status") == "dated_official_rank_metadata"
            for item in evidence_by_score.values()
        ),
        "skill_group_assignment": "none_public_rank_is_recorded_not_converted_to_tier",
        "duplicate_public_player_id_count": len(public_player_ids) - len(set(public_player_ids)),
        "human_provenance_status": "not_verified_public_submission_archive",
        "human_training_admission": "blocked_pending_reviewed_human_provenance_and_tiers",
        "candidate_pool_status": "quarantined_not_decoded",
        "decoded_movement_windows": 0,
        "labelled_players": {group: 0 for group in ("beginner", "intermediate", "expert", "competitive")},
        "labelled_windows": {group: 0 for group in ("beginner", "intermediate", "expert", "competitive")},
        "records": reviewed,
    }


def audit_decoded_candidates(
    candidate_report: Mapping[str, Any],
    decoded_root: str | Path,
    map_plan_root: str | Path,
    *,
    max_gap_ms: float = 75.0,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure quarantined decoded replays without creating admitted corpus rows."""
    import numpy as np

    from .execution_training import _extract_phase_label_arrays, transition_route
    from .io import load_map_plan

    decoded_root = Path(decoded_root)
    map_plan_root = Path(map_plan_root)
    per_replay: list[dict[str, Any]] = []
    aggregate_rejections: Counter[str] = Counter()
    total_windows = 0
    records = list(candidate_report.get("records", []))
    progress_file = Path(progress_path) if progress_path else None
    completed_by_hash: dict[str, dict[str, Any]] = {}
    if progress_file and progress_file.exists():
        try:
            checkpoint = json.loads(progress_file.read_text(encoding="utf-8"))
            if checkpoint.get("candidate_records") == len(records):
                completed_by_hash = {
                    str(item.get("replay_sha256")): item
                    for item in checkpoint.get("records", [])
                    if item.get("replay_sha256")
                }
                print(
                    f"[audit] resumed_checkpoint={len(completed_by_hash)}/{len(records)} path={progress_file}",
                    flush=True,
                )
        except (OSError, ValueError, TypeError):
            completed_by_hash = {}
    def _write_checkpoint(status: str) -> None:
        if not progress_file:
            return
        write_json(progress_file, {
            "schema_version": 1,
            "status": status,
            "candidate_records": len(records),
            "processed_records": len(per_replay),
            "decoded_replays": sum("decoded_frame_count" in item for item in per_replay),
            "decoded_movement_windows": total_windows,
            "records": per_replay,
        })

    for record_index, record in enumerate(records, 1):
        header_metadata = dict(record.get("replay_header", {}))
        replay_digest = str(header_metadata.get("sha256", ""))
        cached = completed_by_hash.get(replay_digest)
        if cached is not None:
            per_replay.append(cached)
            total_windows += int(cached.get("usable_movement_windows", 0))
            aggregate_rejections.update(cached.get("rejection_reasons", {}))
            print(
                f"[audit {record_index}/{len(records)}] sha256={replay_digest[:12]} resumed=true "
                f"windows={cached.get('usable_movement_windows', 0)}",
                flush=True,
            )
            continue
        output_path = decoded_root / f"{replay_digest}.ndjson"
        result: dict[str, Any] = {
            "replay": record.get("replay"),
            "replay_sha256": replay_digest,
            "player_name": header_metadata.get("player_name"),
            "beatmap_md5": header_metadata.get("beatmap_md5"),
            "online_score_id": header_metadata.get("online_score_id"),
            "skill_group": None,
            "admission": "quarantined",
            "decoded_path": str(output_path),
            "usable_movement_windows": 0,
            "window_measurement_context": "circle_transition_geometry_only_skill_unknown",
            "rejection_reasons": {},
        }
        rejects: Counter[str] = Counter()
        if not output_path.exists():
            rejects["decoded_output_missing"] += 1
            result["rejection_reasons"] = dict(rejects)
            per_replay.append(result)
            aggregate_rejections.update(rejects)
            continue
        try:
            with output_path.open("r", encoding="utf-8") as stream:
                decoded_header = json.loads(stream.readline())
                frames = [json.loads(line) for line in stream if line.strip()]
        except (OSError, ValueError, TypeError) as error:
            rejects[f"decoded_output_invalid:{type(error).__name__}"] += 1
            result["rejection_reasons"] = dict(rejects)
            per_replay.append(result)
            aggregate_rejections.update(rejects)
            continue

        times = np.asarray([float(frame["time_ms"]) for frame in frames], dtype=float)
        positions = np.asarray([[float(frame["x"]), float(frame["y"])] for frame in frames], dtype=float)
        finite = bool(np.all(np.isfinite(times)) and np.all(np.isfinite(positions))) if len(frames) else False
        gaps = np.diff(times) if len(times) > 1 else np.asarray([], dtype=float)
        result.update(
            {
                "decoded_frame_count": len(frames),
                "header_frame_count": decoded_header.get("frame_count"),
                "header_frame_count_matches": decoded_header.get("frame_count") == len(frames),
                "decoded_beatmap_md5": decoded_header.get("beatmap_md5"),
                "decoded_map_md5_matches_candidate": decoded_header.get("beatmap_md5") == header_metadata.get("beatmap_md5"),
                "player_hash_present": bool(decoded_header.get("player_hash")),
                "duration_ms": float(times[-1] - times[0]) if len(times) else 0.0,
                "median_cadence_ms": float(np.median(gaps)) if len(gaps) else None,
                "p95_cadence_ms": float(np.percentile(gaps, 95)) if len(gaps) else None,
                "max_gap_ms": float(np.max(gaps)) if len(gaps) else None,
                "gaps_over_threshold": int(np.sum(gaps > max_gap_ms)) if len(gaps) else 0,
                "duplicate_timestamps": int(np.sum(gaps == 0)) if len(gaps) else 0,
                "non_monotone_timestamps": int(np.sum(gaps < 0)) if len(gaps) else 0,
                "nonfinite_frames": int(len(frames) - np.sum(np.isfinite(times) & np.all(np.isfinite(positions), axis=1))) if len(frames) else 0,
                "out_of_playfield_frames": int(np.sum((positions[:, 0] < 0) | (positions[:, 0] > 512) | (positions[:, 1] < 0) | (positions[:, 1] > 384))) if len(frames) else 0,
                "key_transition_count": int(sum(
                    bool(frames[index].get("k1")) != bool(frames[index - 1].get("k1"))
                    or bool(frames[index].get("k2")) != bool(frames[index - 1].get("k2"))
                    for index in range(1, len(frames))
                )),
            }
        )
        if not finite or len(frames) < 4:
            rejects["sparse_or_nonfinite_frames"] += 1
        if result["non_monotone_timestamps"]:
            rejects["timestamps_go_backwards"] += 1
        if not result["decoded_map_md5_matches_candidate"]:
            rejects["decoded_map_md5_mismatch"] += 1
        try:
            plan = load_map_plan(map_plan_root / f"{header_metadata.get('beatmap_md5')}.map.ndjson.gz")
            if plan.beatmap_md5.lower() != str(header_metadata.get("beatmap_md5", "")).lower():
                rejects["map_plan_md5_mismatch"] += 1
            if not finite or result["non_monotone_timestamps"]:
                raise ValueError("invalid_frame_stream")
            for index in range(1, len(plan.objects)):
                try:
                    # The skill value here is only a neutral route-construction
                    # parameter; the candidate remains explicitly unlabelled.
                    route = transition_route(plan, index, skill_group="intermediate", record_id=f"quarantine:{replay_digest}:{index}")
                    _extract_phase_label_arrays(route, times, positions, max_gap_ms=max_gap_ms)
                    result["usable_movement_windows"] += 1
                except ValueError as error:
                    rejects[f"phase_window:{error}"] += 1
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejects[f"map_or_window_analysis:{type(error).__name__}"] += 1
        total_windows += int(result["usable_movement_windows"])
        result["rejection_reasons"] = dict(rejects)
        per_replay.append(result)
        aggregate_rejections.update(rejects)
        _write_checkpoint("in_progress")
        print(
            f"[audit {record_index}/{len(records)}] sha256={replay_digest[:12]} resumed=false "
            f"decoded={('decoded_frame_count' in result)} windows={result['usable_movement_windows']}",
            flush=True,
        )
    _write_checkpoint("complete")
    return {
        "schema_version": 1,
        "candidate_pool_status": "quarantined_decoded_skill_unknown",
        "human_training_admission": "blocked_pending_reviewed_human_provenance_and_tiers",
        "candidate_replays": len(per_replay),
        "decoded_replays": sum("decoded_frame_count" in result for result in per_replay),
        "decoded_movement_windows": total_windows,
        "labelled_players": {group: 0 for group in ("beginner", "intermediate", "expert", "competitive")},
        "labelled_windows": {group: 0 for group in ("beginner", "intermediate", "expert", "competitive")},
        "aggregate_rejection_reasons": dict(aggregate_rejections),
        "records": per_replay,
    }
