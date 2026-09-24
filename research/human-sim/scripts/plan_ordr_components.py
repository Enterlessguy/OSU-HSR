"""Inspect candidate identity components without opening replay movement."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from human_sim.execution_acquisition import iter_central_entries


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def root(self, key: str) -> str:
        parent = self.parent.setdefault(key, key)
        if parent != key:
            self.parent[key] = self.root(parent)
        return self.parent[key]

    def join(self, a: str, b: str) -> None:
        ra, rb = self.root(a), self.root(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main() -> None:
    audit_path = OUT / "metadata-audit-v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    entries = {
        entry.name
        for entry in iter_central_entries((OUT / "index-snapshot" / "central-directory.bin").read_bytes())
    }
    rows = audit["candidate_pool"]
    counts: Counter[str] = Counter()
    valid: list[dict] = []
    for row in rows:
        replay_path = f"replays/osr/{row['replay_md5']}.osr"
        map_path = f"beatmaps/{row['map_md5']}.osu"
        if replay_path not in entries:
            counts["replay_missing_from_archive"] += 1
        elif map_path not in entries:
            counts["map_missing_from_archive"] += 1
        else:
            valid.append(row)
    uf = UnionFind()
    for row in valid:
        uf.join("p:" + row["player_name"].casefold(), "f:" + row["map_family"])
    components: dict[str, list[dict]] = defaultdict(list)
    for row in valid:
        components[uf.root("p:" + row["player_name"].casefold())].append(row)
    sizes = sorted((len(rows) for rows in components.values()), reverse=True)
    groups = []
    for component_rows in components.values():
        players = sorted({row["player_name"].casefold() for row in component_rows})
        families = sorted({row["map_family"] for row in component_rows})
        digest = hashlib.sha256(json.dumps({"players": players, "families": families}, sort_keys=True).encode()).hexdigest()[:16]
        groups.append({"component_id": digest, "rows": len(component_rows), "players": len(players), "families": len(families), "replays": len({row["replay_md5"] for row in component_rows})})
    groups.sort(key=lambda item: (-item["rows"], item["component_id"]))
    report = {
        "schema_version": "ordr-component-inventory-v1",
        "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "movement_opened": False,
        "identity_limit": "player names are provisional until public stable IDs are resolved",
        "counts": {**counts, "candidate_rows": len(rows), "exact_archive_pairs": len(valid), "components": len(components)},
        "component_sizes_top20": sizes[:20],
        "components_at_least_2_rows": sum(size >= 2 for size in sizes),
        "components_at_least_5_rows": sum(size >= 5 for size in sizes),
        "largest_components": groups[:25],
    }
    path = OUT / "component-inventory-v1.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
