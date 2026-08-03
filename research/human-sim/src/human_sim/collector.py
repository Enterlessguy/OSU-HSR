from __future__ import annotations

import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


API_ROOT = "https://osu.ppy.sh/api/v2"


def _oauth_token() -> str:
    client_id = os.environ.get("OSU_CLIENT_ID")
    client_secret = os.environ.get("OSU_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Set OSU_CLIENT_ID and OSU_CLIENT_SECRET; credentials are never written to disk")
    payload = json.dumps(
        {"client_id": int(client_id), "client_secret": client_secret, "grant_type": "client_credentials", "scope": "public"}
    ).encode()
    request = Request("https://osu.ppy.sh/oauth/token", data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        return str(json.load(response)["access_token"])


def collect_replays(score_ids_file: str | Path, output_dir: str | Path, delay_seconds: float = 0.25) -> dict[str, int]:
    token = _oauth_token()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    score_ids = [line.strip() for line in Path(score_ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    downloaded = skipped = unavailable = 0
    for score_id in score_ids:
        target = destination / f"{score_id}.osr"
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue
        request = Request(f"{API_ROOT}/scores/{score_id}/download", headers={"Authorization": f"Bearer {token}"})
        try:
            with urlopen(request, timeout=45) as response:
                payload = response.read()
        except HTTPError as error:
            if error.code in {404, 422}:
                unavailable += 1
                continue
            raise
        temporary = target.with_suffix(".osr.part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        downloaded += 1
        time.sleep(max(0.0, delay_seconds))
    summary = {"requested": len(score_ids), "downloaded": downloaded, "skipped": skipped, "unavailable": unavailable}
    (destination / "collection-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
