"""Create local blinded clips without distributing human replay coordinates.

Selection is SHA-256 ordered by map family and uses the first supported
four-circle geometry, independent of arm score or learned acceptance. Rating
files are external human input. The local key must be kept away from raters.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from human_sim.coherent_execution import _circle_windows
from human_sim.io import load_map_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-tag", required=True)
    args = parser.parse_args()
    if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.case_tag):
        raise ValueError("Invalid case tag")
    cases = OUT / "development-cases/validation" / args.case_tag
    dest = cases / "blinded-review"
    dest.mkdir(exist_ok=True)
    manifest = json.loads((OUT / "development-batches/validation/window-manifest.json").read_text())
    ordered = sorted(manifest["records"], key=lambda r: hashlib.sha256(f"release-review-20260926:{r['map_family']}".encode()).hexdigest())
    clips, key = [], []
    for row in ordered:
        if len(clips) >= 12:
            break
        plan = load_map_plan(OUT / f"development-plans/{row['map']}.map.ndjson.gz")
        with (OUT / f"development-decoded/{row['replay_sha256']}.ndjson").open(encoding="utf-8") as stream:
            next(stream)
            frames = [json.loads(line) for line in stream if line.strip()]
        human_t = np.asarray([f["time_ms"] for f in frames])
        human_xy = np.asarray([[f["x"], f["y"]] for f in frames])
        chosen = None
        for indices in _circle_windows(plan):
            start, end = plan.objects[indices[0]].start_time_ms, plan.objects[indices[-1]].start_time_ms
            native = human_t[(human_t >= start) & (human_t <= end)]
            if human_t[0] <= start < end <= human_t[-1] and len(native) >= 8 and np.max(np.diff(native)) <= 40:
                chosen = (indices, start, end)
                break
        if chosen is None:
            continue
        indices, start, end = chosen
        clip_id = f"clip-{len(clips)+1:02d}"
        case = cases / f"{row['map']}-seed42.npz"
        summary = json.loads(case.with_suffix(".json").read_text())
        if hashlib.sha256(case.read_bytes()).hexdigest() != summary["trace_sha256"]:
            raise ValueError("Trace identity changed")
        with np.load(case) as arrays:
            sample_times = np.arange(start, end + .01, 1000 / 60)
            sources = {"human": (human_t, human_xy), "math": (arrays["times_ms"], arrays["math_positions"]),
                       "hybrid": (arrays["times_ms"], arrays["hybrid_positions"])}
            paths = {arm: np.column_stack([np.interp(sample_times, t, xy[:, a]) for a in range(2)]).round(4).tolist()
                     for arm, (t, xy) in sources.items()}
        arms = sorted(paths, key=lambda arm: hashlib.sha256(f"blinding-20260926:{clip_id}:{arm}".encode()).hexdigest())
        clips.append({"id": clip_id, "duration": float(end - start), "paths": [paths[a] for a in arms],
                      "objects": [[float(plan.objects[i].start_time_ms-start), plan.objects[i].position.x,
                                   plan.objects[i].position.y, plan.objects[i].radius] for i in indices]})
        key.append({"id": clip_id, "arms": arms, "map": row["map"], "family": row["map_family"],
                    "trace_sha256": summary["trace_sha256"], "model_sha256": summary["model_file_sha256"]})
    if len(clips) < 12:
        raise ValueError("Fewer than 12 independent preselected supported clips")
    payload = json.dumps(clips, separators=(",", ":"))
    html = '''<!doctype html><meta charset="utf-8"><title>HSR blinded movement review</title>
<style>body{background:#141821;color:#eee;font:16px system-ui;margin:24px}canvas{background:#090d14;width:32%;max-width:512px}button,input{padding:8px;margin:8px}p{max-width:950px}.panels{display:flex;gap:12px}label{display:inline-block;margin:10px}</style>
<h1>Blinded movement review</h1><p>Local research clips. Each clip presents three cursor paths in random order, including a human reference. All use the same 60 Hz rendering, hit circles and time scale. Rate natural movement from 1 (very unnatural) to 7 (very natural); consider flow, corrections, pauses and repetition. Do not try to identify an arm. Replay as needed. Keep the mapping file hidden.</p>
<label>Anonymous rater code <input id="rater" maxlength="40" placeholder="rater-01"></label><select id="clip"></select><button id="play">Replay</button>
<div class="panels"><canvas width="512" height="384"></canvas><canvas width="512" height="384"></canvas><canvas width="512" height="384"></canvas></div>
<div id="ratings"></div><button id="save">Save ratings JSON</button><p id="status"></p>
<script>const clips=PAYLOAD;const select=document.querySelector('#clip');const canvases=[...document.querySelectorAll('canvas')];const answers={};let current=0,started=performance.now();
clips.forEach((c,i)=>select.add(new Option(c.id,i)));select.onchange=()=>{store();current=+select.value;load();started=performance.now()};document.querySelector('#play').onclick=()=>started=performance.now();
function store(){answers[clips[current].id]=[...document.querySelectorAll('.score')].map(e=>e.value===''?null:+e.value)}
function load(){document.querySelector('#ratings').replaceChildren();['A','B','C'].forEach((name,i)=>{const l=document.createElement('label');l.textContent=name+' naturalness (1–7) ';const input=document.createElement('input');input.type='number';input.min=1;input.max=7;input.step=1;input.className='score';input.value=answers[clips[current].id]?.[i]??'';l.append(input);document.querySelector('#ratings').append(l)})}load();
function draw(now){const c=clips[current],elapsed=Math.min(Math.max(now-started-400,0),c.duration),frame=Math.min(Math.floor(elapsed/1000*60),c.paths[0].length-1);canvases.forEach((canvas,arm)=>{const ctx=canvas.getContext('2d');ctx.clearRect(0,0,512,384);ctx.fillStyle='#eee';ctx.fillText(['A','B','C'][arm],12,20);c.objects.forEach(([t,x,y,r])=>{ctx.strokeStyle=elapsed>=t?'#526079':'#8aa2bf';ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.stroke()});ctx.strokeStyle='#ffc96980';ctx.beginPath();for(let i=Math.max(0,frame-10);i<=frame;i++){const[x,y]=c.paths[arm][i];i===Math.max(0,frame-10)?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.stroke();const[x,y]=c.paths[arm][frame];ctx.fillStyle='#ffe3a3';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill()});requestAnimationFrame(draw)}requestAnimationFrame(draw);
document.querySelector('#save').onclick=()=>{store();const rater=document.querySelector('#rater').value.trim();const complete=clips.every(c=>answers[c.id]?.every(v=>Number.isInteger(v)&&v>=1&&v<=7));if(!rater||!complete){document.querySelector('#status').textContent='Enter a rater code and rate all 12 clips in all three panels.';return}const data={schema_version:'hsr-blinded-naturalness-ratings-v1',study_sha256:'STUDY_HASH',rater,ratings:answers};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='hsr-ratings.json';a.click();URL.revokeObjectURL(a.href)};</script>'''
    html = html.replace("PAYLOAD", payload).replace("STUDY_HASH", hashlib.sha256(payload.encode()).hexdigest())
    (dest / "review.html").write_text(html, encoding="utf-8")
    (dest / "private-arm-key.json").write_text(json.dumps({"study_sha256": hashlib.sha256(payload.encode()).hexdigest(),
         "selection_rule": "SHA256 family order; first geometry with human native support; no arm-score selection", "clips": key}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"review": str(dest / "review.html"), "clips": len(clips), "independent_human_ratings": "required_not_generated"}))


if __name__ == "__main__":
    main()
