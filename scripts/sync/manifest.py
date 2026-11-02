"""Artifact manifest for the sync (scripts/sync/): what exists, how big it is, and the hashes that let the other
computer check what it received. Runs with the system python (-I), no project imports.

  python3 -I scripts/sync/manifest.py            # write runs/sync/manifest.json, print a summary
  python3 -I scripts/sync/manifest.py --light    # list the light-tier selection with sizes (what push_artifacts.sh uploads)
  python3 -I scripts/sync/manifest.py --verify   # hash every episode .npz and compare with data_sha256 in its .json
"""
import fnmatch, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIGHT = [("runs/training", []), ("runs/splits", []), ("runs/diagnostics", []), ("runs/sync/pack", []), ("docs/figures", [])]
SKIP_DATASET = ("aborted", "interrupted", "smoke_check")


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_size(path, excludes=()):
    total = 0; files = 0
    for p in (ROOT / path).rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(ROOT))
            if any(fnmatch.fnmatch(rel, f"{path}/{e}") or fnmatch.fnmatch(rel, e) for e in excludes):
                continue
            total += p.stat().st_size; files += 1
    return total, files


def datasets():
    out = {}
    for d in sorted((ROOT / "datasets").glob("*/*/dataset.json")) + sorted((ROOT / "datasets").glob("*/dataset.json")):
        run = d.parent; rel = str(run.relative_to(ROOT))
        if any(s in rel for s in SKIP_DATASET):
            continue
        eps = sorted((run / "episodes").glob("*.npz")) if (run / "episodes").is_dir() else []
        out[rel] = {"episodes": len(eps), "bytes": sum(p.stat().st_size for p in run.rglob("*") if p.is_file()),
                    "dataset_json_sha256": sha256(d), "tier": "heavy (episodes) + light (metadata)"}
    return out


def training_runs():
    out = {}
    for r in sorted((ROOT / "runs/training").glob("*/run.json")):
        m = json.load(open(r)); out[r.parent.name] = {"status": m.get("status"), "model_sha256": m.get("model_sha256"), "created_at": m.get("created_at")}
    return out


def evaluations():
    out = {}
    for e in sorted((ROOT / "runs/evaluation").iterdir()):
        if e.is_dir():
            traces = sum(p.stat().st_size for p in (e / "traces").rglob("*") if p.is_file()) if (e / "traces").is_dir() else 0
            out[e.name] = {"episodes": len(list((e / "episodes").glob("*"))) if (e / "episodes").is_dir() else 0, "traces_bytes": traces}
    return out


def verify():
    bad = 0; n = 0
    for js in sorted((ROOT / "datasets").glob("*/*/episodes/*.json")):
        meta = json.load(open(js)); npz = js.with_name(meta.get("data_file", js.stem + ".npz"))
        if not npz.exists():
            print("MISSING", npz.relative_to(ROOT)); bad += 1; continue
        n += 1
        if meta.get("data_sha256") and sha256(npz) != meta["data_sha256"]:
            print("MISMATCH", npz.relative_to(ROOT)); bad += 1
    print(f"verified {n} episodes, {bad} problems"); return bad


def main(argv):
    if "--verify" in argv:
        return 1 if verify() else 0
    if "--light" in argv:
        total = 0
        for path, ex in LIGHT:
            if (ROOT / path).is_dir():
                b, f = tree_size(path, ex); total += b; print(f"{path:20s} {f:6d} files {b / 1e9:7.3f} GB   excludes {ex}")
        print(f"{'light tier total':20s} {'':6s}       {total / 1e9:7.3f} GB"); return 0
    man = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
           "datasets": datasets(), "training_runs": training_runs(), "evaluations": evaluations()}
    (ROOT / "runs/sync").mkdir(parents=True, exist_ok=True)
    json.dump(man, open(ROOT / "runs/sync/manifest.json", "w"), indent=1)
    heavy = sum(v["bytes"] for v in man["datasets"].values()) + sum(v["traces_bytes"] for v in man["evaluations"].values())
    print(f"manifest: {len(man['datasets'])} datasets, {len(man['training_runs'])} training runs, {len(man['evaluations'])} evaluations; heavy tier {heavy / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
