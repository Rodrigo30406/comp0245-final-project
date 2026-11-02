"""Pack the many-small-files parts of the light tier into one archive each, so the cloud repo holds ~200 objects instead
of 130 000 (Hugging Face recommends < 10 000 files per folder). Incremental: an archive is rebuilt only when a file
inside its folder is newer than the archive. System python (-I), no project imports.

  python3 -I scripts/sync/pack.py            # (re)build runs/sync/pack/{evaluation,datasets}/*.tar.gz
  python3 -I scripts/sync/pack.py --unpack   # on the other computer: extract every archive into place (never overwrites
                                             # a newer local file)
"""
import os, sys, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "runs/sync/pack"
EVAL_EXCLUDE = ("traces",)                       # traces are the heavy tier
DATASET_EXCLUDE_DIRS = ("executions", "logs")    # dataset metadata: everything but the recordings and the collector logs
DATASET_SKIP = ("aborted", "interrupted", "smoke_check")


def newest(path, exclude_dirs=(), exclude_suffix=()):
    t = 0.0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for f in filenames:
            if f.endswith(exclude_suffix):
                continue
            t = max(t, os.stat(os.path.join(dirpath, f)).st_mtime)
    return t


def build(src, archive, exclude_dirs=(), exclude_suffix=()):
    if archive.exists() and archive.stat().st_mtime >= newest(src, exclude_dirs, exclude_suffix):
        return False
    archive.parent.mkdir(parents=True, exist_ok=True); tmp = archive.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        def flt(info):
            parts = Path(info.name).parts
            if any(d in parts[1:] for d in exclude_dirs) or info.name.endswith(exclude_suffix):
                return None
            return info
        tar.add(src, arcname=str(src.relative_to(ROOT)), filter=flt)
    os.replace(tmp, archive); return True


def pack():
    built = 0; total = 0
    for e in sorted((ROOT / "runs/evaluation").iterdir()):
        if e.is_dir():
            total += 1; built += build(e, PACK / "evaluation" / f"{e.name}.tar.gz", EVAL_EXCLUDE)
    for d in sorted((ROOT / "datasets").glob("*/*/dataset.json")):
        run = d.parent
        if any(s in str(run) for s in DATASET_SKIP):
            continue
        total += 1; built += build(run, PACK / "datasets" / f"{run.parent.name}__{run.name}.meta.tar.gz", DATASET_EXCLUDE_DIRS, (".npz",))
    size = sum(p.stat().st_size for p in PACK.rglob("*.tar.gz"))
    print(f"packed {built} of {total} archives rebuilt; {size / 1e9:.2f} GB in {PACK.relative_to(ROOT)}")


def unpack():
    n = 0
    for archive in sorted(PACK.rglob("*.tar.gz")):
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                dest = ROOT / m.name
                if not str(dest.resolve()).startswith(str(ROOT)):
                    raise ValueError(f"unsafe path in {archive}: {m.name}")
                if m.isfile() and (not dest.exists() or dest.stat().st_mtime < m.mtime):
                    tar.extract(m, ROOT, filter="data"); n += 1
    print(f"unpacked {n} files from {len(list(PACK.rglob('*.tar.gz')))} archives")


if __name__ == "__main__":
    unpack() if "--unpack" in sys.argv else pack()
