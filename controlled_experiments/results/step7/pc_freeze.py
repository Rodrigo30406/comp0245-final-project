"""Experiment PC preregistration freeze and protection of pre-existing files.

Only results/step7 is written.  In particular, the requested report insertion is
made in report_PC_frozen.txt because the user's no-existing-file-edits rule takes
precedence over the plan's request to edit report.txt.  No data directory is ever
traversed or opened here.  Symlinks are inventoried but never followed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "results" / "step7"
REFERENCE_MARKER = "\n[References] 参考文献（已核对出处）"
REPORT_HEADER = (
    "[Methodology] 第 7 步预注册：实验 PC（P 在 HARD 与 CONFIRM 上的确认）"
    "（2026-11-03 写下，在 P 的任何 HARD / CONFIRM 运行之前；"
    "冻结记录 results/step7/prereg_PC_frozen.txt）"
)
SECTION_HEADING = "## 3. 实验 PC 预注册（冻结时把本节原文复制进 report.txt）"
IMPLEMENTATION_PATHS = (
    "results/step7/pc_freeze.py",
    "results/step7/pc_pipeline.py",
    "results/step7/PC_mechanisms.py",
    "results/step7/pc_smoke.py",
    "results/step7/run_expPC.sh",
    "tests/test_step7.py",
)
FROZEN_SOURCE_INPUTS = (
    "analyze_rollouts.py", "evaluate_policy.py", "train_policy.py", "bc_features.py",
    "policies/bc_policy.py", "policies/pd_layer.py",
    "results/step6/gp_common.py", "results/step6/step6_pipeline.py",
    "results/step6/gp_smoke.py",
)
PROTECTED_TREES = ("results", "policies", "trained", "tests", "configs", "report", "__pycache__")
EXCLUDED_RELATIVE = {"results/step7", "tests/test_step7.py"}
BASELINE_NAME = "existing_files_baseline.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locations(repo: Path | str | None) -> tuple[Path, Path]:
    root = Path(repo).resolve() if repo is not None else REPO
    return root, root / "results" / "step7"


def _excluded(relative: str) -> bool:
    return any(relative == prefix or relative.startswith(prefix + "/") for prefix in EXCLUDED_RELATIVE)


def _inventory(root: Path) -> dict[str, dict]:
    """List only authorized non-data paths, without following any symlink."""
    files: dict[str, dict] = {}

    def add(path: Path) -> None:
        relative = path.relative_to(root).as_posix()
        if _excluded(relative):
            return
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            files[relative] = {"kind": "symlink", "target": os.readlink(path)}
        elif stat.S_ISREG(info.st_mode):
            files[relative] = {
                "kind": "file", "size": info.st_size,
                "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
                "mode": stat.S_IMODE(info.st_mode),
            }

    # Do not descend through arbitrary root entries: data is a root symlink.
    for path in sorted(root.iterdir()):
        if path.name == ".git":
            continue
        if path.is_symlink() or path.is_file():
            add(path)
    for name in PROTECTED_TREES:
        tree = root / name
        if not tree.exists() or tree.is_symlink():
            continue
        for directory, subdirs, filenames in os.walk(tree, followlinks=False):
            directory_path = Path(directory)
            kept = []
            for name in sorted(subdirs):
                path = directory_path / name
                relative = path.relative_to(root).as_posix()
                if _excluded(relative):
                    continue
                if path.is_symlink():
                    add(path)
                else:
                    kept.append(name)
            subdirs[:] = kept
            for name in sorted(filenames):
                add(directory_path / name)
    return dict(sorted(files.items()))


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _implementation_paths(root: Path) -> tuple[str, ...]:
    """Every PC Python/shell source present at freezing, plus the real test."""
    discovered = {
        path.relative_to(root).as_posix()
        for path in (root / "results" / "step7").rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    }
    discovered.add("tests/test_step7.py")
    return tuple(sorted(discovered | set(IMPLEMENTATION_PATHS)))


def _frozen_input_paths() -> tuple[str, ...]:
    models = (
        f"trained/{label}_s{seed}.npz"
        for label in ("ego-novw-trig", "ego-novw-trig-pd")
        for seed in (*range(5), *range(100, 105))
    )
    return tuple(sorted((*FROZEN_SOURCE_INPUTS, *models)))


def capture_baseline(repo: Path | str | None = None) -> dict:
    """Create the immutable initial inventory; never replace an existing one.

    All regular files are hashed, including large rollout files.  A 510-second
    budget prevents a command approaching ten minutes; files left after that
    deadline retain full stat metadata and are explicitly marked unhashed.
    """
    root, output = _locations(repo)
    destination = output / BASELINE_NAME
    if destination.exists():
        raise RuntimeError(f"Baseline already exists; use integrity: {destination}")
    started = time.monotonic()
    inventory = _inventory(root)
    # Hash code, models, and compact tables first if the volume is unexpectedly
    # too large to finish.  This does not depend on any experimental outcome.
    order = sorted(inventory, key=lambda relative: (
        inventory[relative].get("size", 0) > 8 * 1024 * 1024,
        inventory[relative].get("size", 0), relative,
    ))
    hashed_bytes = 0
    omitted = []
    for relative in order:
        entry = inventory[relative]
        if entry["kind"] != "file":
            continue
        if time.monotonic() - started > 510:
            entry["sha256"] = None
            omitted.append(relative)
            continue
        path = root / relative
        entry["sha256"] = _file_sha(path)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
            entry["size"], entry["mtime_ns"], entry["ctime_ns"]
        ):
            raise RuntimeError(f"File changed while taking baseline: {relative}")
        hashed_bytes += entry["size"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "protected_trees": list(PROTECTED_TREES),
        "excluded": sorted(EXCLUDED_RELATIVE),
        "data_access": "none; data symlink target is not followed or inspected",
        "hash_algorithm": "sha256", "hashed_bytes": hashed_bytes,
        "unhashed_files": omitted, "elapsed_seconds": time.monotonic() - started,
        "files": inventory,
    }
    _exclusive_write(destination, _json_bytes(manifest))
    return manifest


def check_integrity(repo: Path | str | None = None) -> dict:
    """Verify contents, metadata, links and additions against the baseline."""
    root, output = _locations(repo)
    baseline_path = output / BASELINE_NAME
    manifest = json.loads(baseline_path.read_text(encoding="utf-8"))
    before = manifest["files"]
    current = _inventory(root)
    failures = []
    for relative in sorted(set(before) | set(current)):
        if relative not in current:
            failures.append(f"deleted: {relative}")
            continue
        if relative not in before:
            failures.append(f"unexpected new file: {relative}")
            continue
        expected = before[relative]
        actual = current[relative]
        metadata = {key: value for key, value in expected.items() if key != "sha256"}
        if actual != metadata:
            failures.append(f"metadata changed: {relative}")
        if expected.get("sha256") is not None:
            if _file_sha(root / relative) != expected["sha256"]:
                failures.append(f"sha256 changed: {relative}")
    if failures:
        raise RuntimeError("Pre-existing file integrity failed:\n" + "\n".join(failures))
    return {
        "ok": True, "baseline_sha256": _file_sha(baseline_path),
        "files_checked": len(before), "hashed_bytes": manifest["hashed_bytes"],
        "unhashed_files": manifest["unhashed_files"],
    }


def _section(plan: bytes) -> bytes:
    heading = SECTION_HEADING.encode("utf-8")
    start = plan.index(heading)
    end = plan.index(b"\n## 4.", start)
    # Preserve the source section byte for byte, including its final newline.
    return plan[start:end + 1]


def _report_section(section: bytes) -> bytes:
    return REPORT_HEADER.encode("utf-8") + b"\n\n" + section


def _report_copy(report: bytes, section: bytes) -> bytes:
    marker = REFERENCE_MARKER.encode("utf-8")
    if report.count(marker) != 1:
        raise RuntimeError("Expected exactly one complete References heading")
    return report.replace(marker, b"\n" + _report_section(section) + b"\n" + marker, 1)


def _frozen_text(manifest: dict, section: bytes) -> bytes:
    lines = [
        "实验 PC 预注册冻结记录", "",
        "created_utc: " + manifest["created_utc"],
        "time_sha256: " + manifest["time_sha256"],
        "plan_sha256: " + manifest["plan_sha256"],
        "section3_sha256: " + manifest["section3_sha256"],
        "report_original_sha256: " + manifest["report_original_sha256"],
        "report_copy_sha256: " + manifest["report_copy_sha256"],
        "report_policy: 原 report.txt 保持不变；插入结果为 report_PC_frozen.txt 副本。",
        "reason: 用户直接指令禁止修改任何已有文件，优先于计划 §2.3 的原文件插入要求。",
        "", "以下为冻结的报告节（含 §3 原文，逐字节保存）：", "",
    ]
    return "\n".join(lines).encode("utf-8") + _report_section(section)


def freeze(repo: Path | str | None = None) -> dict:
    """Freeze only after the caller has completed validation. Never overwrite."""
    root, output = _locations(repo)
    manifest_path = output / "prereg_PC_frozen.json"
    if manifest_path.exists():
        return check_freeze(repo=root, require_implementation=True)
    integrity = check_integrity(repo=root)
    plan = (root / "STEP7_PLAN_PROMPT.md").read_bytes()
    section = _section(plan)
    report = (root / "report.txt").read_bytes()
    report_copy = _report_copy(report, section)
    timestamp = datetime.now(timezone.utc).isoformat()
    time_bytes = (timestamp + "\n").encode("utf-8")
    implementations = {relative: _file_sha(root / relative) for relative in _implementation_paths(root)}
    frozen_inputs = {relative: _file_sha(root / relative) for relative in _frozen_input_paths()}
    manifest = {
        "schema_version": 1, "created_utc": timestamp,
        "time_sha256": _sha(time_bytes), "plan_sha256": _sha(plan),
        "section3_sha256": _sha(section), "section3_byte_length": len(section),
        "section_sha256": _sha(_report_section(section)),
        "report_header": REPORT_HEADER, "report_original_sha256": _sha(report),
        "report_copy_sha256": _sha(report_copy),
        "baseline_sha256": integrity["baseline_sha256"],
        "report_policy": "unchanged original; insert verbatim section in report_PC_frozen.txt copy",
        "deviations": ["§2.3 report.txt insertion is represented in a complete copy because user §1/no-existing-file-edits takes precedence"],
        "implementation_sha256": implementations,
        "frozen_input_sha256": frozen_inputs,
    }
    frozen = _frozen_text(manifest, section)
    manifest["frozen_sha256"] = _sha(frozen)
    artifacts = {
        "prereg_PC_time.txt": time_bytes,
        "report_PC_section.txt": _report_section(section),
        "report_PC_frozen.txt": report_copy,
        "prereg_PC_frozen.txt": frozen,
    }
    already = [name for name in artifacts if (output / name).exists()]
    if already:
        raise RuntimeError("Refusing to overwrite existing freeze artifacts: " + ", ".join(already))
    for name, content in artifacts.items():
        _exclusive_write(output / name, content)
    _exclusive_write(manifest_path, _json_bytes(manifest))
    return check_freeze(repo=root, require_implementation=True)


def _implementation_amendment(root: Path, output: Path, manifest: dict, failures: list) -> tuple[dict, dict | None]:
    """Validate the one permitted auditable implementation correction.

    Preregistration and frozen inputs remain checked against the original
    manifest.  This fixed record only supplies replacement implementation
    digests; it cannot add/remove source paths or replace the original freeze.
    """
    original = manifest["implementation_sha256"]
    path = output / "implementation_PC_amend1.json"
    digest_path = output / "implementation_PC_amend1.sha256"
    if not path.exists() and not digest_path.exists():
        return original, None
    if path.is_symlink() or digest_path.is_symlink() or not path.is_file() or not digest_path.is_file():
        failures.append("amend1 record and SHA256 sidecar must both be regular non-symlink files")
        return original, None
    raw = path.read_bytes()
    digest = _sha(raw)
    if digest_path.read_bytes() != (digest + "\n").encode("ascii"):
        failures.append("amend1 record SHA256")
        return original, None
    try:
        amendment = json.loads(raw)
        if not isinstance(amendment, dict):
            raise ValueError("record is not an object")
        if amendment.get("schema_version") != 1 or amendment.get("amendment_id") != "PC-amend1":
            raise ValueError("unknown amendment schema or identifier")
        expected_anchors = {key: manifest[key] for key in ("frozen_sha256", "plan_sha256", "section3_sha256")}
        if amendment.get("anchors") != expected_anchors:
            raise ValueError("original freeze/plan/§3 anchors differ")
        if amendment.get("before_implementation_sha256") != original:
            raise ValueError("before implementation map differs from original freeze")
        after = amendment.get("after_implementation_sha256")
        if not isinstance(after, dict) or set(after) != set(original):
            raise ValueError("after implementation map changes the source path list")
        if any(not isinstance(value, str) or len(value) != 64 or
               any(character not in "0123456789abcdef" for character in value)
               for value in after.values()):
            raise ValueError("after implementation map contains an invalid SHA256")
        reason = amendment.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason is required")
        amended_time = datetime.fromisoformat(amendment["created_utc"])
        if amended_time.tzinfo is None or amended_time < datetime.fromisoformat(manifest["created_utc"]):
            raise ValueError("amendment timestamp must include timezone and follow the original freeze")
        changed = {relative for relative in original if after[relative] != original[relative]}
        backups = amendment.get("backups")
        if not changed or not isinstance(backups, dict) or set(backups) != changed:
            raise ValueError("backups must cover exactly the changed implementation sources")
        for relative in sorted(changed):
            backup = backups[relative]
            if not isinstance(backup, dict) or backup.get("sha256") != original[relative]:
                raise ValueError("backup old SHA256 differs: " + relative)
            backup_relative = Path(backup["path"])
            backup_path = root / backup_relative
            if (backup_relative.is_absolute() or ".." in backup_relative.parts
                    or backup_path.suffix != ".txt" or backup_path.is_symlink()
                    or not backup_path.resolve().is_relative_to(output.resolve())
                    or not backup_path.is_file()):
                raise ValueError("backup must be a regular .txt within step7: " + relative)
            if _file_sha(backup_path) != original[relative]:
                raise ValueError("backup bytes differ from the frozen original: " + relative)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        failures.append("amend1 invalid: " + str(exc))
        return original, None
    metadata = {
        "amendment_id": "PC-amend1", "created_utc": amendment["created_utc"],
        "reason": reason, "sha256": digest,
        "path": path.relative_to(root).as_posix(), "changed_sources": sorted(changed),
    }
    return after, metadata


def check_freeze(require_implementation: bool = True, repo: Path | str | None = None) -> dict:
    """Check source preregistration, timestamp, all frozen artifacts and code."""
    root, output = _locations(repo)
    manifest = json.loads((output / "prereg_PC_frozen.json").read_text(encoding="utf-8"))
    plan = (root / "STEP7_PLAN_PROMPT.md").read_bytes()
    section = _section(plan)
    report = (root / "report.txt").read_bytes()
    failures = []

    def equal(label: str, actual, expected) -> None:
        if actual != expected:
            failures.append(label)

    equal("plan SHA256", _sha(plan), manifest["plan_sha256"])
    equal("§3 SHA256", _sha(section), manifest["section3_sha256"])
    equal("§3 length", len(section), manifest["section3_byte_length"])
    equal("original report unchanged", _sha(report), manifest["report_original_sha256"])
    expected_time = (manifest["created_utc"] + "\n").encode("utf-8")
    time_bytes = (output / "prereg_PC_time.txt").read_bytes()
    equal("timestamp text", time_bytes, expected_time)
    equal("timestamp SHA256", _sha(time_bytes), manifest["time_sha256"])
    report_section = (output / "report_PC_section.txt").read_bytes()
    equal("report section text", report_section, _report_section(section))
    equal("report section SHA256", _sha(report_section), manifest["section_sha256"])
    copy = (output / "report_PC_frozen.txt").read_bytes()
    equal("report copy text", copy, _report_copy(report, section))
    equal("report copy SHA256", _sha(copy), manifest["report_copy_sha256"])
    frozen = (output / "prereg_PC_frozen.txt").read_bytes()
    equal("frozen text", frozen, _frozen_text(manifest, section))
    equal("frozen SHA256", _sha(frozen), manifest["frozen_sha256"])
    equal("baseline SHA256", _file_sha(output / BASELINE_NAME), manifest["baseline_sha256"])
    effective_implementation, amendment = _implementation_amendment(root, output, manifest, failures)
    if require_implementation:
        equal("implementation path list", set(effective_implementation), set(_implementation_paths(root)))
        for relative, expected in effective_implementation.items():
            path = root / relative
            if path.is_symlink() or not path.is_file():
                failures.append("implementation missing: " + relative)
            elif _file_sha(path) != expected:
                failures.append("implementation SHA256: " + relative)
    equal("frozen input path list", set(manifest["frozen_input_sha256"]), set(_frozen_input_paths()))
    for relative, expected in manifest["frozen_input_sha256"].items():
        path = root / relative
        if not path.is_file():
            failures.append("frozen input missing: " + relative)
        elif _file_sha(path) != expected:
            failures.append("frozen input SHA256: " + relative)
    if failures:
        raise RuntimeError("PC freeze verification failed: " + "; ".join(failures))
    if amendment is not None:
        return {**manifest, "effective_implementation_sha256": effective_implementation,
                "implementation_amendment": amendment}
    return manifest


def _summary(value: dict) -> dict:
    return {key: item for key, item in value.items() if key not in {"files", "implementation_sha256", "effective_implementation_sha256", "frozen_input_sha256"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "freeze", "check", "integrity"))
    args = parser.parse_args()
    functions = {"capture": capture_baseline, "freeze": freeze, "check": check_freeze, "integrity": check_integrity}
    result = functions[args.command]()
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
