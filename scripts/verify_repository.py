"""Validate the committed experiment data and repository hygiene."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "ms640_final_paper_v1"
ALGORITHMS = (
    "single_auto_cov_dual",
    "auto_dual",
    "auto_nonauto_cov_triple",
)
STALE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dylib", ".log", ".pid"}
SECRET_PATTERNS = (
    re.compile(r"WANDB_API_KEY\s*=\s*['\"][^'\"]+", re.IGNORECASE),
    re.compile(r"GITHUB_TOKEN\s*=\s*['\"][^'\"]+", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
)
ATTRIBUTION_MARKERS = {
    "CITATION.cff": ("Elliot", "Steffen", "ACSolverES"),
    "LICENSE": ("Copyright (c) 2026 Elliot Steffen", "MIT License"),
    "LICENSE-DATA": ("CC BY 4.0", "Math & AI Lab at Caltech"),
    "THIRD_PARTY_NOTICES.md": (
        "ACSolverX",
        "Copyright (c) 2026 Math & AI Lab at Caltech",
        "10.1090/conm/250/03848",
        "arXiv:2606.21611",
        "arXiv:2408.15332",
    ),
    "data/README.md": ("gssub_baseline_ms640.csv", "external benchmark material"),
    "analysis/unsolved_124/data/README.md": (
        "124",
        "D319DDF4C5E1449377C2BE867DA48110C21EC2E75FC9E8985DA53833599CF914",
    ),
}


def fail(message: str) -> None:
    raise AssertionError(message)


def rows(path: Path) -> list[dict[str, str]]:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2_147_483_647)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_final_results() -> None:
    baseline_path = ROOT / "data" / "gssub_baseline_ms640.csv"
    baseline = rows(baseline_path)
    if len(baseline) != 640:
        fail(f"baseline has {len(baseline)} rows, expected 640")
    baseline_by_id = {row["pres_id"]: row for row in baseline}
    if len(baseline_by_id) != 640:
        fail("baseline presentation IDs are not unique")

    config = json.loads((RESULT_DIR / "config.json").read_text(encoding="utf-8"))
    if Path(config["input_csv"]).is_absolute():
        fail("config.json contains a stale absolute input path")
    if config["input_sha256"] != sha256(baseline_path):
        fail("baseline checksum does not match config.json")
    if config["presentation_ids"] != [row["pres_id"] for row in baseline]:
        fail("config presentation order differs from the baseline")

    checksum_path = RESULT_DIR / "CHECKSUMS.sha256"
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        expected, relative = line.split(maxsplit=1)
        artifact = ROOT / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            fail(f"checksum mismatch: {relative}")

    for algorithm in ALGORITHMS:
        path = RESULT_DIR / f"{algorithm}.csv"
        result = rows(path)
        if len(result) != 640:
            fail(f"{path.name} has {len(result)} rows, expected 640")
        ids = [row["pres_id"] for row in result]
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        if duplicates:
            fail(f"{path.name} contains duplicate IDs: {duplicates[:5]}")
        if set(ids) != set(baseline_by_id):
            fail(f"{path.name} does not contain the baseline ID set")

        for row in result:
            source = baseline_by_id[row["pres_id"]]
            if (row["r1"], row["r2"]) != (source["r1"], source["r2"]):
                fail(f"{path.name} input mismatch at pres_id={row['pres_id']}")
            if row["status"] != "ok" or row["solved"].lower() != "true" or row["error"].strip():
                fail(f"{path.name} records a failed case at pres_id={row['pres_id']}")
            path_data = json.loads(row["path_json"])
            moves_data = json.loads(row["moves_json"])
            if int(row["path_length"]) != len(path_data) - 1:
                fail(f"{path.name} path length mismatch at pres_id={row['pres_id']}")
            if len(moves_data) != int(row["path_length"]):
                fail(f"{path.name} move count mismatch at pres_id={row['pres_id']}")


def verify_unsolved_analysis() -> None:
    analysis = ROOT / "analysis" / "unsolved_124"
    source = rows(analysis / "data" / "wandb_unsolved124_dual_triple.csv")
    labels = Counter(row["label"] for row in source)
    if len(labels) != 124 or set(labels.values()) != {2}:
        fail("unsolved-124 source is not exactly two rows for each of 124 labels")
    paired = rows(analysis / "outputs" / "processed_paired_results.csv")
    if len(paired) != 124:
        fail(f"processed unsolved analysis has {len(paired)} rows, expected 124")
    manifest = json.loads((analysis / "outputs" / "artifact_manifest.json").read_text(encoding="utf-8"))
    if Path(manifest["input_csv"]).is_absolute():
        fail("unsolved analysis manifest contains a stale absolute input path")
    for relative in (*manifest["tables"], *manifest["figures"]):
        if not (analysis / relative).is_file():
            fail(f"unsolved analysis artifact is missing: {relative}")


def verify_attribution() -> None:
    for relative, markers in ATTRIBUTION_MARKERS.items():
        path = ROOT / relative
        if not path.is_file():
            fail(f"required attribution file is missing: {relative}")
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            fail(f"{relative} is missing attribution markers: {missing}")


def verify_hygiene() -> None:
    stale = []
    oversized = []
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        paths = (ROOT / name for name in sorted(set(result.stdout.split("\0"))) if name)
    else:
        paths = ROOT.rglob("*")
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in STALE_SUFFIXES or "__pycache__" in path.parts:
            stale.append(str(relative))
        if path.stat().st_size >= 90 * 1024 * 1024:
            oversized.append(str(relative))
        if path.suffix.lower() in {".py", ".md", ".toml", ".json", ".ipynb", ".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    fail(f"possible embedded secret in {relative}")
    if stale:
        fail(f"stale runtime/build files present: {stale[:10]}")
    if oversized:
        fail(f"files require Git LFS or removal: {oversized}")


def main() -> None:
    verify_final_results()
    verify_unsolved_analysis()
    verify_attribution()
    verify_hygiene()
    print("Repository verification passed.")
    print("Final benchmark: 640 aligned, unique, solved rows per algorithm.")
    print("Unsolved analysis: 124 paired presentations and all artifacts present.")
    print("Attribution: software, upstream code, and dataset notices are present.")
    print("Hygiene: publishable files contain no stale binaries/logs/caches, embedded secrets, or >=90 MiB files.")


if __name__ == "__main__":
    main()
