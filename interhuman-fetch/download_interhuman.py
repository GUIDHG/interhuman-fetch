#!/usr/bin/env python3
"""Fetch InterHuman motions from public Google Drive by clip ID.

Self-contained on purpose: it only needs ``requests`` and the published
``profiles/interhuman.json`` (Drive ID + md5 + byte count per file), so it can
run on a GitHub Actions runner without the rest of the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DRIVE_UC = "https://drive.google.com/uc"
USER_AGENT = "Mozilla/5.0 (compatible; interhuman-fetch/1.0)"
_print_lock = threading.Lock()


def log(message):
    with _print_lock:
        print(message, flush=True)


def md5_of(path):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def motion_records(profile, clip_ids):
    """Pick one record per requested clip, ignoring Drive's "(1)" duplicates."""
    motions = {}
    for record in profile["files"]:
        path = record["path"]
        if not path.startswith("motions/") or not path.endswith(".pkl"):
            continue
        stem = Path(path).stem
        if "(" in stem:
            continue
        motions[stem] = record
    requested = list(dict.fromkeys(clip_ids))
    missing = [clip for clip in requested if clip not in motions]
    if missing:
        raise ValueError(f"Unknown clip IDs: {missing[:10]}")
    return [motions[clip] for clip in requested]


def helper_records(profile, clip_ids):
    """Split files, the licence and the annotations for the selected clips."""
    wanted_annots = {"annots/" + clip + ".txt" for clip in clip_ids}
    return [
        record
        for record in profile["files"]
        if record["path"] in wanted_annots
        or record["path"].startswith("split/")
        or record["path"] == "LICENSE.md"
    ]


def download_one(record, target_root, retries):
    """Fetch one file and verify the published md5 and size."""
    path = target_root / record["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_md5 = record.get("md5")
    expected_bytes = record.get("bytes")
    if path.is_file() and expected_bytes == path.stat().st_size:
        if expected_md5 is None or md5_of(path) == expected_md5:
            return record["path"], "cached"

    temporary = path.with_name(path.name + ".part")
    last_error = None
    for attempt in range(retries):
        if attempt:
            time.sleep(min(2 ** attempt, 30))
        try:
            offset = temporary.stat().st_size if temporary.is_file() else 0
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(
                DRIVE_UC,
                params={"export": "download", "id": record["google_drive_id"]},
                headers=headers,
                stream=True,
                timeout=(15, 120),
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    raise RuntimeError(
                        "Drive returned the HTML confirmation page "
                        f"(content-type={content_type})"
                    )
                append = response.status_code == 206 and offset > 0
                with temporary.open("ab" if append else "wb") as stream:
                    for block in response.iter_content(1 << 18):
                        if block:
                            stream.write(block)
            actual_bytes = temporary.stat().st_size
            if expected_bytes is not None and actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"size {actual_bytes} != published {expected_bytes}"
                )
            if expected_md5 is not None and md5_of(temporary) != expected_md5:
                raise RuntimeError("md5 differs from the published checksum")
            temporary.replace(path)
            return record["path"], "downloaded"
        except Exception as error:  # noqa: BLE001 - reported per file
            last_error = error
            if temporary.is_file() and temporary.stat().st_size and "md5" in str(
                error
            ):
                temporary.unlink()
    raise RuntimeError(f"{record['path']}: {type(last_error).__name__} {last_error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--clip-id", action="append", default=[], dest="clip_ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selection and total bytes without touching the network",
    )
    args = parser.parse_args(argv)

    clip_ids = list(args.clip_ids)
    if args.ids_file:
        clip_ids += [
            line.strip()
            for line in args.ids_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not clip_ids:
        parser.error("Provide --ids-file or --clip-id")
    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]

    profile = json.loads(args.profile.read_text())
    motions = motion_records(profile, clip_ids)
    helpers = helper_records(profile, clip_ids)
    total = sum(int(r.get("bytes") or 0) for r in motions + helpers)
    log(
        f"selection: {len(motions)} motions + {len(helpers)} helper files "
        f"({total / 2**20:.1f} MB)"
    )
    if args.dry_run:
        for record in motions[:5] + helpers[:6]:
            log(f"  {record['path']}  {record['bytes']} B")
        log(f"  ... {max(0, len(motions) + len(helpers) - 11)} more")
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    order = motions + helpers
    results, failures = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one, record, args.output, args.retries
            ): record
            for record in order
        }
        for index, future in enumerate(futures, 1):
            record = futures[future]
            try:
                name, status = future.result()
                results[name] = status
            except Exception as error:  # noqa: BLE001
                failures.append({"path": record["path"], "error": str(error)})
            if index % 20 == 0 or index == len(order):
                log(
                    f"progress {index}/{len(order)} failed={len(failures)}"
                )
    (args.output / "fetch-report.json").write_text(
        json.dumps(
            {
                "requested_clips": len(motions),
                "files": len(results),
                "failed": failures,
                "results": results,
            },
            indent=2,
        )
    )
    if failures:
        log(f"FAILED {len(failures)} file(s); see fetch-report.json")
        for row in failures[:10]:
            log(f"  {row['path']}: {row['error']}")
        return 1
    log("all files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
