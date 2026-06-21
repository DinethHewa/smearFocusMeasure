from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
GENERATED = {"PACKAGE_MANIFEST.json", "FILE_INVENTORY.csv", "MANIFEST_SHA256.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_excluding(*names: str) -> list[Path]:
    excluded = set(names)
    return sorted(
        [path for path in ROOT.rglob("*") if path.is_file() and path.name not in excluded],
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def rebuild_metadata() -> None:
    manifest_path = ROOT / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = files_excluding(*GENERATED)
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["payload_file_count"] = len(payload)
    manifest["payload_total_size_bytes"] = sum(path.stat().st_size for path in payload)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    inventory_paths = files_excluding("FILE_INVENTORY.csv", "MANIFEST_SHA256.txt")
    with (ROOT / "FILE_INVENTORY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in inventory_paths:
            writer.writerow([path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)])

    checksum_paths = files_excluding("MANIFEST_SHA256.txt")
    with (ROOT / "MANIFEST_SHA256.txt").open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")


def build_archive() -> Path:
    archive_path = ROOT.parent / f"{ROOT.name}.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        for path in files_excluding():
            archive.write(path, arcname=(Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sidecar.write_text(f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild release inventory, checksums, and optional ZIP archive")
    parser.add_argument("--archive", action="store_true", help="Also rebuild the package ZIP and checksum sidecar")
    args = parser.parse_args()
    rebuild_metadata()
    print(f"Rebuilt metadata for {ROOT.name}")
    if args.archive:
        print(f"Built archive: {build_archive()}")


if __name__ == "__main__":
    main()
