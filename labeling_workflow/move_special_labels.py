import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


TARGETS = {
    "multiple": "MULTIPLE",
    "none": "NONE",
}

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return records



def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)



def resolve_image_path(base_dir: Path, record: Dict[str, Any]) -> Path | None:
    raw = record.get("local_image_path")
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()



def move_record_image(base_dir: Path, record: Dict[str, Any], dry_run: bool = False) -> Tuple[bool, str]:
    label = str(record.get("label", "")).strip().lower()
    if label not in TARGETS:
        return False, "label not targeted"

    source = resolve_image_path(base_dir, record)
    if source is None:
        return False, "missing local_image_path"
    if not source.exists():
        return False, f"missing file: {source}"

    target_dir = source.parent / TARGETS[label]
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source.name

    counter = 1
    while target_path.exists():
        target_path = target_dir / f"{source.stem}_{counter}{source.suffix}"
        counter += 1

    if not dry_run:
        shutil.move(str(source), str(target_path))
        try:
            record["local_image_path"] = str(target_path.relative_to(base_dir))
        except ValueError:
            record["local_image_path"] = str(target_path)

    return True, f"moved to {target_path}"



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move images whose label is MULTIPLE or NONE into subfolders and update local_image_path in the JSONL."
    )
    parser.add_argument("jsonl_path", help="Path to the JSONL file")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory used to resolve relative local_image_path values",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without changing anything")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()

    records = load_jsonl(jsonl_path)
    moved = 0
    skipped = 0

    for idx, record in enumerate(records, start=1):
        changed, message = move_record_image(base_dir, record, dry_run=args.dry_run)
        if changed:
            moved += 1
            print(f"[{idx}] {message}")
        else:
            skipped += 1

    if not args.dry_run:
        write_jsonl(jsonl_path, records)

    print(f"Done. moved={moved}, skipped={skipped}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
