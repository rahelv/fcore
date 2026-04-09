import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

st.set_page_config(page_title="JSONL Image Reviewer", layout="wide")

SPECIAL_FOLDERS = {"MULTIPLE", "NONE"}


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



def record_is_validated(record: Dict[str, Any]) -> bool:
    value = record.get("validated", False)
    return bool(value)



def normalize_image_path(base_dir: Path, record: Dict[str, Any]) -> Optional[Path]:
    raw = record.get("local_image_path")
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()



def get_visible_indices(records: List[Dict[str, Any]], only_unvalidated: bool) -> List[int]:
    indices = []
    for i, record in enumerate(records):
        if only_unvalidated and record_is_validated(record):
            continue
        indices.append(i)
    return indices



def move_relative(visible_indices: List[int], current_real_index: int, step: int) -> int:
    if not visible_indices:
        return 0
    try:
        pos = visible_indices.index(current_real_index)
    except ValueError:
        return visible_indices[0]
    new_pos = max(0, min(len(visible_indices) - 1, pos + step))
    return visible_indices[new_pos]



def ensure_current_index(records: List[Dict[str, Any]], visible_indices: List[int]) -> None:
    if "current_index" not in st.session_state:
        st.session_state.current_index = visible_indices[0] if visible_indices else 0
        return
    if not records:
        st.session_state.current_index = 0
        return
    if visible_indices:
        if st.session_state.current_index not in visible_indices:
            st.session_state.current_index = visible_indices[0]
    else:
        st.session_state.current_index = 0



def refresh_editor_state(record: Dict[str, Any], real_index: int) -> None:
    if st.session_state.get("editor_record_index") != real_index:
        st.session_state.editor_record_index = real_index
        st.session_state.label_input = record.get("label", "")
        st.session_state.franchise_input = record.get("franchise", "")
        st.session_state.validated_input = bool(record.get("validated", False))



def save_records_and_advance(
    jsonl_path: Path,
    records: List[Dict[str, Any]],
    current_real_index: int,
    next_real_index: Optional[int] = None,
) -> None:
    write_jsonl(jsonl_path, records)
    if next_real_index is not None:
        st.session_state.current_index = next_real_index



def find_images_root(image_path: Path) -> Path:
    current = image_path.parent
    while True:
        if current.name == "images":
            return current
        if current.parent == current:
            return image_path.parent
        current = current.parent



def updated_local_image_path(raw_path: str, dest_abs: Path, base_dir: Path) -> str:
    raw = Path(raw_path)
    if raw.is_absolute():
        return str(dest_abs)

    parts = list(raw.parts)
    if "images" in parts:
        images_idx = parts.index("images")
        prefix = parts[: images_idx + 1]
        tail = list(dest_abs.relative_to(find_images_root(dest_abs)).parts)
        return str(Path(*prefix, *tail))

    try:
        return str(dest_abs.relative_to(base_dir))
    except ValueError:
        return str(dest_abs)



def move_record_to_label_folder(
    jsonl_path: Path,
    base_dir: Path,
    records: List[Dict[str, Any]],
    real_index: int,
    target_label: str,
) -> None:
    if target_label not in SPECIAL_FOLDERS:
        raise ValueError(f"Unsupported target folder: {target_label}")

    record = records[real_index]
    image_path = normalize_image_path(base_dir, record)
    if image_path is None:
        raise ValueError("This record has no local_image_path")
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    images_root = find_images_root(image_path)
    dest_dir = images_root / target_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / image_path.name

    if image_path.resolve() != dest_path.resolve():
        if dest_path.exists():
            raise FileExistsError(f"Destination already exists: {dest_path}")
        shutil.move(str(image_path), str(dest_path))

    record["label"] = target_label
    record["local_image_path"] = updated_local_image_path(
        str(record.get("local_image_path", "")), dest_path, base_dir
    )
    record["franchise"] = st.session_state.get("franchise_input", record.get("franchise", ""))
    record["validated"] = False
    write_jsonl(jsonl_path, records)



def delete_record_and_image(
    jsonl_path: Path,
    records: List[Dict[str, Any]],
    real_index: int,
    image_path: Optional[Path],
) -> None:
    if image_path and image_path.exists() and image_path.is_file():
        image_path.unlink()
    del records[real_index]
    write_jsonl(jsonl_path, records)



def go_to_next_after_action(records_count_after: int, previous_real_index: int) -> None:
    if records_count_after <= 0:
        st.session_state.current_index = 0
    else:
        st.session_state.current_index = min(previous_real_index, records_count_after - 1)



def handle_validate() -> None:
    real_index = st.session_state.current_index
    jsonl_path = Path(st.session_state.jsonl_path_value)
    base_dir = Path(st.session_state.base_dir_value)
    only_unvalidated = st.session_state.only_unvalidated_value

    records = load_jsonl(jsonl_path)
    visible_indices = get_visible_indices(records, only_unvalidated)
    next_real_index = move_relative(visible_indices, real_index, 1) if visible_indices else real_index

    records[real_index]["label"] = st.session_state.get("label_input", "").strip()
    records[real_index]["franchise"] = st.session_state.get("franchise_input", "").strip()
    records[real_index]["validated"] = True
    save_records_and_advance(jsonl_path, records, real_index, next_real_index)

    if only_unvalidated:
        refreshed = load_jsonl(jsonl_path)
        refreshed_visible = get_visible_indices(refreshed, only_unvalidated)
        if refreshed_visible:
            if st.session_state.current_index not in refreshed_visible:
                current = st.session_state.current_index
                candidates = [idx for idx in refreshed_visible if idx >= current]
                st.session_state.current_index = candidates[0] if candidates else refreshed_visible[-1]
        else:
            st.session_state.current_index = 0


st.markdown(
    """
    <style>
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #b91c1c;
        color: white;
        border: 1px solid #b91c1c;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #991b1b;
        color: white;
        border: 1px solid #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("JSONL Image Reviewer")
st.caption("Review each image from local_image_path and verify label/franchise directly in the JSONL file.")

with st.sidebar:
    st.header("Setup")
    jsonl_input = st.text_input(
        "Path to JSONL file",
        value=st.session_state.get("jsonl_path_value", "metadata.jsonl"),
        help="Example: /home/ubuntu/data/commons_cosplay_dataset/metadata/metadata.jsonl",
        key="jsonl_path_value",
    )
    base_dir_input = st.text_input(
        "Base directory for relative image paths",
        value=st.session_state.get("base_dir_value", "."),
        help="If local_image_path is relative, it will be resolved from here.",
        key="base_dir_value",
    )
    only_unvalidated = st.checkbox(
        "Show only unvalidated records",
        value=st.session_state.get("only_unvalidated_value", False),
        key="only_unvalidated_value",
    )
    st.info("Checking validated marks the current row as true in the JSONL and jumps to the next image.")

jsonl_path = Path(jsonl_input).expanduser().resolve()
base_dir = Path(base_dir_input).expanduser().resolve()

if not jsonl_path.exists():
    st.warning("Enter a valid JSONL path in the sidebar to begin.")
    st.stop()

try:
    records = load_jsonl(jsonl_path)
except Exception as e:
    st.error(f"Could not load JSONL: {e}")
    st.stop()

if not records:
    st.info("The JSONL file is empty.")
    st.stop()

visible_indices = get_visible_indices(records, only_unvalidated)
ensure_current_index(records, visible_indices)

if not visible_indices:
    st.success("No records match the current filter.")
    st.stop()

real_index = st.session_state.current_index
record = records[real_index]
refresh_editor_state(record, real_index)
image_path = normalize_image_path(base_dir, record)
current_pos = visible_indices.index(real_index) + 1

top1, top2, top3, top4 = st.columns([1, 1, 1, 2])
with top1:
    if st.button("⟵ Previous", use_container_width=True, type="primary"):

        st.session_state.current_index = move_relative(visible_indices, real_index, -1)
        st.rerun()
with top2:
    if st.button("Next ⟶", use_container_width=True, type="primary"):

        st.session_state.current_index = move_relative(visible_indices, real_index, 1)
        st.rerun()
with top3:
    jump_to = st.number_input(
        f"Record ({current_pos}/{len(visible_indices)})",
        min_value=1,
        max_value=len(visible_indices),
        value=current_pos,
        step=1,
    )
    if jump_to != current_pos:
        st.session_state.current_index = visible_indices[jump_to - 1]
        st.rerun()
with top4:
    st.write("")
    st.write(f"**Title:** {record.get('title', '')}")
    st.write(f"**Image path:** `{record.get('local_image_path', '')}`")

left, right = st.columns([3, 2])

with left:
    st.subheader("Image")
    if image_path is None:
        st.error("This record has no local_image_path.")
    elif not image_path.exists():
        st.error(f"Image file not found: {image_path}")
    else:
        st.image(str(image_path), use_container_width=True)

with right:
    st.subheader("Review fields")
    st.text_input("label", key="label_input")
    st.text_input("franchise", key="franchise_input")
    st.checkbox(
        "validated",
        key="validated_input",
        on_change=handle_validate,
        help="When you check this, the app writes validated=true and jumps to the next image.",
    )

    with st.expander("More context"):
        st.json(
            {
                "costume_hint": record.get("costume_hint", ""),
                "image_description": record.get("image_description", ""),
                "object_name": record.get("object_name", ""),
                "categories": record.get("categories", []),
                "file_page_url": record.get("file_page_url", ""),
                "image_url": record.get("image_url", ""),
            },
            expanded=False,
        )

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button('Move to "MULTIPLE"', use_container_width=True, type="primary"):

            try:
                records[real_index]["franchise"] = st.session_state.get("franchise_input", "").strip()
                move_record_to_label_folder(jsonl_path, base_dir, records, real_index, "MULTIPLE")
                st.session_state.label_input = "MULTIPLE"
                st.session_state.validated_input = False
                st.success("Moved image to MULTIPLE and updated JSONL.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not move to MULTIPLE: {e}")

    with c2:
        if st.button('Move to "NONE"', use_container_width=True, type="primary"):

            try:
                records[real_index]["franchise"] = st.session_state.get("franchise_input", "").strip()
                move_record_to_label_folder(jsonl_path, base_dir, records, real_index, "NONE")
                st.session_state.label_input = "NONE"
                st.session_state.validated_input = False
                st.success("Moved image to NONE and updated JSONL.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not move to NONE: {e}")

    with c3:
        if st.button("Delete", use_container_width=True, type="secondary"):
            try:
                delete_record_and_image(jsonl_path, records, real_index, image_path)
                go_to_next_after_action(len(records), real_index)
                st.success("Image and JSONL row deleted.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not delete: {e}")

st.divider()
with st.expander("Current raw JSONL record"):
    st.json(record, expanded=False)
