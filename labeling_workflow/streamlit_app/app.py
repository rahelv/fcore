import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import streamlit as st

st.set_page_config(page_title="JSONL Image Reviewer", layout="wide")

SPECIAL_FOLDERS = {"MULTIPLE", "NONE"}


class ReviewerError(Exception):
    pass


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ReviewerError(f"Invalid JSON on line {line_no}: {e}") from e
            if not isinstance(item, dict):
                raise ReviewerError(f"Line {line_no} is not a JSON object.")
            records.append(item)
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def record_is_validated(record: Dict[str, Any]) -> bool:
    return bool(record.get("validated", False))


def normalize_image_path(base_dir: Path, record: Dict[str, Any]) -> Optional[Path]:
    raw = record.get("local_image_path")
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def get_visible_indices(
    records: List[Dict[str, Any]], only_unvalidated: bool
) -> List[int]:
    return [
        i
        for i, record in enumerate(records)
        if not only_unvalidated or not record_is_validated(record)
    ]


def move_relative(
    visible_indices: List[int], current_real_index: int, step: int
) -> int:
    if not visible_indices:
        return 0
    try:
        pos = visible_indices.index(current_real_index)
    except ValueError:
        return visible_indices[0]
    new_pos = max(0, min(len(visible_indices) - 1, pos + step))
    return visible_indices[new_pos]


def choose_best_index(visible_indices: List[int], preferred_index: int) -> int:
    if not visible_indices:
        return 0
    for idx in visible_indices:
        if idx >= preferred_index:
            return idx
    return visible_indices[-1]


def ensure_current_index(
    records: List[Dict[str, Any]], visible_indices: List[int]
) -> None:
    if not records:
        st.session_state.current_index = 0
        return

    if "current_index" not in st.session_state:
        st.session_state.current_index = visible_indices[0] if visible_indices else 0
        return

    current_index = int(st.session_state.current_index)
    if current_index < 0 or current_index >= len(records):
        st.session_state.current_index = visible_indices[0] if visible_indices else 0
        return

    if visible_indices and current_index not in visible_indices:
        st.session_state.current_index = choose_best_index(
            visible_indices, current_index
        )
    elif not visible_indices:
        st.session_state.current_index = 0


def reset_editor_binding() -> None:
    st.session_state.pop("editor_record_index", None)


def refresh_editor_state(record: Dict[str, Any], real_index: int) -> None:
    if st.session_state.get("editor_record_index") != real_index:
        st.session_state.editor_record_index = real_index
        st.session_state.label_input = str(record.get("label", ""))
        st.session_state.franchise_input = str(record.get("franchise", ""))
        st.session_state.validated_input = bool(record.get("validated", False))


def apply_editor_state_to_record(record: Dict[str, Any]) -> None:
    record["label"] = st.session_state.get("label_input", "").strip()
    record["franchise"] = st.session_state.get("franchise_input", "").strip()
    record["validated"] = bool(st.session_state.get("validated_input", False))


def build_google_search_url(label: str, franchise: str) -> Optional[str]:
    character_name = (label or "").strip()
    franchise_name = (franchise or "").strip()

    if not character_name:
        return None

    if franchise_name and franchise_name.lower() != "unknown":
        query = f"{character_name} {franchise_name}"
    else:
        query = character_name

    return f"https://www.google.com/search?q={quote_plus(query)}"


def position_after_reload(
    jsonl_path: Path, only_unvalidated: bool, preferred_index: int
) -> None:
    records = load_jsonl(jsonl_path)
    visible_indices = get_visible_indices(records, only_unvalidated)
    st.session_state.current_index = choose_best_index(visible_indices, preferred_index)
    reset_editor_binding()


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


def persist_current_record(
    jsonl_path: Path, records: List[Dict[str, Any]], real_index: int
) -> None:
    if real_index < 0 or real_index >= len(records):
        raise ReviewerError("The selected record no longer exists.")
    apply_editor_state_to_record(records[real_index])
    write_jsonl(jsonl_path, records)


def get_sidecar_jsonl_path(jsonl_path: Path, target_name: str) -> Path:
    return jsonl_path.parent / target_name


def move_record_to_label_folder_and_jsonl(
    jsonl_path: Path,
    base_dir: Path,
    records: List[Dict[str, Any]],
    real_index: int,
    target_label: str,
) -> None:
    if target_label not in SPECIAL_FOLDERS:
        raise ReviewerError(f"Unsupported target folder: {target_label}")
    if real_index < 0 or real_index >= len(records):
        raise ReviewerError("The selected record no longer exists.")

    record = records[real_index]
    apply_editor_state_to_record(record)

    image_path = normalize_image_path(base_dir, record)
    if image_path is None:
        raise ReviewerError("This record has no local_image_path.")
    if not image_path.exists() or not image_path.is_file():
        raise ReviewerError(f"Image not found: {image_path}")

    images_root = find_images_root(image_path)
    dest_dir = images_root / target_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / image_path.name

    if image_path.resolve() != dest_path.resolve():
        if dest_path.exists():
            raise ReviewerError(f"Destination already exists: {dest_path}")
        shutil.move(str(image_path), str(dest_path))

    record["label"] = target_label
    record["local_image_path"] = updated_local_image_path(
        str(record.get("local_image_path", "")), dest_path, base_dir
    )
    record["validated"] = False

    target_jsonl = get_sidecar_jsonl_path(
        jsonl_path,
        "multiple.jsonl" if target_label == "MULTIPLE" else "none.jsonl",
    )
    append_jsonl_record(target_jsonl, record)

    del records[real_index]
    write_jsonl(jsonl_path, records)


def move_record_to_validated_jsonl(
    jsonl_path: Path,
    records: List[Dict[str, Any]],
    real_index: int,
) -> None:
    if real_index < 0 or real_index >= len(records):
        raise ReviewerError("The selected record no longer exists.")

    record = records[real_index]
    apply_editor_state_to_record(record)
    record["validated"] = True

    target_jsonl = get_sidecar_jsonl_path(jsonl_path, "validated.jsonl")
    append_jsonl_record(target_jsonl, record)

    del records[real_index]
    write_jsonl(jsonl_path, records)


def delete_record_and_image(
    jsonl_path: Path,
    records: List[Dict[str, Any]],
    real_index: int,
    image_path: Optional[Path],
) -> None:
    if real_index < 0 or real_index >= len(records):
        raise ReviewerError("The selected record no longer exists.")

    if image_path and image_path.exists() and image_path.is_file():
        image_path.unlink()

    del records[real_index]
    write_jsonl(jsonl_path, records)


def handle_validated_toggle() -> None:
    jsonl_path = Path(st.session_state.jsonl_path_value).expanduser().resolve()
    only_unvalidated = bool(st.session_state.only_unvalidated_value)
    real_index = int(st.session_state.current_index)

    records = load_jsonl(jsonl_path)
    if real_index < 0 or real_index >= len(records):
        raise ReviewerError("The selected record no longer exists.")

    if bool(st.session_state.validated_input):
        next_index = choose_best_index(
            get_visible_indices(records, only_unvalidated),
            real_index,
        )
        move_record_to_validated_jsonl(jsonl_path, records, real_index)
        position_after_reload(jsonl_path, only_unvalidated, next_index)
    else:
        apply_editor_state_to_record(records[real_index])
        write_jsonl(jsonl_path, records)
        position_after_reload(jsonl_path, only_unvalidated, real_index)


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
st.caption(
    "Review each image from local_image_path and verify label/franchise directly in the JSONL file."
)

with st.sidebar:
    st.header("Setup")
    jsonl_input = st.text_input(
        "Path to JSONL file",
        value=st.session_state.get(
            "jsonl_path_value",
            "/Users/rahel/Code/FS26/playground/data/commons_cosplay_dataset/metadata/split/rest.jsonl",
        ),
        help="Example: /home/ubuntu/data/commons_cosplay_dataset/metadata/metadata.jsonl",
        key="jsonl_path_value",
    )
    base_dir_input = st.text_input(
        "Base directory for relative image paths",
        value=st.session_state.get(
            "base_dir_value", "/Users/rahel/Code/FS26/playground/data"
        ),
        help="If local_image_path is relative, it will be resolved from here.",
        key="base_dir_value",
    )
    only_unvalidated = st.checkbox(
        "Show only unvalidated records",
        value=st.session_state.get("only_unvalidated_value", False),
        key="only_unvalidated_value",
    )
    st.info(
        "Checking validated moves the row to validated.jsonl and moves on. Unchecking validated saves validated=false and stays here."
    )

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

real_index = int(st.session_state.current_index)
record = records[real_index]
refresh_editor_state(record, real_index)
image_path = normalize_image_path(base_dir, record)
current_pos = visible_indices.index(real_index) + 1

top1, top2, top3, top4 = st.columns([1, 1, 1, 2])
with top1:
    if st.button("⟵ Previous", use_container_width=True, type="primary"):
        st.session_state.current_index = move_relative(visible_indices, real_index, -1)
        reset_editor_binding()
        st.rerun()
with top2:
    if st.button("Next ⟶", use_container_width=True, type="primary"):
        st.session_state.current_index = move_relative(visible_indices, real_index, 1)
        reset_editor_binding()
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
        reset_editor_binding()
        st.rerun()
with top4:
    st.write("")
    st.write(f"**Title:** {record.get('title', '')}")
    st.write(f"**Image path:** `{record.get('local_image_path', '')}`")

left, right = st.columns([2, 2])

with left:
    st.subheader("Image")
    if image_path is None:
        st.error("This record has no local_image_path.")
    elif not image_path.exists():
        st.error(f"Image file not found: {image_path}")
    else:
        st.image(str(image_path), width=500)

with right:
    st.subheader("Review fields")
    st.text_input("label", key="label_input")
    st.text_input("franchise", key="franchise_input")

    google_search_url = build_google_search_url(
        st.session_state.get("label_input", ""),
        st.session_state.get("franchise_input", ""),
    )
    if google_search_url:
        st.markdown(f"[Open Google search]({google_search_url})")

    st.checkbox(
        "validated",
        key="validated_input",
        on_change=handle_validated_toggle,
        help="When checked, the app moves the current row to validated.jsonl and shows the next applicable record. When unchecked, it saves validated=false and stays here.",
    )

    save1, save2 = st.columns(2)
    with save1:
        if st.button("Save", use_container_width=True):
            try:
                persist_current_record(jsonl_path, records, real_index)
                position_after_reload(jsonl_path, only_unvalidated, real_index)
                st.rerun()
            except Exception as e:
                st.error(f"Could not save: {e}")
    with save2:
        if st.button("Save + Next", use_container_width=True):
            try:
                persist_current_record(jsonl_path, records, real_index)
                next_index = move_relative(visible_indices, real_index, 1)
                position_after_reload(jsonl_path, only_unvalidated, next_index)
                st.rerun()
            except Exception as e:
                st.error(f"Could not save and move on: {e}")

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
                next_index = choose_best_index(visible_indices, real_index)
                move_record_to_label_folder_and_jsonl(
                    jsonl_path, base_dir, records, real_index, "MULTIPLE"
                )
                position_after_reload(jsonl_path, only_unvalidated, next_index)
                st.rerun()
            except Exception as e:
                st.error(f"Could not move to MULTIPLE: {e}")

    with c2:
        if st.button('Move to "NONE"', use_container_width=True, type="primary"):
            try:
                next_index = choose_best_index(visible_indices, real_index)
                move_record_to_label_folder_and_jsonl(
                    jsonl_path, base_dir, records, real_index, "NONE"
                )
                position_after_reload(jsonl_path, only_unvalidated, next_index)
                st.rerun()
            except Exception as e:
                st.error(f"Could not move to NONE: {e}")

    with c3:
        if st.button("Delete", use_container_width=True, type="secondary"):
            try:
                delete_record_and_image(jsonl_path, records, real_index, image_path)
                position_after_reload(jsonl_path, only_unvalidated, real_index)
                st.rerun()
            except Exception as e:
                st.error(f"Could not delete: {e}")

st.divider()
with st.expander("Current raw JSONL record"):
    st.json(record, expanded=False)
