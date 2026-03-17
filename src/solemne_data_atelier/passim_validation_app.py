from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class Paths:
    project_root: Path
    task_dir: Path
    problems_dir: Path
    solutions_dir: Path
    solutions_add_dir: Path
    solutions_merged_dir: Path
    passim_runs_dir: Path


def find_project_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "task").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing data/task")


def get_paths() -> Paths:
    root = find_project_root()
    task = root / "data" / "task"
    return Paths(
        project_root=root,
        task_dir=task,
        problems_dir=task / "problems",
        solutions_dir=task / "solutions",
        solutions_add_dir=task / "solutions_add",
        solutions_merged_dir=task / "solutions_merged",
        passim_runs_dir=root / "passim_runs",
    )


def normalize_reference(ref: str) -> str:
    return str(ref or "").strip().lower()


def parse_resolved_references(value: Any) -> List[str]:
    if isinstance(value, list):
        refs = [normalize_reference(x) for x in value if str(x).strip()]
    elif isinstance(value, str):
        chunks = re.split(r"[,\n;\s]+", value)
        refs = [normalize_reference(x) for x in chunks if str(x).strip()]
    else:
        refs = []
    return sorted(set([x for x in refs if x]))


def canonical_solution_entry(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        span_start = int(raw.get("span_start"))
        span_end = int(raw.get("span_end"))
    except Exception:
        return None

    refs = parse_resolved_references(raw.get("resolved_references", []))
    if not refs:
        return None
    if span_end <= span_start:
        return None

    return {
        "span_start": span_start,
        "span_end": span_end,
        "resolved_references": refs,
    }


def dedupe_solution_entries(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[int, int, Tuple[str, ...]]] = set()
    out: List[Dict[str, Any]] = []
    for row in entries:
        canon = canonical_solution_entry(row)
        if canon is None:
            continue
        key = (
            int(canon["span_start"]),
            int(canon["span_end"]),
            tuple(canon["resolved_references"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(canon)
    out.sort(key=lambda x: (int(x["span_start"]), int(x["span_end"]), ",".join(x["resolved_references"])))
    return out


def load_solution_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return dedupe_solution_entries(payload)


def save_solution_file(path: Path, entries: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dedupe_solution_entries(entries)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def list_problem_ids(problems_dir: Path) -> List[str]:
    if not problems_dir.exists():
        return []
    return sorted([p.stem for p in problems_dir.glob("*.txt")])


def list_solution_ids(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    return sorted([p.stem for p in folder.glob("*.json")])


def load_problem_text(problems_dir: Path, problem_id: str) -> str:
    p = problems_dir / f"{problem_id}.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


@st.cache_data(show_spinner=False)
def _load_visual_matches_cached(path_str: str, mtime_ns: int) -> List[Dict[str, Any]]:
    _ = mtime_ns
    path = Path(path_str)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return payload


@st.cache_data(show_spinner=False)
def _load_task_chunks_cached(path_str: str, mtime_ns: int) -> Dict[str, Dict[str, Any]]:
    _ = mtime_ns
    path = Path(path_str)
    rows = _read_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid:
            out[rid] = row
    return out


def load_visual_matches(run_dir: Path) -> List[Dict[str, Any]]:
    p = run_dir / "reshaped" / "visual_matches.json"
    if not p.exists():
        return []
    return _load_visual_matches_cached(str(p), p.stat().st_mtime_ns)


def load_task_chunk_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    p = run_dir / "input" / "task_chunked.jsonl"
    if not p.exists():
        return {}
    return _load_task_chunks_cached(str(p), p.stat().st_mtime_ns)


def list_run_dirs(passim_runs_dir: Path) -> List[Path]:
    if not passim_runs_dir.exists():
        return []
    return sorted([p for p in passim_runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def render_highlighted_slice_html(text: str, start: int, end: int, color: str = "#fff59d") -> str:
    t = str(text or "")
    n = len(t)
    a = max(0, min(n, int(start)))
    b = max(a, min(n, int(end)))
    return f"{escape(t[:a])}<span style='background:{color};'>{escape(t[a:b])}</span>{escape(t[b:])}"


def render_text_with_spans_html(
    text: str,
    spans: Sequence[Tuple[int, int, str, str]],
) -> str:
    """
    Render full text with multiple highlighted spans.
    spans: [(start, end, label, color), ...]
    """
    t = str(text or "")
    n = len(t)
    clean: List[Tuple[int, int, str, str]] = []
    for s, e, label, color in spans:
        a = max(0, min(n, int(s)))
        b = max(a, min(n, int(e)))
        if b > a:
            clean.append((a, b, str(label or ""), str(color or "#fff59d")))
    clean.sort(key=lambda x: (x[0], x[1]))

    cursor = 0
    chunks: List[str] = []
    for s, e, label, color in clean:
        if s < cursor:
            s = cursor
        if e <= s:
            continue
        if s > cursor:
            chunks.append(escape(t[cursor:s]))
        snippet = escape(t[s:e])
        if label:
            snippet = f"{snippet}<sup style='font-size:10px'>{escape(label)}</sup>"
        chunks.append(f"<span style='background:{color};'>{snippet}</span>")
        cursor = e
    if cursor < n:
        chunks.append(escape(t[cursor:]))
    return "".join(chunks)


def build_merged_solutions(
    solutions_dir: Path,
    solutions_add_dir: Path,
    solutions_merged_dir: Path,
) -> Dict[str, int]:
    solution_ids = set(list_solution_ids(solutions_dir))
    addition_ids = set(list_solution_ids(solutions_add_dir))
    all_ids = sorted(solution_ids | addition_ids)

    written = 0
    total_rows = 0

    solutions_merged_dir.mkdir(parents=True, exist_ok=True)
    for pid in all_ids:
        base_rows = load_solution_file(solutions_dir / f"{pid}.json")
        add_rows = load_solution_file(solutions_add_dir / f"{pid}.json")
        merged_rows = dedupe_solution_entries([*base_rows, *add_rows])
        save_solution_file(solutions_merged_dir / f"{pid}.json", merged_rows)
        written += 1
        total_rows += len(merged_rows)

    return {
        "problems_written": int(written),
        "rows_written": int(total_rows),
    }


def _solution_row_key(row: Dict[str, Any]) -> Tuple[int, int, Tuple[str, ...]]:
    refs = parse_resolved_references(row.get("resolved_references", []))
    return (int(row.get("span_start", 0)), int(row.get("span_end", 0)), tuple(refs))


def _infer_source_label(
    row: Dict[str, Any],
    base_keys: set[Tuple[int, int, Tuple[str, ...]]],
    add_keys: set[Tuple[int, int, Tuple[str, ...]]],
) -> str:
    key = _solution_row_key(row)
    in_base = key in base_keys
    in_add = key in add_keys
    if in_base and in_add:
        return "solution+add"
    if in_base:
        return "solution"
    if in_add:
        return "add"
    return "manual"


def _color_for_source(label: str) -> str:
    if label == "solution":
        return "#c8e6c9"
    if label == "add":
        return "#ffe0b2"
    if label == "solution+add":
        return "#bbdefb"
    return "#f8bbd0"


def render_passim_validation_tab(paths: Paths) -> None:
    st.subheader("Passim Validation -> solutions_add")

    run_dirs = list_run_dirs(paths.passim_runs_dir)
    default_run = str(run_dirs[0]) if run_dirs else str(paths.passim_runs_dir)
    run_dir_input = st.text_input("Passim run folder", value=default_run)
    run_dir = Path(run_dir_input).expanduser().resolve()

    rows = load_visual_matches(run_dir)
    if not rows:
        st.warning("No visual matches found. Expected: <run>/reshaped/visual_matches.json")
        return

    task_chunk_map = load_task_chunk_map(run_dir)

    prepared: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        chunk_start = row.get("chunk_start", 0)
        try:
            chunk_start_int = int(chunk_start if chunk_start is not None else 0)
        except Exception:
            chunk_start_int = 0
        try:
            rel_start = int(row.get("aligned_chunk_start", 0) or 0)
            rel_end = int(row.get("aligned_chunk_end", 0) or 0)
        except Exception:
            rel_start, rel_end = 0, 0

        abs_start = chunk_start_int + rel_start
        abs_end = chunk_start_int + rel_end
        ref = normalize_reference(row.get("reference", ""))
        chunk_text = str(row.get("chunk_text", "") or "")
        verse_text = str(row.get("verse_text", "") or "")
        chunk_id = str(row.get("chunk_id", "") or "")
        if (not chunk_text) and chunk_id and (chunk_id in task_chunk_map):
            chunk_text = str(task_chunk_map[chunk_id].get("text", "") or "")

        prepared.append(
            {
                "row_id": f"{idx}",
                "accept": False,
                "problem_id": str(row.get("problem_id", "")),
                "chunk_index": row.get("chunk_index"),
                "chunk_id": chunk_id,
                "reference": ref,
                "resolved_references_text": ref,
                "score": float(row.get("score", 0.0) or 0.0),
                "is_ground_truth_reference": bool(row.get("is_ground_truth_reference", False)),
                "span_start": int(abs_start),
                "span_end": int(abs_end),
                "aligned_chunk_start": rel_start,
                "aligned_chunk_end": rel_end,
                "aligned_verse_start": int(row.get("aligned_verse_start", 0) or 0),
                "aligned_verse_end": int(row.get("aligned_verse_end", 0) or 0),
                "chunk_text": chunk_text,
                "verse_text": verse_text,
                "chunk_preview": chunk_text[:120],
                "verse_preview": verse_text[:120],
            }
        )

    df = pd.DataFrame(prepared)
    if df.empty:
        st.info("No rows to inspect.")
        return

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        red_only = st.checkbox("Filter red only", value=True)
    with c2:
        min_score = st.number_input("Min score", value=float(df["score"].min()), step=1.0)
    with c3:
        max_rows = st.number_input("Max rows", value=500, min_value=10, step=50)
    with c4:
        problem_filter = st.selectbox("Problem filter", options=["(all)"] + sorted(df["problem_id"].dropna().unique().tolist()))

    ref_substring = st.text_input("Reference contains", value="")

    filtered = df.copy()
    if red_only:
        filtered = filtered[~filtered["is_ground_truth_reference"]]
    filtered = filtered[filtered["score"] >= float(min_score)]
    if problem_filter != "(all)":
        filtered = filtered[filtered["problem_id"] == problem_filter]
    if ref_substring.strip():
        needle = normalize_reference(ref_substring)
        filtered = filtered[filtered["reference"].str.contains(re.escape(needle), regex=True)]
    filtered = filtered.head(int(max_rows))

    st.caption(f"Rows after filters: {len(filtered)}")
    if filtered.empty:
        st.info("No rows match filters.")
        return

    editor_cols = [
        "accept",
        "row_id",
        "problem_id",
        "chunk_index",
        "reference",
        "score",
        "is_ground_truth_reference",
        "span_start",
        "span_end",
        "resolved_references_text",
        "chunk_preview",
        "verse_preview",
    ]
    edited = st.data_editor(
        filtered[editor_cols],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key=f"passim_editor_{run_dir}_{problem_filter}_{red_only}_{min_score}_{max_rows}_{ref_substring}",
        column_config={
            "accept": st.column_config.CheckboxColumn("accept"),
            "row_id": st.column_config.TextColumn("row_id", disabled=True),
            "problem_id": st.column_config.TextColumn("problem_id", disabled=True),
            "chunk_index": st.column_config.NumberColumn("chunk_index", disabled=True),
            "reference": st.column_config.TextColumn("reference", disabled=True),
            "score": st.column_config.NumberColumn("score", disabled=True, format="%.2f"),
            "is_ground_truth_reference": st.column_config.CheckboxColumn("is_gt", disabled=True),
            "span_start": st.column_config.NumberColumn("span_start", min_value=0, step=1),
            "span_end": st.column_config.NumberColumn("span_end", min_value=0, step=1),
            "resolved_references_text": st.column_config.TextColumn("resolved_references (comma-separated)"),
            "chunk_preview": st.column_config.TextColumn("chunk_preview", disabled=True),
            "verse_preview": st.column_config.TextColumn("verse_preview", disabled=True),
        },
    )

    id_to_row = {str(r["row_id"]): r for r in prepared}
    inspect_id = st.selectbox("Inspect row_id", options=edited["row_id"].tolist())
    inspect_row = id_to_row.get(str(inspect_id))
    if inspect_row:
        st.markdown("**Alignment preview**")
        left, right = st.columns(2)
        with left:
            st.markdown("Chunk text")
            st.markdown(
                "<div style='white-space:pre-wrap;border:1px solid #ddd;padding:10px;'>"
                + render_highlighted_slice_html(
                    inspect_row["chunk_text"],
                    int(inspect_row["aligned_chunk_start"]),
                    int(inspect_row["aligned_chunk_end"]),
                    color="#fff176",
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("Verse text")
            st.markdown(
                "<div style='white-space:pre-wrap;border:1px solid #ddd;padding:10px;'>"
                + render_highlighted_slice_html(
                    inspect_row["verse_text"],
                    int(inspect_row["aligned_verse_start"]),
                    int(inspect_row["aligned_verse_end"]),
                    color="#fff176",
                )
                + "</div>",
                unsafe_allow_html=True,
            )

    if st.button("Save accepted rows to solutions_add", type="primary"):
        accepted = edited[edited["accept"] == True]  # noqa: E712
        if accepted.empty:
            st.warning("No accepted rows selected.")
            return

        by_problem: Dict[str, List[Dict[str, Any]]] = {}
        invalid_rows = 0
        for _, row in accepted.iterrows():
            refs = parse_resolved_references(row.get("resolved_references_text", ""))
            if not refs:
                refs = [normalize_reference(row.get("reference", ""))]
            item = canonical_solution_entry(
                {
                    "span_start": row.get("span_start"),
                    "span_end": row.get("span_end"),
                    "resolved_references": refs,
                }
            )
            if item is None:
                invalid_rows += 1
                continue
            pid = str(row.get("problem_id", "")).strip()
            if not pid:
                invalid_rows += 1
                continue
            by_problem.setdefault(pid, []).append(item)

        if not by_problem:
            st.error("All accepted rows were invalid (check span_start/span_end and references).")
            return

        paths.solutions_add_dir.mkdir(parents=True, exist_ok=True)
        written_files = 0
        total_rows = 0
        for pid, new_rows in by_problem.items():
            add_path = paths.solutions_add_dir / f"{pid}.json"
            existing = load_solution_file(add_path)
            merged = dedupe_solution_entries([*existing, *new_rows])
            save_solution_file(add_path, merged)
            written_files += 1
            total_rows += len(new_rows)

        st.success(
            f"Saved additions to {written_files} files in {paths.solutions_add_dir}. "
            f"Accepted rows: {int(len(accepted))}. Invalid skipped: {invalid_rows}. New rows submitted: {total_rows}."
        )


def render_merged_tab(paths: Paths) -> None:
    st.subheader("Merged Data (solutions + solutions_add) -> editable solutions_merged")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption(f"solutions: `{paths.solutions_dir}`")
        st.caption(f"solutions_add: `{paths.solutions_add_dir}`")
        st.caption(f"solutions_merged: `{paths.solutions_merged_dir}`")
    with c2:
        if st.button("Rebuild solutions_merged from solutions + solutions_add"):
            stats = build_merged_solutions(paths.solutions_dir, paths.solutions_add_dir, paths.solutions_merged_dir)
            st.success(
                f"Merged rebuilt. Problems: {stats['problems_written']}, rows: {stats['rows_written']}."
            )

    merged_ids = list_solution_ids(paths.solutions_merged_dir)
    if not merged_ids:
        st.info("No files in solutions_merged yet. Click rebuild first.")
        return

    problem_id = st.selectbox("Problem id", options=merged_ids)
    merged_path = paths.solutions_merged_dir / f"{problem_id}.json"
    base_path = paths.solutions_dir / f"{problem_id}.json"
    add_path = paths.solutions_add_dir / f"{problem_id}.json"

    merged_rows = load_solution_file(merged_path)
    base_rows = load_solution_file(base_path)
    add_rows = load_solution_file(add_path)
    problem_text = load_problem_text(paths.problems_dir, problem_id)

    base_keys = set([_solution_row_key(r) for r in base_rows])
    add_keys = set([_solution_row_key(r) for r in add_rows])

    st.caption(
        f"Rows: solution={len(base_rows)} | additions={len(add_rows)} | merged={len(merged_rows)}"
    )

    editable_rows: List[Dict[str, Any]] = []
    for row in merged_rows:
        editable_rows.append(
            {
                "span_start": int(row["span_start"]),
                "span_end": int(row["span_end"]),
                "resolved_references_text": ", ".join(parse_resolved_references(row.get("resolved_references", []))),
                "source": _infer_source_label(row, base_keys, add_keys),
            }
        )
    df = pd.DataFrame(editable_rows)
    if df.empty:
        df = pd.DataFrame(columns=["span_start", "span_end", "resolved_references_text", "source"])

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        key=f"merged_editor_{problem_id}",
        column_config={
            "span_start": st.column_config.NumberColumn("span_start", min_value=0, step=1),
            "span_end": st.column_config.NumberColumn("span_end", min_value=0, step=1),
            "resolved_references_text": st.column_config.TextColumn("resolved_references (comma-separated)"),
            "source": st.column_config.TextColumn("source", disabled=True),
        },
    )

    # Build a visualization from currently edited rows (not yet saved).
    vis_spans: List[Tuple[int, int, str, str]] = []
    for _, row in edited.iterrows():
        refs = parse_resolved_references(row.get("resolved_references_text", ""))
        label = refs[0] if refs else ""
        source = str(row.get("source", "manual"))
        vis_spans.append(
            (
                int(row.get("span_start", 0) or 0),
                int(row.get("span_end", 0) or 0),
                label,
                _color_for_source(source),
            )
        )

    st.markdown("**Merged visualization**")
    st.markdown(
        "<div style='white-space:pre-wrap;border:1px solid #ddd;padding:10px;'>"
        + render_text_with_spans_html(problem_text, vis_spans)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Color legend: solution=green, add=orange, solution+add=blue, manual=pink.")

    if st.button("Save edited merged file", type="primary"):
        to_save: List[Dict[str, Any]] = []
        invalid_rows = 0
        for _, row in edited.iterrows():
            item = canonical_solution_entry(
                {
                    "span_start": row.get("span_start"),
                    "span_end": row.get("span_end"),
                    "resolved_references": row.get("resolved_references_text", ""),
                }
            )
            if item is None:
                invalid_rows += 1
                continue
            to_save.append(item)

        save_solution_file(merged_path, to_save)
        st.success(
            f"Saved merged edits: {merged_path}. Valid rows saved: {len(dedupe_solution_entries(to_save))}, invalid skipped: {invalid_rows}."
        )


def render_files_overview_tab(paths: Paths) -> None:
    st.subheader("Data Overview")
    counts = {
        "problems": len(list_problem_ids(paths.problems_dir)),
        "solutions": len(list_solution_ids(paths.solutions_dir)),
        "solutions_add": len(list_solution_ids(paths.solutions_add_dir)),
        "solutions_merged": len(list_solution_ids(paths.solutions_merged_dir)),
        "passim_runs": len(list_run_dirs(paths.passim_runs_dir)),
    }
    st.json(counts)

    st.markdown("**Storage locations**")
    st.code(
        "\n".join(
            [
                f"problems: {paths.problems_dir}",
                f"solutions: {paths.solutions_dir}",
                f"solutions_add: {paths.solutions_add_dir}",
                f"solutions_merged: {paths.solutions_merged_dir}",
                f"passim_runs: {paths.passim_runs_dir}",
            ]
        )
    )


def main() -> None:
    st.set_page_config(page_title="Passim Validation", layout="wide")
    st.title("Passim Validation and Ground-Truth Augmentation")

    try:
        paths = get_paths()
    except Exception as exc:
        st.error(str(exc))
        return

    tab_validate, tab_merged, tab_overview = st.tabs(
        ["Passim -> additions", "Merged viewer/editor", "Overview"]
    )
    with tab_validate:
        render_passim_validation_tab(paths)
    with tab_merged:
        render_merged_tab(paths)
    with tab_overview:
        render_files_overview_tab(paths)


if __name__ == "__main__":
    main()
