from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

from solemne_data_atelier.evaluation import normalize_reference
from solemne_data_atelier.utils import split_into_chunks

logger = logging.getLogger(__name__)


def _chunk_text_with_spans(
    text: str,
    *,
    mode: str,
    sentences_per_chunk: int,
    sentence_stride: int,
    char_chunk_size: int,
    char_chunk_overlap: int,
    min_chunk_chars: int,
) -> List[Dict[str, Any]]:
    text = str(text or "")
    if not text:
        return []

    mode_norm = mode.lower().strip()
    out: List[Dict[str, Any]] = []

    if mode_norm in {"full", "none", "document", "whole"}:
        whole = text.strip()
        if not whole or len(whole) < int(min_chunk_chars):
            return []
        leading_ws = len(text) - len(text.lstrip())
        start = leading_ws
        end = start + len(whole)
        return [{"chunk_index": 0, "text": whole, "start": start, "end": end}]

    if mode_norm == "char":
        size = max(1, int(char_chunk_size))
        overlap = max(0, int(char_chunk_overlap))
        step = max(1, size - overlap)
        for idx, i in enumerate(range(0, len(text), step)):
            raw_piece = text[i : i + size]
            stripped = raw_piece.strip()
            if len(stripped) < int(min_chunk_chars):
                continue
            leading_ws = len(raw_piece) - len(raw_piece.lstrip())
            start = i + leading_ws
            end = start + len(stripped)
            out.append({"chunk_index": idx, "text": stripped, "start": start, "end": end})
        return out

    chunks = split_into_chunks(
        text,
        mode=mode_norm,
        sentences_per_chunk=sentences_per_chunk,
        sentence_stride=sentence_stride,
        char_chunk_size=char_chunk_size,
        char_chunk_overlap=char_chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )

    cursor = 0
    for idx, chunk in enumerate(chunks):
        start = text.find(chunk, cursor)
        if start < 0:
            start = text.find(chunk)
        if start < 0:
            start = cursor
        end = start + len(chunk)
        cursor = max(cursor, end)
        out.append({"chunk_index": idx, "text": chunk, "start": start, "end": end})
    return out


def _resolve_passim_runner(passim_runner: Optional[Sequence[str]], project_root: Path) -> List[str]:
    if passim_runner:
        return [str(x) for x in passim_runner]

    try:
        import passim  # noqa: F401

        return [sys.executable, "-m", "passim.seriatim"]
    except Exception:
        pass

    for cmd in ["passim", "seriatim"]:
        p = shutil.which(cmd)
        if p:
            return [p]

    venv_python = Path(project_root) / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python), "-m", "passim.seriatim"]

    raise RuntimeError("Could not resolve a Passim runner. Install passim/seriatim or pass passim_runner.")


def _option_to_flag(name: str) -> str:
    n = str(name).strip()
    if not n:
        raise ValueError("Empty passim option name is not allowed.")
    if n.startswith("-"):
        return n
    n = n.replace("_", "-")
    if len(n) == 1:
        return f"-{n}"
    return f"--{n}"


def _build_passim_args(
    passim_options: Optional[Dict[str, Any]],
    passim_extra_args: Optional[Sequence[str]],
) -> List[str]:
    args: List[str] = []
    for key, value in dict(passim_options or {}).items():
        if value is None:
            continue
        flag = _option_to_flag(key)
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            args.append(flag)
            args.extend([str(v) for v in value])
            continue
        args.extend([flag, str(value)])

    if passim_extra_args:
        args.extend([str(x) for x in passim_extra_args])
    return args


def _write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_passim_out_json_rows(raw_output_dir: Path) -> List[Dict[str, Any]]:
    out_dir = raw_output_dir / "out.json"
    if not out_dir.exists():
        raise FileNotFoundError(f"Passim output folder not found: {out_dir}")
    rows: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("part-*.json")):
        rows.extend(_read_jsonl(path))
    return rows


def _dump_yaml_fallback(data: Dict[str, Any]) -> str:
    lines = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, str):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


def _save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        payload = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000000)
    except Exception:
        payload = _dump_yaml_fallback(data)
    path.write_text(payload, encoding="utf-8")


def save_passim_metrics(
    *,
    run_dir: Path,
    metrics: Dict[str, float],
    predictions_count: int,
) -> Path:
    """
    Save evaluation metrics for a passim run into its artifact folder.
    """
    payload = {
        **{k: float(v) for k, v in metrics.items()},
        "predictions_produced": int(predictions_count),
    }
    out_path = Path(run_dir) / "metrics.yaml"
    _save_yaml(payload, out_path)
    return out_path


def _parse_task_chunk_id(chunk_id: str) -> Tuple[Optional[str], Optional[int]]:
    m = re.match(r"^task::(.+?)::chunk::(\d+)$", str(chunk_id))
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _parse_bible_id(bible_id: str) -> Optional[str]:
    s = str(bible_id or "")
    if not s.startswith("bible::"):
        return None
    return normalize_reference(s.split("bible::", 1)[1])


def _highlight_span_html(text: str, start: int, end: int) -> str:
    t = str(text or "")
    n = len(t)
    a = max(0, min(n, int(start)))
    b = max(a, min(n, int(end)))
    return f"{escape(t[:a])}<span class='aligned'>{escape(t[a:b])}</span>{escape(t[b:])}"


def _render_visual_html(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    css = """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
    th { background: #f6f8fa; position: sticky; top: 0; }
    tr.tp { background: #eaf7ea; }
    tr.fp { background: #fdecec; }
    .aligned { background: #fff176; font-weight: 600; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .wrap { white-space: pre-wrap; word-break: break-word; }
    </style>
    """

    header = (
        "<tr>"
        "<th>problem_id</th>"
        "<th>chunk_index</th>"
        "<th>reference</th>"
        "<th>score</th>"
        "<th>GT</th>"
        "<th>chunk</th>"
        "<th>verse</th>"
        "</tr>"
    )

    body_rows: List[str] = []
    for row in rows:
        row_class = "tp" if row.get("is_ground_truth_reference") else "fp"
        chunk_html = _highlight_span_html(
            row.get("chunk_text", ""),
            int(row.get("aligned_chunk_start", 0) or 0),
            int(row.get("aligned_chunk_end", 0) or 0),
        )
        verse_html = _highlight_span_html(
            row.get("verse_text", ""),
            int(row.get("aligned_verse_start", 0) or 0),
            int(row.get("aligned_verse_end", 0) or 0),
        )
        cells = [
            f"<td class='mono'>{escape(str(row.get('problem_id', '')))}</td>",
            f"<td class='mono'>{escape(str(row.get('chunk_index', '')))}</td>",
            f"<td class='mono'>{escape(str(row.get('reference', '')))}</td>",
            f"<td class='mono'>{escape(str(round(float(row.get('score', 0.0) or 0.0), 4)))}</td>",
            f"<td class='mono'>{'yes' if row.get('is_ground_truth_reference') else 'no'}</td>",
            f"<td class='wrap'>{chunk_html}</td>",
            f"<td class='wrap'>{verse_html}</td>",
        ]
        body_rows.append(f"<tr class='{row_class}'>" + "".join(cells) + "</tr>")

    html = (
        "<html><head><meta charset='utf-8'>"
        + css
        + "</head><body>"
        + "<h2>Passim Reuse Inspection</h2>"
        + "<table>"
        + header
        + "\n".join(body_rows)
        + "</table></body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _reshape_passim_docwise_output(
    raw_rows: Sequence[Dict[str, Any]],
    *,
    chunk_by_id: Dict[str, Dict[str, Any]],
    bible_text_by_ref: Dict[str, str],
    ground_truth_by_problem: Dict[str, Sequence[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    best_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    visual_rows: List[Dict[str, Any]] = []

    gt_norm: Dict[str, set[str]] = {
        pid: {normalize_reference(r) for r in refs} for pid, refs in ground_truth_by_problem.items()
    }

    for doc in raw_rows:
        task_chunk_id = str(doc.get("id", ""))
        if not task_chunk_id.startswith("task::"):
            continue

        chunk_meta = chunk_by_id.get(task_chunk_id, {})
        problem_id, chunk_index_from_id = _parse_task_chunk_id(task_chunk_id)
        if not problem_id:
            continue

        chunk_index = chunk_meta.get("chunk_index", chunk_index_from_id)
        chunk_text_default = str(chunk_meta.get("text", doc.get("text", "")))

        for line in doc.get("lines", []) or []:
            line_text = str(line.get("text", ""))
            base_chunk_text = line_text or chunk_text_default

            for wit in line.get("wits", []) or []:
                reference = _parse_bible_id(wit.get("id", ""))
                if not reference:
                    continue

                score = float(wit.get("matches", 0) or 0.0)
                pair_key = (problem_id, reference)
                prev = best_by_pair.get(pair_key)
                if prev is None or score > float(prev.get("score", 0.0)):
                    best_by_pair[pair_key] = {
                        "problem_id": problem_id,
                        "reference": reference,
                        "score": score,
                        "method": "passim",
                    }

                alg_src = str(wit.get("alg", ""))
                alg_dst = str(wit.get("alg2", ""))
                aligned_chunk_start = int(wit.get("begin2", 0) or 0)
                aligned_verse_start = int(wit.get("begin", 0) or 0)
                aligned_chunk_end = aligned_chunk_start + len(alg_dst.replace("-", ""))
                aligned_verse_end = aligned_verse_start + len(alg_src.replace("-", ""))

                verse_text = bible_text_by_ref.get(reference)
                if verse_text is None:
                    verse_text = str(wit.get("text", ""))

                visual_rows.append(
                    {
                        "problem_id": problem_id,
                        "chunk_id": task_chunk_id,
                        "chunk_index": chunk_index,
                        "chunk_start": chunk_meta.get("start"),
                        "chunk_end": chunk_meta.get("end"),
                        "reference": reference,
                        "score": score,
                        "is_ground_truth_reference": reference in gt_norm.get(problem_id, set()),
                        "chunk_text": base_chunk_text,
                        "verse_text": verse_text,
                        "aligned_chunk_start": aligned_chunk_start,
                        "aligned_chunk_end": aligned_chunk_end,
                        "aligned_verse_start": aligned_verse_start,
                        "aligned_verse_end": aligned_verse_end,
                        "alg": alg_src,
                        "alg2": alg_dst,
                    }
                )

    evaluation_rows = sorted(best_by_pair.values(), key=lambda r: (str(r["problem_id"]), str(r["reference"])))

    by_problem: Dict[str, List[Dict[str, Any]]] = {}
    for row in evaluation_rows:
        by_problem.setdefault(row["problem_id"], []).append(row)

    visual_rows.sort(
        key=lambda r: (
            str(r.get("problem_id", "")),
            int(r.get("chunk_index", 0) or 0),
            -float(r.get("score", 0.0) or 0.0),
            str(r.get("reference", "")),
        )
    )
    return evaluation_rows, by_problem, visual_rows


def build_passim_method_context(
    *,
    problems_by_id: Dict[str, str],
    ground_truth_by_problem: Dict[str, Sequence[str]],
    bible_df: pd.DataFrame,
    project_root: Path,
    passim_runs_dir: Path,
    passim_runner: Optional[Sequence[str]] = None,
    passim_options: Optional[Dict[str, Any]] = None,
    passim_extra_args: Optional[Sequence[str]] = None,
    mode: str = "sentence_window",
    chunking_enabled: bool = True,
    sentences_per_chunk: int = 2,
    sentence_stride: int = 1,
    char_chunk_size: int = 500,
    char_chunk_overlap: int = 100,
    min_chunk_chars: int = 30,
    problem_ids: Optional[Sequence[str]] = None,
    max_problems: Optional[int] = None,
) -> Dict[str, Any]:
    selected_problem_ids = list(problem_ids) if problem_ids is not None else sorted(problems_by_id.keys())
    if max_problems is not None:
        selected_problem_ids = selected_problem_ids[: int(max_problems)]

    now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(passim_runs_dir) / now_ts
    input_dir = run_dir / "input"
    raw_output_dir = run_dir / "raw_output"
    reshaped_dir = run_dir / "reshaped"
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    reshaped_dir.mkdir(parents=True, exist_ok=True)

    task_records: List[Dict[str, Any]] = []
    chunk_by_id: Dict[str, Dict[str, Any]] = {}
    effective_mode = str(mode).strip().lower()
    if not bool(chunking_enabled):
        effective_mode = "full"

    for problem_id in tqdm(selected_problem_ids, desc="Chunking task data", unit="problem"):
        text = problems_by_id.get(problem_id, "")
        chunks = _chunk_text_with_spans(
            text,
            mode=effective_mode,
            sentences_per_chunk=sentences_per_chunk,
            sentence_stride=sentence_stride,
            char_chunk_size=char_chunk_size,
            char_chunk_overlap=char_chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )
        for c in chunks:
            chunk_id = f"task::{problem_id}::chunk::{int(c['chunk_index'])}"
            row = {
                "id": chunk_id,
                "group": "task",
                "problem_id": problem_id,
                "chunk_index": int(c["chunk_index"]),
                "chunk_start": int(c["start"]),
                "chunk_end": int(c["end"]),
                "text": str(c["text"]),
            }
            task_records.append(row)
            chunk_by_id[chunk_id] = {
                "problem_id": problem_id,
                "chunk_index": int(c["chunk_index"]),
                "start": int(c["start"]),
                "end": int(c["end"]),
                "text": str(c["text"]),
            }

    if not task_records:
        raise ValueError("No task chunks were produced. Adjust chunking parameters.")

    bible_records: List[Dict[str, Any]] = []
    bible_text_by_ref: Dict[str, str] = {}
    for _, row in bible_df.iterrows():
        reference = normalize_reference(row["reference"])
        verse_text = str(row.get("text", "") or "").strip()
        bible_id = f"bible::{reference}"
        bible_records.append({"id": bible_id, "group": "bible", "reference": reference, "text": verse_text})
        bible_text_by_ref[reference] = verse_text

    task_path = input_dir / "task_chunked.jsonl"
    bible_path = input_dir / "bible_verses.jsonl"
    merged_path = input_dir / "passim_input.jsonl"
    _write_jsonl(task_records, task_path)
    _write_jsonl(bible_records, bible_path)
    _write_jsonl([*task_records, *bible_records], merged_path)

    runner = _resolve_passim_runner(passim_runner=passim_runner, project_root=Path(project_root))
    merged_options = {
        "docwise": True,
        "fields": ["group"],
        "filterpairs": "group = 'bible' AND group2 = 'task'",
        "output-format": "json",
    }
    merged_options.update(passim_options or {})

    passim_args = _build_passim_args(merged_options, passim_extra_args)
    command = [*runner, *passim_args, str(merged_path), str(raw_output_dir)]

    env = os.environ.copy()
    env.setdefault("PYSPARK_PYTHON", sys.executable)
    env.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    logger.info("Launching passim command: %s", shlex.join(command))
    completed = subprocess.run(command, cwd=str(project_root), env=env, capture_output=True, text=True)
    (run_dir / "passim.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (run_dir / "passim.stderr.log").write_text(completed.stderr or "", encoding="utf-8")

    if completed.returncode != 0:
        stderr_tail = "\n".join((completed.stderr or "").splitlines()[-40:])
        raise RuntimeError(
            "Passim execution failed. "
            f"Run dir: {run_dir}. Exit code: {completed.returncode}\n{stderr_tail}"
        )

    raw_rows = _load_passim_out_json_rows(raw_output_dir)
    eval_rows, preds_by_problem, visual_rows = _reshape_passim_docwise_output(
        raw_rows,
        chunk_by_id=chunk_by_id,
        bible_text_by_ref=bible_text_by_ref,
        ground_truth_by_problem=ground_truth_by_problem,
    )

    (reshaped_dir / "evaluation_predictions.json").write_text(
        json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reshaped_dir / "visual_matches.json").write_text(
        json.dumps(visual_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _render_visual_html(visual_rows, reshaped_dir / "visual_matches.html")

    config_payload = {
        "timestamp": now_ts,
        "run_dir": str(run_dir),
        "command": shlex.join(command),
        "passim_runner": runner,
        "passim_options": merged_options,
        "passim_extra_args": list(passim_extra_args or []),
        "chunking": {
            "enabled": bool(chunking_enabled),
            "mode_requested": str(mode),
            "mode_effective": effective_mode,
            "sentences_per_chunk": int(sentences_per_chunk),
            "sentence_stride": int(sentence_stride),
            "char_chunk_size": int(char_chunk_size),
            "char_chunk_overlap": int(char_chunk_overlap),
            "min_chunk_chars": int(min_chunk_chars),
        },
        "selected_problem_count": len(selected_problem_ids),
        "task_chunk_count": len(task_records),
        "bible_verse_count": len(bible_records),
        "prediction_pair_count": len(eval_rows),
        "visual_match_count": len(visual_rows),
    }
    _save_yaml(config_payload, run_dir / "run_config.yaml")

    return {
        "method_name": "passim",
        "run_dir": run_dir,
        "selected_problem_ids": selected_problem_ids,
        "predictions": eval_rows,
        "predictions_by_problem": preds_by_problem,
        "visual_rows": visual_rows,
        "task_chunk_count": len(task_records),
        "bible_verse_count": len(bible_records),
        "command": shlex.join(command),
        "config_path": run_dir / "run_config.yaml",
        "raw_output_dir": raw_output_dir,
        "reshaped_dir": reshaped_dir,
    }


def passim_method(problem_id: str, _problem_text: str, method_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(method_context.get("predictions_by_problem", {}).get(problem_id, []))
