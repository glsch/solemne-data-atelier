from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm


def find_project_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "task").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing data/task")


def normalize_reference(ref: str) -> str:
    return str(ref or "").strip().lower()


def load_task_problems(task_dir: Path) -> Dict[str, str]:
    problems_dir = Path(task_dir) / "problems"
    if not problems_dir.exists():
        raise FileNotFoundError(f"Missing problems directory: {problems_dir}")

    problems: Dict[str, str] = {}
    for path in sorted(problems_dir.glob("*.txt")):
        problems[path.stem] = path.read_text(encoding="utf-8")
    return problems


def load_task_ground_truth(task_dir: Path) -> Dict[str, List[str]]:
    solutions_dir = Path(task_dir) / "solutions"
    if not solutions_dir.exists():
        raise FileNotFoundError(f"Missing solutions directory: {solutions_dir}")

    out: Dict[str, List[str]] = {}
    for path in sorted(solutions_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        refs = set()
        for item in payload:
            for ref in item.get("resolved_references", []):
                refs.add(normalize_reference(ref))
        out[path.stem] = sorted(refs)
    return out


def load_bible_tsv(project_root: Path) -> pd.DataFrame:
    candidates = [
        project_root / "data" / "raw" / "bible.tsv",
        project_root / "data" / "bible.tsv",
        project_root / "data" / "hackathon_dataset" / "bible.tsv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Could not locate bible.tsv in: {candidates}")

    df = pd.read_csv(path, sep="\t")
    df["reference"] = (
        df["book_code"].astype(str).str.strip().str.lower()
        + "_"
        + df["chapter_number"].astype(str)
        + ":"
        + df["verse_index"].astype(str)
    )
    return df


def flatten_truth_pairs(ground_truth_by_problem: Dict[str, Sequence[str]]) -> set[tuple[str, str]]:
    pairs = set()
    for problem_id, refs in ground_truth_by_problem.items():
        for ref in refs:
            pairs.add((problem_id, normalize_reference(ref)))
    return pairs


def flatten_prediction_pairs(predictions: Sequence[Dict[str, Any]]) -> set[tuple[str, str]]:
    pairs = set()
    for row in predictions:
        pid = str(row.get("problem_id", "")).strip()
        ref = normalize_reference(row.get("reference", ""))
        if pid and ref:
            pairs.add((pid, ref))
    return pairs


def score_predictions(
    predictions: Sequence[Dict[str, Any]],
    ground_truth_by_problem: Dict[str, Sequence[str]],
) -> Dict[str, float]:
    pred_pairs = flatten_prediction_pairs(predictions)
    true_pairs = flatten_truth_pairs(ground_truth_by_problem)

    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


MethodFn = Callable[[str, str, Dict[str, Any]], List[Dict[str, Any]]]


def run_method_on_dataset(
    method_fn: MethodFn,
    problems_by_id: Dict[str, str],
    method_context: Dict[str, Any],
    *,
    problem_ids: Optional[Sequence[str]] = None,
    max_problems: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    selected = list(problem_ids) if problem_ids is not None else sorted(problems_by_id.keys())
    if max_problems is not None:
        selected = selected[: max(0, int(max_problems))]

    iterator: Iterable[str] = selected
    if show_progress:
        iterator = tqdm(selected, desc="Running method", unit="problem")

    out: List[Dict[str, Any]] = []
    for problem_id in iterator:
        text = problems_by_id[problem_id]
        rows = method_fn(problem_id, text, method_context)
        out.extend(rows)
    return out

