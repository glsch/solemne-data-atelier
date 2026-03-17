from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from solemne_data_atelier.evaluation import normalize_reference
from solemne_data_atelier.utils import split_into_chunks

logger = logging.getLogger(__name__)

_REF_PATTERN = re.compile(r"^([1-3]?[a-z]+)(?:[_\s-]?)(\d+):(\d+)(?:-(\d+))?$", re.IGNORECASE)


def _normalize_alias(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").strip().lower())


def _load_book_mapping(book_mapping_path: Path) -> Tuple[List[str], Dict[str, str]]:
    if not Path(book_mapping_path).exists():
        raise FileNotFoundError(f"book_mapping.tsv not found: {book_mapping_path}")

    book_codes: set[str] = set()
    alias_to_code: Dict[str, str] = {}

    with Path(book_mapping_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            code = str(row.get("book_code", "")).strip().lower()
            if not code:
                continue
            book_codes.add(code)
            alias_to_code[_normalize_alias(code)] = code

            work_name = str(row.get("work_name", "")).strip().lower()
            if work_name:
                alias_to_code[_normalize_alias(work_name)] = code

    codes_sorted = sorted(book_codes)
    if not codes_sorted:
        raise ValueError(f"No book codes found in mapping: {book_mapping_path}")
    return codes_sorted, alias_to_code


def _build_bible_verse_index(bible_df: Optional[Any]) -> Dict[Tuple[str, int], set[int]]:
    if bible_df is None:
        return {}

    verse_index: Dict[Tuple[str, int], set[int]] = {}
    for _, row in bible_df.iterrows():
        ref = normalize_reference(row.get("reference", ""))
        m = re.match(r"^([1-3]?[a-z]+)_(\d+):(\d+)$", ref)
        if not m:
            continue
        book = m.group(1)
        chapter = int(m.group(2))
        verse = int(m.group(3))
        verse_index.setdefault((book, chapter), set()).add(verse)
    return verse_index


def _import_openai_chat():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI


def _import_anthropic_chat():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic


def _import_google_chat():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI


def _build_chat_model(
    *,
    provider: str,
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    timeout: float,
    api_key: Optional[str],
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    provider_norm = str(provider or "").strip().lower()
    kwargs = dict(model_kwargs or {})

    if provider_norm == "openai":
        ChatOpenAI = _import_openai_chat()
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for provider='openai'.")
        return ChatOpenAI(
            model=model_name,
            temperature=float(temperature),
            max_tokens=int(max_output_tokens),
            timeout=float(timeout),
            api_key=key,
            **kwargs,
        )

    if provider_norm in {"anthropic", "claude"}:
        ChatAnthropic = _import_anthropic_chat()
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required for provider='anthropic'.")
        return ChatAnthropic(
            model=model_name,
            temperature=float(temperature),
            max_tokens=int(max_output_tokens),
            timeout=float(timeout),
            api_key=key,
            **kwargs,
        )

    if provider_norm in {"google", "gemini"}:
        ChatGoogleGenerativeAI = _import_google_chat()
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required for provider='google'.")
        call_kwargs = {
            "model": model_name,
            "temperature": float(temperature),
            "google_api_key": key,
            "timeout": float(timeout),
            **kwargs,
        }
        if "max_output_tokens" not in call_kwargs and "max_tokens" not in call_kwargs:
            call_kwargs["max_output_tokens"] = int(max_output_tokens)
        try:
            return ChatGoogleGenerativeAI(**call_kwargs)
        except TypeError:
            if "max_output_tokens" in call_kwargs and "max_tokens" not in call_kwargs:
                call_kwargs["max_tokens"] = call_kwargs.pop("max_output_tokens")
            return ChatGoogleGenerativeAI(**call_kwargs)

    raise ValueError("provider must be one of: openai, anthropic, google")


def _estimate_tokens(text: str) -> int:
    return max(1, int(math.ceil(len(str(text or "")) / 4.0)))


def _split_by_token_budget(
    text: str,
    *,
    max_input_tokens: int,
    overlap_chars: int,
) -> List[str]:
    txt = str(text or "").strip()
    if not txt:
        return []
    if _estimate_tokens(txt) <= int(max_input_tokens):
        return [txt]

    max_chars = max(200, int(max_input_tokens) * 4)
    overlap = max(0, int(overlap_chars))
    step = max(1, max_chars - overlap)

    out: List[str] = []
    for i in range(0, len(txt), step):
        piece = txt[i : i + max_chars].strip()
        if piece:
            out.append(piece)
    return out


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "\n".join(parts)
    return str(content or "")


def _extract_json_payload(raw_text: str) -> Any:
    text = str(raw_text or "").strip()
    if not text:
        return {"matches": []}

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            return {"matches": json.loads(arr_match.group(0))}
        except Exception:
            pass

    return {"matches": []}


def _iter_matches(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ["matches", "references", "predictions", "items", "results"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _split_reference_candidates(raw_ref: str) -> List[str]:
    value = str(raw_ref or "").strip()
    if not value:
        return []
    parts = re.split(r"[;,|]\s*", value)
    return [p.strip() for p in parts if p.strip()]


def _parse_reference(
    candidate: str,
    *,
    alias_to_code: Dict[str, str],
    allowed_book_codes: set[str],
) -> Optional[Tuple[str, int, int, int]]:
    c = str(candidate or "").strip().lower()
    if not c:
        return None
    c = c.replace("–", "-").replace("—", "-").replace("−", "-")
    c = c.replace(".", "").replace(" ", "")

    m = _REF_PATTERN.match(c)
    if not m:
        return None

    raw_book = m.group(1)
    chapter = int(m.group(2))
    verse_start = int(m.group(3))
    verse_end = int(m.group(4)) if m.group(4) else verse_start

    canonical_book = alias_to_code.get(_normalize_alias(raw_book), raw_book)
    if canonical_book not in allowed_book_codes:
        return None

    if verse_end < verse_start:
        verse_start, verse_end = verse_end, verse_start

    return canonical_book, chapter, verse_start, verse_end


def _expand_reference(
    *,
    book: str,
    chapter: int,
    verse_start: int,
    verse_end: int,
    bible_verse_index: Dict[Tuple[str, int], set[int]],
) -> List[str]:
    key = (book, int(chapter))
    if key in bible_verse_index:
        available = sorted(v for v in bible_verse_index[key] if int(verse_start) <= v <= int(verse_end))
        return [f"{book}_{chapter}:{v}" for v in available]
    return [f"{book}_{chapter}:{v}" for v in range(int(verse_start), int(verse_end) + 1)]


def _resolve_references(
    raw_reference: str,
    *,
    alias_to_code: Dict[str, str],
    allowed_book_codes: set[str],
    bible_verse_index: Dict[Tuple[str, int], set[int]],
) -> List[str]:
    out: List[str] = []
    for candidate in _split_reference_candidates(raw_reference):
        parsed = _parse_reference(
            candidate,
            alias_to_code=alias_to_code,
            allowed_book_codes=allowed_book_codes,
        )
        if not parsed:
            continue
        book, chapter, verse_start, verse_end = parsed
        out.extend(
            _expand_reference(
                book=book,
                chapter=chapter,
                verse_start=verse_start,
                verse_end=verse_end,
                bible_verse_index=bible_verse_index,
            )
        )
    return out


def _book_codes_for_prompt(book_codes: Sequence[str], *, per_line: int = 16) -> str:
    lines: List[str] = []
    row: List[str] = []
    for code in book_codes:
        row.append(code)
        if len(row) >= int(per_line):
            lines.append(", ".join(row))
            row = []
    if row:
        lines.append(", ".join(row))
    return "\n".join(lines)


def _prepare_chunks(
    text: str,
    *,
    mode: str,
    sentences_per_chunk: int,
    sentence_stride: int,
    char_chunk_size: int,
    char_chunk_overlap: int,
    min_chunk_chars: int,
    max_input_tokens: int,
    context_split_overlap_chars: int,
) -> List[str]:
    base_chunks = split_into_chunks(
        text,
        mode=mode,
        sentences_per_chunk=sentences_per_chunk,
        sentence_stride=sentence_stride,
        char_chunk_size=char_chunk_size,
        char_chunk_overlap=char_chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
    if not base_chunks:
        return []

    expanded: List[str] = []
    for chunk in base_chunks:
        expanded.extend(
            _split_by_token_budget(
                chunk,
                max_input_tokens=max_input_tokens,
                overlap_chars=context_split_overlap_chars,
            )
        )

    return [c for c in expanded if len(c) >= int(min_chunk_chars)]


def build_direct_prompting_method_context(
    *,
    provider: str,
    model_name: str,
    book_mapping_path: Path,
    bible_df: Optional[Any] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 500,
    timeout: float = 120.0,
    api_key: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    book_codes, alias_to_code = _load_book_mapping(Path(book_mapping_path))
    bible_verse_index = _build_bible_verse_index(bible_df)
    chat_model = _build_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
        timeout=float(timeout),
        api_key=api_key,
        model_kwargs=model_kwargs,
    )

    return {
        "method_name": "direct_prompting",
        "provider": str(provider).strip().lower(),
        "model_name": str(model_name).strip(),
        "chat_model": chat_model,
        "book_mapping_path": str(book_mapping_path),
        "allowed_book_codes": set(book_codes),
        "book_codes_prompt_block": _book_codes_for_prompt(book_codes),
        "alias_to_code": alias_to_code,
        "bible_verse_index": bible_verse_index,
    }


def direct_prompting_method(
    problem_id: str,
    problem_text: str,
    method_context: Dict[str, Any],
    *,
    mode: str = "full",
    sentences_per_chunk: int = 4,
    sentence_stride: int = 2,
    char_chunk_size: int = 3000,
    char_chunk_overlap: int = 400,
    min_chunk_chars: int = 30,
    context_window_tokens: int = 8192,
    prompt_token_reserve: int = 1400,
    max_output_tokens: int = 500,
    context_split_overlap_chars: int = 120,
) -> List[Dict[str, Any]]:
    max_input_tokens = max(
        256,
        int(context_window_tokens) - int(prompt_token_reserve) - int(max_output_tokens),
    )
    if max_input_tokens <= 0:
        raise ValueError("Invalid token budget: context_window_tokens is too small for configured reserves.")

    chunks = _prepare_chunks(
        problem_text,
        mode=mode,
        sentences_per_chunk=sentences_per_chunk,
        sentence_stride=sentence_stride,
        char_chunk_size=char_chunk_size,
        char_chunk_overlap=char_chunk_overlap,
        min_chunk_chars=min_chunk_chars,
        max_input_tokens=max_input_tokens,
        context_split_overlap_chars=context_split_overlap_chars,
    )
    if not chunks:
        return []

    system_prompt = (
        "You detect biblical verse reuse in Latin texts.\n"
        "Return JSON only, no prose, no markdown.\n"
        "Output schema:\n"
        '{"matches":[{"reference":"book_chapter:verse_or_range","confidence":0.0}]}\n'
        "Use only these book codes from book_mapping.tsv:\n"
        f"{method_context['book_codes_prompt_block']}\n"
        "Reference format examples: gen_1:1, ps_22:2-4, 1cor_13:4-7.\n"
        "If no biblical verse is present, return {\"matches\":[]}."
    )

    best_scores: Dict[str, float] = {}
    for chunk_idx, chunk in enumerate(chunks):
        human_prompt = (
            "Identify explicit or highly probable biblical verse reuse in this task text.\n"
            "Return JSON only.\n\n"
            f"chunk_index: {chunk_idx}\n"
            "task_text:\n"
            f"{chunk}"
        )
        try:
            response = method_context["chat_model"].invoke(
                [
                    ("system", system_prompt),
                    ("human", human_prompt),
                ]
            )
        except Exception as exc:
            logger.warning(
                "LLM call failed for problem_id=%s chunk_index=%s (%s).",
                problem_id,
                chunk_idx,
                exc,
            )
            continue

        payload = _extract_json_payload(_to_text(getattr(response, "content", response)))
        for match in _iter_matches(payload):
            raw_ref = str(match.get("reference", match.get("ref", ""))).strip()
            if not raw_ref:
                continue

            try:
                confidence = float(match.get("confidence", match.get("score", 0.5)))
            except Exception:
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            resolved_refs = _resolve_references(
                raw_ref,
                alias_to_code=method_context["alias_to_code"],
                allowed_book_codes=method_context["allowed_book_codes"],
                bible_verse_index=method_context["bible_verse_index"],
            )
            for ref in resolved_refs:
                prev = best_scores.get(ref)
                if prev is None or confidence > prev:
                    best_scores[ref] = confidence

    result = [
        {
            "problem_id": problem_id,
            "reference": ref,
            "score": score,
            "method": "direct_prompting",
        }
        for ref, score in sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    print(result)

    return result
