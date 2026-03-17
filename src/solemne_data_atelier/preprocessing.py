from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import pandas as pd

from solemne_data_atelier import DATA_DIR

logger = logging.getLogger(__name__)


BOOK_MAPPING_PATH = DATA_DIR / "raw" / "reference_mapping.json"
BIBLE_TSV_PATH = DATA_DIR / "raw" / "bible.tsv"
TEI_NS = "http://www.tei-c.org/ns/1.0"
TEI_NS_MAP = {"tei": TEI_NS}
SOURCE_TOKEN_STRIP_CHARS = " ,;:.()[]{}<>\"'"
QUOTE_MARKER_START_PREFIX = "__QSTART_"
QUOTE_MARKER_END_PREFIX = "__QEND_"
QUOTE_MARKER_SUFFIX = "__"

REFERENCE_MAPPING_CANDIDATES = (
    DATA_DIR / "raw" / "reference_mapping.json",
    DATA_DIR / "reference_mapping.json",
    DATA_DIR / "hackathon_dataset" / "reference_mapping.json",
)
BIBLE_TSV_CANDIDATES = (
    DATA_DIR / "raw" / "bible.tsv",
    DATA_DIR / "bible.tsv",
    DATA_DIR / "hackathon_dataset" / "bible.tsv",
)


def _resolve_path_with_fallback(primary: Path, candidates: Sequence[Path], what: str) -> Path:
    primary = Path(primary)
    if primary.exists():
        return primary
    for c in candidates:
        c = Path(c)
        if c.exists():
            return c
    searched = [str(primary), *[str(Path(c)) for c in candidates]]
    raise FileNotFoundError(f"{what} not found. Checked: {searched}")


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def compute_quote_source_statistics(
        xml_dir: Path = DATA_DIR / "raw",
        mapping_path: Path = BOOK_MAPPING_PATH,
) -> Dict[str, Any]:
    """
    Compute statistics for <quote> elements with @source in XML files.
    """
    xml_dir = Path(xml_dir)
    mapping_path = _resolve_path_with_fallback(
        primary=Path(mapping_path),
        candidates=REFERENCE_MAPPING_CANDIDATES,
        what="Reference mapping JSON",
    )

    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    with mapping_path.open(encoding="utf-8") as fh:
        reference_mapping = json.load(fh)

    reference_keys = set(reference_mapping.keys())
    all_xml_files = sorted(xml_dir.glob("*.xml"))

    file_quote_counts: Dict[str, int] = {}
    file_bible_abbrev_quote_counts: Dict[str, int] = {}
    parse_errors: List[Dict[str, str]] = []

    for xml_file in all_xml_files:
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError as exc:
            parse_errors.append({"file": xml_file.name, "error": str(exc)})
            continue

        quotes = root.findall(".//tei:quote[@source]", TEI_NS_MAP)
        total_in_file = len(quotes)
        bible_abbrev_in_file = 0

        for quote in quotes:
            source = quote.attrib.get("source", "").strip()
            references = [token.strip(SOURCE_TOKEN_STRIP_CHARS) for token in source.split() if token.strip()]
            books = {ref.split("_", 1)[0] for ref in references}
            if books & reference_keys:
                bible_abbrev_in_file += 1

        if total_in_file > 0:
            file_quote_counts[xml_file.name] = total_in_file
        if bible_abbrev_in_file > 0:
            file_bible_abbrev_quote_counts[xml_file.name] = bible_abbrev_in_file

    file_quote_counts = dict(
        sorted(file_quote_counts.items(), key=lambda item: item[1], reverse=True)
    )
    file_bible_abbrev_quote_counts = dict(
        sorted(file_bible_abbrev_quote_counts.items(), key=lambda item: item[1], reverse=True)
    )

    files_with_quote_source_before_resolution = len(file_quote_counts)
    total_quote_source_before_resolution = int(sum(file_quote_counts.values()))
    files_with_mapped_bible_abbrev_quotes_before_resolution = len(file_bible_abbrev_quote_counts)
    total_mapped_bible_abbrev_quotes_before_resolution = int(sum(file_bible_abbrev_quote_counts.values()))

    return {
        "xml_files_scanned": len(all_xml_files),
        "files_with_quote_source_before_resolution": files_with_quote_source_before_resolution,
        "total_quote_source_before_resolution": total_quote_source_before_resolution,
        "files_with_mapped_bible_abbrev_quotes_before_resolution": (
            files_with_mapped_bible_abbrev_quotes_before_resolution
        ),
        "total_mapped_bible_abbrev_quotes_before_resolution": (
            total_mapped_bible_abbrev_quotes_before_resolution
        ),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "file_quote_counts": file_quote_counts,
        "file_mapped_bible_abbrev_quote_counts": file_bible_abbrev_quote_counts,
        # Backward-compatible aliases.
        "files_with_quote_source": files_with_quote_source_before_resolution,
        "total_quote_source": total_quote_source_before_resolution,
        "files_with_mapped_bible_abbrev_quotes": files_with_mapped_bible_abbrev_quotes_before_resolution,
        "total_mapped_bible_abbrev_quotes": total_mapped_bible_abbrev_quotes_before_resolution,
    }


def _expand_verse_spec_strict_for_bible(verse_spec: str, token: str) -> List[str]:
    verse_spec = verse_spec.strip()
    m_single = re.fullmatch(r"(\d+)", verse_spec)
    if m_single:
        return [str(int(m_single.group(1)))]

    m_range = re.fullmatch(r"(\d+)-(\d+)", verse_spec)
    if m_range:
        start, end = int(m_range.group(1)), int(m_range.group(2))
        if end < start:
            raise ValueError(f"Invalid descending verse range in reference token: {token!r}")
        return [str(v) for v in range(start, end + 1)]

    raise ValueError(f"Invalid verse format for bible.tsv conversion in token: {token!r}")


def _load_bible_book_codes(bible_tsv_path: Path) -> set[str]:
    bible_tsv_path = Path(bible_tsv_path)
    if not bible_tsv_path.exists():
        raise FileNotFoundError(f"Bible TSV not found: {bible_tsv_path}")

    df = pd.read_csv(
        bible_tsv_path,
        sep="\t",
        usecols=["book_code"],
        dtype={"book_code": str},
    )
    return {
        str(v).strip().lower()
        for v in df["book_code"].dropna().astype(str).tolist()
        if str(v).strip()
    }


def _load_bible_verse_lookup(bible_tsv_path: Path) -> Dict[str, str]:
    bible_tsv_path = Path(bible_tsv_path)
    if not bible_tsv_path.exists():
        raise FileNotFoundError(f"Bible TSV not found: {bible_tsv_path}")

    df = pd.read_csv(
        bible_tsv_path,
        sep="\t",
        usecols=["book_code", "chapter_number", "verse_index", "text"],
    )
    df = df.copy()
    df["book_code"] = df["book_code"].astype(str).str.strip().str.lower()
    df["chapter_number"] = pd.to_numeric(df["chapter_number"], errors="raise").astype(int)
    df["verse_index"] = pd.to_numeric(df["verse_index"], errors="raise").astype(int)
    df["reference"] = (
        df["book_code"]
        + "_"
        + df["chapter_number"].astype(str)
        + ":"
        + df["verse_index"].astype(str)
    )
    return {
        str(row["reference"]).strip().lower(): str(row["text"])
        for _, row in df.iterrows()
    }


def _import_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required when retrieving similar verses. Install it with: pip install chromadb"
        ) from exc
    return chromadb


def _list_chroma_collection_names(client: Any) -> List[str]:
    names: List[str] = []
    for item in client.list_collections():
        if isinstance(item, str):
            names.append(item)
            continue
        name = getattr(item, "name", None)
        if name:
            names.append(str(name))
    return names


def _parse_preferred_model_key(preferred_model_key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    model_key = str(preferred_model_key or "").strip()
    if not model_key:
        return None, None
    if ":" in model_key:
        provider, model_name = model_key.split(":", 1)
        return provider.strip().lower() or None, model_name.strip().lower() or None
    return None, model_key.lower()


def _get_collection_metadata(chroma_client: Any, name: str) -> Dict[str, Any]:
    try:
        collection = chroma_client.get_collection(name)
        metadata = getattr(collection, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
    except Exception:
        logger.debug("Could not fetch metadata for collection %s", name, exc_info=True)
    return {}


def _resolve_chroma_collection_name(
        chroma_client: Any,
        collection_name: Optional[str],
        preferred_model_key: Optional[str],
) -> str:
    available = sorted(set(_list_chroma_collection_names(chroma_client)))
    if not available:
        raise ValueError("No collections found in Chroma persist directory.")

    if collection_name:
        if collection_name not in available:
            raise ValueError(
                f"Requested collection {collection_name!r} not found in Chroma. "
                f"Available: {sorted(available)}"
            )
        return collection_name

    preferred_provider, preferred_model = _parse_preferred_model_key(preferred_model_key)
    candidates: List[Tuple[str, Optional[str], Optional[str]]] = []

    for name in available:
        metadata = _get_collection_metadata(chroma_client, name)
        provider = str(metadata.get("provider", "")).strip().lower() or None
        model_name = str(metadata.get("model_name", "")).strip().lower() or None
        if provider is None or model_name is None:
            continue
        candidates.append((name, provider, model_name))

    if not candidates:
        raise ValueError(
            "No collections with metadata keys 'provider' and 'model_name' found in Chroma."
        )

    if preferred_model is not None:
        matching = []
        for name, provider, model_name in candidates:
            if model_name != preferred_model:
                continue
            if preferred_provider is not None and provider != preferred_provider:
                continue
            matching.append(name)
        if matching:
            return sorted(matching)[0]
        raise ValueError(
            f"No Chroma collection metadata match for model key {preferred_model_key!r}."
        )

    default_key = ("openai", "text-embedding-3-large")
    for name, provider, model_name in candidates:
        if provider == default_key[0] and model_name == default_key[1]:
            return name

    # Fallback: first collection in sorted order.
    return available[0]


def _get_top_similar_references_from_collection(
        collection: Any,
        reference: str,
        top_k: int,
        min_cosine_similarity: float,
        oversample: int = 10,
) -> List[Tuple[str, float]]:
    normalized_ref = str(reference).strip().lower()
    if not normalized_ref:
        return []

    fetched = collection.get(
        where={"reference": normalized_ref},
        limit=1,
        include=["embeddings", "metadatas"],
    )
    embeddings = fetched.get("embeddings", []) if isinstance(fetched, dict) else []
    if embeddings is None:
        return []
    if hasattr(embeddings, "__len__") and len(embeddings) == 0:
        return []

    query_embedding = embeddings[0]
    n_results = max(int(top_k) + 1, int(top_k) + int(oversample))
    queried = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["distances", "metadatas"],
    )

    ids_batch = queried.get("ids", []) if isinstance(queried, dict) else []
    dists_batch = queried.get("distances", []) if isinstance(queried, dict) else []
    metas_batch = queried.get("metadatas", []) if isinstance(queried, dict) else []

    ids = ids_batch[0] if ids_batch else []
    dists = dists_batch[0] if dists_batch else []
    metas = metas_batch[0] if metas_batch else []

    out: List[Tuple[str, float]] = []
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        cand_ref = str(meta.get("reference", ids[i])).strip().lower()
        if cand_ref == normalized_ref:
            continue

        dist = float(dists[i]) if i < len(dists) else 1.0
        cosine_sim = 1.0 - dist
        if cosine_sim < float(min_cosine_similarity):
            continue

        out.append((cand_ref, cosine_sim))
        if len(out) >= int(top_k):
            break

    return out


def resolve_biblical_source_references(
        source: Optional[str],
        reference_mapping: Dict[str, str],
        bible_book_codes: Optional[set[str]] = None,
) -> List[str]:
    """
    Resolve a <quote @source> value to verse-level references in bible.tsv format.
    """
    if not source:
        return []

    resolved: List[str] = []
    seen: set[str] = set()

    for raw_token in source.split():
        token = raw_token.strip(SOURCE_TOKEN_STRIP_CHARS)
        if not token:
            continue

        if token.lower().startswith("cf_"):
            token = token[3:]

        book_candidate = token.split("_", 1)[0]
        if book_candidate not in reference_mapping:
            continue

        parts = token.split("_")
        if len(parts) == 3:
            book, chapter, verse_spec = parts
        elif len(parts) == 2 and ":" in parts[1]:
            book = parts[0]
            chapter, verse_spec = parts[1].split(":", 1)
        else:
            print(f"Biblical reference token not convertible: {token!r}")
            continue

        if book != book_candidate:
            raise ValueError(f"Unexpected book token mismatch for reference: {token!r}")
        if not chapter.isdigit():
            raise ValueError(f"Invalid chapter format for bible.tsv conversion in token: {token!r}")

        mapped_book = str(reference_mapping[book]).strip().lower()
        if not mapped_book:
            raise ValueError(f"Missing book mapping for biblical token: {token!r}")
        if bible_book_codes is not None and mapped_book not in bible_book_codes:
            raise ValueError(
                f"Mapped book code {mapped_book!r} from token {token!r} not found in bible.tsv"
            )

        for verse in _expand_verse_spec_strict_for_bible(verse_spec, token):
            ref = f"{mapped_book}_{int(chapter)}:{verse}"
            if ref not in seen:
                seen.add(ref)
                resolved.append(ref)

    return resolved


def _strip_markers_and_collect_spans(
        marked_text: str,
        marker_to_refs: Dict[int, List[str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    start_re = re.compile(rf"{QUOTE_MARKER_START_PREFIX}(\d+){QUOTE_MARKER_SUFFIX}")
    end_re = re.compile(rf"{QUOTE_MARKER_END_PREFIX}(\d+){QUOTE_MARKER_SUFFIX}")

    plain_tokens: List[str] = []
    open_spans: Dict[int, int] = {}
    spans: List[Dict[str, Any]] = []
    current_len = 0

    for token in marked_text.split():
        m_start = start_re.fullmatch(token)
        if m_start:
            marker_id = int(m_start.group(1))
            open_spans[marker_id] = current_len
            continue

        m_end = end_re.fullmatch(token)
        if m_end:
            marker_id = int(m_end.group(1))
            span_start = open_spans.pop(marker_id, None)
            if span_start is None:
                continue
            if current_len <= span_start:
                continue
            spans.append(
                {
                    "span_start": span_start,
                    "span_end": current_len,
                    "resolved_references": marker_to_refs.get(marker_id, []),
                }
            )
            continue

        if plain_tokens:
            current_len += 1
        plain_tokens.append(token)
        current_len += len(token)

    plain_text = " ".join(plain_tokens)
    spans.sort(key=lambda x: (x["span_start"], x["span_end"]))
    return plain_text, spans


def _extract_paragraph_text_and_biblical_spans(
        para: ET.Element,
        reference_mapping: Dict[str, str],
        bible_book_codes: set[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    parts: List[str] = []
    marker_to_refs: Dict[int, List[str]] = {}
    marker_counter = 0

    def walk(node: ET.Element):
        nonlocal marker_counter

        if _local_name(node.tag) == "note" and node.attrib.get("type") == "source":
            return

        marker_id: Optional[int] = None
        if _local_name(node.tag) == "quote":
            refs = resolve_biblical_source_references(
                node.attrib.get("source"),
                reference_mapping,
                bible_book_codes,
            )
            if refs:
                marker_id = marker_counter
                marker_counter += 1
                marker_to_refs[marker_id] = refs
                parts.append(f" {QUOTE_MARKER_START_PREFIX}{marker_id}{QUOTE_MARKER_SUFFIX} ")

        if node.text:
            parts.append(node.text)

        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)

        if marker_id is not None:
            parts.append(f" {QUOTE_MARKER_END_PREFIX}{marker_id}{QUOTE_MARKER_SUFFIX} ")

    walk(para)
    marked_text = _normalize_whitespace("".join(parts))
    if not marked_text:
        return "", []

    return _strip_markers_and_collect_spans(marked_text, marker_to_refs)


def prepare_dataset(
        xml_dir: Path = DATA_DIR / "raw",
        output_dir: Path = DATA_DIR / "preprocessed",
        mapping_path: Path = BOOK_MAPPING_PATH,
        bible_tsv_path: Path = BIBLE_TSV_PATH,
) -> Dict[str, int]:
    """
    Convert raw XML files into a competition-style preprocessed dataset.
    """
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    mapping_path = _resolve_path_with_fallback(
        primary=Path(mapping_path),
        candidates=REFERENCE_MAPPING_CANDIDATES,
        what="Reference mapping JSON",
    )
    bible_tsv_path = _resolve_path_with_fallback(
        primary=Path(bible_tsv_path),
        candidates=BIBLE_TSV_CANDIDATES,
        what="Bible TSV",
    )

    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")

    with mapping_path.open(encoding="utf-8") as fh:
        reference_mapping = json.load(fh)
    bible_book_codes = _load_bible_book_codes(bible_tsv_path)

    problems_dir = output_dir / "problems"
    solutions_dir = output_dir / "solutions"
    problems_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "xml_files_scanned": 0,
        "xml_files_with_biblical_quotes_after_resolution": 0,
        "total_biblical_quote_spans_after_resolution": 0,
        "parse_errors": 0,
    }
    unique_biblical_verses_after_resolution: set[str] = set()
    unique_biblical_books_after_resolution: set[str] = set()
    unique_problem_verse_pairs_after_resolution: set[Tuple[str, str]] = set()

    for xml_file in sorted(xml_dir.glob("*.xml")):
        stats["xml_files_scanned"] += 1
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError as exc:
            logger.warning("Skipping %s due to parse error: %s", xml_file.name, exc)
            stats["parse_errors"] += 1
            continue

        body = root.find(".//tei:body", TEI_NS_MAP)
        if body is None:
            continue

        paragraphs = body.findall(".//tei:p", TEI_NS_MAP)
        if not paragraphs:
            continue

        doc_parts: List[str] = []
        doc_spans: List[Dict[str, Any]] = []
        doc_len = 0

        for para in paragraphs:
            try:
                para_text, para_spans = _extract_paragraph_text_and_biblical_spans(
                    para,
                    reference_mapping,
                    bible_book_codes,
                )
            except ValueError as exc:
                raise ValueError(f"{xml_file.name}: {exc}") from exc

            if doc_parts:
                doc_len += 1

            para_offset = doc_len
            doc_parts.append(para_text)
            doc_len += len(para_text)

            for span in para_spans:
                refs = span.get("resolved_references", [])
                if not refs:
                    continue
                for ref in refs:
                    normalized_ref = str(ref).strip().lower()
                    if not normalized_ref:
                        continue
                    unique_biblical_verses_after_resolution.add(normalized_ref)
                    unique_problem_verse_pairs_after_resolution.add((xml_file.stem, normalized_ref))
                    if "_" in normalized_ref:
                        unique_biblical_books_after_resolution.add(
                            normalized_ref.split("_", 1)[0]
                        )
                doc_spans.append(
                    {
                        "span_start": para_offset + int(span["span_start"]),
                        "span_end": para_offset + int(span["span_end"]),
                        "resolved_references": refs,
                    }
                )

        if not doc_spans:
            continue

        problem_path = problems_dir / f"{xml_file.stem}.txt"
        solution_path = solutions_dir / f"{xml_file.stem}.json"

        problem_path.write_text("\n".join(doc_parts), encoding="utf-8")
        solution_path.write_text(
            json.dumps(doc_spans, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        stats["xml_files_with_biblical_quotes_after_resolution"] += 1
        stats["total_biblical_quote_spans_after_resolution"] += len(doc_spans)

    stats["xml_files_with_biblical_quotes"] = stats["xml_files_with_biblical_quotes_after_resolution"]
    stats["total_biblical_quote_spans"] = stats["total_biblical_quote_spans_after_resolution"]
    stats["total_individual_verses_after_resolution"] = len(unique_biblical_verses_after_resolution)
    stats["total_biblical_books_after_resolution"] = len(unique_biblical_books_after_resolution)
    stats["total_verses_true_positives_after_resolution"] = len(
        unique_problem_verse_pairs_after_resolution
    )

    # Simple summary fields.
    stats["files"] = stats["xml_files_with_biblical_quotes_after_resolution"]
    stats["spans"] = stats["total_biblical_quote_spans_after_resolution"]
    stats["individual_verses"] = stats["total_individual_verses_after_resolution"]
    stats["biblical_books"] = stats["total_biblical_books_after_resolution"]
    stats["total_verses_true_positives"] = stats["total_verses_true_positives_after_resolution"]
    return stats


def produce_visual_validation_data(
        dataset_dir: Path = DATA_DIR / "preprocessed",
        context_chars: int = 50,
        max_rows: Optional[int] = None,
        output_tsv: Optional[Path] = None,
        output_html: Optional[Path] = None,
        include_verse_texts: bool = False,
        bible_tsv_path: Path = BIBLE_TSV_PATH,
        include_similar_verses: bool = False,
        similar_verses_top_k: int = 2,
        similar_verses_min_cosine_similarity: float = 0.8,
        similar_verses_collection_name: Optional[str] = None,
        similar_verses_model_key: Optional[str] = None,
        chroma_persist_directory: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Visual validation helper for span-based datasets.

    Optional features:
    - include_verse_texts=True:
      add a column with resolved references paired with verse text from bible.tsv.
    - include_similar_verses=True:
      for each resolved reference, retrieve top-N nearest biblical verses from
      Chroma (excluding self) with cosine similarity >= threshold.
    - collection resolution uses Chroma collection metadata (provider/model_name).
    """
    dataset_dir = Path(dataset_dir)
    problems_dir = dataset_dir / "problems"
    solutions_dir = dataset_dir / "solutions"

    if not problems_dir.exists():
        raise FileNotFoundError(f"Problems directory not found: {problems_dir}")
    if not solutions_dir.exists():
        raise FileNotFoundError(f"Solutions directory not found: {solutions_dir}")

    verse_lookup: Optional[Dict[str, str]] = None
    if include_verse_texts:
        dataset_local_bible_candidates = (
            dataset_dir.parent / "raw" / "bible.tsv",
            dataset_dir.parent / "bible.tsv",
        )
        resolved_bible_tsv_path = _resolve_path_with_fallback(
            primary=Path(bible_tsv_path),
            candidates=(*dataset_local_bible_candidates, *BIBLE_TSV_CANDIDATES),
            what="Bible TSV",
        )
        verse_lookup = _load_bible_verse_lookup(resolved_bible_tsv_path)

    similar_verse_cache: Dict[str, List[Tuple[str, float]]] = {}
    similar_verse_collection: Optional[Any] = None
    if include_similar_verses:
        if similar_verses_top_k < 1:
            raise ValueError("similar_verses_top_k must be >= 1")
        if not 0.0 <= float(similar_verses_min_cosine_similarity) <= 1.0:
            raise ValueError("similar_verses_min_cosine_similarity must be in [0, 1]")

        if chroma_persist_directory is None:
            chroma_candidates = (
                dataset_dir.parent / "vectorstores" / "chroma",
                DATA_DIR / "vectorstores" / "chroma",
            )
            resolved_chroma_dir = next((Path(p) for p in chroma_candidates if Path(p).exists()), None)
            if resolved_chroma_dir is None:
                raise FileNotFoundError(
                    "Could not find Chroma persist directory. "
                    f"Checked: {[str(Path(p)) for p in chroma_candidates]}"
                )
        else:
            resolved_chroma_dir = Path(chroma_persist_directory)
            if not resolved_chroma_dir.exists():
                raise FileNotFoundError(f"Chroma persist directory not found: {resolved_chroma_dir}")

        chromadb = _import_chromadb()
        chroma_client = chromadb.PersistentClient(path=str(resolved_chroma_dir))
        resolved_collection_name = _resolve_chroma_collection_name(
            chroma_client=chroma_client,
            collection_name=similar_verses_collection_name,
            preferred_model_key=similar_verses_model_key,
        )
        similar_verse_collection = chroma_client.get_collection(resolved_collection_name)

    rows: List[Dict[str, Any]] = []

    for solution_path in sorted(solutions_dir.glob("*.json")):
        stem = solution_path.stem
        problem_path = problems_dir / f"{stem}.txt"
        if not problem_path.exists():
            logger.warning("Missing problem file for solution %s", solution_path.name)
            continue

        text = problem_path.read_text(encoding="utf-8")
        text_len = len(text)

        with solution_path.open(encoding="utf-8") as fh:
            spans = json.load(fh)

        if not isinstance(spans, list):
            logger.warning("Skipping %s because it is not a list", solution_path.name)
            continue

        for idx, span in enumerate(spans):
            if not isinstance(span, dict):
                continue

            start = span.get("span_start")
            end = span.get("span_end")
            refs = span.get("resolved_references", [])

            is_valid = isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= text_len
            quote_text = text[start:end] if is_valid else ""

            if is_valid:
                left_context = text[max(0, start - context_chars):start]
                right_context = text[end:min(text_len, end + context_chars)]
            else:
                left_context = ""
                right_context = ""

            row: Dict[str, Any] = {
                "document": stem,
                "span_index": idx,
                "span_start": start,
                "span_end": end,
                "span_length": (end - start) if is_valid else None,
                "valid_span": is_valid,
                "quote_text": quote_text,
                "resolved_references": " ".join(refs) if isinstance(refs, list) else str(refs),
                # "left_context": left_context,
                # "right_context": right_context,
                "visual_snippet": f"{left_context}[{quote_text}]{right_context}" if is_valid else "",
            }

            if include_verse_texts:
                refs_list = refs if isinstance(refs, list) else []
                pairs: List[str] = []
                for ref in refs_list:
                    normalized_ref = str(ref).strip().lower()
                    verse_text = (verse_lookup or {}).get(normalized_ref, "")
                    pairs.append(f"{normalized_ref} :: {verse_text}" if verse_text else f"{normalized_ref} :: [MISSING]")
                row["resolved_reference_text_pairs"] = " || ".join(pairs)

            if include_similar_verses:
                refs_list = refs if isinstance(refs, list) else []
                per_ref_similar: List[str] = []
                for ref in refs_list:
                    normalized_ref = str(ref).strip().lower()
                    if normalized_ref not in similar_verse_cache:
                        similar_verse_cache[normalized_ref] = _get_top_similar_references_from_collection(
                            collection=similar_verse_collection,
                            reference=normalized_ref,
                            top_k=similar_verses_top_k,
                            min_cosine_similarity=similar_verses_min_cosine_similarity,
                        )
                    neighbors = similar_verse_cache.get(normalized_ref, [])
                    if not neighbors:
                        per_ref_similar.append(f"{normalized_ref} => []")
                        continue
                    formatted = ", ".join(f"{cand_ref} ({score:.3f})" for cand_ref, score in neighbors)
                    per_ref_similar.append(f"{normalized_ref} => [{formatted}]")
                row["resolved_reference_top_similar"] = " || ".join(per_ref_similar)

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["document", "span_start", "span_end"], na_position="last").reset_index(drop=True)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if output_tsv is not None:
        output_tsv = Path(output_tsv)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_tsv, sep="\t", index=False)

    if output_html is not None:
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        html_df = df.copy()
        html_df["visual_snippet"] = html_df["visual_snippet"].apply(
            lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        html_df["visual_snippet"] = html_df["visual_snippet"].str.replace(
            r"\[(.*?)\]",
            r"<mark>\1</mark>",
            regex=True,
        )
        html_df.to_html(output_html, index=False, escape=False)

    return df


def create_hackathon_archive(
        data_dir: Path = DATA_DIR,
        archive_path: Path = DATA_DIR / "hackathon_data.zip",
) -> Path:
    """
    Create a zip archive consumable by _download_and_save extraction logic.

    The archive stores required entries at top level:
    - raw/
    - task/
    - vectorstores/
    - bible.tsv
    - book_mapping.tsv
    - reference_mapping.json
    """
    data_dir = Path(data_dir)
    archive_path = Path(archive_path)
    required_entries = [
        Path("raw"),
        Path("task"),
        Path("vectorstores"),
        Path("bible.tsv"),
        Path("book_mapping.tsv"),
        Path("reference_mapping.json"),
    ]

    missing = [str(rel) for rel in required_entries if not (data_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot build archive because required paths are missing in "
            f"{data_dir}: {missing}"
        )

    def _should_skip(path: Path | str) -> bool:
        name = Path(path).name
        if name == ".DS_Store":
            return True
        if name.startswith("__MACOSX"):
            return True
        if name == "__pycache__":
            return True
        return False

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    files_added = 0
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in required_entries:
            src = data_dir / rel
            if src.is_file():
                if _should_skip(src):
                    continue
                zf.write(src, arcname=rel.as_posix())
                files_added += 1
                continue

            for file_path in sorted(src.rglob("*")):
                if not file_path.is_file():
                    continue
                if any(_should_skip(part) for part in file_path.parts):
                    continue
                arcname = file_path.relative_to(data_dir).as_posix()
                zf.write(file_path, arcname=arcname)
                files_added += 1

    logger.info(
        "Created archive %s with %d file(s) from %s",
        archive_path,
        files_added,
        data_dir,
    )
    return archive_path
