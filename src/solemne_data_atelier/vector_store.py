from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

from solemne_data_atelier import DATA_DIR
from solemne_data_atelier.preprocessing import BIBLE_TSV_PATH

logger = logging.getLogger(__name__)


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


def load_bible_tsv_with_references(bible_tsv_path: Path = BIBLE_TSV_PATH) -> pd.DataFrame:
    bible_tsv_path = _resolve_path_with_fallback(
        primary=Path(bible_tsv_path),
        candidates=BIBLE_TSV_CANDIDATES,
        what="Bible TSV",
    )

    df = pd.read_csv(bible_tsv_path, sep="\t")
    required = {"book_code", "chapter_number", "verse_index", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"bible.tsv is missing required columns: {sorted(missing)}")

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
    return df


def _import_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for vector-store building. Install it with: pip install chromadb"
        ) from exc
    return chromadb


def _import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for HF embedding models. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer


def _import_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for OpenAI embeddings. Install it with: pip install openai"
        ) from exc
    return OpenAI


def _sanitize_component(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return token or "default"


def get_biblical_collection_name(
        provider: str,
        model_name: str,
        collection_prefix: str = "biblical",
        max_len: int = 63,
) -> str:
    prefix = _sanitize_component(collection_prefix)
    provider_part = _sanitize_component(provider)
    model_part = _sanitize_component(model_name)
    base = f"{prefix}__{provider_part}__{model_part}"

    if len(base) <= max_len:
        return base

    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    keep = max_len - len(digest) - 2
    trimmed = base[:keep].rstrip("_-")
    name = f"{trimmed}__{digest}"
    if len(name) < 3:
        name = f"{prefix[:1] or 'b'}__{digest}"
    return name


def _iter_batches(values: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, int, Sequence[Any]]]:
    step = max(1, int(batch_size))
    for start in range(0, len(values), step):
        end = min(len(values), start + step)
        yield start, end, values[start:end]


def _build_unique_ids(references: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, ref in enumerate(references):
        base = str(ref).strip() or f"row_{i}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}__{count}")
    return out


def _embed_batch_openai(
        texts: Sequence[str],
        model_name: str,
        client: Any,
        openai_dimensions: Optional[int] = None,
) -> List[List[float]]:
    kwargs: Dict[str, Any] = {"model": model_name, "input": list(texts)}
    if openai_dimensions is not None:
        kwargs["dimensions"] = int(openai_dimensions)
    response = client.embeddings.create(**kwargs)
    payload = sorted(response.data, key=lambda item: getattr(item, "index", 0))
    return [list(item.embedding) for item in payload]


def build_biblical_vectorstores(
        hf_models: Sequence[str],
        openai_models: Sequence[str],
        bible_tsv_path: Path = BIBLE_TSV_PATH,
        persist_directory: Path = DATA_DIR / "vectorstores" / "chroma",
        collection_prefix: str = "biblical",
        rebuild_collections: bool = False,
        hf_batch_size: int = 64,
        openai_batch_size: int = 128,
        chroma_upsert_batch_size: int = 1024,
        device: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_organization: Optional[str] = None,
        openai_dimensions: Optional[int] = None,
        show_progress: bool = True,
) -> Dict[str, Any]:
    hf_models = [str(m).strip() for m in hf_models if str(m).strip()]
    openai_models = [str(m).strip() for m in openai_models if str(m).strip()]
    if not hf_models and not openai_models:
        raise ValueError("No embedding models provided. Use --hf-model and/or --openai-model.")

    bible_df = load_bible_tsv_with_references(bible_tsv_path=bible_tsv_path)
    references = bible_df["reference"].astype(str).tolist()
    texts = bible_df["text"].fillna("").astype(str).tolist()
    row_ids = _build_unique_ids(references)

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    chromadb = _import_chromadb()
    chroma_client = chromadb.PersistentClient(path=str(persist_directory))

    openai_client = None
    if openai_models:
        OpenAI = _import_openai_client()
        resolved_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI models requested but API key is missing. "
                "Provide --openai-api-key or set OPENAI_API_KEY."
            )
        openai_client = OpenAI(
            api_key=resolved_api_key,
            base_url=openai_base_url,
            organization=openai_organization,
        )

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "persist_directory": str(persist_directory),
        "bible_tsv_path": str(_resolve_path_with_fallback(Path(bible_tsv_path), BIBLE_TSV_CANDIDATES, "Bible TSV")),
        "collection_prefix": collection_prefix,
        "total_verses": len(texts),
        "collections": [],
    }

    for provider, model_name in [("openai", m) for m in openai_models] + [("hf", m) for m in hf_models]:
        collection_name = get_biblical_collection_name(
            provider=provider,
            model_name=model_name,
            collection_prefix=collection_prefix,
        )
        logger.info("Building collection %s for %s model %s", collection_name, provider, model_name)

        if rebuild_collections:
            try:
                chroma_client.delete_collection(collection_name)
                logger.info("Deleted existing collection %s", collection_name)
            except Exception:
                pass

        metadata = {
            "hnsw:space": "cosine",
            "provider": provider,
            "model_name": model_name,
            "collection_prefix": collection_prefix,
        }
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
        )
        # Keep provider/model identity in DB so lookup does not depend on external mapping files.
        try:
            existing_metadata = collection.metadata or {}
            merged_metadata = {**existing_metadata, **metadata}
            collection.modify(metadata=merged_metadata)
        except Exception:
            logger.debug("Could not update metadata for collection %s", collection_name, exc_info=True)

        hf_model = None
        if provider == "hf":
            SentenceTransformer = _import_sentence_transformers()
            hf_model = SentenceTransformer(model_name, device=device)

        if provider == "openai":
            provider_batch_size = min(max(1, int(openai_batch_size)), max(1, int(chroma_upsert_batch_size)))
        else:
            provider_batch_size = max(1, int(chroma_upsert_batch_size))

        total_batches = (len(texts) + provider_batch_size - 1) // provider_batch_size
        batch_iter = _iter_batches(texts, provider_batch_size)
        if show_progress:
            batch_iter = tqdm(
                batch_iter,
                total=total_batches,
                desc=f"{provider}:{model_name}",
                unit="batch",
            )

        for start, end, text_batch in batch_iter:
            if provider == "openai":
                embeddings = _embed_batch_openai(
                    texts=text_batch,
                    model_name=model_name,
                    client=openai_client,
                    openai_dimensions=openai_dimensions,
                )
            else:
                embeddings_arr = hf_model.encode(
                    list(text_batch),
                    batch_size=max(1, int(hf_batch_size)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                embeddings = embeddings_arr.tolist()

            metadatas = [
                {
                    "reference": references[i],
                    "provider": provider,
                    "model_name": model_name,
                    "row_index": int(i),
                }
                for i in range(start, end)
            ]
            collection.upsert(
                ids=row_ids[start:end],
                documents=list(text_batch),
                embeddings=embeddings,
                metadatas=metadatas,
            )

        record = {
            "collection_name": collection_name,
            "provider": provider,
            "model_name": model_name,
            "count": int(collection.count()),
        }
        summary["collections"].append(record)
        logger.info("Collection %s ready (count=%s)", collection_name, collection.count())

    return summary
