from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from solemne_data_atelier.evaluation import normalize_reference
from solemne_data_atelier.utils import split_into_chunks


def _import_chromadb():
    import chromadb

    return chromadb


def _import_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def _import_openai_client():
    from openai import OpenAI

    return OpenAI


def _list_collection_names(client: Any) -> List[str]:
    out = []
    for item in client.list_collections():
        if isinstance(item, str):
            out.append(item)
        else:
            name = getattr(item, "name", None)
            if name:
                out.append(name)
    return out


def _resolve_collection_name(
    client: Any,
    *,
    provider: str,
    model_name: str,
    collection_prefix: str,
    chroma_dir: Path,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_organization: Optional[str] = None,
) -> str:
    provider_norm = str(provider).strip().lower()
    model_norm = str(model_name).strip().lower()
    prefix_norm = str(collection_prefix).strip().lower()

    if provider_norm not in {"hf", "openai"}:
        raise ValueError("provider must be 'hf' or 'openai'")

    def _matches(metadata: Any) -> bool:
        if not isinstance(metadata, dict):
            return False
        p = str(metadata.get("provider", "")).strip().lower()
        m = str(metadata.get("model_name", "")).strip().lower()
        pref = str(metadata.get("collection_prefix", "")).strip().lower()
        return p == provider_norm and m == model_norm and pref == prefix_norm

    for name in sorted(set(_list_collection_names(client))):
        try:
            collection = client.get_collection(name)
        except Exception:
            continue
        if _matches(getattr(collection, "metadata", None)):
            return name

    from solemne_data_atelier.vector_store import build_biblical_vectorstores

    build_biblical_vectorstores(
        hf_models=[model_name] if provider_norm == "hf" else [],
        openai_models=[model_name] if provider_norm == "openai" else [],
        persist_directory=Path(chroma_dir),
        collection_prefix=prefix_norm,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_organization=openai_organization,
        show_progress=True,
    )

    for name in sorted(set(_list_collection_names(client))):
        try:
            collection = client.get_collection(name)
        except Exception:
            continue
        if _matches(getattr(collection, "metadata", None)):
            return name

    raise ValueError(
        f"No biblical collection found for provider={provider_norm}, model={model_norm}, prefix={prefix_norm}."
    )


def _build_query_embedder(
    provider: str,
    model_name: str,
    *,
    hf_batch_size: int = 64,
    openai_batch_size: int = 128,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_organization: Optional[str] = None,
    query_prompt: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Callable[[Sequence[str]], List[List[float]]]:
    provider_norm = provider.lower().strip()

    if provider_norm == "hf":
        SentenceTransformer = _import_sentence_transformer()
        model = SentenceTransformer(model_name, device=device)

        def embed(texts: Sequence[str]) -> List[List[float]]:
            encode_kwargs: Dict[str, Any] = {
                "batch_size": max(1, int(hf_batch_size)),
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
            }
            if isinstance(query_prompt, dict):
                encode_kwargs.update(query_prompt)
            vecs = model.encode(list(texts), **encode_kwargs)
            return vecs.tolist()

        return embed

    if provider_norm == "openai":
        OpenAI = _import_openai_client()
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI provider requires OPENAI_API_KEY or openai_api_key.")
        client = OpenAI(api_key=api_key, base_url=openai_base_url, organization=openai_organization)

        def embed(texts: Sequence[str]) -> List[List[float]]:
            all_embeddings: List[List[float]] = []
            batch_size = max(1, int(openai_batch_size))
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i : i + batch_size])
                response = client.embeddings.create(model=model_name, input=batch)
                data = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([list(item.embedding) for item in data])
            return all_embeddings

        return embed

    raise ValueError("provider must be 'hf' or 'openai'")


def build_embedding_method_context(
    *,
    provider: str,
    model_name: str,
    chroma_dir: Path,
    collection_prefix: str = "biblical",
    hf_batch_size: int = 64,
    openai_batch_size: int = 128,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_organization: Optional[str] = None,
    query_prompt: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    chromadb = _import_chromadb()
    client = chromadb.PersistentClient(path=str(chroma_dir))

    collection_name = _resolve_collection_name(
        client,
        provider=provider,
        model_name=model_name,
        collection_prefix=collection_prefix,
        chroma_dir=Path(chroma_dir),
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_organization=openai_organization,
    )
    collection = client.get_collection(collection_name)

    embed_query = _build_query_embedder(
        provider=provider,
        model_name=model_name,
        hf_batch_size=hf_batch_size,
        openai_batch_size=openai_batch_size,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_organization=openai_organization,
        query_prompt=query_prompt,
        device=device,
    )

    return {
        "method_name": "simple_embedding",
        "provider": provider,
        "model_name": model_name,
        "collection_name": collection_name,
        "collection": collection,
        "embed_query": embed_query,
    }


def simple_embedding_method(
    problem_id: str,
    problem_text: str,
    method_context: Dict[str, Any],
    *,
    mode: str = "sentence",
    sentences_per_chunk: int = 2,
    sentence_stride: int = 1,
    char_chunk_size: int = 500,
    char_chunk_overlap: int = 100,
    min_chunk_chars: int = 30,
    top_k: int = 5,
    similarity_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    chunks = split_into_chunks(
        problem_text,
        mode=mode,
        sentences_per_chunk=sentences_per_chunk,
        sentence_stride=sentence_stride,
        char_chunk_size=char_chunk_size,
        char_chunk_overlap=char_chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
    if not chunks:
        return []

    query_embeddings = method_context["embed_query"](chunks)
    query_result = method_context["collection"].query(
        query_embeddings=query_embeddings,
        n_results=max(1, int(top_k)),
        include=["distances", "metadatas"],
    )

    best_scores: Dict[str, float] = {}
    ids_batch = query_result.get("ids", [])
    dists_batch = query_result.get("distances", [])
    metas_batch = query_result.get("metadatas", [])

    for i in range(len(chunks)):
        ids_i = ids_batch[i] if i < len(ids_batch) else []
        dists_i = dists_batch[i] if i < len(dists_batch) else []
        metas_i = metas_batch[i] if i < len(metas_batch) else []

        for j in range(len(ids_i)):
            meta = metas_i[j] if j < len(metas_i) and isinstance(metas_i[j], dict) else {}
            ref = normalize_reference(meta.get("reference", ids_i[j]))
            dist = float(dists_i[j]) if j < len(dists_i) else 1.0
            similarity = 1.0 - dist

            if similarity >= float(similarity_threshold):
                prev = best_scores.get(ref)
                if prev is None or similarity > prev:
                    best_scores[ref] = similarity

    return [
        {
            "problem_id": problem_id,
            "reference": ref,
            "score": score,
            "method": "simple_embedding",
        }
        for ref, score in sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    ]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dump_yaml_fallback(data: Dict[str, Any]) -> str:
    lines = []
    for key, value in data.items():
        rendered = json.dumps(_to_jsonable(value), ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


def _save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_data = _to_jsonable(data)
    try:
        import yaml

        payload = yaml.safe_dump(payload_data, sort_keys=False, allow_unicode=True, width=1000000)
    except Exception:
        payload = _dump_yaml_fallback(payload_data)
    path.write_text(payload, encoding="utf-8")


def save_simple_embedding_run(
    *,
    simple_embedding_runs_dir: Path,
    method_context: Dict[str, Any],
    method_kwargs: Optional[Dict[str, Any]],
    metrics: Dict[str, float],
    predictions_count: int,
    selected_problem_ids: Sequence[str],
) -> Path:
    """
    Persist simple embedding artifacts in a timestamped run folder (YYYYMMDD_HHMMSS),
    analogous to passim runs.
    """
    run_dir = Path(simple_embedding_runs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    params_payload = {
        "timestamp": run_dir.name,
        "run_dir": str(run_dir),
        "method": "simple_embedding",
        "provider": method_context.get("provider"),
        "model_name": method_context.get("model_name"),
        "collection_name": method_context.get("collection_name"),
        "method_kwargs": dict(method_kwargs or {}),
        "selected_problem_count": int(len(list(selected_problem_ids))),
        "selected_problem_ids": list(selected_problem_ids),
    }
    metrics_payload = {
        **{k: float(v) for k, v in metrics.items()},
        "predictions_produced": int(predictions_count),
    }

    _save_yaml(params_payload, run_dir / "params.yaml")
    _save_yaml(metrics_payload, run_dir / "metrics.yaml")
    return run_dir
