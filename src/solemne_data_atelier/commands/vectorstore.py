import os
from pathlib import Path

import solemne_data_atelier
from solemne_data_atelier import DATA_DIR


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "vectorstore",
        help="Build Chroma biblical embedding collections for OpenAI and HF models.",
    )
    parser.add_argument(
        "--hf-model",
        action="append",
        dest="hf_models",
        default=[],
        help="Hugging Face SentenceTransformer model name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--openai-model",
        action="append",
        dest="openai_models",
        default=[],
        help="OpenAI embedding model name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--bible-tsv-path",
        type=Path,
        default=DATA_DIR / "bible.tsv",
        help="Path to bible.tsv.",
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DATA_DIR / "vectorstores" / "chroma",
        help="Directory for ChromaDB persistence.",
    )
    parser.add_argument(
        "--collection-prefix",
        default="biblical",
        help="Prefix for created collection names.",
    )
    parser.add_argument(
        "--rebuild-collections",
        action="store_true",
        help="Delete existing collections with matching names before rebuilding.",
    )
    parser.add_argument(
        "--hf-batch-size",
        type=int,
        default=64,
        help="Embedding batch size used for HF models.",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=128,
        help="Embedding batch size used for OpenAI models.",
    )
    parser.add_argument(
        "--chroma-upsert-batch-size",
        type=int,
        default=1024,
        help="Upsert batch size for Chroma collection writes.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device for HF models (e.g. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="Optional OpenAI API key; falls back to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional custom base URL for OpenAI-compatible APIs.",
    )
    parser.add_argument(
        "--openai-organization",
        default=None,
        help="Optional OpenAI organization value.",
    )
    parser.add_argument(
        "--openai-dimensions",
        type=int,
        default=None,
        help="Optional OpenAI embedding dimension override.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during embedding/upsert.",
    )
    parser.set_defaults(func=run_vectorstore)


def run_vectorstore(args):
    from solemne_data_atelier.vector_store import build_biblical_vectorstores

    openai_models = args.openai_models or []
    hf_models = args.hf_models or []

    if not openai_models and not hf_models:
        print("Vector store build failed: provide at least one --hf-model or --openai-model.")
        return 1

    try:
        mapping = build_biblical_vectorstores(
            hf_models=hf_models,
            openai_models=openai_models,
            bible_tsv_path=args.bible_tsv_path,
            persist_directory=args.persist_directory,
            collection_prefix=args.collection_prefix,
            rebuild_collections=args.rebuild_collections,
            hf_batch_size=args.hf_batch_size,
            openai_batch_size=args.openai_batch_size,
            chroma_upsert_batch_size=args.chroma_upsert_batch_size,
            device=args.device,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            openai_organization=args.openai_organization,
            openai_dimensions=args.openai_dimensions,
            show_progress=not args.no_progress,
        )
    except Exception as exc:
        print(f"Vector store build failed: {exc}")
        return 1

    print("\nBuilt collections:")
    for item in mapping.get("collections", []):
        print(
            f"- {item['collection_name']} "
            f"[{item['provider']} | {item['model_name']}] count={item['count']}"
        )
    print("\nCollection identity is stored in Chroma metadata (provider/model_name).")
    return 0
