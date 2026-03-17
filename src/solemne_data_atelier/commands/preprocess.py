import sys
from pathlib import Path
import json

from solemne_data_atelier import DATA_DIR

def add_parser(subparsers):
    parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess and restructure the raw data.",
    )

    parser.add_argument(
        "--visual-context-chars",
        type=int,
        default=50,
        help="Context window size for visual validation snippets.",
    )
    parser.add_argument(
        "--visual-max-rows",
        type=int,
        default=None,
        help="Optional row cap for visual validation output.",
    )
    parser.add_argument(
        "--visual-exclude-verse-texts",
        action="store_false",
        dest="visual_include_verse_texts",
        default=True,
        help="Exclude verse-text enrichment from visual validation output (included by default).",
    )
    parser.add_argument(
        "--visual-bible-tsv-path",
        type=Path,
        default=None,
        help="Optional bible.tsv path override for verse-text enrichment.",
    )
    parser.add_argument(
        "--visual-include-similar-verses",
        action="store_true",
        help="Include top similar verses from Chroma for each resolved reference.",
    )
    parser.add_argument(
        "--visual-similar-top-k",
        type=int,
        default=2,
        help="Number of similar verses to keep per reference.",
    )
    parser.add_argument(
        "--visual-similar-min-cosine-similarity",
        type=float,
        default=0.80,
        help="Minimum cosine similarity threshold for similar-verse inclusion.",
    )
    parser.add_argument(
        "--visual-similar-collection-name",
        type=str,
        default=None,
        help="Optional explicit Chroma collection name for similar-verse retrieval.",
    )
    parser.add_argument(
        "--visual-similar-model-key",
        type=str,
        default=None,
        help="Optional mapping key (provider:model) used to resolve Chroma collection.",
    )
    parser.add_argument(
        "--visual-chroma-persist-directory",
        type=Path,
        default=None,
        help="Optional Chroma persist directory override.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help=(
            "Create data/hackathon_data.zip with raw/, task/, vectorstores/, "
            "bible.tsv, book_mapping.tsv, and reference_mapping.json."
        ),
    )

    parser.set_defaults(func=run_preprocess)


def run_preprocess(args):
    from solemne_data_atelier.preprocessing import (
        BIBLE_TSV_PATH,
        compute_quote_source_statistics,
        create_hackathon_archive,
        prepare_dataset,
        produce_visual_validation_data,
    )

    input_dir = Path(DATA_DIR / "raw")
    output_dir = Path(DATA_DIR / "task")

    xml_files = sorted(input_dir.glob("*.xml"))

    if not xml_files:
        print(f"No .xml files found in {input_dir}")
        sys.exit(1)

    print(f"Scanning {len(xml_files)} XML file(s) in {input_dir} …")
    quote_stats = compute_quote_source_statistics(xml_dir=input_dir)

    print("\nQuote-source statistics:")
    print(f"- XML files scanned: {quote_stats['xml_files_scanned']}")
    print(
        "- Files with <quote source=\"...\"> (before resolution): "
        f"{quote_stats['files_with_quote_source_before_resolution']}"
    )
    print(
        "- Total quotations with source attribute (before resolution): "
        f"{quote_stats['total_quote_source_before_resolution']}"
    )
    print(
        "- Files with quotations containing mapped Bible abbreviations (before resolution): "
        f"{quote_stats['files_with_mapped_bible_abbrev_quotes_before_resolution']}"
    )
    print(
        "- Total quotations containing mapped Bible abbreviations (before resolution): "
        f"{quote_stats['total_mapped_bible_abbrev_quotes_before_resolution']}"
    )
    if quote_stats["parse_error_count"] > 0:
        print(f"- Parse errors: {quote_stats['parse_error_count']}")

    print(f"\nPreprocessing data into {output_dir} …")
    dataset_stats = prepare_dataset(xml_dir=input_dir, output_dir=output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preprocess_report.json"
    report_payload = {
        "quote_source_statistics": quote_stats,
        "dataset_generation_statistics": dataset_stats,
    }
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nComputed statistics:")
    print(f"- Files: {dataset_stats['files']}")
    print(f"- Spans: {dataset_stats['spans']}")
    print(f"- Individual verses: {dataset_stats['individual_verses']}")
    print(f"- Biblical books: {dataset_stats['biblical_books']}")
    print(f"- Total verses (True Positives): {dataset_stats['total_verses_true_positives']}")
    print(f"\nReport written: {report_path}")

    produce_visual_validation_data(
        dataset_dir=DATA_DIR / "task",
        context_chars=args.visual_context_chars,
        max_rows=args.visual_max_rows,
        include_verse_texts=args.visual_include_verse_texts,
        bible_tsv_path=args.visual_bible_tsv_path if args.visual_bible_tsv_path is not None else BIBLE_TSV_PATH,
        include_similar_verses=args.visual_include_similar_verses,
        similar_verses_top_k=args.visual_similar_top_k,
        similar_verses_min_cosine_similarity=args.visual_similar_min_cosine_similarity,
        similar_verses_collection_name=args.visual_similar_collection_name,
        similar_verses_model_key=args.visual_similar_model_key,
        chroma_persist_directory=args.visual_chroma_persist_directory,
        output_tsv=DATA_DIR / "task/validation_preview.tsv",
        output_html=DATA_DIR / "task/validation_preview.html",
    )
    if args.archive:
        archive_path = create_hackathon_archive(
            data_dir=DATA_DIR,
            archive_path=DATA_DIR / "hackathon_data.zip",
        )
        print(f"Archive written: {archive_path}")
    print("Done.")
    return 0
