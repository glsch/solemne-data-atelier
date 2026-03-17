from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import zipfile

from typing import List, Optional

import gdown
from solemne_data_atelier import DATA_DIR
from solemne_data_atelier.preprocessing import (
    BIBLE_TSV_PATH,
    BOOK_MAPPING_PATH,
    compute_quote_source_statistics,
    prepare_dataset,
    produce_visual_validation_data,
    resolve_biblical_source_references,
)

logger = logging.getLogger(__name__)


def split_into_chunks(
    text: str,
    *,
    mode: str = "sentence",
    sentences_per_chunk: int = 2,
    sentence_stride: int = 1,
    char_chunk_size: int = 500,
    char_chunk_overlap: int = 100,
    min_chunk_chars: int = 30,
) -> List[str]:
    """
    Shared chunking utility used by multiple methods.

    Supported modes:
    - full (aliases: none, document, whole)
    - sentence
    - sentence_window
    - char
    """
    text = str(text or "")
    if not text:
        return []

    mode_norm = mode.lower().strip()
    chunks: List[str] = []

    if mode_norm in {"full", "none", "document", "whole"}:
        whole = text.strip()
        if not whole:
            return []
        return [whole] if len(whole) >= int(min_chunk_chars) else []

    if mode_norm in {"sentence", "sentence_window"}:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", text) if s.strip()]
        if not sentences:
            whole = text.strip()
            return [whole] if whole and len(whole) >= int(min_chunk_chars) else []

        if mode_norm == "sentence":
            chunks = sentences
        else:
            win = max(1, int(sentences_per_chunk))
            step = max(1, int(sentence_stride))
            for i in range(0, len(sentences), step):
                piece = " ".join(sentences[i : i + win]).strip()
                if piece:
                    chunks.append(piece)
    elif mode_norm == "char":
        size = max(1, int(char_chunk_size))
        overlap = max(0, int(char_chunk_overlap))
        step = max(1, size - overlap)
        for i in range(0, len(text), step):
            piece = text[i : i + size].strip()
            if piece:
                chunks.append(piece)
    else:
        raise ValueError("mode must be one of: full, sentence, sentence_window, char")

    return [c for c in chunks if len(c) >= int(min_chunk_chars)]


def _download_and_save(
        link: str = None,
        target_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Downloads processed and raw data.
    :param target_dir:
    :param name:
    :return:
    """
    if link is None:
        logger.error("No link provided!")
    if target_dir is None:
        target_dir = Path("../../downloads")

    os.makedirs(os.path.expanduser("~/.cache/gdown"), exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_path = target_dir / f"data.zip"
    logger.info(f"Checking for archive at {archive_path}")

    if not archive_path.exists():
        logger.info("Archive not found. Downloading...")
        try:
            gdown.download(id=link, output=str(archive_path), quiet=False)
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            raise
    else:
        logger.info("Archive already exists")

    extracted_files: List[Path] = []
    logger.info("Extracting archive contents")
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if '__MACOSX' not in member:
                    zip_ref.extract(member, str(target_dir))
                    extracted_files.append(target_dir / member)
    except zipfile.BadZipFile:
        logger.error("Invalid or corrupted zip file")
        raise
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise

    logger.info(f"Successfully extracted {len(extracted_files)} files")
    return extracted_files

def download_data():
    _download_and_save(
        link=os.getenv("DATA_DOWNLOAD_LINK"),
        target_dir=DATA_DIR,
    )

__all__ = [
    "BIBLE_TSV_PATH",
    "BOOK_MAPPING_PATH",
    "compute_quote_source_statistics",
    "download_data",
    "prepare_dataset",
    "produce_visual_validation_data",
    "resolve_biblical_source_references",
    "split_into_chunks",
]

if __name__ == "__main__":
    pass
