"""
pdf_chunk_all.py

Scan the current directory for all PDF files, chunk each PDF into overlapping
text chunks, and save each PDF's chunks as a JSONL file.

- Input:  all *.pdf files in the same folder as this script
- Output: one JSONL per PDF, named "<pdf_stem>_chunks.jsonl"

Each JSONL line looks like:
{
    "document_id": "sample",        # PDF filename without extension
    "chunk_index": 0,
    "content": "...",
    "token_count": 512
}

Dependencies:
    pip install pypdf tiktoken
"""

import json
import math
import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional
from pathlib import Path

from pypdf import PdfReader

# ------------------------------------------------------------
# Tokenizer (tiktoken if available, otherwise fallback)
# ------------------------------------------------------------
try:
    import tiktoken

    def get_tokenizer(model_name: str = "gpt-4o-mini") -> Callable[[str], List[int]]:
        """
        Return a tokenizer function using tiktoken.
        """
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        return lambda text: enc.encode(text)

except ImportError:
    def get_tokenizer(model_name: str = "gpt-4o-mini") -> Callable[[str], List[int]]:
        """
        Fallback tokenizer if tiktoken is not installed.

        Very simple: split on words and punctuation.
        """
        import re
        token_pattern = re.compile(r"\w+|\S")

        def tokenize(text: str) -> List[int]:
            tokens = token_pattern.findall(text)
            # Only the count matters; IDs can be fake.
            return list(range(len(tokens)))

        return tokenize


# ------------------------------------------------------------
# Data structure
# ------------------------------------------------------------

@dataclass
class TextChunk:
    document_id: str
    chunk_index: int
    content: str
    token_count: int


# ------------------------------------------------------------
# PDF extraction
# ------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from all pages of a PDF.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


# ------------------------------------------------------------
# Chunking logic (base 450–700 tokens, 10–15% overlap)
# ------------------------------------------------------------

def chunk_text(
    text: str,
    document_id: str,
    tokenizer_fn: Callable[[str], List[int]],
    base_min_tokens: int = 450,
    base_max_tokens: int = 700,
    overlap_ratio_min: float = 0.10,
    overlap_ratio_max: float = 0.15,
) -> List[TextChunk]:
    """
    Chunk text into windows of tokens with overlap.

    Simplified policy:
    - Target chunk size: 450–700 tokens
    - Overlap: ~12.5% of the chunk token length
    """
    import re

    # Split text into word-like units
    word_pattern = re.compile(r"\w+|\S")
    words = word_pattern.findall(text)

    chunks: List[TextChunk] = []
    total_words = len(words)
    word_idx = 0
    chunk_idx = 0

    while word_idx < total_words:
        current_end = word_idx
        chunk_text_str = ""
        current_token_len = 0

        # Grow the chunk until we hit base_max_tokens or run out of words
        for i in range(word_idx, total_words):
            candidate = " ".join(words[word_idx : i + 1])
            tokens = tokenizer_fn(candidate)
            if len(tokens) > base_max_tokens:
                break
            chunk_text_str = candidate
            current_end = i + 1
            current_token_len = len(tokens)

        # If it's too small and we're not at the end, grab the rest
        if current_token_len < base_min_tokens and current_end < total_words:
            candidate = " ".join(words[word_idx:total_words])
            tokens = tokenizer_fn(candidate)
            chunk_text_str = candidate
            current_end = total_words
            current_token_len = len(tokens)

        # Safety check
        if not chunk_text_str.strip():
            break

        chunks.append(
            TextChunk(
                document_id=document_id,
                chunk_index=chunk_idx,
                content=chunk_text_str,
                token_count=current_token_len,
            )
        )
        chunk_idx += 1

        # Overlap ~ average of min and max ratio
        overlap_ratio = (overlap_ratio_min + overlap_ratio_max) / 2.0
        overlap_tokens = int(math.floor(current_token_len * overlap_ratio))
        # Approx: 1 token ≈ 1 word
        overlap_words = overlap_tokens

        next_start = max(word_idx, current_end - overlap_words)
        if next_start >= current_end:
            next_start = current_end  # avoid infinite loop

        word_idx = next_start

    return chunks


# ------------------------------------------------------------
# Save chunks to JSONL
# ------------------------------------------------------------

def save_jsonl(chunks: List[TextChunk], output_path: Path) -> None:
    """
    Save chunks to a JSONL file, one JSON object per line.
    """
    with output_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "token_count": c.token_count,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  -> Saved {len(chunks)} chunks to {output_path.name}")


# ------------------------------------------------------------
# Main: process all PDFs in current directory
# ------------------------------------------------------------

def _load_json_file(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _import_workflow_process_single_reading():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from workflow import process_single_reading  # type: ignore
    return process_single_reading


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory to scan for PDFs (default: this script's directory)",
    )
    parser.add_argument(
        "--chunks-output-dir",
        default=None,
        help="Directory to write *_chunks.jsonl files (default: same as input-dir)",
    )
    parser.add_argument(
        "--generate-scaffolds",
        action="store_true",
        help="After chunking each PDF, generate scaffolds using workflow.py",
    )
    parser.add_argument(
        "--scaffolds-output-dir",
        default=None,
        help="Directory to write scaffold outputs (default: <chunks-output-dir>/scaffolds)",
    )
    parser.add_argument(
        "--class-profile-json",
        default=None,
        help="Path to a JSON file for class_profile (required if --generate-scaffolds)",
    )
    parser.add_argument(
        "--reading-info-json",
        default=None,
        help="Path to a JSON file for reading_info (optional; if omitted, uses PDF stem)",
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    args = parser.parse_args(argv)

    # Use the directory where this script is located by default
    script_dir = Path(__file__).parent.resolve()
    input_dir = Path(args.input_dir).resolve() if args.input_dir else script_dir
    chunks_output_dir = (
        Path(args.chunks_output_dir).resolve() if args.chunks_output_dir else input_dir
    )
    chunks_output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    print(f"Found {len(pdf_files)} PDF file(s):")
    for p in pdf_files:
        print(f" - {p.name}")

    tokenizer = get_tokenizer("gpt-4o-mini")

    process_single_reading = None
    class_profile = None
    reading_info_template = None
    scaffolds_output_dir = None
    if args.generate_scaffolds:
        class_profile = _load_json_file(args.class_profile_json)
        if class_profile is None:
            raise ValueError("--class-profile-json is required when --generate-scaffolds is set")

        reading_info_template = _load_json_file(args.reading_info_json)
        scaffolds_output_dir = (
            Path(args.scaffolds_output_dir).resolve()
            if args.scaffolds_output_dir
            else (chunks_output_dir / "scaffolds")
        )
        scaffolds_output_dir.mkdir(parents=True, exist_ok=True)
        process_single_reading = _import_workflow_process_single_reading()

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        document_id = pdf_path.stem  # filename without extension
        output_path = chunks_output_dir / f"{document_id}_chunks.jsonl"

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(
            text=text,
            document_id=document_id,
            tokenizer_fn=tokenizer,
        )

        print(f"  Generated {len(chunks)} chunks for {pdf_path.name}")
        save_jsonl(chunks, output_path)

        if args.generate_scaffolds and process_single_reading is not None:
            reading_info = dict(reading_info_template or {})
            reading_info.setdefault("assignment_id", document_id)
            reading_info.setdefault("session_description", f"Auto-generated from {pdf_path.name}")
            reading_info.setdefault("assignment_description", f"Reading from {pdf_path.name}")
            reading_info.setdefault("assignment_objective", "Generate annotation scaffolds.")

            per_pdf_output_dir = scaffolds_output_dir / document_id
            per_pdf_output_dir.mkdir(parents=True, exist_ok=True)
            process_single_reading(
                reading_id=document_id,
                chunks_file=str(output_path),
                class_profile=class_profile,
                reading_info=reading_info,
                output_dir=str(per_pdf_output_dir),
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                save_outputs=True,
            )

    print("\nDone. All PDFs in this folder have been chunked.")


if __name__ == "__main__":
    main()
