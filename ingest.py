#!/usr/bin/env python3
"""
ingest.py — Arabic IT/PC Troubleshooting RAG Pipeline
=====================================================
End-to-end ingestion with safe resume and full Vision OCR:
  1. Discover PDFs in ./dataset/ (sorted by numeric prefix)
  2. Load OpenAI API (gpt-4o-mini)
  3. Full Vision OCR via OpenAI
  4. Save intermediate Markdown to ./extracted_md/ (page-level resume)
  5. Semantic chunking (MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter)
  6. Embed with BAAI/bge-m3
  7. Upsert into persistent ChromaDB at ./chroma_db (never deletes)
"""

import os
import re
import sys
import time
import base64
import traceback
from io import BytesIO
from pathlib import Path

# OpenAI API & Retries
import dotenv
from openai import OpenAI, RateLimitError, APIConnectionError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Force UTF-8 output on Windows (cp1252 cannot encode emoji/Arabic)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pymupdf as fitz  # pymupdf (fitz API)
from PIL import Image


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
EXTRACTED_MD_DIR = BASE_DIR / "extracted_md"
CHROMA_DIR = BASE_DIR / "chroma_db"


# ──────────────────────────────────────────────
# Global Tracker for OpenAI Tier 1 Limits
# ──────────────────────────────────────────────
TOTAL_REQUESTS_MADE = 0
REQUEST_LIMIT_WARNING = 9500


# ──────────────────────────────────────────────
# Environment Setup
# ──────────────────────────────────────────────
def setup_openai():
    env_path = BASE_DIR / ".env"
    dotenv.load_dotenv(dotenv_path=env_path, override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found in .env file.")
        sys.exit(1)
    print("✅ OpenAI API configured.")
    return OpenAI(api_key=api_key)


# ──────────────────────────────────────────────
# 2. PDF Discovery (sorted by numeric prefix)
# ──────────────────────────────────────────────
def _extract_prefix_number(path: Path) -> int:
    """Extract the leading integer from a filename like '3_example.pdf' → 3."""
    match = re.match(r"^(\d+)", path.name)
    return int(match.group(1)) if match else 9999


def discover_pdfs() -> list[Path]:
    """Return PDFs sorted by their numeric filename prefix (ascending)."""
    if not DATASET_DIR.exists():
        print(f"[WARN] Dataset directory not found: {DATASET_DIR}")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  → Created empty directory. Place your PDFs there and re-run.")
        return []
    pdfs = sorted(DATASET_DIR.glob("**/*.pdf"))
    pdfs_upper = sorted(DATASET_DIR.glob("**/*.PDF"))
    all_pdfs = sorted(set(pdfs + pdfs_upper), key=_extract_prefix_number)
    print(f"\n📂 Found {len(all_pdfs)} PDF(s) in {DATASET_DIR}")
    for p in all_pdfs:
        size_mb = p.stat().st_size / (1024 * 1024)
        prefix = _extract_prefix_number(p)
        print(f"   • [{prefix:>2}] {p.name}  ({size_mb:.1f} MB)")
    return all_pdfs


# ──────────────────────────────────────────────
# 4. OpenAI Extraction Prompts & Functions
# ──────────────────────────────────────────────

SLOW_VISION_PROMPT = (
    "You are an expert in extracting Arabic text for RAG systems. Accuracy is paramount. "
    "Convert this page image into clean Markdown, strictly following these rules: "
    "1. Use # for main titles and ## for subtitles. "
    "2. Completely ignore Table of Contents, dotted lines (....), separators (---), and page numbers. "
    "3. Merge broken lines into continuous sentences. "
    "4. Format steps as clear numbered lists (1. 2. 3.). "
    "5. Preserve tables in standard Markdown format. "
    "Output ONLY the clean Markdown text.\n"
    "CRITICAL RULE: If the image is a blank page, a decorative cover, or contains no readable text, you MUST return the exact string [EMPTY_PAGE] as your response. Do not apologize, do not invent placeholder titles, and do not explain yourself. Output only [EMPTY_PAGE]."
)


def encode_pil_to_base64(image: Image.Image) -> str:
    """Save PIL Image to BytesIO and encode to Base64 data URI."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def extract_slow_vision(client: OpenAI, image: Image.Image) -> tuple[str, int]:
    """Slow path: full vision OCR from rendered page image via OpenAI."""
    b64_image = encode_pil_to_base64(image)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=3000,
        messages=[
            {"role": "system", "content": SLOW_VISION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract the text from this image."},
                {"type": "image_url", "image_url": {"url": b64_image}}
            ]}
        ]
    )
    
    text = response.choices[0].message.content.strip() if response.choices else ""
    tokens = response.usage.total_tokens if response.usage else 0
    return text, tokens


# ──────────────────────────────────────────────
# 5. Process All PDFs → Markdown (Hybrid + Safe Resume)
# ──────────────────────────────────────────────
def check_request_limit(client: OpenAI) -> OpenAI:
    global TOTAL_REQUESTS_MADE
    if TOTAL_REQUESTS_MADE > REQUEST_LIMIT_WARNING:
        print("\n" + "!" * 60)
        print(f"⚠️ Request limit approaching (Requests made: {TOTAL_REQUESTS_MADE}).")
        print("Please update OPENAI_API_KEY in the .env file with a fresh key or wait for quota reset.")
        print("!" * 60)
        input("Press Enter to continue after updating the key...")
        
        env_path = BASE_DIR / ".env"
        dotenv.load_dotenv(dotenv_path=env_path, override=True)
        api_key = os.environ.get("OPENAI_API_KEY")
        TOTAL_REQUESTS_MADE = 0
        print("✅ Refreshed OpenAI API key. Resuming...")
        return OpenAI(api_key=api_key)
    return client


def is_toc_page(native_text: str) -> bool:
    """Pre-filter: Skip Table of Contents pages."""
    text_prefix = native_text[:200]
    if "الفهرس" in text_prefix or "المحتوى" in text_prefix:
        return True
    
    # Check if mostly dots/numbers
    dots_count = text_prefix.count('.')
    numbers_count = sum(c.isdigit() for c in text_prefix)
    if dots_count > 20 and numbers_count > 5:
        return True
        
    return False


def process_pdfs(pdf_paths: list[Path], client: OpenAI) -> list[Path]:
    """
    For each PDF, process every page using OpenAI.
    Supports page-level resume: appends to existing .md files.
    """
    global TOTAL_REQUESTS_MADE
    EXTRACTED_MD_DIR.mkdir(parents=True, exist_ok=True)
    md_files = []

    # Count total pages across all PDFs for progress reporting
    total_all_pages = 0
    for p in pdf_paths:
        doc = fitz.open(str(p))
        total_all_pages += len(doc)
        doc.close()
    print(f"\n📊 Total pages across all PDFs: {total_all_pages}")

    processed_all_pages = 0

    for pdf_idx, pdf_path in enumerate(pdf_paths, 1):
        md_filename = pdf_path.stem + ".md"
        md_path = EXTRACTED_MD_DIR / md_filename

        # Resume support: check how many pages are already extracted
        start_page = 1
        if md_path.exists() and md_path.stat().st_size > 0:
            content = md_path.read_text(encoding="utf-8")
            page_matches = re.findall(r"<!-- page:(\d+)", content)
            if page_matches:
                start_page = int(page_matches[-1]) + 1

        print(f"\n📄 [{pdf_idx}/{len(pdf_paths)}] Processing: {pdf_path.name}")

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            print(f"   Found {total_pages} page(s)")

            if start_page > total_pages:
                print(f"⏭️  Skipping {pdf_path.name} (all {total_pages} pages already extracted)")
                md_files.append(md_path)
                processed_all_pages += total_pages
                doc.close()
                continue

            # Skip already-processed pages in the global counter
            processed_all_pages += (start_page - 1)

            # Process each page with OpenAI API
            with open(md_path, "a", encoding="utf-8") as f:
                for page_num in range(start_page, total_pages + 1):
                    client = check_request_limit(client)
                    
                    page = doc.load_page(page_num - 1)
                    processed_all_pages += 1
                    print(
                        f"   🔍 Page {page_num}/{total_pages} "
                        f"(global: {processed_all_pages}/{total_all_pages}) ...",
                        end=" ", flush=True
                    )
                    t0 = time.time()

                    # Pre-filter
                    native_text = page.get_text().strip()
                    if is_toc_page(native_text):
                        print("skipped [TOC]")
                        continue

                    try:
                        TOTAL_REQUESTS_MADE += 1
                        zoom = 120 / 72.0
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                        text, tokens = extract_slow_vision(client, img)
                        elapsed = time.time() - t0
                        print(f"done [OPENAI-VISION] ({elapsed:.1f}s, {len(text)} chars) (Tokens: {tokens})")

                        if "[EMPTY_PAGE]" in text:
                            print("   ↪ skipped [EMPTY_PAGE]")
                            continue

                        # Write with page marker for resume support
                        page_text = f"<!-- page:{page_num} source:{pdf_path.name} -->\n\n{text}"
                        if page_num > 1 or start_page > 1:
                            f.write("\n\n---\n\n")
                        f.write(page_text)
                        f.flush()
                    except Exception as e:
                        print(f"failed [ERROR]: {e}")
                        continue
                    finally:
                        # Rate limiting (1.5s sleep)
                        time.sleep(1.5)

            doc.close()
            print(f"   💾 Saved → {md_path.relative_to(BASE_DIR)}")
            md_files.append(md_path)

        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            traceback.print_exc()
            continue

    return md_files


# ──────────────────────────────────────────────
# 6. Semantic Chunking (Header-aware + size-limited)
# ──────────────────────────────────────────────
def chunk_markdown_files(md_files: list[Path]) -> list[dict]:
    """
    Two-stage chunking:
      1. MarkdownHeaderTextSplitter splits by # and ## to preserve semantic sections.
      2. RecursiveCharacterTextSplitter handles any oversized sections.
    Returns a list of dicts: {text, source, page, header}.
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # Stage 1: Split by Markdown headers
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep headers in the chunk text for context
    )

    # Stage 2: Split oversized chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
        keep_separator=True,
    )

    all_chunks = []

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        source_name = md_path.stem + ".pdf"

        # Split into page blocks by our marker
        page_blocks = re.split(r"(?=<!-- page:\d+)", content)

        for block in page_blocks:
            if not block.strip():
                continue

            # Extract page number from marker
            page_match = re.search(r"<!-- page:(\d+)", block)
            page_num = int(page_match.group(1)) if page_match else 0

            # Remove the HTML comment marker before chunking
            clean_block = re.sub(r"<!--.*?-->", "", block).strip()
            if not clean_block or len(clean_block) < 10:
                continue

            # Stage 1: Header-aware split
            header_docs = header_splitter.split_text(clean_block)

            for doc in header_docs:
                # Build the header from metadata returned by the splitter
                h1 = doc.metadata.get("h1", "")
                h2 = doc.metadata.get("h2", "")
                header = " > ".join(filter(None, [h1, h2]))

                doc_text = doc.page_content.strip()
                if len(doc_text) < 10:
                    continue

                # Stage 2: Size-limit split
                sub_chunks = text_splitter.split_text(doc_text)

                for chunk_text in sub_chunks:
                    chunk_text = chunk_text.strip()
                    if len(chunk_text) < 10:
                        continue
                    all_chunks.append({
                        "text": chunk_text,
                        "source": source_name,
                        "page": page_num,
                        "header": header,
                    })

    print(f"\n✂️  Created {len(all_chunks)} chunks from {len(md_files)} Markdown file(s)")
    return all_chunks


# ──────────────────────────────────────────────
# 7. Embedding & ChromaDB Storage (upsert, never delete)
# ──────────────────────────────────────────────
def build_vector_store(chunks: list[dict]) -> None:
    """Embed chunks with BAAI/bge-m3 and upsert into persistent ChromaDB.
    Never deletes the existing collection — builds up across sessions."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    if not chunks:
        print("\n⚠️  No chunks to embed. Skipping vector store creation.")
        return

    # Load embedding model
    embed_model_name = "BAAI/bge-m3"
    print(f"\n🧠 Loading embedding model: {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)
    print("   ✅ Embedding model loaded.")

    # Prepare data
    texts = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page": c["page"], "header": c["header"]}
        for c in chunks
    ]
    ids = [f"chunk_{i:05d}" for i in range(len(chunks))]

    # Embed in batches
    print(f"   Embedding {len(texts)} chunks ...")
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = embed_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
        done = min(i + batch_size, len(texts))
        print(f"   Embedded {done}/{len(texts)}")

    # ChromaDB persistent client
    print(f"\n💾 Storing in ChromaDB at {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name="arabic_it_troubleshooting",
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches (safe for repeated runs — overwrites by ID)
    chroma_batch = 5000
    for i in range(0, len(ids), chroma_batch):
        end = min(i + chroma_batch, len(ids))
        collection.upsert(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=all_embeddings[i:end],
            metadatas=metadatas[i:end],
        )
    print(f"   ✅ Collection now has {collection.count()} chunks total")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  Arabic IT/PC Troubleshooting — RAG Ingestion Pipeline")
    print("  (Safe Resume · OpenAI API · Smart Sorting)")
    print("=" * 60)

    # Step 1: Discover PDFs (sorted by numeric prefix)
    pdf_paths = discover_pdfs()
    if not pdf_paths:
        print("\n⚠️  No PDFs found in ./dataset/ — nothing to process.")
        print("   Place your Arabic IT troubleshooting PDFs in the dataset/ folder and re-run.")
        print("   Exiting.")
        return

    # Step 2: Load OpenAI API Client
    client = setup_openai()

    # Step 3: Extraction → Markdown (resumes from last page)
    md_files = process_pdfs(pdf_paths, client)

    # If no new markdown was produced, check for existing ones
    if not md_files:
        existing_md = sorted(EXTRACTED_MD_DIR.glob("*.md")) if EXTRACTED_MD_DIR.exists() else []
        if existing_md:
            print(f"\n📁 Found {len(existing_md)} existing Markdown file(s) in extracted_md/")
            md_files = existing_md
        else:
            print("\n⚠️  No Markdown files available. Exiting.")
            return

    # Step 4: Semantic chunk
    chunks = chunk_markdown_files(md_files)

    # Step 5: Embed & upsert (never deletes previous data)
    build_vector_store(chunks)

    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete!")
    print(f"  📂 Markdown files : {EXTRACTED_MD_DIR}")
    print(f"  💾 Vector database : {CHROMA_DIR}")
    print(f'  🔎 Test with       : python test_query.py "حل مشكلة الشاشة الزرقاء"')
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
