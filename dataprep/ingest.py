#!/usr/bin/env python3
"""
ingest.py — Arabic IT/PC Troubleshooting RAG Pipeline (Two-Pass Hybrid)
========================================================================
End-to-end ingestion with safe resume and hybrid OCR:
  Pass 1 (Local VLM — Qwen2.5-VL via Ollama):
    • Render PDF pages to images via PyMuPDF
    • Full-page Arabic OCR locally
    • Save page image + insert <!-- image_N --> placeholder when visuals detected
  Pass 2 (Targeted Cloud — GPT-4o):
    • Run once per document after all pages are OCR'd
    • Send Markdown + saved page images to ChatGPT
    • Replace only <!-- image_N --> placeholders with rich descriptions
  Then:
    • Semantic chunking (MarkdownHeader → RecursiveCharacter)
    • Embed with BAAI/bge-m3
    • Upsert into persistent ChromaDB (never deletes)
"""

import os
import re
import sys
import json
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

from pdf2image import convert_from_path
from PIL import Image


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataprep"
EXTRACTED_MD_DIR = BASE_DIR / "extracted_md"
CHROMA_DIR = BASE_DIR / "chroma_db"
PAGE_IMAGES_DIR = BASE_DIR / "extracted_images"


# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
LOCAL_MODEL = "qwen2.5vl-32k"
CLOUD_MODEL = "gpt-4o"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


# ──────────────────────────────────────────────
# Environment & Client Setup
# ──────────────────────────────────────────────
def setup_clients() -> tuple[OpenAI, OpenAI]:
    """
    Return (client_local, client_cloud).
    - client_local: Ollama OpenAI-compatible endpoint for Qwen2.5-VL
    - client_cloud: Official OpenAI client for GPT-4o enrichment
    """
    # Local Ollama client (no real API key needed)
    client_local = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",  # placeholder — Ollama ignores this
    )
    print(f"✅ Local VLM client configured ({OLLAMA_BASE_URL}, model={LOCAL_MODEL})", flush=True)

    # Cloud OpenAI client
    env_path = BASE_DIR / ".env"
    dotenv.load_dotenv(dotenv_path=env_path, override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[WARN] OPENAI_API_KEY not found in .env — Pass 2 enrichment will be skipped.", flush=True)
        client_cloud = None
    else:
        client_cloud = OpenAI(api_key=api_key)
        print(f"✅ Cloud client configured (model={CLOUD_MODEL})", flush=True)

    return client_local, client_cloud


# ──────────────────────────────────────────────
# PDF Discovery (sorted by numeric prefix)
# ──────────────────────────────────────────────
def _extract_prefix_number(path: Path) -> int:
    """Extract the leading integer from a filename like '3_example.pdf' → 3."""
    match = re.match(r"^(\d+)", path.name)
    return int(match.group(1)) if match else 9999


def discover_pdfs() -> list[Path]:
    """Return PDFs sorted by their numeric filename prefix (ascending)."""
    if not DATASET_DIR.exists():
        print(f"[WARN] Dataset directory not found: {DATASET_DIR}", flush=True)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  → Created empty directory. Place your PDFs there and re-run.", flush=True)
        return []
    pdfs = sorted(DATASET_DIR.glob("**/*.pdf"))
    pdfs_upper = sorted(DATASET_DIR.glob("**/*.PDF"))
    all_pdfs = sorted(set(pdfs + pdfs_upper), key=_extract_prefix_number)
    print(f"\n📂 Found {len(all_pdfs)} PDF(s) in {DATASET_DIR}", flush=True)
    for p in all_pdfs:
        size_mb = p.stat().st_size / (1024 * 1024)
        prefix = _extract_prefix_number(p)
        print(f"   • [{prefix:>2}] {p.name}  ({size_mb:.1f} MB)", flush=True)
    return all_pdfs


# ──────────────────────────────────────────────
# Pass 1: Local VLM OCR & Image Detection
# ──────────────────────────────────────────────

LOCAL_VLM_PROMPT = (
    "Extract ONLY the useful Arabic content from this page image into clean Markdown "
    "for a RAG knowledge base. Follow these rules strictly:\n\n"
    "SKIP entirely:\n"
    "- Decorative titles, English-only headings, document cover text, logos\n"
    "- Table of contents, indexes, page numbers, dotted lines (....)\n"
    "- Repeated headers/footers, separators (---), watermarks\n"
    "- Any text that is not actual instructional or informational content\n\n"
    "FORMAT rules:\n"
    "- Use ## for section headings (questions or topic titles)\n"
    "- Each question/answer or topic MUST be a separate paragraph with a blank line between them\n"
    "- Numbered steps: use 1. 2. 3. with each step on its own line\n"
    "- Tables: standard Markdown table format\n"
    "- Merge broken/wrapped lines into complete sentences\n"
    "- NEVER output a wall of text — always use proper line breaks\n\n"
    "IMAGES: If you detect an image, diagram, UI screenshot, or schematic on the page, do NOT attempt to describe it. "
    "Instead, insert exactly <!-- image_{image_counter} --> at the logical reading position in the Markdown.\\n\\n"
    "EMPTY: If the page has no useful instructional content, return only [EMPTY_PAGE].\n\n"
    "Output clean Markdown only. No commentary."
)


def encode_pil_to_base64(image: Image.Image) -> str:
    """Save PIL Image to BytesIO and encode to Base64 data URI."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


VERIFICATION_PROMPT = (
    "You are a quality checker for Arabic OCR output. You receive:\n"
    "1. A draft Markdown extracted from a page image\n"
    "2. The original page image\n\n"
    "Compare the draft carefully against the original image and FIX all issues:\n"
    "- Garbled or incorrect Arabic words → correct them by reading the image\n"
    "- Incomplete sentences or steps that are cut off → complete them from the image\n"
    "- Repeated/duplicated text → remove duplicates, keep one clean copy\n"
    "- Hallucinated content not in the image → remove it entirely\n"
    "- Mixed-language gibberish (random English/Russian in Arabic text) → fix or remove\n"
    "- Missing content visible in the image but absent from draft → add it\n\n"
    "Keep the same Markdown structure (##, numbered lists, etc). "
    "Keep <!-- image_N --> placeholders exactly as they are.\n"
    "If the draft says [EMPTY_PAGE], verify the image is truly empty. "
    "If it has content, extract it properly.\n\n"
    "Output the corrected Markdown only. No commentary, no explanations."
)


def extract_local_vlm(
    client_local: OpenAI, image: Image.Image, image_counter: int,
    prev_page_md: str = ""
) -> tuple[str, int, bool]:
    """
    Two-pass local OCR via Qwen2.5-VL:
      Pass A: Initial extraction from page image → draft markdown
      Pass B: Verify draft against original image → corrected markdown
    Both passes receive the tail of the previous page for continuity.

    Returns:
        (markdown_text, total_token_count, has_image_placeholder)
    """
    b64_image = encode_pil_to_base64(image)

    # Inject the current image counter into the system prompt
    prompt = LOCAL_VLM_PROMPT.replace("{image_counter}", str(image_counter))

    # Build context from previous page (last ~500 chars for continuity)
    context_prefix = ""
    if prev_page_md:
        tail = prev_page_md[-500:]
        context_prefix = (
            f"The previous page ended with:\n"
            f"```\n{tail}\n```\n\n"
            "Continue naturally from where it left off if the content connects. "
            "Do not repeat text from the previous page.\n\n"
        )

    # ── Pass A: Initial OCR extraction ──
    user_text = context_prefix + "Extract the text from this page image."
    response_a = client_local.chat.completions.create(
        model=LOCAL_MODEL,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": b64_image}}
            ]}
        ]
    )

    draft = response_a.choices[0].message.content.strip() if response_a.choices else ""
    tokens_a = response_a.usage.total_tokens if response_a.usage else 0

    # Skip verification for empty pages
    if not draft or "[EMPTY_PAGE]" in draft:
        has_image = bool(re.search(r"<!-- image_\d+ -->", draft))
        return draft, tokens_a, has_image

    # ── Pass B: Verification & correction ──
    verify_text = (
        f"{context_prefix}"
        f"Here is the draft Markdown:\n\n"
        f"```markdown\n{draft}\n```\n\n"
        "And here is the original page image. "
        "Verify and correct the draft against the image:"
    )
    response_b = client_local.chat.completions.create(
        model=LOCAL_MODEL,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": VERIFICATION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": verify_text},
                {"type": "image_url", "image_url": {"url": b64_image}}
            ]}
        ]
    )

    corrected = response_b.choices[0].message.content.strip() if response_b.choices else draft
    tokens_b = response_b.usage.total_tokens if response_b.usage else 0

    # Clean up: remove markdown code fences if the model wrapped the output
    corrected = re.sub(r"^```(?:markdown)?\s*\n?", "", corrected)
    corrected = re.sub(r"\n?\s*```$", "", corrected)

    total_tokens = tokens_a + tokens_b

    # Check if the model inserted an image placeholder
    has_image = bool(re.search(r"<!-- image_\d+ -->", corrected))

    return corrected, total_tokens, has_image


# ──────────────────────────────────────────────
# Pass 2: Targeted Cloud Enrichment with ChatGPT
# ──────────────────────────────────────────────

ENRICHMENT_PROMPT = (
    "You receive a Markdown document and page images. "
    "The Markdown has <!-- image_n --> placeholders where UI screenshots or diagrams were.\\n\\n"
    "For each placeholder, examine the corresponding image and write a concise, "
    "factual description of the key elements: button labels, menu items, field names, "
    "values shown, and workflow steps visible. Keep it short and searchable — "
    "this text will be used for RAG retrieval.\\n\\n"
    "Use Arabic if the image content is Arabic, English otherwise.\\n"
    "Format each as: > **[وصف]:** (concise description)\\n\\n"
    "Return valid JSON only mapping each exact placeholder string to its description.\\n"
    '{"<!-- image_1 -->": "> **[وصف]:** ...", "<!-- image_2 -->": "> **[وصف]:** ..."}\\n\\n'
    "Only include placeholders present in the Markdown. Do NOT include any text outside the JSON."
)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def _call_chatgpt_enrichment(
    client_cloud: OpenAI, md_content: str, image_paths: dict[int, Path]
) -> dict[str, str]:
    """
    Send markdown + images to GPT-4o and get back a JSON mapping
    of IMAGE_N → description.
    """
    # Build the message content: text context + all images
    user_content = [
        {"type": "text", "text": (
            "Here is the Markdown document:\n\n"
            f"```markdown\n{md_content}\n```\n\n"
            "And here are the corresponding page images:"
        )}
    ]

    for img_num in sorted(image_paths.keys()):
        img_path = image_paths[img_num]
        if img_path.exists():
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            user_content.append({
                "type": "text",
                "text": f"\n--- image_{img_num}.png ---"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client_cloud.chat.completions.create(
        model=CLOUD_MODEL,
        response_format={"type": "json_object"},
        max_tokens=4000,
        messages=[
            {"role": "system", "content": ENRICHMENT_PROMPT},
            {"role": "user", "content": user_content}
        ]
    )

    raw = response.choices[0].message.content.strip() if response.choices else "{}"

    # Parse JSON — handle markdown code fences if GPT wraps it
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"   ⚠️  Failed to parse GPT enrichment JSON. Raw response:\n{raw[:500]}", flush=True)
        return {}


def enrich_markdown_with_gpt(
    client_cloud: OpenAI, md_content: str, saved_images: dict[int, Path]
) -> str:
    """
    Pass 2: Scan finished markdown for <!-- image_N --> tags.
    If found, call ChatGPT once with all images to get descriptions,
    then substitute placeholders.

    Returns the enriched markdown string.
    """
    # Find all image placeholders
    placeholders = re.findall(r"<!-- IMAGE_(\d+) -->", md_content)
    if not placeholders:
        return md_content  # Nothing to enrich — no API call

    if client_cloud is None:
        print("   ⚠️  No cloud client — skipping image enrichment.", flush=True)
        return md_content

    image_nums = [int(n) for n in placeholders]
    # Filter to only images we actually saved
    relevant_images = {n: saved_images[n] for n in image_nums if n in saved_images}

    if not relevant_images:
        print("   ⚠️  Placeholders found but no saved images — skipping enrichment.", flush=True)
        return md_content

    print(
        f"   🌐 Pass 2: Enriching {len(relevant_images)} image placeholder(s) via {CLOUD_MODEL}...",
        end=" ", flush=True
    )
    t0 = time.time()

    descriptions = _call_chatgpt_enrichment(client_cloud, md_content, relevant_images)

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s, {len(descriptions)} descriptions)", flush=True)

    # Substitute placeholders
    enriched = md_content
    for img_key, description in descriptions.items():
        # Handle both "IMAGE_1" and "1" as keys
        num = re.search(r"\d+", str(img_key))
        if num:
            tag = f"<!-- image_{num.group()} -->"
            enriched = enriched.replace(tag, description)

    return enriched


# ──────────────────────────────────────────────
# Process All PDFs → Markdown (Two-Pass Hybrid + Safe Resume)
# ──────────────────────────────────────────────
def process_pdfs(
    pdf_paths: list[Path], client_local: OpenAI, client_cloud: OpenAI | None
) -> list[Path]:
    EXTRACTED_MD_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    md_files = []

    # Using pdf2image, we have to convert PDFs to find total pages.
    # We will do this lazily per PDF to save memory.
    processed_all_pages = 0

    for pdf_idx, pdf_path in enumerate(pdf_paths, 1):
        md_filename = pdf_path.stem + ".md"
        md_path = EXTRACTED_MD_DIR / md_filename

        # Reset image counter for this PDF
        image_counter = 1
        
        # Setup specific folder for this PDF's images
        pdf_image_dir = PAGE_IMAGES_DIR / pdf_path.stem
        pdf_image_dir.mkdir(parents=True, exist_ok=True)

        # Resume support: check how many pages are already extracted
        start_page = 1
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
            matches = re.findall(r"<!-- page:(\d+)", content)
            if matches:
                start_page = max(int(m) for m in matches) + 1

        try:
            print(f"\n📄 [{pdf_idx}/{len(pdf_paths)}] Processing: {pdf_path.name}", flush=True)
            
            # ── Pass 1: Local VLM OCR ──
            print("   Converting PDF to images (200 DPI)...", flush=True)
            poppler_bin = BASE_DIR / "poppler-24.08.0" / "Library" / "bin"
            pages = convert_from_path(pdf_path, dpi=200, poppler_path=str(poppler_bin))
            total_pages = len(pages)
            print(f"   Found {total_pages} page(s)", flush=True)

            if start_page > total_pages:
                print(f"   ✅ All {total_pages} pages already extracted.", flush=True)
                md_files.append(md_path)
                processed_all_pages += total_pages
                continue

            processed_all_pages += (start_page - 1)
            pdf_saved_images: dict[int, Path] = {}

            print(f"   ── Pass 1: Local OCR with {LOCAL_MODEL} (2-pass + continuity) ──", flush=True)
            prev_page_text = ""
            
            with open(md_path, "a", encoding="utf-8") as f:
                for page_num, img in enumerate(pages, start=1):
                    if page_num < start_page:
                        continue
                        
                    processed_all_pages += 1
                    print(
                        f"   🔍 Page {page_num}/{total_pages} ...",
                        end=" ", flush=True
                    )
                    t0 = time.time()

                    try:
                        text, tokens, has_image = extract_local_vlm(
                            client_local, img, image_counter,
                            prev_page_md=prev_page_text
                        )
                        elapsed = time.time() - t0
                        print(
                            f"done [2-PASS VLM] ({elapsed:.1f}s, {len(text)} chars, {tokens} tok)",
                            flush=True
                        )

                        if "[EMPTY_PAGE]" in text:
                            print("   ↪ skipped [EMPTY_PAGE]", flush=True)
                            continue

                        prev_page_text = text

                        # If image placeholder was inserted, save the page image
                        if has_image:
                            img_filename = f"image_{image_counter}.png"
                            img_save_path = pdf_image_dir / img_filename
                            img.save(str(img_save_path), format="PNG")
                            pdf_saved_images[image_counter] = img_save_path
                            print(
                                f"   📷 Saved page image → {img_filename}",
                                flush=True
                            )
                            image_counter += 1

                        # Write with page marker for resume support
                        page_text = f"<!-- page:{page_num} source:{pdf_path.name} -->\n\n{text}"
                        if page_num > 1 or start_page > 1:
                            f.write("\n\n---\n\n")
                        f.write(page_text)
                        f.flush()

                    except Exception as e:
                        print(f"failed [ERROR]: {e}", flush=True)
                        continue

            print(f"   💾 Pass 1 complete → {md_path.relative_to(BASE_DIR)}", flush=True)

            # ── Pass 2: Targeted Cloud Enrichment ──
            md_content = md_path.read_text(encoding="utf-8")
            if "<!-- image_" in md_content and pdf_saved_images:
                print(f"   ── Pass 2: Cloud enrichment ({len(pdf_saved_images)} images) ──", flush=True)
                enriched = enrich_markdown_with_gpt(client_cloud, md_content, pdf_saved_images)
                md_path.write_text(enriched, encoding="utf-8")
                print(f"   💾 Pass 2 complete → enriched markdown saved", flush=True)
            else:
                print(f"   ⏭️  Pass 2: No image placeholders — skipping enrichment", flush=True)

            md_files.append(md_path)

        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}", flush=True)
            traceback.print_exc()
            continue

    return md_files


# ──────────────────────────────────────────────
# Semantic Chunking (Header-aware + size-limited)
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

    print(f"\n✂️  Created {len(all_chunks)} chunks from {len(md_files)} Markdown file(s)", flush=True)
    return all_chunks


# ──────────────────────────────────────────────
# Embedding & ChromaDB Storage (upsert, never delete)
# ──────────────────────────────────────────────
def build_vector_store(chunks: list[dict]) -> None:
    """Embed chunks with BAAI/bge-m3 and upsert into persistent ChromaDB.
    Never deletes the existing collection — builds up across sessions."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    if not chunks:
        print("\n⚠️  No chunks to embed. Skipping vector store creation.", flush=True)
        return

    # Load embedding model
    embed_model_name = "BAAI/bge-m3"
    print(f"\n🧠 Loading embedding model: {embed_model_name}", flush=True)
    embed_model = SentenceTransformer(embed_model_name)
    print("   ✅ Embedding model loaded.", flush=True)

    # Prepare data
    texts = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page": c["page"], "header": c["header"]}
        for c in chunks
    ]
    ids = [f"chunk_{i:05d}" for i in range(len(chunks))]

    # Embed in batches
    print(f"   Embedding {len(texts)} chunks ...", flush=True)
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = embed_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
        done = min(i + batch_size, len(texts))
        print(f"   Embedded {done}/{len(texts)}", flush=True)

    # ChromaDB persistent client
    print(f"\n💾 Storing in ChromaDB at {CHROMA_DIR}", flush=True)
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
    print(f"   ✅ Collection now has {collection.count()} chunks total", flush=True)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 60, flush=True)
    print("  Arabic IT/PC Troubleshooting — RAG Ingestion Pipeline", flush=True)
    print("  (Two-Pass Hybrid · Local Qwen + Targeted ChatGPT · Safe Resume)", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Discover PDFs (sorted by numeric prefix)
    pdf_paths = discover_pdfs()
    if not pdf_paths:
        print("\n⚠️  No PDFs found in ./dataset/ — nothing to process.", flush=True)
        print("   Place your Arabic IT troubleshooting PDFs in the dataset/ folder and re-run.", flush=True)
        print("   Exiting.", flush=True)
        return

    # Step 2: Setup dual clients (local VLM + cloud GPT)
    client_local, client_cloud = setup_clients()

    # Step 3: Two-pass extraction → Markdown
    md_files = process_pdfs(pdf_paths, client_local, client_cloud)

    # If no new markdown was produced, check for existing ones
    if not md_files:
        existing_md = sorted(EXTRACTED_MD_DIR.glob("*.md")) if EXTRACTED_MD_DIR.exists() else []
        if existing_md:
            print(f"\n📁 Found {len(existing_md)} existing Markdown file(s) in extracted_md/", flush=True)
            md_files = existing_md
        else:
            print("\n⚠️  No Markdown files available. Exiting.", flush=True)
            return

    # Step 4: Semantic chunk
    chunks = chunk_markdown_files(md_files)

    # Step 5: Embed & upsert (never deletes previous data)
    build_vector_store(chunks)

    print("\n" + "=" * 60, flush=True)
    print("  ✅ Ingestion complete!", flush=True)
    print(f"  📂 Markdown files  : {EXTRACTED_MD_DIR}", flush=True)
    print(f"  📷 Page images     : {PAGE_IMAGES_DIR}", flush=True)
    print(f"  💾 Vector database : {CHROMA_DIR}", flush=True)
    print(f'  🔎 Test with       : python test_query.py "حل مشكلة الشاشة الزرقاء"', flush=True)
    print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    main()
