"""
app/services/ingest.py — Shared patent PDF ingestion pipeline.

Called by both the web API endpoint (POST /api/v1/ingest) and the CLI script
(scripts/ingest_patents.py). Keeping one implementation here ensures both paths
stay in sync.

Pipeline (in order):
  1. Open PDF, extract header text (first META_EXTRACT_PAGES pages)
  2. Detect source language (patent-prefix heuristic → langdetect fallback)
  3. Translate header to English for metadata extraction
  4. Regex-extract metadata (no quota); LLM fills only fields regex left blank
  5. Upsert patent_documents row in Supabase (on_conflict=patent_number)
  6. Extract + OCR all pages, split into labelled paragraphs
  7. Translate all chunks to English (no-op if already English)
  8. Generate 384-dim BGE embeddings in batches of BATCH_SIZE
  9. Bulk-insert patent_chunks rows (embedding as pgvector string "[x,y,z,...]")

Functions:
  ingest_pdf(pdf_bytes, filename, supabase, embed_model, [overrides]) -> dict
    Returns a summary dict with patent_id, chunks_inserted, language, translated,
    and the final metadata. Raises RuntimeError on PDF open failure so the caller
    can translate it to an HTTP 400.
    This function is synchronous — call it with asyncio.to_thread() from async routes.
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

from app.config import BATCH_SIZE, META_EXTRACT_PAGES, UPLOAD_DIR
from app.services.llm import llm_json
from app.utils.metadata import extract_metadata
from app.utils.pdf import extract_page_text, extract_figure_pages, split_into_chunks
from app.utils.translation import detect_language, translate_chunks, translate_to_english

log = logging.getLogger(__name__)

_ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

def _normalise_date(raw: str) -> Optional[str]:
    """Return an ISO YYYY-MM-DD string, or None if the input is blank or unparseable."""
    if not raw:
        return None
    if _ISO_DATE_RE.match(raw):
        return raw
    from datetime import datetime
    for fmt in ('%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%b. %d, %Y', '%Y%m%d'):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    log.warning("Unrecognised date format %r — storing as NULL", raw)
    return None


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    supabase,
    embed_model,
    patent_number: str = "",
    title: str = "",
    assignee: str = "",
    jurisdiction: str = "",
    pub_date: str = "",
) -> Dict[str, Any]:
    safe_stem = Path(filename).stem.replace("/", "_").replace("\\", "_")
    tmp_path = UPLOAD_DIR / f"ingest_{safe_stem}.pdf"

    try:
        tmp_path.write_bytes(pdf_bytes)
        doc = fitz.open(str(tmp_path))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"PDF open failed: {exc}") from exc

    try:
        # Extract header text for language detection and metadata
        header_text = ""
        for i in range(min(META_EXTRACT_PAGES, len(doc))):
            try:
                header_text += doc[i].get_text("text").strip() + "\n\n"
            except Exception:
                pass

        src_lang = detect_language(header_text, patent_number or safe_stem)
        is_english = src_lang.startswith("en")

        # Translate header to English so regex + LLM metadata extraction works
        header_en = header_text
        if not is_english and header_text.strip():
            header_en = translate_to_english(header_text[:4000], src_lang)

        # Regex-first — instant and quota-free
        auto_meta = extract_metadata(header_en, safe_stem)

        # LLM enrichment only for fields regex couldn't fill
        if not auto_meta["patent_number"] or not auto_meta["title"]:
            try:
                missing = [k for k in ("patent_number", "title", "assignee", "publication_date")
                           if not auto_meta.get(k)]
                prompt = (
                    f"Extract these missing patent fields: {missing}.\n"
                    "Respond ONLY with minified JSON:\n"
                    '{"patent_number":"","title":"","assignee":"","jurisdiction":"","publication_date":""}\n'
                    f"TEXT:\n{header_en[:2500]}"
                )
                enriched = llm_json(prompt)
                for k, v in enriched.items():
                    if k in auto_meta and not auto_meta[k] and isinstance(v, str) and v.strip():
                        auto_meta[k] = v.strip()
            except Exception as exc:
                log.warning("LLM metadata enrichment skipped: %s", exc)

        # User-supplied form values win over auto-extracted
        final_number = patent_number.strip() or auto_meta["patent_number"] or safe_stem
        final_title  = title.strip()         or auto_meta["title"]          or "Untitled Patent"
        final_assign = assignee.strip()      or auto_meta["assignee"]       or ""
        final_jx     = jurisdiction.strip()  or auto_meta["jurisdiction"]   or "US"
        final_date   = _normalise_date(pub_date.strip() or auto_meta["publication_date"] or "")

        log.info("Metadata → number=%s title=%s assignee=%s jx=%s date=%s",
                 final_number, final_title, final_assign, final_jx, final_date)

        db_resp = supabase.table("patent_documents").upsert(
            {
                "patent_number":    final_number,
                "title":            final_title,
                "assignee":         final_assign,
                "jurisdiction":     final_jx,
                "publication_date": final_date,
            },
            on_conflict="patent_number",
        ).execute()
        patent_id: str = db_resp.data[0]["id"]
        log.info("Upserted patent_documents → id=%s", patent_id)

        # Extract text from all pages into one full-document string first (Fix 1).
        # Processing page-by-page caused claims that span page boundaries to be
        # split into two incomplete fragments, neither of which was classifiable.
        full_text = ""
        total_pages = len(doc)
        for page_num in range(total_pages):
            try:
                page_text = extract_page_text(doc[page_num], src_lang)
                if page_text:
                    full_text += page_text + "\n\n"
            except Exception as exc:
                log.warning("Page %d error (skipping): %s", page_num + 1, exc)

        all_chunks = split_into_chunks(full_text)
        all_chunks = [c for c in all_chunks if len(c["content"]) >= 10]
        log.info("Extracted %d chunks from %d pages", len(all_chunks), total_pages)

        # Extract figure pages first — even image-only PDFs (0 text chunks) still have figures.
        # PostgREST requires bytea to be sent as a \x-prefixed hex string.
        figures = extract_figure_pages(doc)
        images_inserted = 0
        if figures:
            image_records = [
                {
                    "patent_id":   patent_id,
                    "page_number": fig["page_number"],
                    "width":       fig["width"],
                    "height":      fig["height"],
                    "image_data":  "\\x" + fig["image_data"].hex(),
                }
                for fig in figures
            ]
            # Insert one image at a time — PNG blobs can be several hundred KB each;
            # batching them together easily exceeds Supabase's statement timeout.
            for record in image_records:
                supabase.table("patent_images").insert(record).execute()
                images_inserted += 1
            log.info("Stored %d figure images for patent=%s", images_inserted, final_number)

        if not all_chunks:
            return {
                "patent_id":       patent_id,
                "chunks_inserted": 0,
                "images_inserted": images_inserted,
                "patent_number":   final_number,
                "language":        src_lang,
                "warning": "No text content extracted from PDF.",
            }

        all_chunks = translate_chunks(all_chunks, src_lang)

        log.info("Generating embeddings for %d chunks…", len(all_chunks))
        texts = [c["content"] for c in all_chunks]
        embeddings = embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=BATCH_SIZE
        )

        # pgvector via Supabase REST requires the string format "[x,y,z,...]" — not a Python list
        records = [
            {
                "patent_id":    patent_id,
                "section_type": chunk["section_type"],
                "content":      chunk["content"],
                "embedding":    "[" + ",".join(f"{v:.8f}" for v in emb.tolist()) + "]",
            }
            for chunk, emb in zip(all_chunks, embeddings)
        ]

        total = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i: i + BATCH_SIZE]
            # supabase-py >= 2.9 defaults to return=minimal (HTTP 204, no body),
            # so resp.data is always []. Count the batch directly; PostgREST raises
            # on any real failure (constraint violation, RLS block, schema mismatch).
            supabase.table("patent_chunks").insert(batch).execute()
            total += len(batch)

        log.info("Ingest complete: patent=%s chunks=%d images=%d lang=%s translated=%s",
                 final_number, total, images_inserted, src_lang, not is_english)

        return {
            "patent_id":       patent_id,
            "chunks_inserted": total,
            "images_inserted": images_inserted,
            "patent_number":   final_number,
            "language":        src_lang,
            "translated":      not is_english,
            "metadata": {
                "title":            final_title,
                "assignee":         final_assign,
                "jurisdiction":     final_jx,
                "publication_date": final_date,
            },
        }

    finally:
        doc.close()
        tmp_path.unlink(missing_ok=True)
