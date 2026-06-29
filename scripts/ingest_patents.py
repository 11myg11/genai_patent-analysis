"""
scripts/ingest_patents.py — CLI for patent PDF ingestion.

Supports two modes (mutually exclusive):

  Single-file mode:
    python scripts/ingest_patents.py --pdf "path/to/patent.pdf"
    All five metadata fields can be overridden via flags; auto-extraction is
    used for any field left blank.

  Directory/batch mode:
    python scripts/ingest_patents.py --dir path/to/folder/
    Ingests every *.pdf in the directory sequentially. A failed PDF is logged
    and skipped — it does not abort the remaining files.
    Use --skip-existing to query Supabase first and skip already-ingested patents.

Internally calls app/services/ingest.py:ingest_pdf() — the same pipeline used
by the web API endpoint POST /api/v1/ingest.

Usage examples:
  python scripts/ingest_patents.py --pdf patent.pdf
  python scripts/ingest_patents.py --pdf patent.pdf --patent-number EP1234567A1
  python scripts/ingest_patents.py --dir /data/patents
  python scripts/ingest_patents.py --dir /data/patents --skip-existing
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Set

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure tesseract — same logic as main.py
def _configure_tesseract() -> None:
    if not shutil.which("tesseract"):
        for candidate in ("/opt/homebrew/bin", "/usr/local/bin"):
            if shutil.which("tesseract", path=candidate):
                os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
                break
    if not os.environ.get("TESSDATA_PREFIX"):
        from pathlib import Path as _Path
        tess_bin = shutil.which("tesseract")
        if tess_bin:
            tessdata = _Path(tess_bin).parent.parent / "share" / "tessdata"
            if tessdata.is_dir():
                os.environ["TESSDATA_PREFIX"] = str(tessdata)

_configure_tesseract()

from sentence_transformers import SentenceTransformer
from supabase import create_client

from app.config import EMBEDDING_MODEL, settings
from app.services.ingest import ingest_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Matches the leading patent-office prefix in filenames like EP3456789A1.pdf.
_PATENT_NUMBER_RE = re.compile(r"^([A-Z]{2}[\d]+[A-Z]?\d?)", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest patent PDFs — single file or whole directory.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to a single patent PDF.")
    group.add_argument("--dir", help="Directory of PDFs to ingest (all *.pdf files).")

    # Metadata overrides — only meaningful in single-file mode
    p.add_argument("--patent-number", default="", help="Override patent number.")
    p.add_argument("--title",         default="", help="Override title.")
    p.add_argument("--assignee",      default="", help="Override assignee.")
    p.add_argument("--jurisdiction",  default="", help="Override jurisdiction (US/EP/CN…).")
    p.add_argument("--pub-date",      default="", help="Override publication date YYYY-MM-DD.")

    p.add_argument("--skip-existing", action="store_true",
                   help="(--dir mode) Skip PDFs whose patent number already exists in Supabase.")
    return p.parse_args()


def _fetch_existing_patent_numbers(supabase) -> Set[str]:
    """Return all patent_number values from patent_documents, paginating past PostgREST's 1000-row cap."""
    numbers: Set[str] = set()
    page_size = 1000
    offset = 0
    try:
        while True:
            resp = (
                supabase.table("patent_documents")
                .select("patent_number")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            for row in batch:
                numbers.add(row["patent_number"])
            if len(batch) < page_size:
                break
            offset += page_size
    except Exception:
        log.warning("Could not fetch existing patent numbers; --skip-existing has no effect.", exc_info=True)
    return numbers


def _predict_patent_number_from_filename(filename: str) -> Optional[str]:
    """Predict the patent number from a filename without reading the PDF.

    Returns the uppercased match (e.g. "EP3456789A1") or None if the filename
    stem doesn't start with a recognisable office-code + digit pattern.
    """
    stem = Path(filename).stem.strip()
    m = _PATENT_NUMBER_RE.match(stem)
    return m.group(1).upper() if m else None


def main() -> None:
    args = _parse_args()

    if not settings.supabase_url or not settings.supabase_anon_key:
        log.error("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        sys.exit(1)

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    supabase = create_client(settings.supabase_url, settings.supabase_anon_key)
    log.info("Supabase connected.")

    # ── Single-file mode ──────────────────────────────────────────────────────
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.is_file():
            log.error("PDF not found: %s", pdf_path)
            sys.exit(1)

        result = ingest_pdf(
            pdf_bytes=pdf_path.read_bytes(),
            filename=pdf_path.name,
            supabase=supabase,
            embed_model=embed_model,
            patent_number=args.patent_number,
            title=args.title,
            assignee=args.assignee,
            jurisdiction=args.jurisdiction,
            pub_date=args.pub_date,
        )
        log.info("Ingestion summary:\n%s", json.dumps(result, indent=2, default=str))
        return

    # ── Directory/batch mode ──────────────────────────────────────────────────
    pdf_dir = Path(args.dir)
    if not pdf_dir.is_dir():
        log.error("Directory not found: %s", pdf_dir)
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDF files found in %s", pdf_dir)
        sys.exit(0)

    log.info("Found %d PDF(s) in %s", len(pdfs), pdf_dir)

    existing: Set[str] = set()
    if args.skip_existing:
        existing = _fetch_existing_patent_numbers(supabase)
        log.info("%d existing patent number(s) fetched from Supabase.", len(existing))

    failed = 0
    for pdf_path in pdfs:
        if args.skip_existing:
            predicted = _predict_patent_number_from_filename(pdf_path.name)
            if predicted and predicted in existing:
                log.info("SKIP  %s  (already ingested as %s)", pdf_path.name, predicted)
                continue

        try:
            result = ingest_pdf(
                pdf_bytes=pdf_path.read_bytes(),
                filename=pdf_path.name,
                supabase=supabase,
                embed_model=embed_model,
            )
            log.info(
                "OK    %s  patent=%s  chunks=%d  images=%d",
                pdf_path.name,
                result.get("patent_number", "?"),
                result.get("chunks_inserted", 0),
                result.get("images_inserted", 0),
            )
        except Exception:
            log.exception("FAIL  %s", pdf_path.name)
            failed += 1

    log.info("Batch done — %d/%d failed.", failed, len(pdfs))


if __name__ == "__main__":
    main()
