"""
scripts/ingest_patents.py — CLI wrapper for the patent ingestion pipeline.

Use this script to ingest patent PDFs from the command line without starting
the web server. Metadata (patent number, title, assignee, etc.) is extracted
automatically from the PDF; all fields can be overridden via flags.

Internally calls app/services/ingest.py:ingest_pdf() — the same pipeline used
by the web API endpoint POST /api/v1/ingest.

Usage:

Usage:
  python scripts/ingest_patents.py --pdf path/to/patent.pdf

  # Override any auto-extracted field:
  python scripts/ingest_patents.py --pdf patent.pdf --patent-number EP1234567A1
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest a patent PDF — metadata is auto-extracted, all args optional.",
    )
    p.add_argument("--pdf",           required=True, help="Path to patent PDF.")
    p.add_argument("--patent-number", default="",    help="Override patent number.")
    p.add_argument("--title",         default="",    help="Override title.")
    p.add_argument("--assignee",      default="",    help="Override assignee.")
    p.add_argument("--jurisdiction",  default="",    help="Override jurisdiction (US/EP/CN…).")
    p.add_argument("--pub-date",      default="",    help="Override publication date YYYY-MM-DD.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not settings.supabase_url or not settings.supabase_anon_key:
        log.error("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        sys.exit(1)

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    supabase = create_client(settings.supabase_url, settings.supabase_anon_key)
    log.info("Supabase connected.")

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


if __name__ == "__main__":
    main()
