"""
app/utils/metadata.py — Regex-based patent metadata extraction (no LLM, no network).

Extracts patent_number, title, assignee, jurisdiction, and publication_date from
the raw text of a patent's first few pages. Handles USPTO, EPO, CNIPA, JPO, and
KIPO header formats. Called first during ingestion; LLM enrichment is only used
afterwards for any fields this function leaves blank.

Functions:
  extract_metadata(text, filename_hint) -> dict
    Parses the given text and returns a dict with keys:
    patent_number, title, assignee, jurisdiction, publication_date.
    filename_hint (e.g. "EP1234567A1.pdf") is used as a fallback for the
    patent number and jurisdiction when no match is found in the text.

Pattern strategy (in order):
  patent_number  — 9 office-specific regex patterns (US/EP/CN/JP/KR/WO/DE/FR/GB)
  publication_date — ISO → US long-form → compact YYYYMMDD → European DD.MM.YYYY
  assignee       — INID code (73) → generic label patterns
  title          — INID code (54) → title label → all-caps line heuristic
"""
import logging
import re
from datetime import datetime
from typing import Dict

log = logging.getLogger(__name__)

_PN_PATTERNS = [
    r'\b(US\s*\d{6,8}\s*[A-Z]\d?)\b',
    r'\b(EP\s*\d{6,7}\s*[A-Z]\d?)\b',
    r'\b(CN\s*\d{8,12}\s*[A-Z]?)\b',
    r'\b(JP\s*\d{7,13}\s*[A-Z]?)\b',
    r'\b(KR\s*\d{7,12}\s*[A-Z]\d?)\b',
    r'\b(WO\s*\d{4}/?\d{4,7}\s*[A-Z]?\d?)\b',
    r'\b(DE\s*\d{9,12}\s*[A-Z]\d?)\b',
    r'\b(FR\s*\d{7,10}\s*[A-Z]?\d?)\b',
    r'\b(GB\s*\d{7}\s*[A-Z]?)\b',
]

_JX_MAP = {
    "CN": "CN", "JP": "JP", "KR": "KR", "DE": "DE", "FR": "FR",
    "GB": "GB", "EP": "EP", "WO": "WO", "US": "US", "AT": "AT", "CH": "CH",
}


def extract_metadata(text: str, filename_hint: str = "") -> Dict[str, str]:
    """
    Regex-only metadata extraction — no network calls, no quota.
    Handles USPTO, EPO, CNIPA, JPO, KIPO header formats.
    """
    meta: Dict[str, str] = {
        "patent_number": "", "title": "", "assignee": "",
        "jurisdiction": "", "publication_date": "",
    }

    # Patent number
    for pat in _PN_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            meta["patent_number"] = re.sub(r'\s+', '', m.group(1)).upper()
            meta["jurisdiction"] = meta["patent_number"][:2].upper()
            break

    if not meta["patent_number"] and filename_hint:
        fn = filename_hint.replace(".pdf", "").replace(".PDF", "").strip()
        m = re.match(r'^([A-Z]{2}[\d]+[A-Z]?\d?)', fn, re.IGNORECASE)
        if m:
            meta["patent_number"] = m.group(1).upper()
            meta["jurisdiction"] = meta["patent_number"][:2].upper()

    # Publication date — try formats in priority order
    m = re.search(
        r'(?:Date of Patent|Publication Date|Pub\.?\s*Date)[:\s]+(\d{4}-\d{2}-\d{2})',
        text, re.IGNORECASE,
    )
    if m:
        meta["publication_date"] = m.group(1)

    if not meta["publication_date"]:
        m = re.search(
            r'(?:Date of Patent|Publication Date)[:\s]+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})',
            text, re.IGNORECASE,
        )
        if m:
            raw = m.group(1).strip()
            for fmt in ('%B %d, %Y', '%b. %d, %Y', '%B %d %Y', '%b %d, %Y'):
                try:
                    meta["publication_date"] = datetime.strptime(raw, fmt).strftime('%Y-%m-%d')
                    break
                except ValueError:
                    pass

    if not meta["publication_date"]:
        m = re.search(r'\b(2\d{3}[01]\d[0-3]\d)\b', text)
        if m:
            try:
                meta["publication_date"] = datetime.strptime(m.group(1), '%Y%m%d').strftime('%Y-%m-%d')
            except ValueError:
                pass

    if not meta["publication_date"]:
        m = re.search(r'\b(\d{2}\.\d{2}\.\d{4})\b', text)
        if m:
            try:
                meta["publication_date"] = datetime.strptime(m.group(1), '%d.%m.%Y').strftime('%Y-%m-%d')
            except ValueError:
                pass

    # Assignee — INID code (73) first
    m = re.search(r'\(73\)\s*(?:Assignee|Applicant)[:\s]+([^\n\r(]{3,80})', text, re.IGNORECASE)
    if m:
        meta["assignee"] = m.group(1).strip().rstrip(',.')
    if not meta["assignee"]:
        m = re.search(
            r'(?:Assignee|Applicant|Anmelder|Titulaire|Patentinhaber)[:\s]+'
            r'([A-Z][^\n\r(]{3,80}?)(?=\s*[\n\r(]|\s*,\s*[A-Z]{2}\b)',
            text, re.IGNORECASE,
        )
        if m:
            meta["assignee"] = m.group(1).strip().rstrip(',.')
    if not meta["assignee"]:
        m = re.search(r'(?:ASSIGNEE|APPLICANT)[:\s]+([A-Z][^\n\r]{3,80})', text)
        if m:
            meta["assignee"] = m.group(1).strip().rstrip(',.')

    # Title — INID code (54) first
    m = re.search(r'\(54\)\s*([A-Z][^\n\r]{10,150})', text)
    if m:
        val = m.group(1).strip().rstrip('.')
        if not any(w in val.upper() for w in ['UNITED STATES', 'PATENT OFFICE', 'APPLICATION']):
            meta["title"] = val.title() if val.isupper() else val
    if not meta["title"]:
        m = re.search(
            r'(?:TITLE OF INVENTION|Title of Invention|Invention Title)[:\s]+([^\n\r]{10,150})',
            text, re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip().rstrip('.')
            meta["title"] = val.title() if val.isupper() else val
    if not meta["title"]:
        for line in text.splitlines():
            line = line.strip()
            if (15 <= len(line) <= 120 and line.isupper()
                    and not any(w in line for w in
                        ['UNITED STATES', 'PATENT', 'OFFICE', 'APPLICATION',
                         'PUBLICATION', 'INTERNATIONAL', 'WORLD'])):
                meta["title"] = line.title()
                break

    # Jurisdiction fallback from filename prefix
    if not meta["jurisdiction"]:
        prefix = (filename_hint or "")[:2].upper()
        meta["jurisdiction"] = _JX_MAP.get(prefix, "US")

    log.info(
        "Regex metadata: number=%s title=%s assignee=%s jx=%s date=%s",
        meta["patent_number"],
        (meta["title"][:40] + "...") if len(meta["title"]) > 40 else meta["title"],
        (meta["assignee"][:40] + "...") if len(meta["assignee"]) > 40 else meta["assignee"],
        meta["jurisdiction"],
        meta["publication_date"],
    )
    return meta
