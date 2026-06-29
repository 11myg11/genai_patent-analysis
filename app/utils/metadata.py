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
    r'\b(US\s*\d{6,9}\s*[A-Z]\d?)\b',
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


# Corporate suffixes that mark a clean stop point between company name and
# address — covers US (Inc/LLC/Corp/Co/Ltd), DE (GmbH/AG/KGaA/Aktien-
# gesellschaft), UK (Plc), NL (N.V./B.V.), FR/IT/ES (S.A./S.p.A./S.L./S.r.l.),
# and Asian (often appearing as the compound form "Co., Ltd.").
_SINGLE_CORP_SUFFIX = (
    r'(?:Inc|LLC|Ltd|Corp(?:oration)?|Co|GmbH|AG|KGaA|Plc|'
    r'Aktiengesellschaft|'
    r'N\.?\s*V\.?|S\.?\s*A\.?(?:\.?\s*p\.?\s*A\.?)?|S\.?\s*L\.?|Pty|'
    r'B\.?\s*V\.?|S\.?\s*r\.?\s*l\.?)'
)
# Full pattern accepts one suffix optionally followed by more suffixes joined
# by commas/spaces ("Co., Ltd.", "Co. Ltd.", "Co., Ltd., LLC"). Without this
# the post-processor would cut "Toyota Motor Co., Ltd." → "Toyota Motor Co.",
# discarding a legitimate part of the company name.
_CORP_SUFFIX_PATTERN = rf'\b{_SINGLE_CORP_SUFFIX}\.?(?:\s*,?\s*{_SINGLE_CORP_SUFFIX}\.?)*'


def _clean_assignee(raw: str) -> str:
    """Post-process a raw regex-captured assignee string.

    Two jobs:
      1. Collapse interleaved whitespace/newlines (OCR'd two-column patent
         layouts inject newlines and citation fragments mid-name).
      2. Drop the trailing address at the first corporate-suffix + punctuation
         boundary, so 'FOO INC., Grand Cayman (KY)' → 'FOO INC.' rather than
         a mid-word slice. Cleaner output than keeping address fragments that
         the OCR may have mangled anyway.

    If no corporate suffix is found (e.g. assignees like 'Bayerische Motoren
    Werke Aktiengesellschaft' that don't end in a recognised abbreviation),
    the value is returned with whitespace collapsed but untrimmed — better to
    keep extra text than to throw away a legitimate name.
    """
    # Strip trailing comma/semicolon/colon only — not '.' — because patent
    # assignees frequently end with an abbreviation period ("INC.", "CO.")
    # that we want to preserve as part of the canonical name.
    val = re.sub(r'\s+', ' ', raw).strip().rstrip(',;:')
    m = re.match(rf'^(.+?{_CORP_SUFFIX_PATTERN})\s*[,.]', val, re.IGNORECASE)
    if m:
        val = m.group(1).rstrip(',;:')
    return val


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

    # Assignee — INID code (73) first.
    # IMPORTANT: char classes here allow \n\r intentionally. OCR'd two-column
    # patent layouts interleave assignee text with citation lines, so the
    # field often wraps mid-name across newlines (e.g.
    #   "GLOBALFOUNDRIES INC., Grand\nCayman (KY)"). The previous
    # [^\n\r(] char class stopped at the first newline and produced
    # mid-word truncations like "GLOBALFOUNDRIES INC., Grand". We now allow
    # the match to span newlines (still bounded by "(" — the next INID code
    # opener — and a generous char cap), then post-process via
    # _clean_assignee to drop the trailing address.
    m = re.search(r'\(73\)\s*(?:Assignee|Applicant)[:\s]+([^(]{3,150})', text, re.IGNORECASE)
    if m:
        meta["assignee"] = _clean_assignee(m.group(1))
    if not meta["assignee"]:
        m = re.search(
            r'(?:Assignee|Applicant|Anmelder|Titulaire|Patentinhaber)[:\s]+'
            r'([A-Z][^(]{3,150}?)(?=\(|\s*,\s*[A-Z]{2}\b)',
            text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            meta["assignee"] = _clean_assignee(m.group(1))
    if not meta["assignee"]:
        m = re.search(r'(?:ASSIGNEE|APPLICANT)[:\s]+([A-Z][^(]{3,150})', text, re.DOTALL)
        if m:
            meta["assignee"] = _clean_assignee(m.group(1))

    # Title — INID code (54) first.
    # USPTO two-column OCR interleaves title lines with reference citations, so we
    # grab all text between (54) and the next INID code then filter to lines that
    # start with an uppercase letter (citations start with digits like "5,994,840 A").
    m = re.search(r'\(54\)(.*?)(?=\(\d{2}\))', text, re.DOTALL)
    if m:
        raw_lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
        title_lines = []
        for line in raw_lines:
            alpha = [c for c in line if c.isalpha()]
            # Must start with a letter, have ≥5 alphabetic chars, and be mostly uppercase
            if not line[0].isalpha() or len(alpha) < 5:
                continue
            upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
            if upper_ratio < 0.8:
                continue
            if any(w in line.upper() for w in [
                'UNITED STATES', 'PATENT OFFICE', 'APPLICATION',
                'OTHER PUBLICATIONS', 'U.S. PATENT', 'PRIOR ART',
            ]):
                continue
            title_lines.append(line)
            if len(title_lines) >= 3:  # patent titles are rarely longer than 3 lines
                break
        val = ' '.join(title_lines).strip().rstrip('.')
        if len(val) >= 10:
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
