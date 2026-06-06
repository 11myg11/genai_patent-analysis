"""
app/routes/api.py — All JSON API endpoints for the Patent Analysis Platform.

Route handlers are kept thin: they validate input, delegate to services/utils,
and map exceptions to HTTP status codes. No business logic lives here.

Endpoints:
  GET  /health                          → {"status": "ok"} — liveness probe
  POST /api/v1/extract-metadata         → extract patent metadata from a PDF upload
                                          (regex-first, LLM fills missing fields)
  POST /api/v1/ingest                   → full ingestion pipeline: PDF → chunks → embeddings → Supabase
  GET  /api/v1/patents                  → list all patents (optional ?jurisdiction= ?assignee= filters)
  GET    /api/v1/patents/{patent_id}      → single patent with all its chunks
  PATCH  /api/v1/patents/{patent_id}      → update metadata fields (partial update)
  DELETE /api/v1/patents/{patent_id}      → delete patent + all chunks + images
  GET  /api/v1/patents/{patent_id}/summary → LLM-generated one-page patent summary
  GET  /api/v1/patents/{patent_id}/images → list figure metadata (id, page_number) — no binary data
  GET  /api/v1/images/{image_id}          → stream a single figure as image/png
  POST /api/v1/compare                  → compare two patents: overlap analysis + similarity score
  POST /api/v1/evaluate-design          → core IP risk assessment + audited design-around proposals

All blocking Supabase calls are wrapped with asyncio.to_thread().
The ingestion route calls services/ingest.py:ingest_pdf() via asyncio.to_thread()
because the full pipeline (PDF parsing, embedding, bulk insert) is synchronous.
"""
import asyncio
import json
import logging
from typing import Optional

import fitz
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.models import (
    ChunkReference,
    CompareRequest,
    DesignEvaluationRequest,
    DesignEvaluationResponse,
    PatentUpdateRequest,
)
from app.services.ingest import ingest_pdf
from app.services.llm import llm_json
from app.services.retrieval import (
    build_context_block,
    call_agent_auditor,
    call_agent_generator,
    fetch_hybrid_matches,
)
from app.state import state
from app.utils.metadata import extract_metadata
from app.utils.translation import detect_language, translate_to_english

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/health", include_in_schema=False)
async def health():
    return JSONResponse({"status": "ok"})


@router.post("/api/v1/extract-metadata")
async def extract_metadata_endpoint(file: UploadFile = File(...), filename_hint: str = Form("")):
    contents = await file.read()
    tmp_name = file.filename or "patent.pdf"
    _name = (filename_hint or tmp_name).strip()

    # Write to a temp file so PyMuPDF can open it reliably across all versions
    import tempfile, os as _os
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with _os.fdopen(tmp_fd, "wb") as f:
            f.write(contents)
        try:
            doc = fitz.open(tmp_path)
            header_text = ""
            for i in range(min(4, len(doc))):
                header_text += doc[i].get_text("text").strip() + "\n\n"

            # Many USPTO patents embed all content as raster images — no text layer.
            # Reuse extract_page_text (which already has the OCR fallback) on the
            # first 2 pages to recover metadata from image-only PDFs.
            if len(header_text.strip()) < 80:
                from app.utils.pdf import extract_page_text
                try:
                    header_text = ""
                    for i in range(min(2, len(doc))):
                        header_text += extract_page_text(doc[i], src_lang) + "\n\n"
                    log.info("extract-metadata: OCR fallback produced %d chars", len(header_text))
                except Exception as ocr_exc:
                    log.warning("extract-metadata: OCR fallback failed: %s", ocr_exc)
            doc.close()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"PDF read failed: {exc}") from exc
    finally:
        _os.unlink(tmp_path)

    src_lang = detect_language(header_text, _name)

    header_en = header_text
    if not src_lang.startswith("en") and header_text.strip():
        header_en = translate_to_english(header_text[:4000], src_lang)

    meta = extract_metadata(header_en, _name)

    # Only call LLM when we have enough text to work with — skip for scanned/empty PDFs
    # to avoid a slow call that returns nothing useful.
    has_text = len(header_en.strip()) > 80
    missing_key_fields = not meta["patent_number"] or not meta["title"] or not meta["assignee"]
    if has_text and missing_key_fields:
        try:
            missing = [k for k in ("patent_number", "title", "assignee", "publication_date")
                       if not meta.get(k)]
            prompt = (
                f"Extract these missing patent metadata fields: {missing}.\n"
                "Respond ONLY with minified JSON matching this schema exactly:\n"
                '{"patent_number":"","title":"","assignee":"","jurisdiction":"","publication_date":""}\n'
                "Return empty string for any field you cannot find.\n\n"
                f"TEXT:\n{header_en[:2500]}"
            )
            enriched = llm_json(prompt)
            for k, v in enriched.items():
                if k in meta and not meta[k] and isinstance(v, str) and v.strip():
                    meta[k] = v.strip()
        except Exception as exc:
            log.warning("LLM metadata enrichment skipped: %s", exc)

    return {**meta, "detected_language": src_lang, "translated": not src_lang.startswith("en")}


@router.post("/api/v1/ingest")
async def ingest_from_browser(
    file:          UploadFile = File(...),
    patent_number: str        = Form(""),
    title:         str        = Form(""),
    assignee:      str        = Form(""),
    jurisdiction:  str        = Form(""),
    pub_date:      str        = Form(""),
):
    contents = await file.read()
    try:
        result = await asyncio.to_thread(
            ingest_pdf,
            pdf_bytes=contents,
            filename=file.filename or "patent.pdf",
            supabase=state.supabase,
            embed_model=state.embed_model,
            patent_number=patent_number,
            title=title,
            assignee=assignee,
            jurisdiction=jurisdiction,
            pub_date=pub_date,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.get("/api/v1/patents")
async def list_patents(jurisdiction: Optional[str] = None, assignee: Optional[str] = None):
    try:
        def _query():
            q = (
                state.supabase.table("patent_documents")
                .select("id,patent_number,title,assignee,jurisdiction,publication_date,created_at")
                .order("created_at", desc=True)
            )
            if jurisdiction:
                q = q.eq("jurisdiction", jurisdiction)
            if assignee:
                q = q.ilike("assignee", f"%{assignee}%")
            return q.execute()
        resp = await asyncio.to_thread(_query)
        return resp.data or []
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/api/v1/patents/{patent_id}")
async def get_patent(patent_id: str):
    try:
        def _query():
            doc = state.supabase.table("patent_documents").select("*").eq("id", patent_id).single().execute()
            chunks = (
                state.supabase.table("patent_chunks")
                .select("id,section_type,content")
                .eq("patent_id", patent_id)
                .order("id").execute()
            )
            return doc, chunks
        doc_resp, chunks_resp = await asyncio.to_thread(_query)
        return {"document": doc_resp.data, "chunks": chunks_resp.data or []}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/api/v1/patents/{patent_id}")
async def update_patent(patent_id: str, body: PatentUpdateRequest):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update.")
    try:
        def _update():
            return (
                state.supabase.table("patent_documents")
                .update(fields)
                .eq("id", patent_id)
                .execute()
            )
        resp = await asyncio.to_thread(_update)
        if not resp.data:
            raise HTTPException(status_code=404, detail="Patent not found.")
        return resp.data[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.delete("/api/v1/patents/{patent_id}", status_code=204)
async def delete_patent(patent_id: str):
    """Delete a patent and all its child rows.

    Uses per-table batch sizes to stay within Supabase's statement timeout:
    - patent_chunks: 50 per batch (rows are small — just text + embedding string)
    - patent_images: 1 per batch (rows contain full PNG bytea, often 200 KB+;
      deleting even a handful at once causes PostgreSQL to exceed the timeout)
    """
    def _delete_in_batches(table: str, id_col: str, batch: int):
        while True:
            rows = (
                state.supabase.table(table)
                .select("id")
                .eq(id_col, patent_id)
                .limit(batch)
                .execute()
            ).data or []
            if not rows:
                break
            ids = [r["id"] for r in rows]
            state.supabase.table(table).delete().in_("id", ids).execute()

    try:
        await asyncio.to_thread(_delete_in_batches, "patent_chunks", "patent_id", 50)
        await asyncio.to_thread(_delete_in_batches, "patent_images", "patent_id", 1)
        await asyncio.to_thread(
            lambda: state.supabase.table("patent_documents").delete().eq("id", patent_id).execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/api/v1/patents/{patent_id}/summary")
async def get_patent_summary(patent_id: str):
    try:
        def _query():
            doc = state.supabase.table("patent_documents").select("*").eq("id", patent_id).single().execute()
            chunks = (
                state.supabase.table("patent_chunks")
                .select("section_type,content")
                .eq("patent_id", patent_id)
                .in_("section_type", ["claim_independent", "claim_dependent", "description"])
                .limit(20).execute()
            )
            return doc, chunks
        doc_resp, chunks_resp = await asyncio.to_thread(_query)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    doc = doc_resp.data
    chunks = chunks_resp.data or []
    context = "\n\n".join(f"[{c['section_type'].upper()}] {c['content']}" for c in chunks)

    prompt = f"""You are a patent analyst. Given the following patent content, produce a JSON summary.
Respond ONLY with minified JSON. No markdown. No explanation outside JSON.

Output schema:
{{"abstract":"2-3 sentence technical abstract","key_claims":["claim1","claim2","claim3"],"technology_domain":"...","novelty_statement":"...","commercial_relevance":"..."}}

PATENT: {doc.get('patent_number')} — {doc.get('title')}
ASSIGNEE: {doc.get('assignee')}
CONTENT:
{context[:3000]}"""

    try:
        summary = llm_json(prompt)
    except Exception:
        summary = {
            "abstract": "Summary generation failed.",
            "key_claims": [], "technology_domain": "",
            "novelty_statement": "", "commercial_relevance": "",
        }

    return {"document": doc, "summary": summary, "chunk_count": len(chunks)}


@router.get("/api/v1/patents/{patent_id}/images")
async def list_patent_images(patent_id: str):
    """Return metadata (id + page_number) for all stored figures of a patent.
    Binary data is NOT included — use GET /api/v1/images/{image_id} to fetch each image.
    """
    try:
        resp = await asyncio.to_thread(
            lambda: (
                state.supabase.table("patent_images")
                .select("id,page_number,width,height")
                .eq("patent_id", patent_id)
                .order("page_number")
                .execute()
            )
        )
        return resp.data or []
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/api/v1/images/{image_id}")
async def serve_patent_image(image_id: str):
    """Stream a single figure as a PNG image.
    PostgREST returns bytea columns as \\x-prefixed hex strings; we decode here.
    """
    try:
        resp = await asyncio.to_thread(
            lambda: (
                state.supabase.table("patent_images")
                .select("image_data")
                .eq("id", image_id)
                .single()
                .execute()
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Image not found: {exc}") from exc

    raw = resp.data.get("image_data", "")
    # PostgREST hex format: "\x89504e47..." — strip the leading \x before decoding
    if isinstance(raw, str) and raw.startswith("\\x"):
        image_bytes = bytes.fromhex(raw[2:])
    elif isinstance(raw, (bytes, bytearray)):
        image_bytes = bytes(raw)
    else:
        raise HTTPException(status_code=500, detail="Unexpected image_data format from database")

    return Response(content=image_bytes, media_type="image/png")


@router.post("/api/v1/compare")
async def compare_patents(req: CompareRequest):
    try:
        def _fetch_docs():
            a = state.supabase.table("patent_documents").select("*").eq("id", req.patent_id_a).single().execute()
            b = state.supabase.table("patent_documents").select("*").eq("id", req.patent_id_b).single().execute()
            return a, b
        a_resp, b_resp = await asyncio.to_thread(_fetch_docs)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Patent lookup failed: {exc}") from exc

    def _get_chunks(patent_id: str):
        rows = (
            state.supabase.table("patent_chunks")
            .select("section_type,content")
            .eq("patent_id", patent_id)
            .in_("section_type", ["claim_independent", "claim_dependent"])
            .limit(20).execute().data or []
        )
        if rows:
            return [r["content"] for r in rows]
        return (
            state.supabase.table("patent_chunks")
            .select("content").eq("patent_id", patent_id)
            .limit(20).execute().data or []
        ) and [r["content"] for r in
               state.supabase.table("patent_chunks").select("content")
               .eq("patent_id", patent_id).limit(20).execute().data or []]

    try:
        a_contents, b_contents = await asyncio.to_thread(
            lambda: (_get_chunks(req.patent_id_a), _get_chunks(req.patent_id_b))
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chunk fetch failed: {exc}") from exc

    a_text = "\n\n".join(a_contents or [])
    b_text = "\n\n".join(b_contents or [])

    log.info("compare | A=%s chunks=%d B=%s chunks=%d",
             a_resp.data.get("patent_number"), len(a_contents or []),
             b_resp.data.get("patent_number"), len(b_contents or []))

    similarity_score = 0.0
    if a_text.strip() and b_text.strip():
        try:
            ea = state.embed_model.encode([a_text[:2000]], normalize_embeddings=True)[0]
            eb = state.embed_model.encode([b_text[:2000]], normalize_embeddings=True)[0]
            similarity_score = float(np.dot(ea, eb))
        except Exception as exc:
            log.warning("Similarity embedding failed: %s", exc)

    a_doc, b_doc = a_resp.data, b_resp.data

    def _block(doc, text):
        if text:
            return f"{doc.get('patent_number','?')} — {doc.get('title','?')}\nAssignee: {doc.get('assignee','?')}\n\n{text[:2000]}"
        return f"{doc.get('patent_number','?')} — {doc.get('title','?')} (no chunk text indexed)"

    schema = (
        '{"overlap_summary":"string",'
        '"overlapping_claims":[{"claim_a":"string","claim_b":"string",'
        '"overlap_type":"exact|semantic|functional","risk_level":"HIGH|MEDIUM|LOW"}],'
        '"differentiating_features":["string"],'
        '"overall_risk":"HIGH|MEDIUM|LOW|CLEAR",'
        '"recommendation":"string"}'
    )
    prompt = (
        "You are a senior patent IP analyst. Compare the two patents below.\n"
        "Respond ONLY with minified JSON matching this schema — no markdown, no preamble:\n"
        f"{schema}\n\n"
        f"PATENT A:\n{_block(a_doc, a_text)}\n\n"
        f"PATENT B:\n{_block(b_doc, b_text)}\n\n"
        "Identify structural and semantic overlaps between the claims. "
        "Assign a risk level to each overlap. List differentiating features. "
        "Give overall_risk and a one-sentence recommendation."
    )

    try:
        analysis = llm_json(prompt)
        for key, default in [
            ("overlap_summary", ""), ("overlapping_claims", []),
            ("differentiating_features", []), ("overall_risk", "UNKNOWN"), ("recommendation", ""),
        ]:
            if key not in analysis:
                analysis[key] = default
        log.info("compare OK | risk=%s overlaps=%d",
                 analysis.get("overall_risk"), len(analysis.get("overlapping_claims", [])))
    except HTTPException:
        raise
    except Exception as exc:
        log.error("compare analysis error: %r", exc)
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}") from exc

    return {
        "patent_a":         a_doc,
        "patent_b":         b_doc,
        "similarity_score": round(similarity_score * 100, 1),
        "analysis":         analysis,
    }


@router.post("/api/v1/evaluate-design", response_model=DesignEvaluationResponse)
async def evaluate_design(request: DesignEvaluationRequest):
    log.info("evaluate-design | product_id=%s jurisdiction=%s",
             request.product_id, request.jurisdiction)

    try:
        query_embedding = state.embed_model.encode(
            [request.proposed_specifications], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}") from exc

    raw_chunks = await asyncio.to_thread(
        fetch_hybrid_matches, query_embedding, request.proposed_specifications, request.jurisdiction
    )

    if not raw_chunks:
        return DesignEvaluationResponse(
            product_id=request.product_id, risk_status="CLEAR",
            infringement_map=[], design_arounds=[],
            matched_chunks=[], token_budget_used=0,
        )

    chunk_refs = [
        ChunkReference(**{k: c.get(k, "") for k in ChunkReference.model_fields},
                       rrf_score=float(c.get("rrf_score", 0.0)))
        for c in raw_chunks
    ]
    context_block = build_context_block(raw_chunks)
    gen_out = call_agent_generator(context_block, request.proposed_specifications, request.component_scope)
    aud_out = call_agent_auditor(gen_out, request.component_scope)
    merged_das = aud_out.get("design_arounds_merged", [])
    token_est = (
        len(context_block) + len(request.proposed_specifications) + len(json.dumps(gen_out))
    ) // 4

    return DesignEvaluationResponse(
        product_id=request.product_id,
        risk_status=aud_out.get("risk_status", "UNKNOWN"),
        infringement_map=aud_out.get("infringement_map", []),
        design_arounds=merged_das,
        matched_chunks=chunk_refs,
        token_budget_used=token_est,
    )
