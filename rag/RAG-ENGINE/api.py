"""
api.py - FMPC Study Assistant Backend

Usage:
    py -3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000
"""

import os
import sys
import json
import hashlib
import requests
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── CONFIG ─────────────────────────────────────────────────────────
CHROMA_PATH  = r"C:\FMPC_Scribe\chroma_db"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 5
N_QCM        = 15

KNOWN_DISCIPLINES = [
    "Toutes", "anatomie", "bacteriologie", "histologie",
    "hematologie", "embryologie", "immunologie", "physiologie", "biochimie", "virologie"
]

EXAM_BLOCK_MAP = {
    "microbiologie":             ["bacteriologie", "virologie"],
    "anatomie":                  ["anatomie"],
    "histologie-embryologie":    ["histologie", "embryologie"],
    "hematologie-immunologie":   ["hematologie", "immunologie"],
    "physiologie":               ["physiologie"],
}

# ── APP ─────────────────────────────────────────────────────────────
app = FastAPI(title="FMPC Study Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CHROMA ─────────────────────────────────────────────────────────
_collections = None
_exam_col    = None

def get_collections():
    global _collections
    if _collections is None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            _collections = {}
            for name in ["slides", "transcripts"]:
                try:
                    _collections[name] = client.get_collection(name=name, embedding_function=ef)
                except Exception:
                    _collections[name] = None
        except Exception as e:
            print(f"[API] ChromaDB init error: {e}")
            _collections = {"slides": None, "transcripts": None}
    return _collections

def get_exam_collection():
    global _exam_col
    if _exam_col is None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            _exam_col = client.get_collection(name="exam_questions", embedding_function=ef)
        except Exception:
            _exam_col = None
    return _exam_col

# ── RETRIEVAL ──────────────────────────────────────────────────────
def retrieve(query, discipline_filter, top_k=TOP_K):
    collections = get_collections()
    where = None
    if discipline_filter and discipline_filter != "Toutes":
        if isinstance(discipline_filter, list):
            where = {"discipline": {"$in": discipline_filter}} if len(discipline_filter) > 1 else {"discipline": discipline_filter[0]}
        else:
            where = {"discipline": discipline_filter}

    results = {"slides": [], "transcripts": []}
    for col_name, col in collections.items():
        if col is None:
            continue
        try:
            kwargs = {"query_texts": [query], "n_results": top_k,
                      "include": ["documents", "metadatas", "distances"]}
            if where:
                kwargs["where"] = where
            res = col.query(**kwargs)
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                results[col_name].append({
                    "text":       doc,
                    "discipline": meta.get("discipline", "?"),
                    "filename":   meta.get("filename", "?"),
                    "page":       meta.get("page", meta.get("chunk_index", "?")),
                    "score":      round((1 - dist) * 100, 1),
                })
        except Exception as e:
            print(f"[API] Retrieval error ({col_name}): {e}")
    return results

def build_context(results):
    parts = []
    if results["transcripts"]:
        parts.append("=== TRANSCRIPTIONS ===")
        for i, r in enumerate(results["transcripts"], 1):
            parts.append(f"[{i} | {r['discipline']} | {r['filename']} | chunk {r['page']} | {r['score']}%]\n{r['text']}")
    if results["slides"]:
        parts.append("=== SLIDES ===")
        for i, r in enumerate(results["slides"], 1):
            parts.append(f"[{i} | {r['discipline']} | {r['filename']} | page {r['page']} | {r['score']}%]\n{r['text']}")
    return "\n\n".join(parts)

def ollama(prompt, timeout=180):
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False
        }, timeout=timeout)
        return res.json().get("response", "").strip()
    except Exception as e:
        raise HTTPException(500, f"Ollama error: {e}")

def extract_json_array(raw):
    s = raw.find("[")
    e = raw.rfind("]") + 1
    if s == -1 or e == 0:
        return None
    return raw[s:e]

def normalize_question(q):
    if not all(k in q for k in ["question", "choices", "correct"]):
        return None
    for l in "ABCDE":
        q["choices"].setdefault(l, "")
    if isinstance(q["correct"], str):
        q["correct"] = sorted(q["correct"].replace(" ", "").upper().split(","))
    else:
        q["correct"] = sorted([c.upper() for c in q["correct"]])
    q["correct"] = [c for c in q["correct"] if c in "ABCDE"] or ["A"]
    q.setdefault("explanation", "")
    q.setdefault("distractors", {})
    q.setdefault("source_claim", "")
    q.setdefault("source", "")
    return q

# ── MODELS ─────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:      str
    discipline: str = "Toutes"
    top_k:      int = TOP_K

class QCMRequest(BaseModel):
    discipline: str   = "Toutes"
    course:     str   = ""    # specific filename to restrict retrieval to
    topic:      str   = ""
    top_k:      int   = TOP_K

# ── ROUTES ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/api/stats")
async def stats():
    cols = get_collections()
    ec   = get_exam_collection()
    return {
        "slides":      cols["slides"].count()    if cols.get("slides")      else 0,
        "transcripts": cols["transcripts"].count() if cols.get("transcripts") else 0,
        "exams":       ec.count()                if ec                      else 0,
        "disciplines": KNOWN_DISCIPLINES,
    }

@app.get("/api/courses")
async def list_courses():
    """Return all available course filenames grouped by discipline."""
    cols = get_collections()
    courses = {}
    for col_name, col in cols.items():
        if col is None:
            continue
        try:
            results = col.get(include=["metadatas"])
            for meta in results["metadatas"]:
                disc  = meta.get("discipline", "?")
                fname = meta.get("filename", "?")
                if fname and fname != "?":
                    courses.setdefault(disc, set()).add(fname)
        except Exception:
            pass
    return {"courses": {d: sorted(fnames) for d, fnames in sorted(courses.items())}}

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    results = retrieve(req.query, req.discipline if req.discipline != "Toutes" else None, req.top_k)
    if not any(results[k] for k in results):
        return {"answer": "Aucun résultat trouvé dans la base de données pour cette question.", "sources": results}

    context = build_context(results)
    prompt  = (
        "Tu es un assistant pédagogique pour les étudiants en 1ère année de médecine à la FMPC. "
        "Réponds en te basant UNIQUEMENT sur le contexte fourni. "
        "Si le contexte est insuffisant, dis-le clairement. "
        "Réponds en français. Sois précis et concis.\n\n"
        "RÈGLES:\n"
        "- Priorité aux transcriptions (paroles du professeur)\n"
        "- Complète avec les slides si nécessaire\n"
        "- Signale toute contradiction\n"
        "- Ne fabrique rien hors du contexte\n\n"
        f"CONTEXTE:\n{context}\n\nQUESTION: {req.query}\n\nRÉPONSE:"
    )
    answer = ollama(prompt, timeout=120)
    return {"answer": answer, "sources": results}

def _retrieve_from_course(filename, query, top_k=6):
    """Retrieve chunks from a specific course file."""
    cols    = get_collections()
    results = {"slides": [], "transcripts": []}
    where   = {"filename": filename}
    for col_name, col in cols.items():
        if col is None:
            continue
        try:
            # First try query-based retrieval filtered to file
            res = col.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                results[col_name].append({
                    "text":       doc,
                    "discipline": meta.get("discipline", "?"),
                    "filename":   meta.get("filename", "?"),
                    "page":       meta.get("page", meta.get("chunk_index", "?")),
                    "score":      round((1 - dist) * 100, 1),
                })
        except Exception:
            # Fall back to fetching all chunks from that file
            try:
                res = col.get(where=where, include=["documents", "metadatas"])
                import random
                pairs = list(zip(res["documents"], res["metadatas"]))
                random.shuffle(pairs)
                for doc, meta in pairs[:top_k]:
                    results[col_name].append({
                        "text":       doc,
                        "discipline": meta.get("discipline", "?"),
                        "filename":   meta.get("filename", "?"),
                        "page":       meta.get("page", meta.get("chunk_index", "?")),
                        "score":      80.0,
                    })
            except Exception:
                pass
    return results

# ── QCM BANK ───────────────────────────────────────────────────────
QCM_BANK_PATH = r"C:\FMPC_Scribe\qcm_bank.json"

def _load_bank():
    try:
        if os.path.exists(QCM_BANK_PATH):
            with open(QCM_BANK_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_bank(bank):
    try:
        with open(QCM_BANK_PATH, "w", encoding="utf-8") as f:
            json.dump(bank, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[BANK] Write error: {e}")

def _all_bank_questions():
    """Flat list of all question texts already in the bank."""
    bank = _load_bank()
    return [q["question"].strip().lower() for session in bank for q in session.get("questions", [])]

@app.post("/api/qcm/generate")
async def generate_qcm(req: QCMRequest):
    disc_filter = req.discipline if req.discipline != "Toutes" else None
    query = req.topic.strip() or f"concept médical {req.discipline if req.discipline != 'Toutes' else 'médecine'}"

    results = retrieve(query, disc_filter, req.top_k)
    if not any(results[k] for k in results):
        raise HTTPException(404, "Aucun contenu trouvé. Ingérez des slides ou transcriptions.")

    context      = build_context(results)
    existing_qs  = _all_bank_questions()   # avoid repeating banked questions
    all_q        = []
    MAX_ATTEMPTS = 10   # safety cap
    attempt      = 0

    # ── Extract claims ONCE before the generation loop ──────────────
    claim_prompt = (
        "Tu es un assistant qui extrait des faits médicaux d'un texte de cours.\n"
        "Extrait UNIQUEMENT les faits, affirmations, et relations anatomiques/physiologiques "
        "présents EXPLICITEMENT dans le texte ci-dessous.\n"
        "NE PAS ajouter de connaissances extérieures. NE PAS inférer. NE PAS compléter.\n"
        "Format: une affirmation par ligne, formulée comme une phrase courte et précise.\n\n"
        f"TEXTE DU COURS ({req.discipline}):\n{context}\n\n"
        "AFFIRMATIONS EXTRAITES (uniquement ce qui est dans le texte):"
    )
    claims_raw = ollama(claim_prompt, timeout=120)
    claims = claims_raw.strip() if len(claims_raw.strip()) > 50 else context

    while len(all_q) < N_QCM and attempt < MAX_ATTEMPTS:
        attempt += 1
        needed    = N_QCM - len(all_q)
        batch_n   = min(4, needed)

        # ── Generate QCMs strictly from extracted claims ──
        avoid_current = [q["question"][:70] for q in all_q]
        avoid_banked  = [q[:70] for q in existing_qs[-20:]]
        avoid_all     = avoid_current + avoid_banked
        avoid_str     = ("\nÉvite ABSOLUMENT ces sujets déjà couverts:\n- " + "\n- ".join(avoid_all)) if avoid_all else ""

        prompt = (
            f"Tu es un professeur de médecine FMPC.{avoid_str}\n\n"
            "RÈGLE ABSOLUE: Chaque question, chaque choix, chaque explication "
            "DOIT être basé EXCLUSIVEMENT sur les affirmations du cours listées ci-dessous.\n"
            "Tu N'as PAS le droit d'utiliser tes connaissances générales. "
            "Si une information n'est pas dans la liste, ne l'utilise pas.\n\n"
            f"AFFIRMATIONS DU COURS ({req.discipline}):\n{claims}\n\n"
            f"Génère EXACTEMENT {batch_n} QCMs à 5 choix (A,B,C,D,E) basés sur ces affirmations.\n\n"
            "FORMAT JSON STRICT:\n"
            "- question: formulée à partir des affirmations\n"
            "- choices A-E: exactement 5, toutes non vides\n"
            "- correct: liste ex: [\"A\"] ou [\"A\",\"D\"]\n"
            "- explanation: citation EXACTE ou quasi-exacte de l'affirmation du cours qui justifie la réponse\n"
            "- distractors: pour chaque mauvaise réponse, pourquoi elle est fausse SELON LE COURS\n"
            "- source_claim: l'affirmation exacte du cours sur laquelle la question est basée\n\n"
            "Réponds UNIQUEMENT avec un tableau JSON valide, rien d'autre:\n"
            '[{"question":"...","choices":{"A":"...","B":"...","C":"...","D":"...","E":"..."},'
            '"correct":["A"],"explanation":"Citation du cours: ...","distractors":{"B":"..."},"source_claim":"..."}]'
        )

        raw = ollama(prompt, timeout=180)
        arr = extract_json_array(raw)
        if arr:
            try:
                batch = json.loads(arr)
                for q in batch:
                    nq = normalize_question(q)
                    if not nq:
                        continue
                    if nq["question"].strip().lower() in existing_qs:
                        continue
                    if any(nq["question"].strip().lower() == x["question"].strip().lower() for x in all_q):
                        continue
                    all_q.append(nq)
            except Exception:
                pass

    return {"questions": all_q[:N_QCM], "total": len(all_q)}

class SaveQCMRequest(BaseModel):
    discipline: str
    topic:      str
    questions:  list

@app.post("/api/qcm/save")
async def save_qcm(req: SaveQCMRequest):
    bank = _load_bank()
    import time as _time
    session = {
        "id":         str(int(_time.time() * 1000)),
        "discipline": req.discipline,
        "topic":      req.topic or "Général",
        "date":       __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
        "questions":  req.questions,
        "count":      len(req.questions),
    }
    bank.insert(0, session)  # newest first
    _save_bank(bank)
    return {"id": session["id"], "saved": len(req.questions)}

@app.get("/api/qcm/bank")
async def list_qcm_bank():
    bank = _load_bank()
    # Return metadata only (no questions) for listing
    return {"sessions": [
        {k: v for k, v in s.items() if k != "questions"}
        for s in bank
    ], "total_questions": sum(s.get("count", 0) for s in bank)}

@app.get("/api/qcm/bank/{session_id}")
async def load_qcm_session(session_id: str):
    bank = _load_bank()
    for s in bank:
        if s["id"] == session_id:
            return s
    raise HTTPException(404, "Session not found")

@app.delete("/api/qcm/bank/{session_id}")
async def delete_qcm_session(session_id: str):
    bank = _load_bank()
    bank = [s for s in bank if s["id"] != session_id]
    _save_bank(bank)
    return {"ok": True}

@app.get("/api/exams/catalog")
async def exam_catalog():
    ec = get_exam_collection()
    if ec is None or ec.count() == 0:
        return {"catalog": {}}
    try:
        results = ec.get(include=["metadatas"])
        catalog = {}
        for meta in results["metadatas"]:
            block = meta.get("block", "unknown")
            year  = meta.get("year",  "unknown")
            fname = meta.get("filename", "unknown")
            catalog.setdefault(block, {}).setdefault(year, set()).add(fname)
        return {"catalog": {b: {y: sorted(fs) for y, fs in yrs.items()} for b, yrs in catalog.items()}}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/exams/questions/{filename}")
async def exam_questions(filename: str):
    ec = get_exam_collection()
    if ec is None:
        raise HTTPException(404, "Exam collection not found")
    try:
        results = ec.get(where={"filename": filename}, include=["metadatas"])
        questions = []
        for meta in results["metadatas"]:
            questions.append({
                "q_num":       meta["q_num"],
                "question":    meta["question"],
                "choices":     {l: meta.get(l, "") for l in "ABCDE"},
                "correct":     json.loads(meta["correct"]),
                "explanation": meta.get("explanation", ""),
            })
        questions.sort(key=lambda x: x["q_num"])
        return {"questions": questions, "total": len(questions)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── EXPLANATION CACHE ──────────────────────────────────────────────
EXPLAIN_CACHE_PATH = r"C:\FMPC_Scribe\explain_cache.json"

def _load_cache():
    try:
        if os.path.exists(EXPLAIN_CACHE_PATH):
            with open(EXPLAIN_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_cache(cache):
    try:
        with open(EXPLAIN_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[CACHE] Write error: {e}")

def _cache_key(question, correct):
    import hashlib
    raw = question.strip().lower() + "|" + ",".join(sorted(correct))
    return hashlib.md5(raw.encode()).hexdigest()

class ExplainRequest(BaseModel):
    question:   str
    choices:    dict
    correct:    list
    discipline: str = "medecine"

@app.post("/api/exams/explain")
async def explain_question(req: ExplainRequest):
    """
    Generate professor-grounded correction for a question.
    Cached to disk after first generation — instant on repeat calls.
    """
    key   = _cache_key(req.question, req.correct)
    cache = _load_cache()

    if key in cache:
        cached = cache[key]
        cached["cached"] = True
        return cached

    # Retrieve professor material
    query   = req.question + " " + " ".join(req.choices.get(l, "") for l in req.correct)
    results = retrieve(query, req.discipline, top_k=5)
    if not any(results[k] for k in results):
        results = retrieve(req.question, None, top_k=4)

    context = build_context(results)

    wrong_letters   = [l for l in "ABCDE" if l not in req.correct and req.choices.get(l)]
    wrong_choices   = "\n".join(f"{l}. {req.choices[l]}" for l in wrong_letters)
    correct_choices = "\n".join(f"{l}. {req.choices[l]}" for l in req.correct)
    distractor_keys = ", ".join(f'"{l}": "..."' for l in wrong_letters)

    prompt = (
        "Tu es un professeur de médecine à la FMPC. "
        "Explique cette question en te basant UNIQUEMENT sur le cours fourni.\n\n"
        f"QUESTION: {req.question}\n\n"
        f"BONNES RÉPONSES:\n{correct_choices}\n\n"
        f"MAUVAISES RÉPONSES:\n{wrong_choices}\n\n"
        f"COURS DU PROFESSEUR:\n{context[:3000]}\n\n"
        "Réponds en JSON valide UNIQUEMENT, rien avant ni après:\n"
        + '{"explanation": "...", "distractors": {' + distractor_keys + "}}"
    )

    try:
        raw = ollama(prompt, timeout=120)
        s = raw.find("{")
        e = raw.rfind("}") + 1
        if s == -1 or e == 0:
            return {"explanation": "", "distractors": {}}
        data = json.loads(raw[s:e])
        data.setdefault("explanation", "")
        data.setdefault("distractors", {})
        cache[key] = data
        _save_cache(cache)
        return data
    except Exception:
        return {"explanation": "", "distractors": {}}


# ═══════════════════════════════════════════════════════════════════
# FEATURE: LECTURE GAP DETECTION
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/gaps/{discipline}")
async def detect_gaps(discipline: str):
    """
    Compare slide content vs transcript content for a discipline.
    Returns concepts found in slides but not transcribed, and vice versa.
    """
    cols = get_collections()
    slides_col      = cols.get("slides")
    transcripts_col = cols.get("transcripts")

    if not slides_col or not transcripts_col:
        raise HTTPException(404, "Collections not found")

    where = {"discipline": discipline} if discipline != "Toutes" else None

    def get_all_text(col, where):
        try:
            kwargs = {"include": ["documents", "metadatas"]}
            if where:
                kwargs["where"] = where
            res = col.get(**kwargs)
            return [
                {"text": doc, "filename": meta.get("filename","?"), "page": meta.get("page","?")}
                for doc, meta in zip(res["documents"], res["metadatas"])
            ]
        except Exception:
            return []

    slides_chunks      = get_all_text(slides_col, where)
    transcript_chunks  = get_all_text(transcripts_col, where)

    if not slides_chunks and not transcript_chunks:
        return {"gaps": [], "only_in_slides": [], "only_in_transcripts": [], "discipline": discipline}

    slides_text      = "\n".join(c["text"] for c in slides_chunks[:30])[:4000]
    transcript_text  = "\n".join(c["text"] for c in transcript_chunks[:30])[:4000]

    prompt = (
        f"Tu es un assistant pédagogique. Compare ces deux sources du cours de {discipline}.\n\n"
        "SOURCE 1 — SLIDES DU PROFESSEUR:\n" + slides_text + "\n\n"
        "SOURCE 2 — TRANSCRIPTION DU COURS:\n" + transcript_text + "\n\n"
        "Identifie:\n"
        "1. Concepts PRÉSENTS dans les slides mais NON mentionnés dans la transcription\n"
        "2. Concepts MENTIONNÉS dans la transcription mais ABSENTS des slides\n"
        "3. Points de CONTRADICTION entre les deux sources\n\n"
        "Réponds en JSON valide UNIQUEMENT:\n"
        '{"only_in_slides":["concept1","concept2"],'
        '"only_in_transcripts":["concept1","concept2"],'
        '"contradictions":["description de la contradiction"],'
        '"coverage_score": 85}'
    )

    try:
        raw = ollama(prompt, timeout=180)
        s = raw.find("{"); e = raw.rfind("}") + 1
        if s == -1 or e == 0:
            return {"error": "parse_failed"}
        data = json.loads(raw[s:e])
        data["discipline"]        = discipline
        data["slides_files"]      = list(set(c["filename"] for c in slides_chunks))
        data["transcript_files"]  = list(set(c["filename"] for c in transcript_chunks))
        return data
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ═══════════════════════════════════════════════════════════════════
# FEATURE: LECTURE TIMELINE SEARCH
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/timeline/search")
async def timeline_search(q: str, discipline: str = "Toutes", limit: int = 20):
    """
    Search for a keyword/concept across all transcripts.
    Returns matching chunks with filename, discipline, and page/timestamp metadata.
    """
    cols = get_collections()
    col  = cols.get("transcripts")
    if col is None:
        raise HTTPException(404, "Transcripts collection not found")

    where = {"discipline": discipline} if discipline != "Toutes" else None
    try:
        kwargs = {
            "query_texts": [q],
            "n_results":   min(limit, col.count() or 1),
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        res = col.query(**kwargs)

        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = round((1 - dist) * 100, 1)
            if score < 40:
                continue
            hits.append({
                "text":       doc,
                "discipline": meta.get("discipline", "?"),
                "filename":   meta.get("filename", "?"),
                "date":       meta.get("date", "?"),
                "chunk":      meta.get("chunk_index", meta.get("page", "?")),
                "score":      score,
            })

        hits.sort(key=lambda x: x["score"], reverse=True)
        return {"hits": hits, "total": len(hits), "query": q}
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ═══════════════════════════════════════════════════════════════════
# FEATURE: WEAK POINTS — ANSWER HISTORY TRACKING
# ═══════════════════════════════════════════════════════════════════

HISTORY_PATH = r"C:\FMPC_Scribe\answer_history.json"

def _load_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_history(h):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(h, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[HISTORY] Write error: {e}")

class AnswerRecord(BaseModel):
    question:    str
    correct:     list
    choices:     dict
    discipline:  str  = "general"
    was_correct: bool
    confident:   bool = True

@app.post("/api/history/record")
async def record_answer(rec: AnswerRecord):
    """Record an answer attempt for weak-point tracking."""
    import time as _t
    key     = _cache_key(rec.question, rec.correct)
    history = _load_history()

    if key not in history:
        history[key] = {
            "question":   rec.question,
            "choices":    rec.choices,
            "correct":    rec.correct,
            "discipline": rec.discipline,
            "attempts":   [],
        }

    history[key]["attempts"].append({
        "ts":          int(_t.time()),
        "was_correct": rec.was_correct,
        "confident":   rec.confident,
    })
    _save_history(history)
    return {"ok": True, "key": key}

@app.get("/api/history/weak")
async def get_weak_points(discipline: str = "Toutes", limit: int = 15):
    """
    Return questions ranked by weakness:
    - Wrong answers weighted higher
    - Correct-but-unconfident weighted medium
    - Recent attempts weighted higher
    """
    import time as _t
    history = _load_history()
    now     = _t.time()
    scored  = []

    for key, entry in history.items():
        if discipline != "Toutes" and entry.get("discipline") != discipline:
            continue
        attempts = entry.get("attempts", [])
        if not attempts:
            continue

        score = 0
        for a in attempts:
            age_days = (now - a.get("ts", now)) / 86400
            recency  = max(0.3, 1 - age_days / 30)  # decays over 30 days
            if not a.get("was_correct"):
                score += 3.0 * recency   # wrong → high priority
            elif not a.get("confident"):
                score += 1.5 * recency   # correct but unsure → medium
            else:
                score -= 0.5 * recency   # correct + confident → reduce priority

        if score > 0:
            scored.append({
                "key":        key,
                "question":   entry["question"],
                "choices":    entry["choices"],
                "correct":    entry["correct"],
                "discipline": entry["discipline"],
                "score":      round(score, 2),
                "attempts":   len(attempts),
                "last_wrong": next((a for a in reversed(attempts) if not a["was_correct"]), None) is not None,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"weak": scored[:limit], "total_tracked": len(history)}


# ═══════════════════════════════════════════════════════════════════
# FEATURE: PRE-EXAM MODE
# ═══════════════════════════════════════════════════════════════════

SCHEDULE_PATH = r"C:\FMPC_Scribe\schedule.json"

def _load_schedule():
    try:
        if os.path.exists(SCHEDULE_PATH):
            with open(SCHEDULE_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("schedule", [])
    except Exception:
        pass
    return []

@app.get("/api/preexam/upcoming")
async def upcoming_exams():
    """
    Scan exam_questions collection to find available exam blocks,
    and cross-reference with schedule to find upcoming exam dates.
    Returns ranked list of disciplines to study with urgency score.
    """
    from datetime import datetime, timedelta
    ec       = get_exam_collection()
    schedule = _load_schedule()
    today    = datetime.now().date()

    # Get all exam blocks available
    blocks = {}
    if ec and ec.count() > 0:
        res = ec.get(include=["metadatas"])
        for meta in res["metadatas"]:
            block = meta.get("block", "unknown")
            blocks.setdefault(block, 0)
            blocks[block] += 1

    # Find upcoming exam dates from schedule (entries with type=exam or keyword in discipline)
    # Since schedule may not have exam entries, we surface all disciplines with ingested exams
    # weighted by how recently they were last practiced in answer_history
    history    = _load_history()
    discipline_practice = {}
    for entry in history.values():
        d = entry.get("discipline", "")
        if d:
            attempts = entry.get("attempts", [])
            discipline_practice[d] = discipline_practice.get(d, 0) + len(attempts)

    results = []
    for block, q_count in blocks.items():
        disciplines = EXAM_BLOCK_MAP.get(block, [block])
        practiced   = sum(discipline_practice.get(d, 0) for d in disciplines)
        # Lower practice = higher urgency
        urgency = max(1, 10 - min(practiced // 5, 9))
        results.append({
            "block":       block,
            "disciplines": disciplines,
            "q_count":     q_count,
            "practiced":   practiced,
            "urgency":     urgency,
        })

    results.sort(key=lambda x: x["urgency"], reverse=True)
    return {"blocks": results, "today": str(today)}

class PreExamRequest(BaseModel):
    blocks:     list   # list of block names to include
    n:          int = 15

@app.post("/api/preexam/generate")
async def generate_preexam_session(req: PreExamRequest):
    """
    Generate a mixed QCM session weighted toward the requested exam blocks.
    Pulls from weak history first, then generates new questions from course material.
    """
    # Pull weak questions for these disciplines first
    all_disciplines = []
    for block in req.blocks:
        all_disciplines.extend(EXAM_BLOCK_MAP.get(block, [block]))

    history  = _load_history()
    weak_qs  = []
    for entry in history.values():
        if entry.get("discipline") in all_disciplines:
            attempts = entry.get("attempts", [])
            wrong    = sum(1 for a in attempts if not a.get("was_correct"))
            if wrong > 0:
                weak_qs.append({
                    "question":    entry["question"],
                    "choices":     entry["choices"],
                    "correct":     entry["correct"],
                    "discipline":  entry["discipline"],
                    "explanation": "",
                    "distractors": {},
                    "source_claim": "",
                    "from_weak":   True,
                })

    # Sort by most wrong, take up to half the session
    weak_qs = sorted(weak_qs, key=lambda x: x.get("score", 0), reverse=True)[:req.n // 2]

    # Fill the rest with newly generated questions
    needed = req.n - len(weak_qs)
    new_qs = []
    if needed > 0 and all_disciplines:
        disc = all_disciplines[0]
        results = retrieve(f"concept médical {disc}", disc, top_k=6)
        if any(results[k] for k in results):
            context = build_context(results)
            existing = _all_bank_questions()
            prompt = (
                f"Génère EXACTEMENT {needed} QCMs à 5 choix sur {', '.join(all_disciplines)}.\n"
                "Basé UNIQUEMENT sur le contexte. Système TOR. JSON valide uniquement.\n"
                '[{"question":"...","choices":{"A":"...","B":"...","C":"...","D":"...","E":"..."},'
                '"correct":["A"],"explanation":"...","distractors":{},"source_claim":""}]\n\n'
                f"CONTEXTE:\n{context}"
            )
            raw = ollama(prompt, timeout=180)
            arr = extract_json_array(raw)
            if arr:
                try:
                    batch = json.loads(arr)
                    for q in batch:
                        nq = normalize_question(q)
                        if nq:
                            nq["from_weak"] = False
                            new_qs.append(nq)
                except Exception:
                    pass

    import random
    all_qs = weak_qs + new_qs
    random.shuffle(all_qs)
    return {"questions": all_qs[:req.n], "weak_count": len(weak_qs), "new_count": len(new_qs)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
