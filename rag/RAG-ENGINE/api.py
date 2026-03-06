"""
api.py - FMPC Study Assistant Backend

Usage:
    py -3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000
"""

import os
import sys
import json
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
    q.setdefault("source", "")
    return q

# ── MODELS ─────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:      str
    discipline: str = "Toutes"
    top_k:      int = TOP_K

class QCMRequest(BaseModel):
    discipline: str   = "Toutes"
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

@app.post("/api/qcm/generate")
async def generate_qcm(req: QCMRequest):
    disc_filter = req.discipline if req.discipline != "Toutes" else None
    query = req.topic.strip() or f"concept médical {req.discipline if req.discipline != 'Toutes' else 'médecine'}"

    results = retrieve(query, disc_filter, req.top_k)
    if not any(results[k] for k in results):
        raise HTTPException(404, "Aucun contenu trouvé. Ingérez des slides ou transcriptions.")

    context = build_context(results)
    all_q   = []
    N_BATCH = 3

    for i in range(0, N_QCM, N_BATCH):
        avoid = "; ".join(q["question"][:60] for q in all_q[-6:])
        avoid_str = f"\nÉvite ces sujets: {avoid}" if avoid else ""

        prompt = (
            f"Tu es un professeur de médecine FMPC. "
            f"Génère {N_BATCH} QCMs DIFFÉRENTS à 5 choix (A,B,C,D,E).{avoid_str}\n\n"
            "RÈGLES:\n"
            "- Exactement 5 propositions (A à E)\n"
            "- Une ou plusieurs bonnes réponses (système TOR)\n"
            "- Chaque question teste un concept DIFFÉRENT\n"
            "- explanation: 1 phrase expliquant POURQUOI la/les bonne(s) réponse(s) sont correctes\n"
            "- distractors: objet avec UNE phrase par mauvaise réponse expliquant POURQUOI elle est fausse\n"
            "- correct: liste JSON ex: [\"A\"] ou [\"A\",\"D\"]\n"
            "- Réponds UNIQUEMENT avec un tableau JSON valide\n\n"
            '[{"question":"...","choices":{"A":"...","B":"...","C":"...","D":"...","E":"..."},'
            '"correct":["A"],"explanation":"Pourquoi A est correct...",'
            '"distractors":{"B":"Pourquoi B est faux...","C":"Pourquoi C est faux..."}}]\n\n'
            f"CONTEXTE ({req.discipline}):\n{context}"
        )

        raw = ollama(prompt)
        arr = extract_json_array(raw)
        if arr:
            try:
                batch = json.loads(arr)
                for q in batch:
                    nq = normalize_question(q)
                    if nq:
                        all_q.append(nq)
            except Exception:
                pass

        if len(all_q) >= N_QCM:
            break

    return {"questions": all_q[:N_QCM]}

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

class ExplainRequest(BaseModel):
    question:   str
    choices:    dict
    correct:    list
    discipline: str = "medecine"

@app.post("/api/exams/explain")
async def explain_question(req: ExplainRequest):
    """
    Given a question + correct answers, retrieve professor context and generate
    grounded explanations: why each correct answer is right, why each wrong answer is wrong.
    """
    # Build a search query from the question text
    query   = req.question + " " + " ".join(req.choices.get(l, "") for l in req.correct)
    results = retrieve(query, req.discipline, top_k=5)

    if not any(results[k] for k in results):
        # No professor material found — fall back to general knowledge
        results = retrieve(req.question, None, top_k=4)

    context = build_context(results)

    wrong_letters = [l for l in "ABCDE" if l not in req.correct and req.choices.get(l)]
    wrong_choices = "
".join(f"{l}. {req.choices[l]}" for l in wrong_letters)
    correct_choices = "
".join(f"{l}. {req.choices[l]}" for l in req.correct)

    prompt = (
        "Tu es un professeur de médecine à la FMPC. "
        "Explique cette question d'examen en te basant UNIQUEMENT sur le cours fourni.\n\n"
        "QUESTION: " + req.question + "\n\n"
        "BONNES RÉPONSES:\n" + correct_choices + "\n\n"
        "MAUVAISES RÉPONSES:\n" + wrong_choices + "\n\n"
        "COURS DU PROFESSEUR:\n" + context[:3000] + "\n\n"
        "Réponds en JSON valide UNIQUEMENT, sans rien avant ni après:\n"
        '{"explanation":"1-2 phrases expliquant POURQUOI les bonnes réponses sont correctes selon le cours",'
        '"distractors":{"' + (wrong_letters[0] if wrong_letters else "B") + '":"1 phrase pourquoi faux selon le cours"'
        + ((',"' + wrong_letters[1] + '":"..."') if len(wrong_letters) > 1 else '')
        + ((',"' + wrong_letters[2] + '":"..."') if len(wrong_letters) > 2 else '')
        + ((',"' + wrong_letters[3] + '":"..."') if len(wrong_letters) > 3 else '')
        + '}}'
    )

    try:
        raw = ollama(prompt, timeout=120)
        s = raw.find("{")
        e = raw.rfind("}") + 1
        if s == -1 or e == 0:
            return {"explanation": "", "distractors": {}, "source": "no_context"}
        data = json.loads(raw[s:e])
        data.setdefault("explanation", "")
        data.setdefault("distractors", {})
        return data
    except Exception:
        return {"explanation": "", "distractors": {}, "source": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
