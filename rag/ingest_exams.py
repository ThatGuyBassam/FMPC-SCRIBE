"""
ingest_exams.py - Parse and ingest past exam PDFs as structured questions

Usage:
    py -3.11 ingest_exams.py                         -> watch mode
    py -3.11 ingest_exams.py list                    -> show all ingested exams
    py -3.11 ingest_exams.py delete "filename"       -> remove one exam
    py -3.11 ingest_exams.py file.pdf [block] [year] -> manual ingest

Folder structure:
    C:\\FMPC_Scribe\\exams\\microbiologie\\examen_2025.pdf
    C:\\FMPC_Scribe\\exams\\anatomie\\examen_2024.pdf

Block folder names:
    microbiologie | anatomie | histologie-embryologie | hematologie-immunologie | physiologie
"""

import sys
import os
import re
import json
import time
import requests
from datetime import datetime
from collections import Counter

# ── CONFIG ─────────────────────────────────────────────────────────
CHROMA_PATH  = r"C:\FMPC_Scribe\chroma_db"
EXAMS_DIR    = r"C:\FMPC_Scribe\exams"
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
QUESTIONS_PER_CHUNK = 7   # how many questions to send to Qwen per call

EXAM_BLOCK_MAP = {
    "microbiologie":             ["bacteriologie", "virologie"],
    "anatomie":                  ["anatomie"],
    "histologie-embryologie":    ["histologie", "embryologie"],
    "hematologie-immunologie":   ["hematologie", "immunologie"],
    "physiologie":               ["physiologie"],
}

# ── DEPENDENCY CHECKS ──────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    print("[EXAMS] ERROR: pdfplumber not installed.")
    print("  Run: py -3.11 -m pip install pdfplumber")
    sys.exit(1)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("[EXAMS] ERROR: chromadb not installed.")
    sys.exit(1)

# ── CHROMA ─────────────────────────────────────────────────────────
def get_collection():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name="exam_questions",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

# ── INFERENCE ──────────────────────────────────────────────────────
def infer_block(filepath):
    parent = os.path.basename(os.path.dirname(filepath)).lower()
    return parent if parent in EXAM_BLOCK_MAP else "microbiologie"

def infer_year(filepath):
    match = re.search(r"(20\d{2})", os.path.basename(filepath))
    return match.group(1) if match else "unknown"

# ── OCR ────────────────────────────────────────────────────────────
def ocr_page(pil_image):
    try:
        import pytesseract
        return pytesseract.image_to_string(pil_image, lang="fra").strip()
    except ImportError:
        return ""
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return ""

# ── PDF EXTRACTION ─────────────────────────────────────────────────
def extract_pdf(path):
    try:
        import fitz
        has_fitz = True
    except ImportError:
        has_fitz = False

    try:
        import pytesseract
        has_ocr = True
    except ImportError:
        has_ocr = False

    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        print(f"[EXAMS] PDF: {total} pages")

        raw_pages = []
        for i, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            raw_pages.append((i, text))

        text_found = sum(1 for _, t in raw_pages if len(t) >= 20)

        if text_found == 0:
            print("[EXAMS] Scanned PDF — switching to OCR")
            if not has_fitz or not has_ocr:
                print("[EXAMS] ERROR: Install: py -3.11 -m pip install pymupdf pytesseract pillow")
                print("[EXAMS]   + Tesseract with French: https://github.com/UB-Mannheim/tesseract/wiki")
                return []

            from PIL import Image
            import io
            doc = fitz.open(path)
            for i, page_obj in enumerate(doc, 1):
                mat = fitz.Matrix(300 / 72, 300 / 72)
                pix = page_obj.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                print(f"[EXAMS]   OCR page {i}/{total}...", end=" ", flush=True)
                text = ocr_page(img)
                if len(text) >= 20:
                    pages.append((i, text))
                    print(f"ok ({len(text)} chars)")
                else:
                    print("skipped")
        else:
            for i, text in raw_pages:
                if len(text) >= 20:
                    pages.append((i, text))

    print(f"[EXAMS] {len(pages)} usable pages extracted")
    return pages

# ── QUESTION BOUNDARY SPLITTER ─────────────────────────────────────
MAX_CHUNK_CHARS   = 2200   # hard cap per chunk sent to Qwen
QUESTIONS_PER_CHUNK = 5   # target questions per chunk

def split_by_question_boundaries(full_text):
    """
    Split exam text at question boundaries, then enforce MAX_CHUNK_CHARS hard cap.
    Detects: '1-', '2.', '10)', '1 -', '1 .' and also '1/' common in OCR.
    """
    boundaries = [m.start() for m in re.finditer(
        r"(?m)^\s*\d{1,2}\s*[\-\.\)\/]\s", full_text
    )]

    if len(boundaries) < 3:
        # Boundary detection failed — split purely by char count
        print("[PARSE] Boundary detection failed — splitting by character count")
        boundaries = []

    raw_chunks = []
    if boundaries:
        print(f"[PARSE] Detected {len(boundaries)} question boundaries")
        for i in range(0, len(boundaries), QUESTIONS_PER_CHUNK):
            start = boundaries[i]
            end   = boundaries[i + QUESTIONS_PER_CHUNK] if (i + QUESTIONS_PER_CHUNK) < len(boundaries) else len(full_text)
            raw_chunks.append(full_text[start:end])
    else:
        # Character-based fallback
        pos = 0
        while pos < len(full_text):
            end = min(pos + MAX_CHUNK_CHARS, len(full_text))
            nl  = full_text.rfind("\n", pos, end)
            if nl > pos + 200:
                end = nl
            raw_chunks.append(full_text[pos:end])
            pos = end

    # Hard cap: split any chunk still over MAX_CHUNK_CHARS at a newline
    final = []
    for chunk in raw_chunks:
        if len(chunk) <= MAX_CHUNK_CHARS:
            final.append(chunk)
        else:
            pos = 0
            while pos < len(chunk):
                end = min(pos + MAX_CHUNK_CHARS, len(chunk))
                nl  = chunk.rfind("\n", pos, end)
                if nl > pos + 200:
                    end = nl
                final.append(chunk[pos:end])
                pos = end

    return final

# ── LLM PARSER ─────────────────────────────────────────────────────
def parse_chunk(chunk, block, year, idx, total):
    print(f"[PARSE] Chunk {idx+1}/{total} ({len(chunk)} chars) ... ", end="", flush=True)

    prompt = (
        "Tu es un assistant qui structure des QCMs d'examens medicaux FMPC.\n"
        "Extrait TOUTES les questions QCM de ce texte. Ne saute aucune question.\n\n"
        "REGLES:\n"
        "- Exactement 5 propositions par question (A, B, C, D, E)\n"
        "- Identifie les bonnes reponses avec tes connaissances medicales\n"
        "- explanation: 1 phrase courte en francais\n"
        "- q_num: numero exact de la question dans l'examen\n"
        "- correct: liste ex: [\"A\"] ou [\"A\",\"C\"]\n"
        "- Reponds UNIQUEMENT avec un tableau JSON valide, rien d'autre\n\n"
        "[{\"q_num\":1,\"question\":\"...\","
        "\"choices\":{\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\",\"E\":\"...\"},"
        "\"correct\":[\"A\"],\"explanation\":\"...\"}]\n\n"
        f"TEXTE ({block}, {year}):\n{chunk}"
    )

    for attempt in range(2):  # retry once on JSON error
        try:
            res = requests.post(OLLAMA_URL, json={
                "model":       OLLAMA_MODEL,
                "prompt":      prompt,
                "stream":      False,
                "num_predict": 4096,   # prevent truncated JSON
            }, timeout=360)
            raw   = res.json().get("response", "").strip()
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end == 0:
                print("no JSON found")
                return []
            parsed = json.loads(raw[start:end])
            print(f"{len(parsed)} questions")
            return parsed
        except json.JSONDecodeError as e:
            if attempt == 0:
                print(f"JSON error (retrying)...", end=" ", flush=True)
            else:
                print(f"JSON error: {e}")
                return []
        except requests.exceptions.Timeout:
            print("timeout — chunk may be too large, skipping")
            return []
        except Exception as e:
            print(f"error: {e}")
            return []
    return []

def parse_questions(full_text, block, year):
    chunks = split_by_question_boundaries(full_text)
    print(f"[PARSE] {len(chunks)} chunks to process for this exam")

    raw_questions = []
    for i, chunk in enumerate(chunks):
        parsed = parse_chunk(chunk, block, year, i, len(chunks))
        for q in parsed:
            if not all(k in q for k in ["q_num", "question", "choices", "correct"]):
                continue
            # Ensure all 5 choices exist
            for letter in "ABCDE":
                q["choices"].setdefault(letter, "")
            # Normalize correct answers
            if isinstance(q["correct"], str):
                q["correct"] = sorted(q["correct"].replace(" ", "").upper().split(","))
            else:
                q["correct"] = sorted([str(c).upper().strip() for c in q["correct"]])
            # Filter invalid letters
            q["correct"] = [c for c in q["correct"] if c in "ABCDE"]
            if not q["correct"]:
                q["correct"] = ["A"]
            q.setdefault("explanation", "")
            raw_questions.append(q)

    # Deduplicate by q_num
    seen, unique = set(), []
    for q in raw_questions:
        num = int(q["q_num"]) if str(q["q_num"]).isdigit() else q["q_num"]
        if num not in seen:
            seen.add(num)
            q["q_num"] = int(num) if str(num).isdigit() else num
            unique.append(q)

    unique.sort(key=lambda x: x["q_num"] if isinstance(x["q_num"], int) else 0)
    print(f"[PARSE] Total: {len(unique)} unique questions")
    return unique

# ── INGEST ─────────────────────────────────────────────────────────
def ingest_exam(path, block, year):
    filename   = os.path.basename(path)
    name_noext = os.path.splitext(filename)[0]
    discipline = EXAM_BLOCK_MAP.get(block, ["medecine"])[0]
    today      = datetime.now().strftime("%Y-%m-%d")

    print(f"\n[EXAMS] ─── Ingestion ───────────────────────────────")
    print(f"[EXAMS] File:  {filename}")
    print(f"[EXAMS] Block: {block}  |  Year: {year}")

    pages = extract_pdf(path)
    if not pages:
        print("[EXAMS] ERROR: No text extracted.")
        return

    full_text = "\n\n".join(text for _, text in pages)
    questions = parse_questions(full_text, block, year)

    if not questions:
        print("[EXAMS] ERROR: No questions parsed. Is Ollama running?")
        return

    col = get_collection()
    documents, metadatas, ids = [], [], []

    for q in questions:
        q_id = f"{name_noext}_q{q['q_num']}"
        try:
            existing = col.get(ids=[q_id])
            if existing and existing["ids"]:
                continue
        except Exception:
            pass

        doc_text = (
            f"Q{q['q_num']}. {q['question']}\n"
            + "\n".join(f"{l}. {q['choices'].get(l, '')}" for l in "ABCDE")
        )
        documents.append(doc_text)
        metadatas.append({
            "block":       block,
            "discipline":  discipline,
            "filename":    name_noext,
            "year":        year,
            "date":        today,
            "q_num":       int(q["q_num"]) if isinstance(q["q_num"], int) else 0,
            "question":    q["question"][:1000],
            "A":           q["choices"].get("A", "")[:500],
            "B":           q["choices"].get("B", "")[:500],
            "C":           q["choices"].get("C", "")[:500],
            "D":           q["choices"].get("D", "")[:500],
            "E":           q["choices"].get("E", "")[:500],
            "correct":     json.dumps(q["correct"]),
            "explanation": q.get("explanation", "")[:500],
        })
        ids.append(q_id)

    if not documents:
        print("[EXAMS] All questions already ingested.")
        return

    BATCH = 50
    for i in range(0, len(documents), BATCH):
        col.add(
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
            ids=ids[i:i + BATCH]
        )
        print(f"[EXAMS]   {min(i + BATCH, len(documents))}/{len(documents)} stored")

    print(f"[EXAMS] Done: {len(documents)} questions. DB total: {col.count()}")

# ── CLI ─────────────────────────────────────────────────────────────
def cmd_list():
    try:
        col     = get_collection()
        results = col.get(include=["metadatas"])
        if not results["ids"]:
            print("[EXAMS] No exams ingested yet.")
            return
        exams = {}
        for meta in results["metadatas"]:
            key = meta.get("filename", "?")
            exams.setdefault(key, {"block": meta.get("block", "?"),
                                   "year":  meta.get("year", "?"),
                                   "count": 0})
            exams[key]["count"] += 1
        print(f"\n[EXAMS] {col.count()} questions across {len(exams)} exams:\n")
        for fname, info in sorted(exams.items()):
            print(f"  {info['block']:28} {info['year']:8} {fname}  ({info['count']} questions)")
    except Exception as e:
        print(f"[EXAMS] Error: {e}")

def cmd_delete(target):
    try:
        col     = get_collection()
        results = col.get(include=["metadatas"])
        to_del  = [rid for rid, meta in zip(results["ids"], results["metadatas"])
                   if meta.get("filename") == target]
        if to_del:
            col.delete(ids=to_del)
            print(f"[DELETE] Removed {len(to_del)} questions for: {target}")
        else:
            print(f"[DELETE] Not found: {target}")
            print("[DELETE] Check filenames with: py -3.11 ingest_exams.py list")
    except Exception as e:
        print(f"[DELETE] Error: {e}")

# ── WATCHER ────────────────────────────────────────────────────────
def cmd_watch():
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("[WATCHER] ERROR: py -3.11 -m pip install watchdog")
        sys.exit(1)

    os.makedirs(EXAMS_DIR, exist_ok=True)
    recently = set()

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory or not event.src_path.lower().endswith(".pdf"):
                return
            if event.src_path in recently:
                return
            time.sleep(2)
            recently.add(event.src_path)
            block = infer_block(event.src_path)
            year  = infer_year(event.src_path)
            print(f"\n[WATCHER] New: {os.path.basename(event.src_path)} ({block}, {year})")
            try:
                ingest_exam(event.src_path, block, year)
            except Exception as e:
                print(f"[WATCHER] ERROR: {e}")

    print(f"[WATCHER] Watching: {EXAMS_DIR}")
    print(f"[WATCHER] Folders: {', '.join(EXAM_BLOCK_MAP.keys())}")
    print(f"[WATCHER] Ctrl+C to stop.\n")

    col = get_collection()
    for root, dirs, files in os.walk(EXAMS_DIR):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            fpath      = os.path.join(root, fname)
            name_noext = os.path.splitext(fname)[0]
            try:
                existing = col.get(ids=[f"{name_noext}_q1"])
                if existing and existing["ids"]:
                    print(f"[WATCHER] Already ingested: {fname}")
                    continue
            except Exception:
                pass
            block = infer_block(fpath)
            year  = infer_year(fpath)
            print(f"[WATCHER] Ingesting: {fname} ({block}, {year})")
            try:
                ingest_exam(fpath, block, year)
            except Exception as e:
                print(f"[WATCHER] ERROR: {e}")

    observer = Observer()
    observer.schedule(Handler(), path=EXAMS_DIR, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[WATCHER] Stopped.")
    observer.join()

# ── ENTRY POINT ────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "watch":
        cmd_watch()

    elif sys.argv[1] == "list":
        cmd_list()

    elif sys.argv[1] == "delete":
        if len(sys.argv) < 3:
            print('Usage: py -3.11 ingest_exams.py delete "filename_without_extension"')
        else:
            cmd_delete(sys.argv[2])

    else:
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"[EXAMS] File not found: {path}")
            sys.exit(1)
        block = sys.argv[2].lower() if len(sys.argv) >= 3 else infer_block(path)
        year  = sys.argv[3]         if len(sys.argv) >= 4 else infer_year(path)
        ingest_exam(path, block, year)
