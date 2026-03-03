import sys, os, json
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────
CHROMA_PATH = r"C:\FMPC_Scribe\chroma_db"
SLIDES_DIR  = r"C:\FMPC_Scribe\slides"  # optional default drop folder

# ─── IMPORTS ───────────────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    print("[INGEST] ERROR: pdfplumber not installed.")
    print("  Run: py -3.11 -m pip install pdfplumber")
    sys.exit(1)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("[INGEST] ERROR: chromadb not installed.")
    print("  Run: py -3.11 -m pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[INGEST] ERROR: sentence-transformers not installed.")
    print("  Run: py -3.11 -m pip install sentence-transformers")
    sys.exit(1)

# ─── EMBEDDING MODEL ───────────────────────────────────────────────
# Multilingual, handles French natively, runs on CPU
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding_function():
    print(f"[INGEST] Loading embedding model: {EMBED_MODEL_NAME}")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )

# ─── CHROMA CLIENT ─────────────────────────────────────────────────
def get_collection(name):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# ─── PDF EXTRACTION ────────────────────────────────────────────────
def extract_pages(pdf_path):
    """
    Extract text from each page of a PDF.
    Returns list of (page_number, text) tuples.
    Skips pages with less than 20 characters (likely image-only slides).
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"[INGEST] PDF has {total} pages.")
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                text = text.strip()
                if len(text) >= 20:
                    pages.append((i, text))
                else:
                    print(f"[INGEST]   Page {i}: skipped (too short — likely image only)")
            else:
                print(f"[INGEST]   Page {i}: skipped (no text extracted)")
    print(f"[INGEST] Extracted {len(pages)} text pages out of {total}.")
    return pages

# ─── INGEST ────────────────────────────────────────────────────────
def ingest_pdf(pdf_path, discipline):
    filename = os.path.basename(pdf_path)
    filename_noext = os.path.splitext(filename)[0]
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"[INGEST] ─── Starting ingestion ───")
    print(f"[INGEST] File:       {filename}")
    print(f"[INGEST] Discipline: {discipline}")
    print(f"[INGEST] Date:       {today}")

    # Extract pages
    pages = extract_pages(pdf_path)
    if not pages:
        print("[INGEST] ERROR: No text could be extracted from this PDF.")
        print("[INGEST] The PDF may be entirely image-based. Consider OCR.")
        sys.exit(1)

    # Connect to ChromaDB slides collection
    print(f"[INGEST] Connecting to ChromaDB at {CHROMA_PATH}...")
    collection = get_collection("slides")

    # Build documents, metadata, ids
    documents = []
    metadatas = []
    ids = []

    for page_num, text in pages:
        doc_id = f"{filename_noext}_p{page_num}"

        # Check if already ingested — skip duplicates
        existing = collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            print(f"[INGEST]   Page {page_num}: already in DB, skipping.")
            continue

        documents.append(text)
        metadatas.append({
            "source":     "slide",
            "discipline": discipline,
            "filename":   filename_noext,
            "date":       today,
            "page":       page_num,
        })
        ids.append(doc_id)

    if not documents:
        print("[INGEST] All pages already ingested. Nothing new to add.")
        return

    # Ingest in batches of 50
    BATCH_SIZE = 50
    total = len(documents)
    print(f"[INGEST] Ingesting {total} pages in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch_docs  = documents[i:i+BATCH_SIZE]
        batch_meta  = metadatas[i:i+BATCH_SIZE]
        batch_ids   = ids[i:i+BATCH_SIZE]
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        end = min(i + BATCH_SIZE, total)
        print(f"[INGEST]   Ingested pages {i+1}–{end} / {total}")

    print(f"[INGEST] ─── DONE: {total} pages ingested into 'slides' collection ───")
    print(f"[INGEST] Collection now has {collection.count()} total entries.")

# ─── TRANSCRIPT INGEST (called from processor.py) ──────────────────
def ingest_transcript(txt_path, discipline, label, professor=None):
    """
    Ingest a transcript .txt file into the transcripts collection.
    Called automatically by processor.py after saving the transcript.
    Chunks by paragraph, skips the header block.
    """
    print(f"[INGEST] Ingesting transcript: {os.path.basename(txt_path)}")

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Skip the header (everything before the === line)
    if "=" * 10 in raw:
        raw = raw.split("=" * 10, 1)[-1].strip()

    paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) >= 30]

    if not paragraphs:
        print("[INGEST] No usable paragraphs found in transcript.")
        return

    collection = get_collection("transcripts")
    today = datetime.now().strftime("%Y-%m-%d")
    filename_noext = os.path.splitext(os.path.basename(txt_path))[0]

    documents = []
    metadatas = []
    ids = []

    for i, para in enumerate(paragraphs):
        doc_id = f"{filename_noext}_chunk{i}"

        existing = collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            continue

        documents.append(para)
        metadatas.append({
            "source":      "transcript",
            "discipline":  discipline,
            "filename":    filename_noext,
            "label":       label,
            "professor":   professor or "unknown",
            "date":        today,
            "chunk_index": i,
        })
        ids.append(doc_id)

    if not documents:
        print("[INGEST] Transcript already fully ingested.")
        return

    BATCH_SIZE = 50
    total = len(documents)
    for i in range(0, total, BATCH_SIZE):
        collection.add(
            documents=documents[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE]
        )
    print(f"[INGEST] Transcript ingested: {total} chunks → 'transcripts' collection")
    print(f"[INGEST] Collection now has {collection.count()} total entries.")

# ─── CLI ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.11 ingest_slides.py <path_to_pdf> [discipline]")
        print("Example: py -3.11 ingest_slides.py slides\\bacterio_morphologie.pdf bacteriologie")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"[INGEST] ERROR: File not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print(f"[INGEST] ERROR: File must be a PDF.")
        sys.exit(1)

    # Discipline: from argument or inferred from filename
    if len(sys.argv) >= 3:
        discipline = sys.argv[2].lower().strip()
    else:
        # Try to infer from filename
        fname = os.path.splitext(os.path.basename(pdf_path))[0].lower()
        known = ["anatomie", "bacteriologie", "histologie", "hematologie",
                 "embryologie", "immunologie", "physiologie", "biochimie"]
        discipline = next((d for d in known if d in fname), "medecine")
        print(f"[INGEST] No discipline specified — inferred from filename: {discipline}")

    ingest_pdf(pdf_path, discipline)
