"""
anki_generator.py - Generate Anki flashcards from ingested slides

Requirements:
    - Anki must be open
    - AnkiConnect addon installed in Anki (code: 2055492159)

Usage:
    py -3.11 anki_generator.py bacteriologie
    py -3.11 anki_generator.py anatomie
    py -3.11 anki_generator.py all
"""

import sys
import os
import json
import time
import requests

# ── CONFIG ─────────────────────────────────────────────────────────
CHROMA_PATH   = r"C:\FMPC_Scribe\chroma_db"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "qwen2.5:7b"
EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
ANKI_URL      = "http://localhost:8765"
ANKI_DECK_ROOT = "FMPC"
CHUNK_SIZE    = 5  # slides per Anki generation batch

KNOWN_DISCIPLINES = [
    "anatomie", "bacteriologie", "histologie", "hematologie",
    "embryologie", "immunologie", "physiologie", "biochimie", "virologie"
]

# ── ANKI CONNECT ───────────────────────────────────────────────────
def anki_request(action, **params):
    payload = {"action": action, "version": 6, "params": params}
    try:
        res = requests.post(ANKI_URL, json=payload, timeout=10)
        data = res.json()
        if data.get("error"):
            raise Exception(data["error"])
        return data.get("result")
    except requests.exceptions.ConnectionError:
        print("[ANKI] ERROR: Cannot connect to Anki.")
        print("[ANKI] Make sure Anki is open and AnkiConnect is installed.")
        print("[ANKI] Install AnkiConnect: Tools > Add-ons > Get Add-ons > code: 2055492159")
        sys.exit(1)

def ensure_deck(deck_name):
    anki_request("createDeck", deck=deck_name)

def card_exists(front, deck):
    """Check if a card with this front already exists in the deck."""
    try:
        note_ids = anki_request("findNotes", query=f'deck:"{deck}" front:"{front}"')
        return len(note_ids) > 0
    except Exception:
        return False

def add_card(deck, front, back, tags):
    note = {
        "deckName": deck,
        "modelName": "Basic",
        "fields": {
            "Front": front,
            "Back": back
        },
        "tags": tags,
        "options": {
            "allowDuplicate": False,
            "duplicateScope": "deck"
        }
    }
    try:
        result = anki_request("addNote", note=note)
        return result is not None
    except Exception as e:
        if "duplicate" in str(e).lower():
            return False
        raise

# ── CHROMA ─────────────────────────────────────────────────────────
def get_slides_for_discipline(discipline):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("[ANKI] ERROR: chromadb not installed.")
        sys.exit(1)

    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        col = client.get_collection(name="slides", embedding_function=ef)
    except Exception:
        print("[ANKI] ERROR: No slides collection found. Ingest slides first.")
        sys.exit(1)

    where = {"discipline": discipline} if discipline != "all" else None
    kwargs = {"include": ["documents", "metadatas"], "where": where} if where else {"include": ["documents", "metadatas"]}

    # Get all matching documents
    results = col.get(**kwargs)

    if not results["ids"]:
        print(f"[ANKI] No slides found for discipline: {discipline}")
        sys.exit(1)

    # Group by filename
    files = {}
    for doc, meta in zip(results["documents"], results["metadatas"]):
        fname = meta.get("filename", "unknown")
        disc  = meta.get("discipline", discipline)
        if fname not in files:
            files[fname] = {"discipline": disc, "slides": []}
        files[fname]["slides"].append(doc)

    print(f"[ANKI] Found {len(results['ids'])} slides across {len(files)} files.")
    return files

# ── CARD GENERATION ────────────────────────────────────────────────
def generate_cards_from_chunk(slides_text, discipline, filename):
    prompt = (
        "Tu es un créateur expert de flashcards Anki pour les étudiants en médecine à la FMPC. "
        "À partir du contenu de cours suivant, génère des flashcards de haute qualité.\n\n"
        "RÈGLES STRICTES:\n"
        "- Génère entre 3 et 8 flashcards par extrait selon la densité du contenu\n"
        "- Chaque flashcard doit porter sur UN SEUL concept précis\n"
        "- Le recto (front): une question claire et précise, ou un terme à définir\n"
        "- Le verso (back): réponse concise, max 3 lignes, faits uniquement\n"
        "- Priorité aux: définitions, rôles, compositions, mécanismes, classifications\n"
        "- Ignore les titres de slides seuls, les numéros de page, les métadonnées\n"
        "- Ne génère PAS de carte si le contenu est trop vague ou incomplet\n"
        "- Réponds UNIQUEMENT en JSON valide, aucun texte avant ou après\n\n"
        "FORMAT JSON EXACT:\n"
        "[\n"
        "  {\"front\": \"question ou terme\", \"back\": \"réponse concise\"},\n"
        "  {\"front\": \"...\", \"back\": \"...\"}\n"
        "]\n\n"
        f"CONTENU ({discipline} - {filename}):\n{slides_text}"
    )

    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=180)
        raw = res.json().get("response", "").strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        cards = json.loads(raw)
        if not isinstance(cards, list):
            return []
        # Validate structure
        valid = []
        for c in cards:
            if isinstance(c, dict) and "front" in c and "back" in c:
                if len(c["front"].strip()) > 3 and len(c["back"].strip()) > 3:
                    valid.append(c)
        return valid

    except json.JSONDecodeError as e:
        print(f"[ANKI]   WARNING: JSON parse failed: {e}")
        return []
    except Exception as e:
        print(f"[ANKI]   WARNING: Generation failed: {e}")
        return []

# ── MAIN LOGIC ─────────────────────────────────────────────────────
def process_discipline(discipline, files):
    total_added   = 0
    total_skipped = 0
    total_cards   = 0

    for filename, data in files.items():
        disc   = data["discipline"]
        slides = data["slides"]
        deck   = f"{ANKI_DECK_ROOT}::{disc}::{filename[:40]}"

        print(f"\n[ANKI] Processing: {filename}")
        print(f"[ANKI] Deck: {deck}")
        print(f"[ANKI] Slides: {len(slides)}")

        ensure_deck(deck)

        # Process in batches of CHUNK_SIZE slides
        all_cards = []
        for i in range(0, len(slides), CHUNK_SIZE):
            batch      = slides[i:i+CHUNK_SIZE]
            batch_text = "\n\n---\n\n".join(batch)
            batch_num  = i // CHUNK_SIZE + 1
            total_batches = (len(slides) + CHUNK_SIZE - 1) // CHUNK_SIZE

            print(f"[ANKI]   Generating cards batch {batch_num}/{total_batches}...")
            cards = generate_cards_from_chunk(batch_text, disc, filename)
            print(f"[ANKI]   Generated {len(cards)} cards from batch {batch_num}")
            all_cards.extend(cards)
            time.sleep(0.5)  # brief pause between Ollama calls

        total_cards += len(all_cards)
        print(f"[ANKI] Total cards generated for {filename}: {len(all_cards)}")
        print(f"[ANKI] Adding to Anki...")

        tags = [disc, filename[:30].replace(" ", "_")]

        for card in all_cards:
            try:
                added = add_card(deck, card["front"], card["back"], tags)
                if added:
                    total_added += 1
                else:
                    total_skipped += 1
            except Exception as e:
                print(f"[ANKI]   ERROR adding card: {e}")
                total_skipped += 1

        print(f"[ANKI] {filename}: {total_added} added, {total_skipped} skipped (duplicates)")

    return total_cards, total_added, total_skipped

def main():
    if len(sys.argv) < 2:
        print("Usage: py -3.11 anki_generator.py <discipline|all>")
        print("Examples:")
        print("  py -3.11 anki_generator.py bacteriologie")
        print("  py -3.11 anki_generator.py all")
        sys.exit(1)

    discipline = sys.argv[1].lower().strip()

    if discipline != "all" and discipline not in KNOWN_DISCIPLINES:
        print(f"[ANKI] Unknown discipline: {discipline}")
        print(f"[ANKI] Known: {', '.join(KNOWN_DISCIPLINES)} or 'all'")
        sys.exit(1)

    # Check Anki is running
    print("[ANKI] Checking Anki connection...")
    try:
        version = anki_request("version")
        print(f"[ANKI] Connected to AnkiConnect v{version}")
    except SystemExit:
        sys.exit(1)

    print(f"\n[ANKI] === Generating cards for: {discipline} ===\n")

    files = get_slides_for_discipline(discipline)
    total_cards, total_added, total_skipped = process_discipline(discipline, files)

    print(f"\n[ANKI] === COMPLETE ===")
    print(f"[ANKI] Cards generated : {total_cards}")
    print(f"[ANKI] Cards added     : {total_added}")
    print(f"[ANKI] Cards skipped   : {total_skipped} (already existed)")
    print(f"[ANKI] Open Anki to review your new cards.")

if __name__ == "__main__":
    main()
