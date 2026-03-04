"""
rag_engine.py - FMPC RAG Study Assistant

Usage:
    py -3.11 -m streamlit run rag_engine.py
"""

import os
import sys
import requests
import streamlit as st

# ── CONFIG ─────────────────────────────────────────────────────────
CHROMA_PATH  = r"C:\FMPC_Scribe\chroma_db"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 4  # chunks retrieved per collection

KNOWN_DISCIPLINES = [
    "Toutes", "anatomie", "bacteriologie", "histologie",
    "hematologie", "embryologie", "immunologie", "physiologie", "biochimie"
]

# ── CHROMA SETUP ───────────────────────────────────────────────────
@st.cache_resource
def load_collections():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        st.error("chromadb not installed. Run: py -3.11 -m pip install chromadb")
        sys.exit(1)

    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collections = {}
    for name in ["slides", "transcripts"]:
        try:
            collections[name] = client.get_collection(name=name, embedding_function=ef)
        except Exception:
            collections[name] = None

    return collections

# ── RETRIEVAL ──────────────────────────────────────────────────────
def retrieve(collections, query, discipline_filter, top_k=TOP_K):
    where = None
    if discipline_filter and discipline_filter != "Toutes":
        where = {"discipline": discipline_filter}

    results = {"slides": [], "transcripts": []}

    for col_name, col in collections.items():
        if col is None:
            continue
        try:
            kwargs = {
                "query_texts": [query],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            if where:
                kwargs["where"] = where

            res = col.query(**kwargs)

            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0]
            ):
                # Convert cosine distance to similarity score (0-100)
                score = round((1 - dist) * 100, 1)
                results[col_name].append({
                    "text":       doc,
                    "discipline": meta.get("discipline", "?"),
                    "filename":   meta.get("filename", "?"),
                    "page":       meta.get("page", meta.get("chunk_index", "?")),
                    "professor":  meta.get("professor", ""),
                    "source":     col_name,
                    "score":      score,
                })
        except Exception as e:
            st.warning(f"Retrieval error ({col_name}): {e}")

    return results

# ── GENERATION ─────────────────────────────────────────────────────
def build_context(results):
    parts = []

    if results["transcripts"]:
        parts.append("=== TRANSCRIPTIONS DE COURS ===")
        for i, r in enumerate(results["transcripts"], 1):
            parts.append(
                f"[Transcription {i} | {r['discipline']} | {r['filename']} | "
                f"chunk {r['page']} | score {r['score']}%]\n{r['text']}"
            )

    if results["slides"]:
        parts.append("\n=== SLIDES DU PROFESSEUR ===")
        for i, r in enumerate(results["slides"], 1):
            parts.append(
                f"[Slide {i} | {r['discipline']} | {r['filename']} | "
                f"page {r['page']} | score {r['score']}%]\n{r['text']}"
            )

    return "\n\n".join(parts)

def generate_answer(query, context):
    prompt = (
        "Tu es un assistant pédagogique pour les étudiants en 1ère année de médecine à la FMPC. "
        "Réponds à la question en te basant UNIQUEMENT sur le contexte fourni ci-dessous. "
        "Si le contexte ne contient pas assez d'informations pour répondre, dis-le clairement. "
        "Réponds en français. Sois précis et concis.\n\n"
        "RÈGLES:\n"
        "- Base ta réponse sur les transcriptions en priorité (paroles du professeur)\n"
        "- Complète avec les slides si nécessaire\n"
        "- Si transcription et slides se contredisent, signale-le explicitement\n"
        "- Ne fabrique pas d'information absente du contexte\n\n"
        f"CONTEXTE:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "RÉPONSE:"
    )

    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=120)
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"Erreur de génération: {e}"

def check_contradiction(results):
    """Flag if transcript and slide both returned results on the same topic."""
    if not results["transcripts"] or not results["slides"]:
        return False
    # Simple heuristic: if both have high scores (>60%), flag for review
    top_transcript = results["transcripts"][0]["score"] if results["transcripts"] else 0
    top_slide      = results["slides"][0]["score"] if results["slides"] else 0
    return top_transcript > 60 and top_slide > 60

# ── STREAMLIT UI ───────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FMPC Assistant",
        page_icon="🩺",
        layout="wide"
    )

    # Header
    st.title("🩺 FMPC Study Assistant")
    st.caption("Posez une question — réponse basée sur vos cours et slides")

    # Load collections
    collections = load_collections()

    # Sidebar — filters and stats
    with st.sidebar:
        st.header("Filtres")
        discipline = st.selectbox("Matière", KNOWN_DISCIPLINES)
        top_k = st.slider("Chunks récupérés", min_value=2, max_value=8, value=TOP_K)
        show_sources = st.toggle("Afficher les sources", value=True)

        st.divider()
        st.header("Base de données")
        for name, col in collections.items():
            if col:
                count = col.count()
                st.metric(name, f"{count} chunks")
            else:
                st.metric(name, "vide")

        st.divider()
        st.caption("Ollama: qwen2.5:7b")
        st.caption("Embed: multilingual-MiniLM")

    # Main query area
    query = st.text_input(
        "Votre question",
        placeholder="Ex: Quel est le rôle de la capsule bactérienne ?",
        key="query_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("Rechercher", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Effacer", use_container_width=True)

    if clear_btn:
        st.session_state.pop("last_result", None)
        st.rerun()

    if search_btn and query.strip():
        with st.spinner("Recherche en cours..."):
            disc_filter = discipline if discipline != "Toutes" else None
            results = retrieve(collections, query, disc_filter, top_k=top_k)

            has_results = any(results[k] for k in results)
            if not has_results:
                st.warning("Aucun résultat trouvé. Essayez une autre matière ou reformulez.")
                st.stop()

            context = build_context(results)

        with st.spinner("Génération de la réponse..."):
            answer = generate_answer(query, context)

        st.session_state["last_result"] = {
            "query":   query,
            "answer":  answer,
            "results": results,
        }

    # Display result
    if "last_result" in st.session_state:
        data = st.session_state["last_result"]

        st.divider()

        # Contradiction warning
        if check_contradiction(data["results"]):
            st.warning(
                "⚠️ Transcription et slides disponibles sur ce sujet — "
                "vérifiez la cohérence dans les sources ci-dessous."
            )

        # Answer
        st.subheader("Réponse")
        st.markdown(data["answer"])

        # Sources
        if show_sources:
            st.divider()
            st.subheader("Sources utilisées")

            tab1, tab2 = st.tabs([
                f"📝 Transcriptions ({len(data['results']['transcripts'])})",
                f"📊 Slides ({len(data['results']['slides'])})"
            ])

            with tab1:
                if data["results"]["transcripts"]:
                    for r in data["results"]["transcripts"]:
                        with st.expander(
                            f"{r['discipline']} | {r['filename']} | chunk {r['page']} | {r['score']}%"
                        ):
                            st.text(r["text"])
                else:
                    st.info("Aucune transcription disponible pour cette matière.")

            with tab2:
                if data["results"]["slides"]:
                    for r in data["results"]["slides"]:
                        with st.expander(
                            f"{r['discipline']} | {r['filename']} | page {r['page']} | {r['score']}%"
                        ):
                            st.text(r["text"])
                else:
                    st.info("Aucun slide disponible pour cette matière.")

if __name__ == "__main__":
    main()
