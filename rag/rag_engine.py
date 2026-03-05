
            collections[name] = client.get_collection(name=name, embedding_function=ef)
        except Exception:
            collections[name] = None

    return collections

@st.cache_resource
def load_exam_collection():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return client.get_collection(name="exams", embedding_function=ef)
    except Exception:
        return None

# ── WHISPER (voice input) ───────────────────────────────────────────
@st.cache_resource
def load_whisper():
    try:
        import whisper
        return whisper.load_model("tiny")
    except ImportError:
        return None

def transcribe_voice(audio_bytes):
    model = load_whisper()
    if model is None:
        return None, "Whisper non installé. Exécutez: py -3.11 -m pip install openai-whisper"
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        result = model.transcribe(tmp_path, language="fr", fp16=False)
        os.unlink(tmp_path)
        return result["text"].strip(), None
    except Exception as e:
        return None, f"Erreur transcription: {e}"

# ── RETRIEVAL ──────────────────────────────────────────────────────
def retrieve(collections, query, discipline_filter, top_k=TOP_K):
    where = None
    if discipline_filter:
        if isinstance(discipline_filter, list):
            if len(discipline_filter) == 1:
                where = {"discipline": discipline_filter[0]}
            else:
                where = {"discipline": {"$in": discipline_filter}}
        elif discipline_filter != "Toutes":
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
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False
        }, timeout=120)
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"Erreur de génération: {e}"

def check_contradiction(results):
    if not results["transcripts"] or not results["slides"]:
        return False
    top_t = results["transcripts"][0]["score"] if results["transcripts"] else 0
    top_s = results["slides"][0]["score"] if results["slides"] else 0
    return top_t > 60 and top_s > 60

# ── MCQ HELPERS ────────────────────────────────────────────────────
def _parse_mcq(raw):
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    mcq = json.loads(raw)
    required = ["question", "choices", "correct", "explanation", "source"]
    if all(k in mcq for k in required):
        # Normalize correct to always be a sorted list
        if isinstance(mcq["correct"], str):
            mcq["correct"] = sorted(mcq["correct"].replace(" ", "").split(","))
        else:
            mcq["correct"] = sorted(mcq["correct"])
        return mcq
    return None

def generate_mcq_from_context(context, discipline):
    prompt = (
        "Tu es un professeur de médecine à la FMPC. "
        "Génère une QCM à 4 choix basée UNIQUEMENT sur le contexte ci-dessous.\n\n"
        "RÈGLES STRICTES:\n"
        "- 4 choix (A, B, C, D)\n"
        "- Il peut y avoir UNE OU PLUSIEURS bonnes réponses (système TOR: tout ou rien)\n"
        "- Les mauvaises réponses doivent être plausibles\n"
        "- L'explication doit citer la phrase exacte du contexte qui justifie chaque bonne réponse\n"
        "- Réponds UNIQUEMENT en JSON valide, sans backticks, sans texte avant ou après\n\n"
        "correct doit être une liste JSON, même si une seule réponse: [\"A\"] ou [\"A\",\"C\"]\n\n"
        '{"question":"...","choices":{"A":"...","B":"...","C":"...","D":"..."},'
        '"correct":["A"],"explanation":"...","source":"fichier | page X"}\n\n'
        f"CONTEXTE ({discipline}):\n{context}"
    )
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False
        }, timeout=120)
        raw = res.json().get("response", "").strip()
        mcq = _parse_mcq(raw)
        return (mcq, None) if mcq else (None, "Format JSON incomplet.")
    except json.JSONDecodeError:
        return None, "Erreur JSON — réessayez."
    except Exception as e:
        return None, f"Erreur: {e}"

def generate_mcq_from_exam(exam_text, discipline, year, professor_context):
    prompt = (
        "Tu es un professeur de médecine à la FMPC. "
        "Voici un extrait d'examen. Génère une QCM à 4 choix basée dessus.\n\n"
        "RÈGLES STRICTES:\n"
        "- 4 choix (A, B, C, D)\n"
        "- Il peut y avoir UNE OU PLUSIEURS bonnes réponses (système TOR: tout ou rien)\n"
        "- Les mauvaises réponses doivent être plausibles\n"
        "- La correction doit être justifiée par le matériel du professeur fourni\n"
        "- Si la réponse n'est pas dans le matériel, base-toi sur le contenu de l'examen\n"
        "- Réponds UNIQUEMENT en JSON valide, sans backticks, sans texte avant ou après\n\n"
        "correct doit être une liste JSON, même si une seule réponse: [\"A\"] ou [\"A\",\"C\"]\n\n"
        '{"question":"...","choices":{"A":"...","B":"...","C":"...","D":"..."},'
        '"correct":["A"],"explanation":"...","source":"fichier | page X"}\n\n'
        f"EXTRAIT D'EXAMEN ({discipline}, {year}):\n{exam_text}\n\n"
        f"MATÉRIEL DU PROFESSEUR:\n{professor_context}"
    )
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False
        }, timeout=120)
        raw = res.json().get("response", "").strip()
        mcq = _parse_mcq(raw)
        return (mcq, None) if mcq else (None, "Format JSON incomplet.")
    except json.JSONDecodeError:
        return None, "Erreur JSON — réessayez."
    except Exception as e:
        return None, f"Erreur: {e}"

# ── MCQ DISPLAY HELPER (TOR system) ───────────────────────────────
def display_mcq(mcq, q_index, state_prefix, answered_key, selected_key):
    """Display a single MCQ with TOR scoring (all-or-nothing, multiple correct answers)."""
    answered = st.session_state.get(answered_key, False)
    selected = st.session_state.get(selected_key, [])  # list of checked letters

    correct = mcq["correct"]  # always a list after _parse_mcq
    n_correct = len(correct)
    choices = mcq["choices"]

    st.markdown(f"### Q{q_index + 1}. {mcq['question']}")
    col_info, col_tor = st.columns([4, 1])
    with col_info:
        st.caption(f"Source: {mcq.get('source', '?')}")
    with col_tor:
        st.caption(f"🎯 TOR — {n_correct} bonne(s) réponse(s)")

    if not answered:
        # Checkboxes for multiple selection
        checked = []
        for letter, text in choices.items():
            if st.checkbox(f"{letter}. {text}", key=f"{state_prefix}_cb_{letter}"):
                checked.append(letter)
        if st.button("Valider", type="primary", key=f"{state_prefix}_submit", use_container_width=True):
            st.session_state[selected_key] = sorted(checked)
            st.session_state[answered_key] = True
            st.rerun()
    else:
        for letter, text in choices.items():
            is_correct_choice = letter in correct
            was_selected = letter in selected
            if is_correct_choice and was_selected:
                st.success(f"✅ {letter}. {text}")
            elif is_correct_choice and not was_selected:
                st.warning(f"⚠️ {letter}. {text} ← réponse manquée")
            elif not is_correct_choice and was_selected:
                st.error(f"❌ {letter}. {text} ← mauvais choix")
            else:
                st.write(f"{letter}. {text}")

        is_correct = sorted(selected) == sorted(correct)
        st.divider()
        if is_correct:
            st.success("🎉 Bonne réponse ! (TOR validé)")
        else:
            correct_str = ", ".join(correct)
            st.error(f"❌ Mauvaise réponse. Réponses correctes: **{correct_str}** (TOR — tout ou rien)")
        with st.expander("📖 Correction basée sur le cours"):
            st.markdown(mcq["explanation"])
            st.caption(f"Source: {mcq.get('source', '?')}")
        return is_correct
    return None

# ── PAGE: RAG ASSISTANT ─────────────────────────────────────────────
def page_rag(collections, discipline, top_k, show_sources):
    st.subheader("Posez votre question")

    voice_query = None
    try:
        from audio_recorder_streamlit import audio_recorder
        st.caption("🎙️ Cliquez pour parler ou tapez directement")
        audio_bytes = audio_recorder(
            text="", recording_color="#e74c3c", neutral_color="#2c3e50",
            icon_size="2x", pause_threshold=2.0,
        )
        if audio_bytes and len(audio_bytes) > 1000:
            with st.spinner("Transcription vocale..."):
                voice_query, err = transcribe_voice(audio_bytes)
                if err:
                    st.warning(err)
                elif voice_query:
                    st.success(f"🎙️ Transcrit: *{voice_query}*")
    except ImportError:
        st.caption("💡 Saisie vocale: `py -3.11 -m pip install audio-recorder-streamlit openai-whisper`")

    query = st.text_input(
        "Votre question",
        value=voice_query if voice_query else "",
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
            if not any(results[k] for k in results):
                st.warning("Aucun résultat trouvé. Essayez une autre matière ou reformulez.")
                st.stop()
            context = build_context(results)

        with st.spinner("Génération de la réponse..."):
            answer = generate_answer(query, context)

        st.session_state["last_result"] = {
            "query": query, "answer": answer, "results": results,
        }

    if "last_result" in st.session_state:
        data = st.session_state["last_result"]
        st.divider()
        if check_contradiction(data["results"]):
            st.warning(
                "⚠️ Transcription et slides disponibles sur ce sujet — "
                "vérifiez la cohérence dans les sources ci-dessous."
            )
        st.subheader("Réponse")
        st.markdown(data["answer"])

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
                        with st.expander(f"{r['discipline']} | {r['filename']} | chunk {r['page']} | {r['score']}%"):
                            st.text(r["text"])
                else:
                    st.info("Aucune transcription disponible pour cette matière.")
            with tab2:
                if data["results"]["slides"]:
                    for r in data["results"]["slides"]:
                        with st.expander(f"{r['discipline']} | {r['filename']} | page {r['page']} | {r['score']}%"):
                            st.text(r["text"])
                else:
                    st.info("Aucun slide disponible pour cette matière.")

# ── PAGE: MCQ (generated from course material) ─────────────────────
def page_mcq(collections, discipline, top_k):
    st.subheader("QCM — Questions à Choix Multiples")
    st.caption("5 questions générées à partir de vos cours et slides")

    topic = st.text_input(
        "Sujet (optionnel)",
        placeholder="Ex: capsule bactérienne, aorte abdominale... ou laissez vide pour aléatoire",
        key="mcq_topic"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        gen_btn = st.button("Générer 5 QCM", type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button("Recommencer", use_container_width=True)

    if reset_btn:
        for key in list(st.session_state.keys()):
            if key.startswith("mcq_"):
                del st.session_state[key]
        st.rerun()

    if gen_btn:
        disc_filter = discipline if discipline != "Toutes" else None
        query = topic.strip() if topic.strip() else f"concept important en {discipline if discipline != 'Toutes' else 'médecine'}"

        with st.spinner("Récupération du contenu..."):
            results = retrieve(collections, query, disc_filter, top_k=top_k)
            if not any(results[k] for k in results):
                st.warning("Aucun contenu trouvé. Ingérez des slides ou transcriptions d'abord.")
                st.stop()
            context = build_context(results)

        questions = []
        progress = st.progress(0, text="Génération des questions...")
        for i in range(N_EXAM_QUESTIONS):
            progress.progress((i + 1) / N_EXAM_QUESTIONS, text=f"Question {i+1}/{N_EXAM_QUESTIONS}...")
            mcq, err = generate_mcq_from_context(context, discipline)
            if mcq:
                questions.append(mcq)
        progress.empty()

        if not questions:
            st.error("Impossible de générer les questions. Réessayez.")
            st.stop()

        st.session_state["mcq_questions"] = questions
        st.session_state["mcq_score"]     = 0
        st.session_state["mcq_current"]   = 0
        for i in range(len(questions)):
            st.session_state.pop(f"mcq_ans_{i}", None)
            st.session_state.pop(f"mcq_sel_{i}", None)

    if "mcq_questions" in st.session_state:
        questions = st.session_state["mcq_questions"]
        total     = len(questions)

        # Score display
        answered_count = sum(1 for i in range(total) if st.session_state.get(f"mcq_ans_{i}", False))
        correct_count  = sum(
            1 for i in range(total)
            if st.session_state.get(f"mcq_ans_{i}", False)
            and st.session_state.get(f"mcq_sel_{i}") == questions[i]["correct"]
        )
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions", f"{answered_count}/{total}")
        c2.metric("Correctes", correct_count)
        c3.metric("Score", f"{round(correct_count/total*100) if total else 0}%")

        # Display all questions
        for i, mcq in enumerate(questions):
            st.divider()
            display_mcq(mcq, i, f"mcq_{i}", f"mcq_ans_{i}", f"mcq_sel_{i}")

        if answered_count == total:
            st.divider()
            st.success(f"🎓 Session terminée ! Score final: {correct_count}/{total}")

# ── PAGE: EXAM PRACTICE ─────────────────────────────────────────────
def page_exams(collections, top_k):
    st.subheader("📚 Examens Anciens — Entraînement")
    st.caption("5 questions tirées de vos examens passés, corrigées par le matériel du professeur")

    exam_col = load_exam_collection()

    if exam_col is None or exam_col.count() == 0:
        st.warning("Aucun examen ingéré.")
        st.markdown("**Setup:**")
        st.code(
            "mkdir C:\\FMPC_Scribe\\exams\\bacteriologie\n"
            "mkdir C:\\FMPC_Scribe\\exams\\anatomie\n"
            "# ... puis deposez vos PDFs d'examens dans les subfolders\n"
            "py -3.11 ingest_exams.py"
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        block_name  = st.selectbox("Bloc d'examen", list(EXAM_BLOCKS.keys()), key="exam_block")
        disciplines = EXAM_BLOCKS[block_name]
    with col2:
        try:
            where    = None
            if disciplines:
                where = {"discipline": {"$in": disciplines}} if len(disciplines) > 1 else {"discipline": disciplines[0]}
            kwargs   = {"include": ["metadatas"]}
            if where:
                kwargs["where"] = where
            all_meta = exam_col.get(**kwargs)["metadatas"]
            years    = sorted(set(m.get("year", "unknown") for m in all_meta), reverse=True)
        except Exception:
            years = []
        year_filter = st.selectbox("Année", ["Toutes"] + years, key="exam_year")
    with col3:
        topic = st.text_input("Sujet (optionnel)", placeholder="Ex: morphologie bactérienne...", key="exam_topic")

    col1, col2 = st.columns([1, 5])
    with col1:
        gen_btn = st.button("Générer 5 QCM", type="primary", use_container_width=True, key="exam_gen")
    with col2:
        reset_btn = st.button("Recommencer", use_container_width=True, key="exam_reset")

    if reset_btn:
        for key in list(st.session_state.keys()):
            if key.startswith("ex_"):
                del st.session_state[key]
        st.rerun()

    if gen_btn:
        query = topic.strip() if topic.strip() else f"question examen {block_name}"

        # Build exam where clause
        where_exam = {}
        if disciplines:
            where_exam["discipline"] = {"$in": disciplines} if len(disciplines) > 1 else disciplines[0]
        if year_filter != "Toutes":
            where_exam["year"] = year_filter

        questions = []
        progress  = st.progress(0, text="Génération des questions...")

        for i in range(N_EXAM_QUESTIONS):
            progress.progress((i + 1) / N_EXAM_QUESTIONS, text=f"Question {i+1}/{N_EXAM_QUESTIONS}...")

            # Get exam chunk
            try:
                kwargs = {
                    "query_texts": [query],
                    "n_results": 3,
                    "include": ["documents", "metadatas", "distances"]
                }
                if where_exam:
                    kwargs["where"] = where_exam
                exam_res  = exam_col.query(**kwargs)
                exam_text = "\n\n".join(exam_res["documents"][0])
                exam_meta = exam_res["metadatas"][0][0] if exam_res["metadatas"][0] else {}
                exam_year = exam_meta.get("year", "?")
                exam_disc = exam_meta.get("discipline", block_name)
            except Exception as e:
                st.warning(f"Question {i+1}: récupération échouée — {e}")
                continue

            # Get professor context
            prof_results = retrieve(collections, query, disciplines, top_k=top_k)
            prof_context = build_context(prof_results)

            mcq, err = generate_mcq_from_exam(exam_text, exam_disc, exam_year, prof_context)
            if mcq:
                questions.append(mcq)

        progress.empty()

        if not questions:
            st.error("Impossible de générer les questions. Vérifiez que des examens sont ingérés.")
            st.stop()

        st.session_state["ex_questions"] = questions
        for i in range(len(questions)):
            st.session_state.pop(f"ex_ans_{i}", None)
            st.session_state.pop(f"ex_sel_{i}", None)

    if "ex_questions" in st.session_state:
        questions = st.session_state["ex_questions"]
        total     = len(questions)

        answered_count = sum(1 for i in range(total) if st.session_state.get(f"ex_ans_{i}", False))
        correct_count  = sum(
            1 for i in range(total)
            if st.session_state.get(f"ex_ans_{i}", False)
            and st.session_state.get(f"ex_sel_{i}") == questions[i]["correct"]
        )

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions", f"{answered_count}/{total}")
        c2.metric("Correctes", correct_count)
        c3.metric("Score", f"{round(correct_count/total*100) if total else 0}%")

        for i, mcq in enumerate(questions):
            st.divider()
            display_mcq(mcq, i, f"ex_{i}", f"ex_ans_{i}", f"ex_sel_{i}")

        if answered_count == total:
            st.divider()
            st.success(f"🎓 Session terminée ! Score final: {correct_count}/{total}")

# ── MAIN ───────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FMPC Assistant",
        page_icon="🩺",
        layout="wide"
    )

    st.title("🩺 FMPC Study Assistant")

    collections = load_collections()

    with st.sidebar:
        page = st.radio(
            "Navigation",
            ["💬 Assistant RAG", "📝 QCM", "📚 Examens"],
            label_visibility="collapsed"
        )

        st.divider()
        st.header("Filtres")
        discipline   = st.selectbox("Matière", KNOWN_DISCIPLINES)
        top_k        = st.slider("Chunks récupérés", min_value=2, max_value=8, value=TOP_K)
        show_sources = st.toggle("Afficher les sources", value=True)

        st.divider()
        st.header("Base de données")
        for name, col in collections.items():
            if col:
                st.metric(name, f"{col.count()} chunks")
            else:
                st.metric(name, "vide")
        exam_col = load_exam_collection()
        if exam_col:
            st.metric("exams", f"{exam_col.count()} chunks")
        else:
            st.metric("exams", "vide")

        st.divider()
        st.caption("Ollama: qwen2.5:7b")
        st.caption("Embed: multilingual-MiniLM")

    if page == "💬 Assistant RAG":
        page_rag(collections, discipline, top_k, show_sources)
    elif page == "📝 QCM":
        page_mcq(collections, discipline, top_k)
    else:
        page_exams(collections, top_k)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
