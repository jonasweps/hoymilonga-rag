"""
=============================================================================
streamlit_app.py — Webgränssnitt för Hoymilonga RAG-chatbot
=============================================================================
Kör med:    streamlit run streamlit_app.py
Förutsätter att rag_chatbot.ipynb har körts först (för att bygga vektordatabasen).

Appen återanvänder modulerna som definieras i notebooken via importer från
en lokal modul `rag_core.py`. Notebooken skriver ut den filen automatiskt
i kapitel 6.
=============================================================================
"""

import os
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ladda miljövariabler (.env) — API-nycklar och defaults
load_dotenv()

# Importera kärnlogiken som notebooken skapar.
# Om filen inte finns, vägleder vi användaren.
try:
    from rag_core import (
        VectorStore,
        HybridRetriever,
        RAGPipeline,
        get_llm_provider,
        load_chunks_for_bm25,
    )
except ImportError:
    st.error(
        "❌ `rag_core.py` saknas. Kör hela notebooken `rag_chatbot.ipynb` "
        "minst en gång — den genererar `rag_core.py` automatiskt."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Sidkonfiguration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hoymilonga RAG-chatbot",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Cachad initialisering av RAG-pipelinen
#   @st.cache_resource gör att vi bara laddar embeddings-modellen och databasen
#   EN gång per Streamlit-process — inte vid varje fråga. Avgörande för UX.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Laddar embeddings-modell och vektordatabas...")
def init_pipeline(provider_name: str, model_name: str, top_k: int, alpha: float):
    """Sätter upp RAG-pipelinen med vald LLM-leverantör."""
    project_root = Path(__file__).parent
    chroma_path = project_root / "data" / "chroma_db"

    # 1. Vektordatabas (ChromaDB med multilingual sentence-transformer)
    vector_store = VectorStore(
        persist_dir=str(chroma_path),
        collection_name="hoymilonga",
    )

    # 2. Hybrid retriever (vektor + BM25)
    chunks, metadatas = load_chunks_for_bm25(vector_store)
    retriever = HybridRetriever(
        vector_store=vector_store,
        all_chunks=chunks,
        all_metadatas=metadatas,
    )

    # 3. LLM-leverantör (OpenAI / Anthropic / Ollama)
    llm = get_llm_provider(provider_name, model_name)

    # 4. Den färdiga RAG-pipelinen med alla delar ihopkopplade
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_provider=llm,
        top_k=top_k,
        alpha=alpha,
    )
    return pipeline


# -----------------------------------------------------------------------------
# Sidopanel — inställningar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Inställningar")

    provider = st.selectbox(
        "LLM-leverantör",
        options=["openai", "anthropic", "ollama"],
        index=["openai", "anthropic", "ollama"].index(
            os.getenv("LLM_PROVIDER", "openai")
        ),
        help="Välj vilken språkmodell som ska generera svar.",
    )

    # Modellnamn beror på leverantör
    default_models = {
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    }
    model_name = st.text_input("Modellnamn", value=default_models[provider])

    st.markdown("---")
    st.subheader("Retrieval-parametrar")
    top_k = st.slider(
        "Antal hämtade chunks (top-k)",
        min_value=1, max_value=15, value=5,
        help="Hur många textstycken hämtas som kontext till modellen.",
    )
    alpha = st.slider(
        "Vikt vektor vs BM25 (α)",
        min_value=0.0, max_value=1.0, value=0.6, step=0.1,
        help="α=1.0 = ren vektorsökning, α=0.0 = ren nyckelords-sökning.",
    )

    st.markdown("---")
    if st.button("🗑️ Återställ chathistorik"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "💡 **Källcitat** visas under varje svar. Klicka på ⓘ för att "
        "se exakt vilka textstycken som användes."
    )


# -----------------------------------------------------------------------------
# Initiera pipelinen
# -----------------------------------------------------------------------------
try:
    rag = init_pipeline(provider, model_name, top_k, alpha)
except Exception as exc:
    st.error(f"❌ Kunde inte starta pipelinen: {exc}")
    st.info(
        "Kontrollera att:\n"
        "1. Du har kört notebooken `rag_chatbot.ipynb` så vektordatabasen finns.\n"
        "2. Din `.env` innehåller rätt API-nyckel för vald leverantör.\n"
        "3. Om du valt Ollama: att Ollama-servern körs (`ollama serve`)."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Huvudgränssnitt
# -----------------------------------------------------------------------------
st.title("💃 Hoymilonga RAG-chatbot")
st.caption(
    f"Svar genereras av **{provider}/{model_name}** baserat på innehåll "
    f"från Hoymilonga.com. Off-topic-frågor avvisas."
)

# Initiera chathistorik i session-state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Visa hela historiken
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Källor ({len(set(msg['sources']))} unika)"):
                for src in sorted(set(msg["sources"])):
                    st.markdown(f"- [{src}]({src})")
        if msg.get("retrieved_chunks"):
            with st.expander("🔍 Hämtade textstycken (debug)"):
                for i, chunk in enumerate(msg["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))


# -----------------------------------------------------------------------------
# Inputfält och svar
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Vad vill du veta om Hoymilonga?"):
    # Logga användarens fråga
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generera svar
    with st.chat_message("assistant"):
        with st.spinner("Söker i kunskapsbasen..."):
            try:
                result = rag.answer(prompt)
            except Exception as exc:
                st.error(f"Något gick fel: {exc}")
                st.stop()

        st.markdown(result["answer"])

        # Visa källor om någon genererades
        if result.get("sources"):
            with st.expander(f"📚 Källor ({len(set(result['sources']))} unika)"):
                for src in sorted(set(result["sources"])):
                    st.markdown(f"- [{src}]({src})")

        if result.get("retrieved_chunks"):
            with st.expander("🔍 Hämtade textstycken (debug)"):
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))

    # Spara svaret i historiken
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "retrieved_chunks": result.get("retrieved_chunks", []),
    })

    # Logga interaktionen till disk för senare analys
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"chat_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model_name,
            "question": prompt,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "top_k": top_k,
            "alpha": alpha,
        }, ensure_ascii=False) + "\n")
