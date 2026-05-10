# Hoymilonga RAG-chatbot

En chatbot baserad på **Retrieval-Augmented Generation (RAG)** som svarar på frågor om
[Hoymilonga.com](https://hoymilonga.com) — en sajt om tango och milongor.

Projektet är byggt som en skoluppgift och demonstrerar hela RAG-pipelinen från
web scraping till evaluering, med stöd för flera LLM-leverantörer.

## Funktioner

- **Web scraping** av Hoymilonga.com med respekt för `robots.txt`, datumstämplade snapshots
- **Hybrid retrieval** (vektorsökning + BM25)
- **Multi-leverantörsstöd**: OpenAI, Anthropic och Ollama (lokalt)
- **Streamlit-gränssnitt** med källcitat och chathistorik
- **Inbyggda guardrails** mot off-topic-frågor
- **Komplett evaluering**: Hit Rate, MRR och LLM-as-judge

## Snabbstart

```bash
# 1. Klona och installera
git clone <ditt-repo>
cd hoymilonga-rag
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2. Konfigurera nycklar
cp .env.example .env
# Redigera .env och lägg in dina API-nycklar

# 3. Bygg databasen (kör notebooken steg för steg)
jupyter notebook rag_chatbot.ipynb

# 4. Starta chatten
streamlit run streamlit_app.py
```

## Projektstruktur

```
hoymilonga-rag/
├── rag_chatbot.ipynb      # Huvudnotebook (skoluppgiften)
├── streamlit_app.py       # Webgränssnittet
├── data/
│   ├── raw/<datum>/       # Scrapade snapshots
│   └── chroma_db/         # Vektordatabas (skapas automatiskt)
├── eval/
│   ├── golden_dataset.json
│   └── results.csv        # Evalueringsresultat
├── logs/                  # Loggar av frågor + svar
├── .env.example
├── .gitignore
└── requirements.txt
```

## Krav

- Python 3.10+
- (Valfritt) [Ollama](https://ollama.com) för lokal körning utan API-kostnad
- (Valfritt) OpenAI- eller Anthropic-API-nyckel

## Licens

Skoluppgift — fri användning för utbildningsändamål.
