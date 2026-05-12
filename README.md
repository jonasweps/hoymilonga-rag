# Hoymilonga RAG-chatbot

> Skoluppgift — Alternativ A: Implementera en chattbot med RAG
> Författare: Jonas Wepsäläinen · 2026

En chatbot baserad på **Retrieval-Augmented Generation (RAG)** som svarar på
frågor om [Hoy-milonga.com](https://www.hoy-milonga.com) — en internationell
sajt där tangodansare hittar och annonserar milongor (tangoevenemang).

Projektet demonstrerar hela RAG-pipelinen från web scraping till evaluering,
med stöd för flera LLM-leverantörer.

---

## Innehåll

- [Snabbsammanfattning](#snabbsammanfattning)
- [Funktioner](#funktioner)
- [Resultat](#resultat)
- [Projektstruktur](#projektstruktur)
- [Snabbstart](#snabbstart)
- [Konfigurera LLM-leverantör](#konfigurera-llm-leverantör)
- [Evaluering](#evaluering)
- [Slutreflektion](#slutreflektion)

---

## Snabbsammanfattning

| Komponent | Val |
|---|---|
| **Datakälla** | hoy-milonga.com (web scraping med snapshot-versionering) |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (lokal, multilingual) |
| **Vektordatabas** | ChromaDB (lokal, persistent) |
| **Retrieval** | Hybrid (vektor + BM25) med justerbar α-vikt |
| **Genereringsmodell** | OpenAI gpt-4o-mini *(testad)*, Ollama llama3.1:8b *(testad)*, Anthropic Claude *(stödjs)* |
| **Gränssnitt** | Streamlit-app + textbaserat i notebooken |
| **Guardrails** | Off-topic-avvisning, "vet inte"-erkännande, källcitat |
| **Evaluering** | Hit Rate, MRR, LLM-as-judge, refusal accuracy, citation rate |

---

## Funktioner

- **Web scraping** av hoy-milonga.com med respekt för `robots.txt`, datumstämplade snapshots
- **Hybrid retrieval**: vektorsökning (semantisk) kombinerad med BM25 (nyckelords)
- **Multi-leverantörsstöd**: OpenAI / Anthropic / Ollama via gemensamt gränssnitt (Strategy Pattern)
- **Streamlit-gränssnitt** med källcitat, chathistorik och justerbara hyperparametrar
- **Guardrails** mot off-topic-frågor och hallucinationer
- **Komplett evaluering** med golden dataset (12 frågor, 6 kategorier)
- **Loggning** av alla interaktioner till JSONL för senare analys

---

## Resultat

Evaluering på golden dataset (12 frågor) med OpenAI gpt-4o-mini:

| Mått | Värde | Tolkning |
|---|---|---|
| **Retrieval — Hit Rate** | **88.9%** | Korrekta källor i top-5 |
| **Retrieval — MRR** | **0.633** | Rätt dokument i snitt på position ~1.6 |
| **Generation — snittbetyg (1-5)** | 2.40 | LLM-as-judge på alla svar |
| **Generation — andel ≥4** | 40% | Tydligt korrekta svar |
| **Refusal accuracy** | **91.7%** | Off-topic-frågor avvisas korrekt |
| **Citation rate (on-topic)** | 40% | Andel svar med källcitat |

### Hyperparameter-sökning för hybrid α

| α | Hit Rate | MRR |
|---|---|---|
| 0.00 (ren BM25) | 88.9% | **0.689** |
| 0.25 | 88.9% | 0.689 |
| 0.50 | 88.9% | 0.670 |
| 0.60 (default) | 88.9% | 0.633 |
| 1.00 (ren vektor) | 88.9% | 0.633 |

**Insikt:** För Hoymilongas lilla, domänspecifika korpus med tydliga
egennamn (städer, "milonga", "tango") presterar klassisk BM25 marginellt
bättre än vektorsökning. Diskuteras i kapitel 10 i notebooken.

---

## Projektstruktur

```
hoymilonga-rag/
├── rag_chatbot.ipynb       # Huvudnotebook — hela pipelinen + reflektion
├── streamlit_app.py        # Webgränssnitt
├── rag_core.py             # Kärnmoduler (genererad av notebooken)
├── build_notebook.py       # Hjälpscript som byggde notebooken
├── eval/
│   ├── golden_dataset.json # 12 utvärderingsfrågor i 6 kategorier
│   └── results.csv         # Evalueringsresultat per fråga
├── data/                   # (gitignored — skapas lokalt)
│   ├── raw/<datum>/        # Datumstämplade scraping-snapshots
│   └── chroma_db/          # Vektordatabas
├── logs/                   # (gitignored — chatlogg-historik)
├── .env.example            # Mall för API-nycklar
├── .gitignore              # Skyddar .env och stora data-mappar
├── requirements.txt        # Python-beroenden
└── README.md               # Denna fil
```

---

## Snabbstart

### 1. Klona och installera

```bash
git clone https://github.com/jonasweps/hoymilonga-rag.git
cd hoymilonga-rag

python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

### 2. Konfigurera API-nyckel

```bash
cp .env.example .env
# Öppna .env och fyll i din OpenAI-nyckel (sk-proj-...)
```

### 3. Bygg vektordatabasen

```bash
jupyter notebook rag_chatbot.ipynb
# Kör alla celler — scraping + indexering + utvärdering
```

### 4. Starta chatten

```bash
streamlit run streamlit_app.py
# Öppnar http://localhost:8501 i webbläsaren
```

---

## Konfigurera LLM-leverantör

Projektet stöder tre LLM-leverantörer som kan jämföras direkt mot samma RAG-pipeline:

### OpenAI (rekommenderas för bästa kvalitet)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
```

### Ollama (rekommenderas för full datasekretess)

```bash
# Installera
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

### Anthropic (stödjs)

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5
```

---

## Evaluering

Evalueringssystemet (kapitel 9 i notebooken) mäter på två nivåer:

**Retrieval-metriker** (hittar vi rätt dokument?)
- **Hit Rate@5** — andel frågor där minst en korrekt källa är i top-5
- **MRR** — Mean Reciprocal Rank, belönar tidig korrekt träff

**Generation-metriker** (är svaret bra?)
- **LLM-as-judge** — domarmodell betygsätter svar 1-5 mot facit
- **Refusal accuracy** — vägrar botten korrekt vid off-topic?
- **Citation rate** — innehåller svar `[Källa N]`?

Alla resultat sparas i `eval/results.csv` för fortsatt analys.

---

## Slutreflektion

Notebookens kapitel 10 innehåller en utförlig reflektion om:

- **Verkliga användningsfall** — för Hoymilonga själva, samt generaliserat till andra branscher
- **Affärsmöjligheter och utmaningar** — kostnad, latens, monitoring
- **Etiska och samhälleliga perspektiv** — hallucinationer, bias, GDPR, scraping-etik, sysselsättning
- **Tekniska begränsningar** — och föreslagna vidareutvecklingar
- **Kritisk analys av evalueringsresultaten** — varför är retrieval bra men generation bara hyfsad?

**Huvudslutsats:** RAG är en mogen och praktisk arkitektur, men dess kvalitet
är fundamentalt begränsad av kvaliteten på underlagskällan. Ett perfekt
språkmodell + perfekt retriever kan inte generera information som inte finns.

---

## Licens

Skoluppgift — fri användning för utbildningsändamål.

## Kontakt

Jonas Wepsäläinen — `jonas.wepsalainen@gmail.com`
