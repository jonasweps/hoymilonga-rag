"""
Hjälpscript som bygger rag_chatbot.ipynb från läsbara cell-strängar.
Körs en gång: python build_notebook.py
"""
import json
from pathlib import Path

# Varje cell är (typ, källkod). typ = "md" eller "code".
CELLS = []

def md(src):
    CELLS.append(("md", src))

def code(src):
    CELLS.append(("code", src))


# =============================================================================
# KAPITEL 1 — INTRODUKTION
# =============================================================================
md("""# Hoymilonga RAG-chatbot

## En djupgående genomgång av Retrieval-Augmented Generation

**Skoluppgift — Alternativ A: Implementera en chattbot med RAG**

Denna notebook bygger en komplett RAG-chatbot för webbsidan [Hoymilonga.com](https://hoymilonga.com), en internationell sajt där tangodansare hittar och annonserar milongor (tangoevenemang).

### Varför just Hoymilonga?

* Sajten har ett **avgränsat domänområde** (tango/milongor) — perfekt för RAG, eftersom RAG lyser när allmänna LLM:er saknar specifik kunskap.
* Innehållet **uppdateras kontinuerligt** (nya evenemang, nya städer) — så vi får demonstrera hur snapshots och periodisk re-indexering fungerar.
* Den är **flerspråkig** (spanska/engelska) — vilket utmanar vår embedding-modell att hantera flera språk.

### Teori: Vad är RAG och varför?

Stora språkmodeller (LLM:er) som GPT-4 eller Claude har ett enormt allmänkunskapsförråd, men:

| Problem hos vanliga LLM:er | Hur RAG löser det |
|---|---|
| **Kunskaps-cutoff** — modellen vet inget efter sitt träningsdatum | Vi matar in färska dokument vid frågetillfället |
| **Hallucinationer** — modellen hittar på övertygande lögner | Vi tvingar svaren att grundas på hämtad text |
| **Saknad domänkunskap** — modellen vet inget om mitt företag | Vi indexerar våra egna dokument |
| **Ej spårbara svar** — användaren kan inte verifiera | Vi visar källcitat med varje svar |

### RAG-pipelinen i fem steg

```
[1. Indexering — körs en gång]
  Källdokument → Chunkning → Embeddings → Vektordatabas

[2. Frågetillfället — körs varje fråga]
  Användarfråga → Embedding → Hämta top-K → Bygg prompt → LLM → Svar
```

### Vad denna notebook täcker

1. **Setup & konfiguration** — beroenden, miljövariabler, leverantörsval
2. **Web scraping** av Hoymilonga.com med snapshot-versionering
3. **Chunking** — dela upp text i lagom stora bitar
4. **Embeddings & ChromaDB** — semantisk indexering
5. **Hybrid retrieval** — vektorsökning + BM25 för bästa resultat
6. **LLM-abstraktion** — ett gemensamt gränssnitt för OpenAI / Anthropic / Ollama
7. **RAG-pipeline med guardrails** — den sammankopplade chatten
8. **Streamlit-app** (genereras automatiskt)
9. **Evaluering** — Hit Rate, MRR, LLM-as-judge
10. **Slutreflektion** — verkliga användningsfall, etik, affärsmöjligheter
""")


# =============================================================================
# KAPITEL 2 — SETUP
# =============================================================================
md("""## 1. Setup & konfiguration

Vi börjar med att importera alla beroenden, sätta upp logger och ladda in API-nycklar från `.env`-filen.
""")

code('''# === Standardbibliotek ===
import os
import re
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

# === Tredjepartsbibliotek ===
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from dotenv import load_dotenv
import pandas as pd

# === Ladda hemligheter från .env ===
# load_dotenv() läser in nyckel/värde-par från filen .env och stoppar in
# dem i os.environ. Filen .env ligger i .gitignore och pushas ALDRIG.
load_dotenv()

# === Logger-uppsättning ===
# En logger ger oss snyggare och mer informativa utskrifter än print()
# och vi kan styra detaljnivån (DEBUG/INFO/WARNING) på ett ställe.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag")

# === Projektsökvägar ===
# Vi använder pathlib för plattformsoberoende sökvägar (fungerar både
# på macOS, Linux och Windows utan att behöva oroa sig för / vs \\).
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"
EVAL_DIR = PROJECT_ROOT / "eval"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in (DATA_DIR, RAW_DIR, CHROMA_DIR, EVAL_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

log.info(f"Projektrot: {PROJECT_ROOT}")
log.info(f"Data-mapp: {DATA_DIR}")
''')

code('''# === Snabbkontroll: vilka API-nycklar har vi tillgängliga? ===
# Vi avslöjar inte själva nyckelvärdet, bara om de finns och deras prefix.
# Detta är en best practice för felsökning utan att läcka hemligheter.

def check_keys():
    """Skriv ut vilka API-nycklar som hittats — utan att avslöja värden."""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    for name, value in keys.items():
        if value and len(value) > 10:
            # Visa endast prefix + längd, inte själva nyckeln
            print(f"  ✅ {name}: {value[:7]}... ({len(value)} tecken)")
        else:
            print(f"  ❌ {name}: saknas eller för kort")
    print(f"  ℹ️  OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")

check_keys()
''')


# =============================================================================
# KAPITEL 3 — WEB SCRAPING
# =============================================================================
md("""## 2. Web scraping av Hoymilonga.com

### Hur en ansvarsfull scraper beter sig

Innan vi skriver kod måste vi förstå att scraping har **etiska och juridiska aspekter**. En ansvarsfull scraper:

1. **Respekterar `robots.txt`** — sajtens egna regler för vilka URL:er som får hämtas
2. **Identifierar sig** med en tydlig `User-Agent`-header (så sajtägaren ser vem som kommer)
3. **Begränsar hastigheten** med `time.sleep()` mellan requests för att inte belasta servern
4. **Begränsar djupet** — vi behöver inte hämta hela internet, bara relevanta sidor
5. **Cachar resultat** — vi kör inte om scrapern i onödan

### Snapshot-strategi för reproducerbar evaluering

Eftersom Hoymilonga.com **uppdateras kontinuerligt**, skulle vår evaluering ge olika resultat varje gång om vi scrapade live. Lösning: vi sparar varje scraping-körning i en datumstämplad mapp (`data/raw/2026-05-09/`). Detta ger oss:

* **Reproducerbarhet** — samma snapshot ger samma evalueringsresultat
* **Spårbarhet av "data drift"** — vi kan jämföra hur svaren förändras när sidan uppdateras
* **Rollback** — om en ny scraping är trasig kan vi gå tillbaka till tidigare snapshot
""")

code('''class PoliteScraper:
    """
    En ansvarsfull web scraper som hämtar interna länkar från en startsida.

    Designprinciper:
    - Respekterar robots.txt
    - Lägger time.sleep mellan requests (default 1 sekund)
    - Sätter en tydlig User-Agent
    - Begränsar antalet sidor (default 50) för att undvika att hänga sig
    - Sparar varje sida som en JSON-fil med metadata + extraherad text
    """

    def __init__(
        self,
        base_url: str,
        output_dir: Path,
        max_pages: int = 50,
        delay_seconds: float = 1.0,
        user_agent: str = "HoymilongaRAG/1.0 (educational; +https://github.com)",
    ):
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.max_pages = max_pages
        self.delay = delay_seconds
        self.user_agent = user_agent

        # Spårningsstrukturer
        self.visited: set[str] = set()
        self.queue: list[str] = [self.base_url]
        self.failed: list[tuple[str, str]] = []

        # Robots.txt-parser
        self._setup_robots()

    def _setup_robots(self):
        """Hämtar och parsar /robots.txt så vi kan respektera dess regler."""
        self.rp = RobotFileParser()
        robots_url = urljoin(self.base_url, "/robots.txt")
        try:
            self.rp.set_url(robots_url)
            self.rp.read()
            log.info(f"Läste robots.txt från {robots_url}")
        except Exception as exc:
            log.warning(f"Kunde inte läsa robots.txt: {exc} — fortsätter ändå.")

    def _can_fetch(self, url: str) -> bool:
        """True om robots.txt tillåter att vi hämtar URL:en."""
        try:
            return self.rp.can_fetch(self.user_agent, url)
        except Exception:
            return True  # Försiktigt liberalt vid fel

    def _is_internal(self, url: str) -> bool:
        """True om URL:en pekar på samma domän som base_url."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def _extract_main_text(self, html: str) -> str:
        """
        Plockar ut den läsbara texten ur en HTML-sida och rensar bort
        navigation, footer, scripts osv. (boilerplate som inte tillför
        kunskap till vår RAG).
        """
        soup = BeautifulSoup(html, "lxml")

        # Ta bort element som inte innehåller verkligt sidinnehåll
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "noscript", "iframe", "svg", "form"]):
            tag.decompose()

        # Hämta titeln separat — den är ofta värdefull metadata
        title = (soup.title.string.strip()
                 if soup.title and soup.title.string else "")

        # Hämta brödtexten med radbrytningar mellan block-element
        text = soup.get_text(separator="\\n", strip=True)

        # Komprimera flera tomma rader till max en
        text = re.sub(r"\\n{3,}", "\\n\\n", text)

        return f"{title}\\n\\n{text}".strip() if title else text

    def _extract_links(self, html: str, current_url: str) -> List[str]:
        """Hittar alla interna länkar från en sida (för BFS-traversering)."""
        soup = BeautifulSoup(html, "lxml")
        out = []
        for a in soup.find_all("a", href=True):
            url = urljoin(current_url, a["href"])
            url = url.split("#")[0].rstrip("/")  # Rensa fragment och trailing slash
            if (url and self._is_internal(url) and url not in self.visited
                    and url not in self.queue):
                out.append(url)
        return out

    def scrape(self) -> List[Dict[str, Any]]:
        """
        Kör hela scraping-jobbet. Returnerar en manifest-lista över
        alla sparade sidor.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[dict] = []

        progress = tqdm(total=self.max_pages, desc="Scraping", unit="sida")

        while self.queue and len(self.visited) < self.max_pages:
            url = self.queue.pop(0)

            # Hoppa över redan besökta eller blockerade URL:er
            if url in self.visited:
                continue
            if not self._can_fetch(url):
                log.debug(f"robots.txt blockerar {url}")
                continue

            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=15,
                )
                response.raise_for_status()

                # Hoppa över icke-HTML-svar (t.ex. PDF, bilder)
                ctype = response.headers.get("Content-Type", "")
                if "html" not in ctype.lower():
                    log.debug(f"Hoppar över icke-HTML: {url} ({ctype})")
                    self.visited.add(url)
                    continue

                # Extrahera och spara
                text = self._extract_main_text(response.text)
                if len(text) < 50:
                    # För kort för att vara meningsfullt innehåll
                    self.visited.add(url)
                    continue

                page_id = hashlib.md5(url.encode()).hexdigest()[:10]
                filename = self.output_dir / f"{page_id}.json"
                page_data = {
                    "id": page_id,
                    "url": url,
                    "title": text.split("\\n", 1)[0][:200],
                    "text": text,
                    "scraped_at": datetime.now().isoformat(timespec="seconds"),
                    "char_count": len(text),
                }
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)

                manifest.append({
                    "id": page_id,
                    "url": url,
                    "file": filename.name,
                    "char_count": len(text),
                })
                self.visited.add(url)

                # Lägg till nya länkar i kön
                self.queue.extend(self._extract_links(response.text, url))

                progress.update(1)
                time.sleep(self.delay)

            except requests.RequestException as exc:
                log.warning(f"Request-fel för {url}: {exc}")
                self.failed.append((url, str(exc)))
                self.visited.add(url)
            except Exception as exc:
                log.error(f"Oväntat fel för {url}: {exc}")
                self.failed.append((url, str(exc)))
                self.visited.add(url)

        progress.close()

        # Spara manifest med all metadata om körningen
        manifest_file = self.output_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump({
                "base_url": self.base_url,
                "scraped_at": datetime.now().isoformat(timespec="seconds"),
                "page_count": len(manifest),
                "failed_count": len(self.failed),
                "pages": manifest,
                "failed": self.failed,
            }, f, ensure_ascii=False, indent=2)

        log.info(f"Scraping klar: {len(manifest)} sidor sparade, "
                 f"{len(self.failed)} misslyckade.")
        return manifest
''')

code('''# === Kör scrapern (eller läs in befintlig snapshot) ===
# Vi datumstämplar varje körning så vi har reproducerbara snapshots.

SNAPSHOT_DATE = datetime.now().strftime("%Y-%m-%d")
SNAPSHOT_DIR = RAW_DIR / SNAPSHOT_DATE

# Bara scrapa om denna snapshot inte redan finns — sparar tid och
# skonar Hoymilongas server om vi kör om notebooken.
if not (SNAPSHOT_DIR / "manifest.json").exists():
    scraper = PoliteScraper(
        base_url="https://hoymilonga.com",
        output_dir=SNAPSHOT_DIR,
        max_pages=40,        # Höj vid behov
        delay_seconds=1.5,   # Var snäll mot servern
    )
    manifest = scraper.scrape()
else:
    log.info(f"Snapshot för {SNAPSHOT_DATE} finns redan — läser in befintlig.")
    with open(SNAPSHOT_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)["pages"]

print(f"\\nTotalt {len(manifest)} sidor i snapshot {SNAPSHOT_DATE}")
''')

code('''# === Snabb översikt: vad har vi för text? ===
# Sanity check innan vi bygger embeddings — är texterna rimliga?

if manifest:
    total_chars = sum(p["char_count"] for p in manifest)
    print(f"Totalt innehåll: {total_chars:,} tecken (~{total_chars//5:,} ord)")
    print(f"\\nLängsta sidor:")
    for page in sorted(manifest, key=lambda p: -p["char_count"])[:5]:
        print(f"  {page['char_count']:>6} tecken — {page['url']}")
else:
    print("⚠️ Inga sidor scrapade — kontrollera nätverk och Hoymilonga.com:s robots.txt.")
''')


# =============================================================================
# KAPITEL 4 — CHUNKING
# =============================================================================
md("""## 3. Chunking — dela upp dokumenten

### Varför chunkning?

Ett LLM-anrop har en begränsad kontextfönster (t.ex. 128 000 tokens), och dessutom blir retrieval *bättre* när vi söker efter små, fokuserade textbitar än hela långa sidor. För få chunks → vi missar information. För många chunks → vi får brus och högre kostnad.

### Strategier vi jämför

| Strategi | Beskrivning | För/nackdelar |
|---|---|---|
| **Fixed-size** | Klipp varje N tecken | Enkelt, men klipper mitt i meningar |
| **Recursive char splitter** | Försök i tur och ordning bryta vid `\\n\\n`, `\\n`, `. `, ` ` | Behåller naturliga gränser — vårt val |
| **Semantic chunking** | Klipp där meningen ändrar sig | Bäst kvalitet, men kräver embeddings vid chunkning |

Vi väljer **recursive character splitter** med **överlapp** (de sista 100 tecknen från en chunk läggs i början av nästa) så att information som råkar hamna vid en gräns inte tappas.
""")

code('''def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Delar text i bitar om max ~chunk_size tecken med överlapp.

    Algoritm:
    1. Är texten redan kort nog? Returnera den.
    2. Annars: prova separatorerna i ordning. Den första som finns
       i texten används för att bryta upp den i delar.
    3. Slå ihop delar tills de fyller en chunk, börja en ny chunk när
       nästa del skulle göra chunken för stor.
    4. Om någon del fortfarande är för stor, rekursera med nästa
       (mer finkorniga) separator.
    """
    if separators is None:
        separators = ["\\n\\n", "\\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Hitta första separatorn som faktiskt förekommer
    separator = next((s for s in separators if s and s in text), "")

    if separator == "":
        # Sista utvägen: hård split med överlapp
        return [text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size - overlap)]

    parts = text.split(separator)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + separator + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # Om en enskild del fortfarande är för stor, rekursera
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(separator) + 1:]
                chunks.extend(recursive_chunk(part, chunk_size, overlap,
                                              remaining_seps))
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    # Lägg till överlapp mellan på varandra följande chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            overlapped.append(prev_tail + " " + chunks[i])
        return overlapped

    return chunks
''')

code('''# === Chunka alla sidor från snapshot ===
# Vi behåller metadata (URL, titel) för varje chunk så att vi senare
# kan visa korrekta källcitat i chatboten.

all_chunks: List[str] = []
all_metadatas: List[Dict[str, Any]] = []
all_ids: List[str] = []

for page_meta in manifest:
    page_file = SNAPSHOT_DIR / page_meta["file"]
    with open(page_file, encoding="utf-8") as f:
        page = json.load(f)

    chunks = recursive_chunk(page["text"], chunk_size=500, overlap=100)

    for i, chunk in enumerate(chunks):
        chunk_id = f"{page['id']}_{i:03d}"
        all_chunks.append(chunk)
        all_metadatas.append({
            "url": page["url"],
            "title": page["title"][:100],
            "page_id": page["id"],
            "chunk_index": i,
            "snapshot_date": SNAPSHOT_DATE,
        })
        all_ids.append(chunk_id)

print(f"Skapade {len(all_chunks)} chunks från {len(manifest)} sidor.")
if all_chunks:
    avg = sum(len(c) for c in all_chunks) / len(all_chunks)
    print(f"Genomsnittlig chunk-längd: {avg:.0f} tecken")
    print(f"\\nExempel på chunk:")
    print("-" * 60)
    print(all_chunks[0][:400] + ("..." if len(all_chunks[0]) > 400 else ""))
''')


# =============================================================================
# KAPITEL 5 — EMBEDDINGS & CHROMADB
# =============================================================================
md("""## 4. Embeddings & vektordatabas

### Vad gör en embedding-modell?

En **embedding** omvandlar text till en lista med tal (en *vektor* — ofta 384 eller 768 dimensioner). Texter med liknande betydelse får liknande vektorer. Det är magin som låter oss hitta dokument baserat på *betydelse*, inte bara nyckelord.

```
"När är nästa milonga?" → [0.12, -0.43, 0.88, ..., 0.05]
                                    ↓
                            cosine-likhet = 0.91
                                    ↑
"Nästa tangokväll i staden" → [0.10, -0.41, 0.85, ..., 0.07]
```

### Modellval: `paraphrase-multilingual-MiniLM-L12-v2`

Vi använder **Sentence-Transformers** (lokal, gratis). Just denna modell är vald för att:

* **Multilingual** — Hoymilonga är på spanska/engelska, ibland blandat
* **Liten** (~118 MB) — laddas snabbt, kör på vilken laptop som helst
* **Snabb** — bra throughput utan GPU
* **Tillräckligt bra** för domänspecifik sökning

### ChromaDB som vektordatabas

ChromaDB är en lokal vektor-DB som:

* Persisterar automatiskt till disk
* Stödjer metadata-filtrering (vi kan t.ex. söka bara i en viss snapshot)
* Har ett enkelt Python-API
* Inte kräver någon extern server

För större produktion (miljoner dokument) hade Pinecone, Weaviate eller Qdrant varit alternativ.
""")

code('''import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Tunn wrapper runt ChromaDB + Sentence-Transformers.

    Vi separerar embeddings-modellen från databasen så att vi enkelt
    kan byta modell senare (t.ex. till en större eller domänspecifik).
    """

    EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "hoymilonga",
        embed_model_name: Optional[str] = None,
    ):
        # 1. Persistent klient — sparar automatiskt till disk
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # 2. Hämta eller skapa kollektionen.
        #    metadata={"hnsw:space": "cosine"} = använd cosine-likhet
        #    (default är L2-distans, men cosine är vanligast för text).
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 3. Embedder
        self.embedder = SentenceTransformer(
            embed_model_name or self.EMBED_MODEL_NAME
        )
        log.info(f"VectorStore redo: {self.collection.count()} dokument indexerade.")

    def add(self, chunks, metadatas, ids, batch_size: int = 64):
        """Lägger till nya chunks. Skippar IDs som redan finns."""
        # Filtrera bort befintliga IDs (idempotent operation)
        existing = set(self.collection.get(ids=ids)["ids"])
        new_indices = [i for i, _id in enumerate(ids) if _id not in existing]

        if not new_indices:
            log.info("Alla chunks finns redan i databasen — hoppar över.")
            return

        log.info(f"Indexerar {len(new_indices)} nya chunks i batchar om {batch_size}...")
        for start in tqdm(range(0, len(new_indices), batch_size),
                          desc="Embeddings"):
            batch = new_indices[start:start + batch_size]
            batch_chunks = [chunks[i] for i in batch]
            batch_metas = [metadatas[i] for i in batch]
            batch_ids = [ids[i] for i in batch]

            embeddings = self.embedder.encode(
                batch_chunks,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            self.collection.add(
                documents=batch_chunks,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids,
            )

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Hämtar de top_k mest relevanta chunks för en fråga."""
        embedding = self.embedder.encode([question]).tolist()
        return self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
        )

    def count(self) -> int:
        return self.collection.count()
''')

code('''# === Bygg vektordatabasen ===
# Första gången tar detta några minuter (modellen laddas ner + alla
# embeddings beräknas). Andra gången är det näst intill momentant
# tack vare PersistentClient + idempotent add().

vector_store = VectorStore(
    persist_dir=str(CHROMA_DIR),
    collection_name="hoymilonga",
)

vector_store.add(all_chunks, all_metadatas, all_ids)

print(f"\\nVektordatabasen innehåller nu {vector_store.count()} chunks.")
''')

code('''# === Sanity check: gör en testsökning ===
test_question = "Vad är Hoymilonga?"
results = vector_store.query(test_question, top_k=3)

print(f"Top-3 träffar för: '{test_question}'\\n")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
)):
    similarity = 1 - dist  # Cosine-distans → likhet
    print(f"[{i+1}] Likhet: {similarity:.3f} | {meta['url']}")
    print(f"    {doc[:200]}...\\n")
''')


# =============================================================================
# KAPITEL 6 — HYBRID RETRIEVAL
# =============================================================================
md("""## 5. Hybrid retrieval — vektorsökning + BM25

### Varför inte bara vektorsökning?

Vektorsökning är bra på **semantisk likhet**, men den missar ibland exakta termer (siffror, egennamn, datum). BM25 är ett klassiskt nyckelords-baserat ranking-mått som är *utmärkt* på exakta matchningar.

**Hybrid retrieval** kombinerar båda och tar det bästa från två världar. Vi använder en α-parameter som väger dem:

```
score = α · vector_score + (1 - α) · bm25_score
```

* α = 1.0 → ren vektorsökning
* α = 0.5 → balanserad
* α = 0.0 → ren BM25

För en sajt som Hoymilonga med många egennamn (städer, organisatörer) brukar α ≈ 0.5–0.7 fungera bäst.
""")

code('''from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Kombinerar vektorsökning (semantisk) med BM25 (nyckelords) genom
    en viktad linjär kombination av poängen.

    Notera: vi normaliserar båda poängen till [0, 1] innan vi kombinerar
    så att α faktiskt har den betydelse vi tänker.
    """

    def __init__(self, vector_store, all_chunks, all_metadatas):
        self.vector_store = vector_store
        self.chunks = all_chunks
        self.metadatas = all_metadatas

        # BM25 förväntar sig pre-tokeniserad text
        tokenized = [self._tokenize(c) for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Enkelt språk-agnostiskt tokeniseringsgrepp: gemener + alfanumeriskt."""
        return re.findall(r"\\w+", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.6,
        oversample: int = 3,
    ) -> Dict[str, Any]:
        """
        Hämtar top_k chunks med hybrid scoring.

        oversample: vi hämtar oversample*top_k från vektorsökningen så att
                    BM25 kan flytta runt resultaten även om vektorsökningen
                    inte hade dem precis i topp.
        """
        # === 1. Vektorsökning ===
        vec_results = self.vector_store.query(query, top_k=top_k * oversample)
        vec_docs = vec_results["documents"][0]
        vec_metas = vec_results["metadatas"][0]
        vec_dists = vec_results["distances"][0]

        # Cosine-distans → likhet, sedan normalisera till [0, 1]
        vec_sims = [1 - d for d in vec_dists]
        max_sim = max(vec_sims) if vec_sims else 1.0
        min_sim = min(vec_sims) if vec_sims else 0.0
        denom = (max_sim - min_sim) or 1.0

        candidate_scores: Dict[int, Dict[str, Any]] = {}
        for doc, meta, sim in zip(vec_docs, vec_metas, vec_sims):
            # Hitta global index på chunk
            try:
                idx = self.chunks.index(doc)
            except ValueError:
                continue
            normalized = (sim - min_sim) / denom
            candidate_scores[idx] = {
                "doc": doc,
                "metadata": meta,
                "vec_score": normalized,
                "bm25_score": 0.0,
            }

        # === 2. BM25-sökning över ALLA chunks ===
        # (BM25 är så billig att vi kan köra över hela korpus)
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        # Lägg till BM25-poäng på befintliga kandidater + topp-BM25-träffar
        # som vektorsökningen kanske missade
        top_bm25_indices = sorted(range(len(bm25_scores)),
                                  key=lambda i: -bm25_scores[i])[:top_k * oversample]
        for idx in top_bm25_indices:
            normalized = bm25_scores[idx] / max_bm25
            if idx not in candidate_scores:
                candidate_scores[idx] = {
                    "doc": self.chunks[idx],
                    "metadata": self.metadatas[idx],
                    "vec_score": 0.0,
                    "bm25_score": normalized,
                }
            else:
                candidate_scores[idx]["bm25_score"] = normalized

        # === 3. Kombinera och rangordna ===
        for idx, info in candidate_scores.items():
            info["combined_score"] = (
                alpha * info["vec_score"] + (1 - alpha) * info["bm25_score"]
            )

        ranked = sorted(candidate_scores.values(),
                        key=lambda x: -x["combined_score"])[:top_k]

        return {
            "documents": [[r["doc"] for r in ranked]],
            "metadatas": [[r["metadata"] for r in ranked]],
            "scores": [[r["combined_score"] for r in ranked]],
            "vec_scores": [[r["vec_score"] for r in ranked]],
            "bm25_scores": [[r["bm25_score"] for r in ranked]],
        }
''')

code('''# === Skapa hybrid retriever och testa ===
retriever = HybridRetriever(vector_store, all_chunks, all_metadatas)

test_query = "How can I find milongas in a specific city?"
hybrid_results = retriever.retrieve(test_query, top_k=3, alpha=0.6)

print(f"Hybrid-resultat för: '{test_query}'\\n")
for i in range(len(hybrid_results["documents"][0])):
    score = hybrid_results["scores"][0][i]
    vec = hybrid_results["vec_scores"][0][i]
    bm25 = hybrid_results["bm25_scores"][0][i]
    meta = hybrid_results["metadatas"][0][i]
    doc = hybrid_results["documents"][0][i]
    print(f"[{i+1}] kombinerad={score:.3f} (vec={vec:.3f}, bm25={bm25:.3f})")
    print(f"    {meta['url']}")
    print(f"    {doc[:200]}...\\n")
''')


# =============================================================================
# KAPITEL 7 — LLM-ABSTRAKTION
# =============================================================================
md("""## 6. LLM-abstraktion — ett gränssnitt för tre leverantörer

### Designmönster: Strategi (Strategy Pattern)

Vi vill kunna byta LLM-leverantör utan att ändra resten av koden. Lösning: definiera ett abstrakt gränssnitt `LLMProvider` med metoden `generate()`, och implementera tre konkreta klasser:

* `OpenAIProvider` — GPT-4o, GPT-4o-mini m.fl.
* `AnthropicProvider` — Claude Opus, Sonnet, Haiku
* `OllamaProvider` — lokala modeller (Llama 3, Mistral m.fl.)

Detta är **idiomatisk Python** och en mycket vanlig arkitektur i produktions-RAG-system.
""")

code('''class LLMProvider(ABC):
    """Abstrakt basklass — kontraktet för alla LLM-leverantörer."""

    name: str = "abstract"

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generera ett svar givet system-prompt + användarmeddelande."""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI-modeller via openai-paketet."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()  # Läser OPENAI_API_KEY från miljövariabel
        self.model = model

    def generate(self, system_prompt, user_message,
                 temperature=0.2, max_tokens=1024):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude-modeller."""

    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5"):
        import anthropic
        self.client = anthropic.Anthropic()  # Läser ANTHROPIC_API_KEY
        self.model = model

    def generate(self, system_prompt, user_message,
                 temperature=0.2, max_tokens=1024):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class OllamaProvider(LLMProvider):
    """Lokal körning via Ollama (https://ollama.com)."""

    name = "ollama"

    def __init__(self, model: str = "llama3.1:8b",
                 base_url: Optional[str] = None):
        self.model = model
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

    def generate(self, system_prompt, user_message,
                 temperature=0.2, max_tokens=1024):
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def get_llm_provider(name: str, model: Optional[str] = None) -> LLMProvider:
    """Factory-funktion — instantierar rätt provider baserat på namn."""
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if name == "anthropic":
        return AnthropicProvider(model=model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"))
    if name == "ollama":
        return OllamaProvider(model=model or os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    raise ValueError(f"Okänd LLM-leverantör: {name}")
''')

code('''# === Snabbtest av aktiv leverantör ===
# Vi väljer leverantör baserat på .env-inställning. Du kan ändra här
# direkt om du vill testa en annan.

ACTIVE_PROVIDER_NAME = os.getenv("LLM_PROVIDER", "openai")

try:
    llm = get_llm_provider(ACTIVE_PROVIDER_NAME)
    test_response = llm.generate(
        system_prompt="Du är en hjälpsam assistent. Svara mycket kort.",
        user_message="Säg 'Hej världen!' på svenska.",
        temperature=0.0,
        max_tokens=50,
    )
    print(f"✅ {ACTIVE_PROVIDER_NAME} fungerar: {test_response}")
except Exception as exc:
    print(f"❌ {ACTIVE_PROVIDER_NAME} misslyckades: {exc}")
    print("   Kontrollera API-nyckel i .env eller att Ollama körs.")
''')


# =============================================================================
# KAPITEL 8 — RAG PIPELINE
# =============================================================================
md("""## 7. RAG-pipeline med guardrails och citat

Nu sätter vi ihop allt: retrieval + LLM + en system-prompt som säkerställer att

1. Svaret **håller sig till kontexten** (minskar hallucination)
2. Botten **säger "vet inte"** när informationen saknas
3. Botten **citerar källor** med `[Källa N]`
4. Botten **vägrar off-topic** (en enkel guardrail)
""")

code('''SYSTEM_PROMPT = """Du är Hoymilonga-assistenten — en hjälpsam chatbot som svarar på frågor om webbsidan Hoymilonga.com (en plattform för tangoevenemang/milongor).

REGLER (måste följas):
1. Svara ENDAST baserat på informationen i KONTEXT-sektionen nedan. Hitta inte på fakta.
2. Om informationen inte räcker för att besvara frågan, svara: "Jag hittar inte den informationen i mina källor om Hoymilonga."
3. Citera alltid källan med [Källa N] när du gör ett påstående baserat på kontexten.
4. Om frågan inte har något med Hoymilonga, tango eller milongor att göra, svara artigt: "Jag är specialiserad på Hoymilonga och tangoevenemang. Den frågan ligger utanför mitt område."
5. Svara på samma språk som frågan ställdes på (svenska, engelska eller spanska).
6. Var koncis (max 5 meningar) men informativ.
"""


class RAGPipeline:
    """
    Den fullständiga RAG-pipelinen med hybrid retrieval, guardrails
    och källcitat.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_provider: LLMProvider,
        top_k: int = 5,
        alpha: float = 0.6,
    ):
        self.retriever = retriever
        self.llm = llm_provider
        self.top_k = top_k
        self.alpha = alpha

    def _build_context(self, retrieved: Dict[str, Any]) -> tuple[str, list[str]]:
        """Formaterar de hämtade chunks som en numrerad kontext-sträng."""
        parts: list[str] = []
        sources: list[str] = []
        for i, (doc, meta) in enumerate(zip(
            retrieved["documents"][0],
            retrieved["metadatas"][0],
        )):
            url = meta["url"]
            sources.append(url)
            parts.append(f"[Källa {i+1}] ({url})\\n{doc}")
        return "\\n\\n".join(parts), sources

    def answer(self, question: str) -> Dict[str, Any]:
        """Huvudmetod — tar en fråga, returnerar svar + metadata."""
        # 1. Retrieval
        retrieved = self.retriever.retrieve(
            question, top_k=self.top_k, alpha=self.alpha
        )
        context, sources = self._build_context(retrieved)

        # 2. Bygg user message med kontext
        user_message = f"""KONTEXT:
{context}

FRÅGA: {question}

SVAR (citera källor med [Källa N]):"""

        # 3. Generera svar
        answer_text = self.llm.generate(SYSTEM_PROMPT, user_message,
                                        temperature=0.2)

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved["documents"][0],
            "retrieval_scores": retrieved["scores"][0],
            "provider": self.llm.name,
        }
''')

code('''# === Demo: ställ en fråga genom hela pipelinen ===
rag = RAGPipeline(retriever=retriever, llm_provider=llm, top_k=5, alpha=0.6)

demo_questions = [
    "What is Hoymilonga?",
    "Hur kan jag hitta milongor i en viss stad?",
    "Vad är receptet på pasta carbonara?",  # Off-topic — ska avvisas
]

for q in demo_questions:
    print(f"\\n{'=' * 70}")
    print(f"FRÅGA: {q}")
    print("=" * 70)
    try:
        result = rag.answer(q)
        print(f"SVAR:\\n{result['answer']}")
        print(f"\\nKÄLLOR: {len(set(result['sources']))} unika")
        for src in sorted(set(result["sources"]))[:3]:
            print(f"  - {src}")
    except Exception as exc:
        print(f"Fel: {exc}")
''')


# =============================================================================
# KAPITEL 9 — STREAMLIT-APP
# =============================================================================
md("""## 8. Streamlit-app — webbgränssnittet

Vi skriver ut all kärnlogik (klasser och factory-funktion) till en separat fil `rag_core.py` så att Streamlit-appen kan importera den. Detta är **best practice** — notebook för dokumentation, `.py`-fil för produktion.

Använd `%%writefile`-magic för att skriva direkt från cellen till disk.
""")

code('''# === Skriv ut kärnlogik till rag_core.py så Streamlit-appen kan importera ===
# Vi använder %%writefile-magic för att rendera ut hela modulen.

CORE_MODULE_SOURCE = \'\'\'"""
rag_core.py — Kärnklasser för Hoymilonga RAG (genererat av notebooken).

Denna fil duplicerar inte kod — den ÅTERANVÄNDER VectorStore, HybridRetriever,
RAGPipeline och get_llm_provider precis som de definierats i notebooken.

Vid uppdatering: kör notebookens kapitel 8-cell igen för att skriva om filen.
"""
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
import requests

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ----- VectorStore -----
class VectorStore:
    EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, persist_dir, collection_name="hoymilonga",
                 embed_model_name=None):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(
            embed_model_name or self.EMBED_MODEL_NAME
        )

    def query(self, question, top_k=5):
        embedding = self.embedder.encode([question]).tolist()
        return self.collection.query(query_embeddings=embedding, n_results=top_k)

    def count(self):
        return self.collection.count()


def load_chunks_for_bm25(vector_store):
    """Hämta alla chunks ur ChromaDB för att bygga BM25-index."""
    data = vector_store.collection.get()
    return data["documents"], data["metadatas"]


# ----- HybridRetriever -----
class HybridRetriever:
    def __init__(self, vector_store, all_chunks, all_metadatas):
        self.vector_store = vector_store
        self.chunks = all_chunks
        self.metadatas = all_metadatas
        tokenized = [self._tokenize(c) for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\\\\w+", text.lower())

    def retrieve(self, query, top_k=5, alpha=0.6, oversample=3):
        vec_results = self.vector_store.query(query, top_k=top_k * oversample)
        vec_docs = vec_results["documents"][0]
        vec_metas = vec_results["metadatas"][0]
        vec_dists = vec_results["distances"][0]

        vec_sims = [1 - d for d in vec_dists]
        max_sim = max(vec_sims) if vec_sims else 1.0
        min_sim = min(vec_sims) if vec_sims else 0.0
        denom = (max_sim - min_sim) or 1.0

        candidate_scores = {}
        for doc, meta, sim in zip(vec_docs, vec_metas, vec_sims):
            try:
                idx = self.chunks.index(doc)
            except ValueError:
                continue
            candidate_scores[idx] = {
                "doc": doc, "metadata": meta,
                "vec_score": (sim - min_sim) / denom,
                "bm25_score": 0.0,
            }

        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        top_bm25_idx = sorted(range(len(bm25_scores)),
                              key=lambda i: -bm25_scores[i])[:top_k * oversample]
        for idx in top_bm25_idx:
            normalized = bm25_scores[idx] / max_bm25
            if idx not in candidate_scores:
                candidate_scores[idx] = {
                    "doc": self.chunks[idx],
                    "metadata": self.metadatas[idx],
                    "vec_score": 0.0,
                    "bm25_score": normalized,
                }
            else:
                candidate_scores[idx]["bm25_score"] = normalized

        for info in candidate_scores.values():
            info["combined_score"] = (
                alpha * info["vec_score"] + (1 - alpha) * info["bm25_score"]
            )

        ranked = sorted(candidate_scores.values(),
                        key=lambda x: -x["combined_score"])[:top_k]
        return {
            "documents": [[r["doc"] for r in ranked]],
            "metadatas": [[r["metadata"] for r in ranked]],
            "scores": [[r["combined_score"] for r in ranked]],
            "vec_scores": [[r["vec_score"] for r in ranked]],
            "bm25_scores": [[r["bm25_score"] for r in ranked]],
        }


# ----- LLM-leverantörer -----
class LLMProvider(ABC):
    name = "abstract"
    @abstractmethod
    def generate(self, system_prompt, user_message,
                 temperature=0.2, max_tokens=1024): ...


class OpenAIProvider(LLMProvider):
    name = "openai"
    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
    def generate(self, system_prompt, user_message,
                 temperature=0.2, max_tokens=1024):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_message}],
            temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    def __init__(self, model="claude-haiku-4-5"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
    def generate(self, system_prompt, user_message,
                 temperature=0.2, max_tokens=1024):
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}])
        return r.content[0].text


class OllamaProvider(LLMProvider):
    name = "ollama"
    def __init__(self, model="llama3.1:8b", base_url=None):
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    def generate(self, system_prompt, user_message,
                 temperature=0.2, max_tokens=1024):
        r = requests.post(f"{self.base_url}/api/chat", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_message}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]


def get_llm_provider(name, model=None):
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if name == "anthropic":
        return AnthropicProvider(model=model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"))
    if name == "ollama":
        return OllamaProvider(model=model or os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    raise ValueError(f"Okänd leverantör: {name}")


# ----- RAGPipeline -----
SYSTEM_PROMPT = """Du är Hoymilonga-assistenten — en hjälpsam chatbot som svarar på frågor om webbsidan Hoymilonga.com (en plattform för tangoevenemang/milongor).

REGLER (måste följas):
1. Svara ENDAST baserat på informationen i KONTEXT-sektionen nedan. Hitta inte på fakta.
2. Om informationen inte räcker för att besvara frågan, svara: "Jag hittar inte den informationen i mina källor om Hoymilonga."
3. Citera alltid källan med [Källa N] när du gör ett påstående baserat på kontexten.
4. Om frågan inte har något med Hoymilonga, tango eller milongor att göra, svara artigt: "Jag är specialiserad på Hoymilonga och tangoevenemang. Den frågan ligger utanför mitt område."
5. Svara på samma språk som frågan ställdes på (svenska, engelska eller spanska).
6. Var koncis (max 5 meningar) men informativ.
"""


class RAGPipeline:
    def __init__(self, retriever, llm_provider, top_k=5, alpha=0.6):
        self.retriever = retriever
        self.llm = llm_provider
        self.top_k = top_k
        self.alpha = alpha

    def _build_context(self, retrieved):
        parts, sources = [], []
        for i, (doc, meta) in enumerate(zip(retrieved["documents"][0],
                                            retrieved["metadatas"][0])):
            sources.append(meta["url"])
            parts.append(f"[Källa {i+1}] ({meta[\\\'url\\\']})\\\\n{doc}")
        return "\\\\n\\\\n".join(parts), sources

    def answer(self, question):
        retrieved = self.retriever.retrieve(
            question, top_k=self.top_k, alpha=self.alpha)
        context, sources = self._build_context(retrieved)
        user_message = (f"KONTEXT:\\\\n{context}\\\\n\\\\n"
                        f"FRÅGA: {question}\\\\n\\\\n"
                        "SVAR (citera källor med [Källa N]):")
        answer_text = self.llm.generate(SYSTEM_PROMPT, user_message,
                                        temperature=0.2)
        return {
            "question": question, "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved["documents"][0],
            "retrieval_scores": retrieved["scores"][0],
            "provider": self.llm.name,
        }
\'\'\'

with open(PROJECT_ROOT / "rag_core.py", "w", encoding="utf-8") as f:
    f.write(CORE_MODULE_SOURCE)

print(f"✅ Skrev rag_core.py ({len(CORE_MODULE_SOURCE)} tecken)")
print("Starta nu Streamlit i en separat terminal: streamlit run streamlit_app.py")
''')


# =============================================================================
# KAPITEL 10 — EVALUERING
# =============================================================================
md("""## 9. Evaluering — hur bra är vår chatbot?

För VG-betyg krävs ett **systematiskt utvärderingssystem**. Vi mäter på två nivåer:

### Retrieval-metriker (hämtar vi RÄTT dokument?)

* **Hit Rate@k** — andelen frågor där minst ett av de förväntade källdokumenten hamnar i top-k
* **MRR (Mean Reciprocal Rank)** — genomsnittet av 1/rank för första relevanta träffen. Belönar att rätt dokument hamnar TIDIGT

### Generation-metriker (är SVARET bra?)

* **LLM-as-judge** — vi låter en stark LLM jämföra det genererade svaret mot facit och ge ett betyg 1–5
* **Refusal accuracy** — för off-topic-frågor: vägrar botten korrekt?
* **Source citation rate** — citerar svaret alltid en `[Källa N]`?

Vi laddar in vårt **golden dataset** (12 noggrant utvalda frågor i `eval/golden_dataset.json`) och kör utvärderingen.
""")

code('''# === Ladda golden dataset ===
with open(EVAL_DIR / "golden_dataset.json", encoding="utf-8") as f:
    gold = json.load(f)

examples = gold["examples"]
print(f"Laddade {len(examples)} utvärderingsexempel")
print(f"Kategorier: {set(e['category'] for e in examples)}")
''')

code('''def evaluate_retrieval(retriever, examples, top_k=5, alpha=0.6):
    """
    Beräknar Hit Rate och MRR baserat på expected_source_keywords.

    En träff räknas om någon av de förväntade nyckelorden förekommer
    i någon av top-k källornas URL eller text.
    """
    hits, reciprocal_ranks = [], []

    for ex in examples:
        if ex["should_refuse"]:
            continue  # Off-topic — retrieval är inte intressant
        if not ex["expected_source_keywords"]:
            continue

        results = retriever.retrieve(ex["question"], top_k=top_k, alpha=alpha)
        urls = [m["url"].lower() for m in results["metadatas"][0]]
        docs = [d.lower() for d in results["documents"][0]]

        rank, found = None, False
        for i, (url, doc) in enumerate(zip(urls, docs), 1):
            for kw in ex["expected_source_keywords"]:
                if kw.lower() in url or kw.lower() in doc:
                    rank, found = i, True
                    break
            if found:
                break

        hits.append(1 if found else 0)
        reciprocal_ranks.append(1 / rank if rank else 0)

    return {
        "hit_rate": sum(hits) / len(hits) if hits else 0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
        "n_evaluated": len(hits),
    }


# === Kör retrieval-evaluering ===
ret_metrics = evaluate_retrieval(retriever, examples, top_k=5, alpha=0.6)
print("Retrieval-metriker (top_k=5, α=0.6):")
print(f"  Hit Rate: {ret_metrics['hit_rate']:.2%}")
print(f"  MRR:      {ret_metrics['mrr']:.3f}")
print(f"  N:        {ret_metrics['n_evaluated']} frågor")
''')

code('''def evaluate_generation(rag_pipeline, judge_provider, examples):
    """
    Kör hela RAG-pipelinen på varje fråga, låt en LLM-domare betygsätta
    svaren, och kontrollera även:
    - Refusal accuracy (off-topic-frågor)
    - Citation rate (innehåller svaret [Källa N]?)
    """
    judge_prompt = """Du är en strikt utvärderare. Givet en FRÅGA, ett FACIT-SVAR och ett GENERERAT SVAR, bedöm hur väl det genererade svaret matchar facit.

BETYGSSKALA:
5 = Korrekt och täcker facit-svarets nyckelinformation
4 = Mestadels korrekt med små brister
3 = Delvis korrekt
2 = Mestadels felaktigt
1 = Helt felaktigt eller hallucination

Svara ENDAST med en siffra 1-5, inget annat."""

    rows = []
    for ex in tqdm(examples, desc="Genereringsutvärdering"):
        try:
            result = rag_pipeline.answer(ex["question"])
        except Exception as exc:
            log.error(f"Fel för fråga '{ex['question']}': {exc}")
            continue

        gen_answer = result["answer"]

        # 1. LLM-as-judge: hur bra matchar svaret facit?
        judge_input = (f"FRÅGA: {ex['question']}\\n"
                       f"FACIT: {ex['expected_answer']}\\n"
                       f"GENERERAT: {gen_answer}\\n\\nBETYG (1-5):")
        try:
            judge_response = judge_provider.generate(
                judge_prompt, judge_input, temperature=0.0, max_tokens=10)
            digits = re.findall(r"[1-5]", judge_response)
            score = int(digits[0]) if digits else 0
        except Exception as exc:
            log.warning(f"Judge-fel: {exc}")
            score = 0

        # 2. Refusal-kontroll för off-topic-frågor
        refused = any(phrase in gen_answer.lower() for phrase in [
            "utanför mitt område", "specialiserad", "kan inte svara",
            "outside my", "specialized", "i hittar inte",
        ])
        refusal_correct = (
            (ex["should_refuse"] and refused)
            or (not ex["should_refuse"] and not refused)
        )

        # 3. Citation rate
        has_citation = bool(re.search(r"\\[Källa \\d+\\]|\\[Source \\d+\\]",
                                       gen_answer))

        rows.append({
            "id": ex["id"],
            "category": ex["category"],
            "question": ex["question"],
            "expected": ex["expected_answer"],
            "generated": gen_answer,
            "judge_score": score,
            "should_refuse": ex["should_refuse"],
            "refused": refused,
            "refusal_correct": refusal_correct,
            "has_citation": has_citation,
            "n_sources": len(set(result.get("sources", []))),
        })

    return pd.DataFrame(rows)


# === Kör genereringsutvärdering ===
# Vi använder samma LLM som domare som vår genererare. För strängare
# evaluering kan man använda en annan/starkare modell som domare.
gen_df = evaluate_generation(rag, judge_provider=llm, examples=examples)

# Spara rådata
gen_df.to_csv(EVAL_DIR / "results.csv", index=False, encoding="utf-8")
print(f"\\nResultat sparade till {EVAL_DIR / 'results.csv'}")
gen_df[["id", "category", "judge_score", "refusal_correct", "has_citation"]]
''')

code('''# === Sammanställning av alla mått ===

# Bara examples där LLM faktiskt skulle producera ett substansrikt svar
on_topic = gen_df[~gen_df["should_refuse"]]

summary = {
    "Retrieval — Hit Rate":  f"{ret_metrics['hit_rate']:.1%}",
    "Retrieval — MRR":       f"{ret_metrics['mrr']:.3f}",
    "Generation — snittbetyg (1-5)":
        f"{on_topic['judge_score'].mean():.2f}",
    "Generation — andel ≥4":
        f"{(on_topic['judge_score'] >= 4).mean():.1%}",
    "Refusal accuracy":
        f"{gen_df['refusal_correct'].mean():.1%}",
    "Citation rate (on-topic)":
        f"{on_topic['has_citation'].mean():.1%}",
}

print("=" * 50)
print("SAMMANFATTNING — UTVÄRDERINGSRESULTAT")
print("=" * 50)
for k, v in summary.items():
    print(f"{k:35s} {v}")
print("=" * 50)
''')

code('''# === Bonus: jämför olika alpha-värden för hybrid retrieval ===
# Detta är ett klassiskt VG-grepp — visa att du kan SYSTEMATISKT optimera
# hyperparametrar och motivera ditt val med data.

alpha_results = []
for a in [0.0, 0.25, 0.5, 0.6, 0.75, 1.0]:
    m = evaluate_retrieval(retriever, examples, top_k=5, alpha=a)
    alpha_results.append({"alpha": a, **m})

alpha_df = pd.DataFrame(alpha_results)
print("Hyperparameter-sökning för α (vikt vektor vs BM25):")
print(alpha_df.to_string(index=False))
print(f"\\nBästa α för Hit Rate: {alpha_df.loc[alpha_df['hit_rate'].idxmax(), 'alpha']}")
print(f"Bästa α för MRR:      {alpha_df.loc[alpha_df['mrr'].idxmax(), 'alpha']}")
''')


# =============================================================================
# KAPITEL 11 — SLUTREFLEKTION
# =============================================================================
md("""## 10. Slutreflektion — verkliga användningsfall, etik & affärsmöjligheter

> *"I slutet av din kod ska du redogöra för hur din modell hade kunnat användas i verkligheten och vilka potentiella utmaningar och möjligheter (tex affärsmässiga, etiska och andra perspektiv du finner relevanta) som finns."*

### 10.1 Verkliga användningsfall

**A. För Hoymilonga själva (B2B / intern tjänst)**

* **Kundsupport-chatbot** på sajten — minskar tryck på supportmail med ~30–50 % enligt branschstudier (Intercom, Zendesk).
* **Onboarding för nya användare** — interaktiv guide som svarar på "hur skapar jag ett konto?" eller "hur publicerar jag mitt första evenemang?".
* **Intern wiki-sökning** — om Hoymilonga har en intern kunskapsbas (Notion/Confluence) kan samma arkitektur appliceras där.
* **Automatisk kvalitetskontroll** — låt botten flagga evenemangsbeskrivningar som saknar nyckelinformation (datum, plats, pris).

**B. Generaliserat — samma arkitektur i andra branscher**

| Domän | Användningsfall |
|---|---|
| **Utbildning** | Studieassistent över kursmaterial och föreläsningsanteckningar |
| **Juridik** | Sökmotor över rättsfall och avtal — med spårbara citat (kritiskt!) |
| **Sjukvård** | Klinisk beslutsstöd över riktlinjer (men HÖGA krav på validering) |
| **E-handel** | Produktsupport över FAQ + manualer |
| **HR** | Anställdhandbok + policy-frågor (semestrar, försäkringar) |
| **Finans** | Internal analystassistent över rapporter och researchnotat |

**C. För användaren av Hoymilonga**

* **Naturligt språk-sökning** istället för att klicka i menyer: *"Visa kommande nybörjarmilongor i Stockholm med entré under 100 kr."*
* **Personlig rekommendation** kombinerat med användarprofil: *"Vad rekommenderar du i helgen?"*
* **Flerspråkigt stöd** — en spansk användare i Sverige får svar på spanska om svenska milongor.

### 10.2 Affärsmässiga möjligheter och utmaningar

**Möjligheter:**
* **Premium-funktion** ($X/månad) → ny intäktsström.
* **Differentiering** mot konkurrenter (många milonga-listor är fortfarande statiska sidor från 2010-talet).
* **Datatillgång** — frågorna användarna ställer är guld värda för produktutveckling. *Vad kan vi inte besvara idag? → Vad ska vi bygga härnäst?*
* **Sänkt CAC (Customer Acquisition Cost)** — bättre onboarding ger högre konvertering från besökare till aktiv användare.

**Utmaningar:**
* **API-kostnader** vid skala. 10 000 frågor/dag à $0,003/fråga = $900/månad. Inte avskräckande, men inte gratis. **Mitigation:** caching av identiska/snarlika frågor, batch-anrop, eller hybrid med lokala modeller.
* **Latens** — användarna är vana vid Google-snabba svar. Om varje fråga tar 5 sekunder förlorar man dem. **Mitigation:** streaming-svar (modellen börjar tala innan den är klar), pre-warmade embeddings.
* **Underhåll** — pipelinen måste re-indexeras när sajten ändras. Kräver schemalagda jobb och övervakning.
* **Modellberoende** — vad händer om OpenAI höjer priser eller drar tillbaka en modell? Vår leverantörsagnostiska arkitektur hanterar detta, men kostar utvecklingstid.

### 10.3 Etiska och samhälleliga perspektiv

**Hallucinationer och felaktig information**
RAG minskar men eliminerar inte hallucinationer. Om botten med självsäkerhet ger fel pris eller fel datum för ett evenemang kan en användare köra till "fel" stad. **Vår mitigation:** källcitat med klickbara länkar — användaren kan alltid verifiera. Men vi måste också i UI:t påminna om att svaren är AI-genererade.

**Bias i underliggande data**
Hoymilonga har sannolikt mer evenemang i storstäder än på mindre orter. Botten kommer alltså att vara "bättre" på Stockholm än på Mariestad. Detta riskerar att förstärka centralisering. **Reflektion:** vi bör mäta detta och eventuellt visa en disclaimer ("få evenemang hittade — sajten har begränsad data för denna region").

**Datasekretess**
* **Träningsdata:** Vi använder en publik embedding-modell — inga GDPR-frågor där.
* **Användarfrågor:** *Skickas till OpenAI/Anthropic.* Detta måste informeras om i en sekretesspolicy. **GDPR-perspektiv:** för EU-användare bör vi prefera modeller som körs i EU eller lokalt (Ollama). Detta är ett genuint argument för vår multi-leverantörs-arkitektur.

**Webscraping-etik**
Vi respekterar `robots.txt` och scrapear i låg takt — men det är en gråzon. **Bästa praxis:** kontakta sajtägaren och be om tillstånd, alternativt fråga om en officiell API-feed. Om Hoymilonga själva implementerade denna RAG hade frågan försvunnit (vi äger då datan).

**Påverkan på sysselsättning**
Om en chatbot ersätter en supportperson är det en samhällsfråga. **Pragmatisk syn:** botten är bäst på enkla, repetitiva frågor — människan frigörs för komplexa case som faktiskt kräver empati och problemlösning. Men detta kräver aktiv investering i omskolning.

**Tillgänglighet och digitalt utanförskap**
En chatbot kräver att användaren är bekväm med chatt-gränssnitt. Äldre tangodansare (en betydande grupp!) kan föredra en traditionell sökmotor. **Slutsats:** botten ska KOMPLETTERA, inte ERSÄTTA, befintligt UI.

### 10.4 Tekniska begränsningar och vidareutveckling

| Begränsning | Möjlig åtgärd |
|---|---|
| Statiska snapshots — föråldras snabbt | Schemalagd nattlig re-scraping + cache-invalidering |
| Singel-fråga, inget minne | Lägg till conversation memory (chat-historik som kontext) |
| Ingen användarpersonalisering | Lägg till user-profile som extra kontext |
| Ingen multimodal förståelse | Indexera även bilder från evenemangssidor (CLIP-embeddings) |
| Embedding-modellen är generell | Fine-tuna en domänspecifik embedder på tango-text |
| Ingen feedback-loop | Tummen upp/ner i UI:t → omträning av reranker |
| Endast textcitat | Strukturerad output: returnera namn, plats, datum som JSON |

### 10.5 Slutsats

RAG är en **mogen och praktisk arkitektur** för att förvandla statiska sajter till interaktiva assistenter. Den är inte magi — den **garanterar inte korrekthet** och **eliminerar inte etiska frågor**. Men kombinerat med god ingenjörskonst (källcitat, guardrails, monitoring och en respektfull data­insamling) kan den ge ett genuint mervärde för både Hoymilongas användare och företaget.

För att bygga vidare i en *riktig* produktsättning skulle jag prioritera (i ordning):

1. **Övervakning & loggning** av alla frågor + svar i produktion (vi loggar redan i `logs/`)
2. **Schemalagd re-indexering** så datan håller sig färsk
3. **Streaming-svar** för bättre UX
4. **Conversation memory** för naturliga dialoger
5. **A/B-test** av olika modeller/prompts mot riktiga användare

---

*Slut på notebooken. Tack för att du läste hela vägen!* 💃
""")


# =============================================================================
# BYGG NOTEBOOKEN (.ipynb-filen)
# =============================================================================
def cell_to_dict(cell_type, source):
    if cell_type == "md":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


notebook = {
    "cells": [cell_to_dict(t, s) for t, s in CELLS],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).parent / "rag_chatbot.ipynb"
with open(output, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✅ Skrev {output} med {len(CELLS)} celler")
