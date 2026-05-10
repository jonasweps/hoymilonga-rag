"""
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
        return re.findall(r"\w+", text.lower())

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
            parts.append(f"[Källa {i+1}] ({meta['url']})\n{doc}")
        return "\n\n".join(parts), sources

    def answer(self, question):
        retrieved = self.retriever.retrieve(
            question, top_k=self.top_k, alpha=self.alpha)
        context, sources = self._build_context(retrieved)
        user_message = (f"KONTEXT:\n{context}\n\n"
                        f"FRÅGA: {question}\n\n"
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
