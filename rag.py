"""Evaluated RAG for ASP.NET Core: the final retrieval + generation pipeline.

Retrieval: hybrid search (vector + BM25 keyword, merged with Reciprocal Rank Fusion),
then a cross-encoder reranker picks the best 5 of 30 candidates.
Generation: an LLM answers using only those 5 chunks, either
  - "ollama": a local model on this computer (default), or
  - "groq":   a hosted model via Groq's API (used by the live app).

Used by the evaluation notebook AND the Streamlit app, so both run exactly the same code.
"""

import os                                               # os: read settings from environment variables
import re                                               # re: text patterns (used to split text into words)
import time                                             # time: measure durations, and wait between retries
from pathlib import Path                                # Path: folders and file paths

import chromadb                                         # chromadb: the vector database
import numpy as np                                      # numpy: fast sorting of scores
import pandas as pd                                     # pandas: tables
from rank_bm25 import BM25Okapi                         # BM25Okapi: keyword search
from sentence_transformers import CrossEncoder, SentenceTransformer   # embedding model and reranker

# ---------------- Settings ----------------
ROOT = Path(__file__).resolve().parent                  # The project folder (this file lives in it), so paths work from anywhere
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                  # Must match the model used to build the database
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # Instruction this model expects in front of questions
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # The reranker
COLLECTION = "aspnetcore_docs_bge_small"                # The ChromaDB collection name

DEFAULT_MODELS = {                                      # The default LLM for each provider
    "ollama": "llama3.2:3b",                            # Local: small enough for a laptop CPU
    "groq": "openai/gpt-oss-20b",                       # Hosted: a production model on Groq
}
FORCE_CPU = os.getenv("RAG_FORCE_CPU", "0") == "1"      # Set RAG_FORCE_CPU=1 on computers whose GPU gives garbage output with Ollama

NOT_FOUND = "I couldn't find this in the documentation."   # The exact refusal sentence (the eval checks for it)

SYSTEM_PROMPT = f"""You answer questions about ASP.NET Core 10 using ONLY the documentation excerpts provided.
Rules:
- Use only information from the excerpts. Do not use outside knowledge.
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Cite the excerpts you used by number in square brackets, like [1] or [2][3].
- Be concise: a short paragraph, plus a short code snippet only if an excerpt contains one."""

OLLAMA_OPTIONS = {                                      # Generation settings for the local model
    "temperature": 0.1,                                 # Very low randomness, but enough to escape repeating loops
    "seed": 42,                                         # Fixed seed: the same question gives the same answer
    "repeat_penalty": 1.15,                             # Discourage repeating recent words
    "num_predict": 400,                                 # Maximum answer length (~300 words)
    "num_ctx": 4096,                                    # How much text the model may read at once
}
if FORCE_CPU:                                           # Only on computers where RAG_FORCE_CPU=1...
    OLLAMA_OPTIONS["num_gpu"] = 0                       # ...run the LLM entirely on the CPU

GROQ_OPTIONS = {                                        # Generation settings for the hosted model
    "temperature": 0.1,                                 # Same low randomness as locally
    "seed": 42,                                         # Same fixed seed
    "max_completion_tokens": 1024,                      # Room for the model's short internal reasoning plus the answer
    "reasoning_effort": "low",                          # gpt-oss "thinks" before answering; "low" keeps it fast and cheap on tokens
}


def tokenize(text):                                     # Split text into lowercase words, for keyword search
    return re.findall(r"\w+", text.lower())


class RagPipeline:                                      # Bundles the loaded models and the functions that use them
    def __init__(self, provider=None, model=None, max_retries=6):
        # provider / model: choose the LLM. If not given, read them from the environment variables
        # RAG_LLM_PROVIDER and RAG_LLM, falling back to Ollama with llama3.2:3b.
        # max_retries: how many times to wait and retry when Groq's rate limit is hit.
        self.provider = (provider or os.getenv("RAG_LLM_PROVIDER", "ollama")).lower()
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider '{self.provider}'. Use 'ollama' or 'groq'.")
        self.model = model or os.getenv("RAG_LLM") or DEFAULT_MODELS[self.provider]
        self.max_retries = max_retries

        if self.provider == "groq":                     # Hosted LLM: connect to Groq
            from groq import Groq                       # Imported only when needed
            api_key = os.getenv("GROQ_API_KEY")         # The key lives in an environment variable, never in the code
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is not set. Save it with setx (locally) or in Streamlit secrets (deployed).")
            self.groq = Groq(api_key=api_key)

        # Retrieval: the same for both providers
        self.chunks = pd.read_json(ROOT / "data/processed/chunks.jsonl", lines=True)   # All chunks
        self.chunk_ids = self.chunks["chunk_id"].tolist()                              # Their IDs, in order
        self.by_id = self.chunks.set_index("chunk_id")                                 # Same table, looked up by chunk ID
        self.embedder = SentenceTransformer(EMBED_MODEL)                               # Embedding model
        self.reranker = CrossEncoder(RERANK_MODEL)                                     # Reranker model
        self.collection = chromadb.PersistentClient(path=str(ROOT / "data/chroma")).get_collection(COLLECTION)   # Vector database
        self.bm25 = BM25Okapi([tokenize(t) for t in self.chunks["embed_text"]])       # Keyword index

    # ---------------- Retrieval ----------------
    def vector_ids(self, question, k):                  # Vector search: top k chunk IDs
        q = self.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True)
        return self.collection.query(query_embeddings=q.tolist(), n_results=k)["ids"][0]

    def bm25_ids(self, question, k):                    # Keyword search: top k chunk IDs
        scores = self.bm25.get_scores(tokenize(question))
        return [self.chunk_ids[i] for i in np.argsort(scores)[::-1][:k]]

    def hybrid_ids(self, question, k, pool=50, c=60):   # Merge both rankings with Reciprocal Rank Fusion
        scores = {}
        for ranking in (self.vector_ids(question, pool), self.bm25_ids(question, pool)):
            for r, cid in enumerate(ranking, start=1):
                scores[cid] = scores.get(cid, 0) + 1 / (c + r)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def retrieve(self, question, k=5, candidates=30):   # The final retrieval: hybrid -> rerank -> top k
        cands = self.hybrid_ids(question, candidates)
        scores = self.reranker.predict([(question, self.by_id.at[c, "embed_text"]) for c in cands])
        hits = []
        for i in np.argsort(scores)[::-1][:k]:
            row = self.by_id.loc[cands[i]]
            hits.append({
                "chunk_id": cands[i],
                "rerank_score": round(float(scores[i]), 3),
                "where": row["title"] + (f" > {row['section']}" if row["section"] else ""),
                "url": row["url"],
                "text": row["text"],
            })
        return hits

    # ---------------- Generation ----------------
    @staticmethod
    def build_prompt(question, hits):                   # Pack the numbered excerpts and the question into one message
        context = "\n\n".join(f"[{i}] {h['where']}\n{h['text']}" for i, h in enumerate(hits, start=1))
        return f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"

    def _messages(self, question, hits):                # The conversation sent to either provider
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.build_prompt(question, hits)}]

    def _generate_ollama(self, question, hits):         # Local model via Ollama
        import ollama                                   # Imported only when needed
        try:
            response = ollama.chat(model=self.model, messages=self._messages(question, hits), options=OLLAMA_OPTIONS)
            return response["message"]["content"].strip()
        except ollama.ResponseError as e:
            return f"[Generation failed: {e}]"
        except ConnectionError:
            return "[Generation failed: Ollama is not running. Start the Ollama app and try again.]"

    def _generate_groq(self, question, hits):           # Hosted model via Groq, with automatic retries on rate limits
        import groq                                     # For Groq's error types
        for attempt in range(self.max_retries + 1):     # First try, plus up to max_retries retries
            try:
                response = self.groq.chat.completions.create(
                    model=self.model, messages=self._messages(question, hits), **GROQ_OPTIONS)
                return (response.choices[0].message.content or "").strip()
            except groq.RateLimitError as e:            # Hit the requests- or tokens-per-minute limit
                if attempt == self.max_retries:
                    return "[Generation failed: Groq rate limit reached. Please try again in a minute.]"
                wait = float(e.response.headers.get("retry-after", 15))   # Groq says how long to wait; default 15 s
                time.sleep(min(wait, 60) + 1)           # Wait (at most a minute), then try again
            except groq.APIError as e:                  # Any other API problem (bad key, network, server error)
                return f"[Generation failed: {e}]"

    def generate(self, question, hits):                 # Ask the chosen LLM; returns (answer, seconds)
        start = time.time()
        if self.provider == "groq":
            answer = self._generate_groq(question, hits)
        else:
            answer = self._generate_ollama(question, hits)
        return answer, round(time.time() - start, 1)

    def ask(self, question, k=5):                       # The full pipeline: retrieve -> generate
        start = time.time()
        hits = self.retrieve(question, k)
        answer, llm_seconds = self.generate(question, hits)
        return {
            "question": question,
            "answer": answer,
            "hits": hits,
            "provider": self.provider,                  # Which provider answered (useful when comparing runs)
            "model": self.model,                        # Which model answered
            "llm_seconds": llm_seconds,
            "total_seconds": round(time.time() - start, 1),
        }