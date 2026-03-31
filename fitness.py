"""
fitness.py
Fitness function for the RAG pipeline optimization.

Scores a candidate parameter set by:
  1. Building a RAG pipeline with those parameters
  2. Querying Ollama with gold-standard questions
  3. Computing cosine similarity between LLM answers and gold answers
  4. Applying a penalty if chunk_overlap >= chunk_size

Uses a dictionary cache to avoid re-evaluating identical parameter sets.
"""

import hashlib
from sentence_transformers import SentenceTransformer, util
from rag_pipeline import build_vectorstore, query_rag
from gold_qa import GOLD_QA

# Similarity model — loaded once globally
_sim_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cache: maps parameter hash → fitness score
_cache: dict = {}

# Mutable counter — tracks total Ollama calls made across all evaluations
llm_call_count = [0]

# Maximum allowed LLM calls (assignment budget)
LLM_BUDGET = 50


def evaluate(params: tuple, document_text: str) -> float:
    """
    Evaluates a candidate parameter configuration.

    Args:
        params:        (chunk_size, chunk_overlap, temperature, top_k)
        document_text: Raw document string loaded from disk

    Returns:
        Fitness score in [0.0, 1.0]
        Returns 0.0 for invalid configurations (overlap >= chunk_size)
        Returns 0.0 if LLM budget is exhausted
    """
    chunk_size, chunk_overlap, temperature, top_k = params

    # Cast to correct types (GA may pass floats for integer params)
    chunk_size    = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    top_k         = int(top_k)
    temperature   = round(float(temperature), 3)

    # ── Penalty: chunk_overlap must be strictly less than chunk_size
    if chunk_overlap >= chunk_size:
        return 0.0

    # ── Cache lookup: avoid re-running identical configurations
    cache_key = hashlib.md5(str((chunk_size, chunk_overlap, temperature, top_k)).encode()).hexdigest()
    if cache_key in _cache:
        print(f"  [CACHE HIT] params={params} → {_cache[cache_key]:.4f}")
        return _cache[cache_key]

    # ── Budget check: do not exceed 50 LLM calls total
    calls_needed = len(GOLD_QA)
    if llm_call_count[0] + calls_needed > LLM_BUDGET:
        print(f"  [BUDGET EXCEEDED] {llm_call_count[0]}/{LLM_BUDGET} calls used")
        return 0.0

    # ── Build RAG pipeline and evaluate
    try:
        vectorstore = build_vectorstore(document_text, chunk_size, chunk_overlap)
        scores = []

        for qa in GOLD_QA:
            llm_call_count[0] += 1
            answer = query_rag(vectorstore, qa["question"], top_k, temperature)

            # Cosine similarity between LLM answer and gold answer
            emb_pred = _sim_model.encode(answer,       convert_to_tensor=True)
            emb_gold = _sim_model.encode(qa["answer"], convert_to_tensor=True)
            score    = float(util.cos_sim(emb_pred, emb_gold))
            scores.append(max(0.0, score))  # clamp negatives to 0

        fitness_score = sum(scores) / len(scores)

    except Exception as e:
        print(f"  [ERROR] Failed to evaluate params={params}: {e}")
        fitness_score = 0.0

    # Store in cache
    _cache[cache_key] = fitness_score
    return fitness_score
