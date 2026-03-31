"""
RAG Pipeline Hyperparameter Optimization using Genetic Algorithm
Dataset Track: Legal/Compliance (Markdown)
Author: [Your Name]
"""

import random
import math
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

# ─────────────────────────────────────────────
# 1. GOLD STANDARD QA PAIRS (Legal/Compliance)
# ─────────────────────────────────────────────
GOLD_QA = [
    {
        "question": "What is the notice period for data deletion?",
        "answer": "Users must submit a deletion request and the company will process it within 30 days."
    },
    {
        "question": "Can user data be shared with third parties?",
        "answer": "User data may be shared with trusted third parties only with explicit user consent."
    },
    {
        "question": "What cookies are used on the platform?",
        "answer": "The platform uses essential cookies for authentication and optional analytics cookies."
    },
    {
        "question": "How can users opt out of marketing communications?",
        "answer": "Users can opt out via the unsubscribe link in any email or through account settings."
    },
    {
        "question": "What is the minimum age to use the service?",
        "answer": "Users must be at least 18 years old or have parental consent to use the service."
    },
]

# ─────────────────────────────────────────────
# 2. LOAD DOCUMENT
# ─────────────────────────────────────────────
def load_document(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# ─────────────────────────────────────────────
# 3. RAG PIPELINE BUILDER
# ─────────────────────────────────────────────
def build_rag_pipeline(text: str, chunk_size: int, chunk_overlap: int):
    """Creates chunked vector store from raw text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def query_rag(vectorstore, question: str, top_k: int, temperature: float) -> str:
    """Retrieves context and queries Ollama."""
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
Be concise and factual.

Context:
{context}

Question: {question}
Answer:"""
    
    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()

# ─────────────────────────────────────────────
# 4. FITNESS FUNCTION (with caching)
# ─────────────────────────────────────────────
sim_model = SentenceTransformer("all-MiniLM-L6-v2")
_cache = {}
llm_call_count = [0]  # mutable to track across calls

def fitness(params: tuple, document_text: str) -> float:
    """
    params = (chunk_size, chunk_overlap, temperature, top_k)
    Returns fitness score in [0, 1].
    Budget: max 50 LLM calls total.
    """
    chunk_size, chunk_overlap, temperature, top_k = params
    chunk_size  = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    top_k       = int(top_k)
    temperature = float(temperature)
    
    # Penalty: invalid overlap
    if chunk_overlap >= chunk_size:
        return 0.0
    
    # Cache key
    key = hashlib.md5(str(params).encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    
    # Budget check
    if llm_call_count[0] >= 50:
        return 0.0
    
    try:
        vectorstore = build_rag_pipeline(document_text, chunk_size, chunk_overlap)
        scores = []
        
        for qa in GOLD_QA:
            llm_call_count[0] += 1
            answer = query_rag(vectorstore, qa["question"], top_k, temperature)
            
            # Cosine similarity
            emb_pred  = sim_model.encode(answer, convert_to_tensor=True)
            emb_gold  = sim_model.encode(qa["answer"], convert_to_tensor=True)
            score     = float(util.cos_sim(emb_pred, emb_gold))
            scores.append(max(0.0, score))
        
        fitness_score = sum(scores) / len(scores)
    except Exception as e:
        print(f"  [ERROR] {e}")
        fitness_score = 0.0
    
    _cache[key] = fitness_score
    return fitness_score

# ─────────────────────────────────────────────
# 5. GENETIC ALGORITHM
# ─────────────────────────────────────────────

# Search space bounds: [chunk_size, chunk_overlap, temperature, top_k]
BOUNDS = [
    (100, 1000),  # chunk_size
    (0,   200),   # chunk_overlap
    (0.0, 1.0),   # temperature
    (1,   10),    # top_k
]

POP_SIZE    = 8   # small: budget-conscious
GENERATIONS = 6   # 8×6 = 48 evaluations (under 50)
CROSSOVER_P = 0.8
MUTATION_P  = 0.2

def random_individual():
    return [
        random.randint(BOUNDS[0][0], BOUNDS[0][1]),
        random.randint(BOUNDS[1][0], BOUNDS[1][1]),
        round(random.uniform(BOUNDS[2][0], BOUNDS[2][1]), 2),
        random.randint(BOUNDS[3][0], BOUNDS[3][1]),
    ]

def clamp(value, low, high):
    return max(low, min(high, value))

def crossover(p1, p2):
    """Single-point crossover."""
    if random.random() < CROSSOVER_P:
        point = random.randint(1, len(p1) - 1)
        child1 = p1[:point] + p2[point:]
        child2 = p2[:point] + p1[point:]
        return child1, child2
    return p1[:], p2[:]

def mutate(individual):
    """Gaussian mutation for continuous, uniform for discrete."""
    ind = individual[:]
    for i in range(len(ind)):
        if random.random() < MUTATION_P:
            low, high = BOUNDS[i]
            if i in (0, 1, 3):  # integers
                ind[i] = clamp(
                    ind[i] + random.randint(-50, 50),
                    low, high
                )
            else:  # float
                ind[i] = clamp(
                    round(ind[i] + random.gauss(0, 0.1), 2),
                    low, high
                )
    return ind

def select_tournament(population, fitnesses, k=3):
    """Tournament selection."""
    candidates = random.sample(list(zip(population, fitnesses)), k)
    return max(candidates, key=lambda x: x[1])[0]

def run_genetic_algorithm(document_text: str):
    """Main GA loop. Returns history of best fitnesses + best params."""
    population    = [random_individual() for _ in range(POP_SIZE)]
    history       = []  # (iteration, best_fitness, best_params)
    best_overall  = None
    best_fitness  = -1.0
    
    for gen in range(GENERATIONS):
        # Evaluate
        fitnesses = [fitness(tuple(ind), document_text) for ind in population]
        
        # Track best
        gen_best_idx = fitnesses.index(max(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_par = population[gen_best_idx][:]
        
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_overall = gen_best_par[:]
        
        history.append({
            "iteration":    gen + 1,
            "best_fitness": round(best_fitness, 4),
            "params":       best_overall[:],
        })
        
        print(f"Gen {gen+1:02d} | Best fitness: {best_fitness:.4f} | "
              f"Params: {best_overall} | LLM calls: {llm_call_count[0]}")
        
        # Evolve
        new_population = []
        # Elitism: keep best
        new_population.append(best_overall[:])
        
        while len(new_population) < POP_SIZE:
            p1 = select_tournament(population, fitnesses)
            p2 = select_tournament(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(c2))
        
        population = new_population
    
    return history, best_overall, best_fitness

# ─────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(history):
    """2D convergence plot + parameter evolution."""
    iterations   = [h["iteration"] for h in history]
    best_fitness = [h["best_fitness"] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GA Convergence on RAG Hyperparameter Optimization", fontsize=13)
    
    # ── Convergence curve
    ax1 = axes[0]
    ax1.plot(iterations, best_fitness, "o-", color="#534AB7", linewidth=2, markersize=6)
    ax1.axhline(0.8, linestyle="--", color="#D85A30", alpha=0.6, label="Target (0.8)")
    ax1.fill_between(iterations, best_fitness, alpha=0.15, color="#534AB7")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness Score")
    ax1.set_title("Convergence over generations")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ── Parameter evolution (chunk_size)
    ax2 = axes[1]
    chunk_sizes = [h["params"][0] for h in history]
    overlaps    = [h["params"][1] for h in history]
    temps       = [h["params"][2] for h in history]
    top_ks      = [h["params"][3] for h in history]
    
    ax2.plot(iterations, chunk_sizes, "s-", label="chunk_size", color="#534AB7")
    ax2.plot(iterations, overlaps,    "^-", label="chunk_overlap", color="#1D9E75")
    ax2.plot(iterations, [t*100 for t in temps], "o-", label="temperature×100", color="#D85A30")
    ax2.plot(iterations, [k*50 for k in top_ks], "D-", label="top_k×50", color="#D4537E")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Parameter value (scaled)")
    ax2.set_title("Parameter evolution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ga_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to ga_convergence.png")

def print_comparison_table(history):
    """Prints the report table."""
    print("\n" + "="*80)
    print(f"{'Iteration':>9} | {'Best Fitness':>12} | {'chunk_size':>10} | "
          f"{'chunk_overlap':>13} | {'temperature':>11} | {'top_k':>6}")
    print("-"*80)
    for h in history:
        p = h["params"]
        print(f"{h['iteration']:>9} | {h['best_fitness']:>12.4f} | {p[0]:>10} | "
              f"{p[1]:>13} | {p[2]:>11.2f} | {p[3]:>6}")
    print("="*80)

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load your document (change path to your actual file)
    DOCUMENT_PATH = "privacy_policy.md"  # or .pdf / .txt
    document_text = load_document(DOCUMENT_PATH)
    
    print(f"Document loaded: {len(document_text)} characters")
    print(f"Starting GA optimization. Budget: 50 LLM calls\n")
    
    history, best_params, best_score = run_genetic_algorithm(document_text)
    
    # Print table
    print_comparison_table(history)
    
    # Plot
    plot_results(history)
    
    # Summary
    print(f"\n{'='*40}")
    print(f"OPTIMAL CONFIGURATION FOUND")
    print(f"{'='*40}")
    print(f"  chunk_size    : {best_params[0]}")
    print(f"  chunk_overlap : {best_params[1]}")
    print(f"  temperature   : {best_params[2]}")
    print(f"  top_k         : {best_params[3]}")
    print(f"  fitness score : {best_score:.4f}")
    print(f"  total LLM calls used: {llm_call_count[0]}/50")