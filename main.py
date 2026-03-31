"""
main.py
Entry point for the RAG Hyperparameter Optimization project.

Usage:
    python main.py

Steps:
    1. Loads the document from DOCUMENT_PATH
    2. Runs the Genetic Algorithm (ga.py) to find optimal parameters
    3. Prints the comparison table (visualize.py)
    4. Saves the convergence plot (visualize.py)
    5. Prints the final optimal configuration summary
"""

from rag_pipeline import load_document
from fitness import llm_call_count
import ga
from visualize import plot_convergence, print_comparison_table

# ── Configuration ────────────────────────────────────────
# Change this to your document's path.
# Supported: .md, .txt, .pdf (PDF requires: pip install pypdf)
DOCUMENT_PATH = "privacy_policy.md"
# ─────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  RAG Pipeline Hyperparameter Optimization")
    print("  Algorithm: Genetic Algorithm")
    print("=" * 60)

    # Step 1: Load document
    print(f"\nLoading document: {DOCUMENT_PATH}")
    document_text = load_document(DOCUMENT_PATH)
    print(f"Document loaded: {len(document_text)} characters\n")

    # Step 2: Run GA
    history, best_params, best_fitness = ga.run(document_text)

    # Step 3: Print comparison table
    print_comparison_table(history)

    # Step 4: Save convergence plot
    plot_convergence(history, save_path="ga_convergence.png")

    # Step 5: Final summary
    print("=" * 60)
    print("  OPTIMAL CONFIGURATION FOUND")
    print("=" * 60)
    print(f"  G1  chunk_size    : {best_params[0]}")
    print(f"  G2  chunk_overlap : {best_params[1]}")
    print(f"  G3  temperature   : {best_params[2]}")
    print(f"  G4  top_k         : {best_params[3]}")
    print(f"  Best fitness score: {best_fitness:.4f}")
    print(f"  Total LLM calls   : {llm_call_count[0]}/50")
    print("=" * 60)

    # Efficiency note for the report
    for h in history:
        if h["best_fitness"] >= 0.8:
            print(f"\n  Target fitness 0.8 first reached at generation {h['iteration']}")
            break
    else:
        print(f"\n  Target fitness 0.8 was not reached within the budget.")
        print(f"  Best achieved: {best_fitness:.4f}")


if __name__ == "__main__":
    main()
