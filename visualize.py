"""
visualize.py
Generates the convergence and parameter evolution charts required by the report.

Produces:
    ga_convergence.png — 2-panel figure:
        Left:  Fitness convergence curve with target line at 0.8
        Right: Parameter evolution across generations (scaled to same axis)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_convergence(history: list, save_path: str = "ga_convergence.png"):
    """
    Plots fitness convergence and parameter evolution from GA history.

    Args:
        history:   List of dicts from ga.run() — each has
                   {iteration, best_fitness, params: [G1, G2, G3, G4]}
        save_path: Output file path for the saved figure
    """
    iterations   = [h["iteration"]    for h in history]
    best_fitness = [h["best_fitness"] for h in history]

    # Extract each parameter across generations
    chunk_sizes  = [h["params"][0]        for h in history]
    overlaps     = [h["params"][1]        for h in history]
    temperatures = [h["params"][2] * 100  for h in history]  # scale to 0-100 for visibility
    top_ks       = [h["params"][3] * 50   for h in history]  # scale to 0-500 for visibility

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "GA Convergence — RAG Hyperparameter Optimization",
        fontsize=13, fontweight="bold", color="#1A237E"
    )

    # ── Left panel: Fitness convergence ──
    ax1.plot(iterations, best_fitness, "o-",
             color="#3949AB", linewidth=2.5, markersize=7, label="Best fitness")
    ax1.fill_between(iterations, best_fitness, alpha=0.12, color="#3949AB")
    ax1.axhline(0.8, linestyle="--", color="#D85A30", linewidth=1.5,
                alpha=0.8, label="Target threshold (0.8)")

    # Annotate first time target is crossed
    for i, f in enumerate(best_fitness):
        if f >= 0.8:
            ax1.annotate(
                f"Crosses 0.8\nat gen {iterations[i]}",
                xy=(iterations[i], f),
                xytext=(iterations[i] + 0.3, f - 0.08),
                fontsize=9, color="#D85A30",
                arrowprops=dict(arrowstyle="->", color="#D85A30", lw=1.2)
            )
            break

    ax1.set_xlabel("Generation", fontsize=11)
    ax1.set_ylabel("Best fitness score", fontsize=11)
    ax1.set_title("Fitness convergence", fontsize=11)
    ax1.set_ylim(0, 1.08)
    ax1.set_xlim(min(iterations) - 0.3, max(iterations) + 0.3)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Right panel: Parameter evolution ──
    ax2.plot(iterations, chunk_sizes,  "s-", color="#3949AB", linewidth=2, markersize=6, label="chunk_size (raw)")
    ax2.plot(iterations, overlaps,     "^-", color="#1D9E75", linewidth=2, markersize=6, label="chunk_overlap (raw)")
    ax2.plot(iterations, temperatures, "o-", color="#D85A30", linewidth=2, markersize=6, label="temperature × 100")
    ax2.plot(iterations, top_ks,       "D-", color="#993556", linewidth=2, markersize=6, label="top_k × 50")

    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Parameter value (scaled)", fontsize=11)
    ax2.set_title("Parameter evolution across generations", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved → {save_path}")


def print_comparison_table(history: list):
    """
    Prints the formatted comparison table required by the report rubric.
    Columns: Iteration | Best Fitness | G1 (chunk_size) | G2 (chunk_overlap) | G3 (temperature) | G4 (top_k)
    """
    print("\n" + "=" * 78)
    print(f"{'Iteration':>9} | {'Best Fitness':>12} | {'G1 chunk_size':>13} | "
          f"{'G2 overlap':>10} | {'G3 temp':>7} | {'G4 top_k':>8}")
    print("-" * 78)
    for h in history:
        p = h["params"]
        print(f"{h['iteration']:>9} | {h['best_fitness']:>12.4f} | {p[0]:>13} | "
              f"{p[1]:>10} | {p[2]:>7.3f} | {p[3]:>8}")
    print("=" * 78 + "\n")
