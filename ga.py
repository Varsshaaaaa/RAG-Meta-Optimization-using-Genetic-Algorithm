"""
ga.py
Genetic Algorithm for RAG hyperparameter optimization.

Search space:
    G1: chunk_size    → int  in [100, 1000]
    G2: chunk_overlap → int  in [0, 200]
    G3: temperature   → float in [0.0, 1.0]
    G4: top_k         → int  in [1, 10]

Strategy:
    - Tournament selection (k=3)
    - Single-point crossover (p=0.8)
    - Gaussian mutation for float, uniform random for int (p=0.2)
    - Elitism: best individual always survives to next generation
"""

import random
from fitness import evaluate, llm_call_count, LLM_BUDGET

# ── Search space bounds: (min, max) for each gene
BOUNDS = [
    (100,  1000),   # G1: chunk_size
    (0,    200),    # G2: chunk_overlap
    (0.0,  1.0),    # G3: temperature
    (1,    10),     # G4: top_k
]

# ── GA hyperparameters (tuned to stay within 50-call budget)
POP_SIZE      = 8     # 8 individuals × 6 generations = 48 max evaluations
N_GENERATIONS = 6
CROSSOVER_P   = 0.8
MUTATION_P    = 0.2
TOURNAMENT_K  = 3


def _clamp(value, low, high):
    return max(low, min(high, value))


def random_individual() -> list:
    """Creates a random candidate within bounds."""
    return [
        random.randint(BOUNDS[0][0], BOUNDS[0][1]),   # chunk_size
        random.randint(BOUNDS[1][0], BOUNDS[1][1]),   # chunk_overlap
        round(random.uniform(BOUNDS[2][0], BOUNDS[2][1]), 2),  # temperature
        random.randint(BOUNDS[3][0], BOUNDS[3][1]),   # top_k
    ]


def tournament_select(population: list, fitnesses: list) -> list:
    """
    Selects one individual via tournament selection.
    Randomly picks TOURNAMENT_K candidates and returns the fittest.
    """
    candidates = random.sample(list(zip(population, fitnesses)), TOURNAMENT_K)
    return max(candidates, key=lambda x: x[1])[0][:]


def crossover(parent1: list, parent2: list) -> tuple:
    """
    Single-point crossover. With probability CROSSOVER_P, splits both
    parents at a random point and swaps tails. Otherwise returns copies.
    """
    if random.random() < CROSSOVER_P:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1[:], parent2[:]


def mutate(individual: list) -> list:
    """
    Applies mutation gene-by-gene with probability MUTATION_P.
    - Integer genes (chunk_size, chunk_overlap, top_k): uniform random shift
    - Float gene (temperature): Gaussian perturbation (sigma=0.1)
    """
    ind = individual[:]
    for i in range(len(ind)):
        if random.random() < MUTATION_P:
            low, high = BOUNDS[i]
            if i == 2:  # temperature — float gene
                ind[i] = _clamp(
                    round(ind[i] + random.gauss(0, 0.1), 3),
                    low, high
                )
            else:  # integer genes
                delta = random.randint(-80, 80) if i == 0 else random.randint(-30, 30)
                ind[i] = _clamp(ind[i] + delta, low, high)
    return ind


def run(document_text: str) -> tuple:
    """
    Runs the Genetic Algorithm.

    Args:
        document_text: Raw text of the document to optimize against

    Returns:
        history      : list of dicts {iteration, best_fitness, params}
        best_params  : [chunk_size, chunk_overlap, temperature, top_k]
        best_fitness : float — highest fitness score found
    """
    print(f"\nStarting Genetic Algorithm")
    print(f"Population: {POP_SIZE}  |  Generations: {N_GENERATIONS}  |  Budget: {LLM_BUDGET} LLM calls\n")

    # Initialise random population
    population   = [random_individual() for _ in range(POP_SIZE)]
    history      = []
    best_params  = None
    best_fitness = -1.0

    for gen in range(N_GENERATIONS):
        print(f"── Generation {gen + 1}/{N_GENERATIONS} ──")

        # Evaluate all individuals
        fitnesses = []
        for ind in population:
            score = evaluate(tuple(ind), document_text)
            fitnesses.append(score)
            print(f"  params={ind}  fitness={score:.4f}  (LLM calls: {llm_call_count[0]}/{LLM_BUDGET})")

        # Track overall best
        gen_best_idx = fitnesses.index(max(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_params  = population[gen_best_idx][:]

        history.append({
            "iteration":    gen + 1,
            "best_fitness": round(best_fitness, 4),
            "params":       best_params[:],
        })

        print(f"  → Generation best: {max(fitnesses):.4f}  |  Overall best: {best_fitness:.4f}\n")

        # Stop early if budget nearly exhausted
        if llm_call_count[0] >= LLM_BUDGET:
            print("Budget reached — stopping early.")
            break

        # ── Evolve next generation ──
        next_population = []

        # Elitism: carry best individual forward unchanged
        next_population.append(best_params[:])

        # Fill the rest via selection → crossover → mutation
        while len(next_population) < POP_SIZE:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            next_population.append(mutate(c1))
            if len(next_population) < POP_SIZE:
                next_population.append(mutate(c2))

        population = next_population

    return history, best_params, best_fitness
