# Comparative Meta-Optimization of RAG Pipelines

A Genetic Algorithm-based optimization project for tuning Retrieval-Augmented Generation (RAG) hyperparameters on a legal/compliance Markdown dataset using a local Ollama LLM.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20Phi--3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project evaluates how meta-heuristic optimization can improve the quality of a RAG pipeline by searching for the best combination of:

- `chunk_size`
- `chunk_overlap`
- `temperature`
- `top_k`

The optimizer uses a Genetic Algorithm and a semantic fitness function based on cosine similarity between generated answers and gold-standard answers.

---

## Objective

The goal is to maximize answer quality from a local LLM while staying within a strict budget of 50 LLM calls.

The final best configuration achieved:

- `chunk_size = 656`
- `chunk_overlap = 175`
- `temperature = 0.09`
- `top_k = 10`
- `best fitness = 0.8497`

---

## Dataset Track

**Track:** Legal / Compliance (Markdown)

Why this track?

- Legal documents require high overlap to avoid splitting clauses.
- Larger chunks help preserve semantic context.
- Low temperature improves precision in generated answers.

---

## Features

- RAG pipeline with document chunking and retrieval.
- Local LLM integration using Ollama.
- Genetic Algorithm for hyperparameter search.
- Fitness evaluation using SentenceTransformers cosine similarity.
- Caching to avoid repeated evaluations.
- Convergence visualization and comparison table.
- Budget-aware optimization under limited LLM calls.

---

## Project Structure

```bash
.
├── main.py
├── rag_pipeline.py
├── fitness.py
├── ga.py
├── visualize.py
├── gold_qa.py
├── data/
│   └── privacy_policy.md
├── output/
│   ├── ga_convergence.png
│   └── RAG_Report_Final.docx
└── README.md
```

---

## Requirements

- Python 3.10+
- Ollama installed locally
- Model pulled in Ollama:
  - `phi3` or `tinyllama`
- SentenceTransformers
- FAISS
- LangChain
- `python-docx`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-meta-optimization.git
cd rag-meta-optimization
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Ollama and pull the model

```bash
ollama pull phi3
ollama serve
```

---

## Usage

Run the main script:

```bash
python main.py
```

This will:

1. Load the legal/compliance document.
2. Build the RAG pipeline.
3. Evaluate candidate hyperparameters.
4. Run the Genetic Algorithm.
5. Print the best configuration and fitness.
6. Generate convergence outputs and the report artifacts.

---

## Validation Questions

The project uses 5 gold-standard QA pairs derived from the legal document, such as:

- What is the notice period for data deletion?
- Can user data be shared with third parties?
- What cookies are used on the platform?
- How can users opt out of marketing communications?
- What is the minimum age to use the service?

---

## Fitness Function

The fitness score is calculated as follows:

- Run the RAG pipeline with a candidate parameter set.
- Query the local LLM with the validation questions.
- Compare generated answers to gold answers using cosine similarity.
- Average the scores to produce a value between 0 and 1.
- Apply a penalty if `chunk_overlap >= chunk_size`.

---

## Results

### Best configuration found

| Iteration | Best Fitness | chunk_size | chunk_overlap | temperature | top_k |
|---|---:|---:|---:|---:|---:|
| 1 | 0.8497 | 656 | 175 | 0.09 | 10 |
| 2 | 0.8497 | 656 | 175 | 0.09 | 10 |

### Final outcome

- Best fitness: `0.8497`
- LLM calls used: `50 / 50`
- Threshold `0.8` was reached immediately in the first recorded generation.
- The search plateaued after the first strong solution.

---

## Visualization

The convergence plot shows:

- Fitness reached 0.8497 quickly.
- The best score remained stable afterward.
- The parameter set favored larger chunks, high overlap, low temperature, and higher top-k.



![GA Convergence](Screenshot-2026-03-31 111334.png")


---

## Inference

The Genetic Algorithm performed well under a tight budget because it found a strong configuration quickly. However, the plateau suggests that the search space may have been constrained by the dataset or that the optimizer converged early.

The legal/compliance domain strongly influenced the best settings:

- Higher `chunk_overlap` preserved clause boundaries.
- Moderate `chunk_size` preserved enough context.
- Low `temperature` improved precision.
- Higher `top_k` helped retrieve multiple relevant clauses.

---

## Conclusion

This project demonstrates that meta-heuristic optimization can improve RAG performance by tuning retrieval and generation parameters. For this legal dataset, the best configuration was achieved quickly and remained stable, showing that the selected GA setup was effective within the available evaluation budget.


---

## License

This project is provided for academic use. Add your preferred license here if needed.
