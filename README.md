# RAG-Meta-Optimization-using-Genetic-Algorithm
🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline and optimizes its hyperparameters using a Genetic Algorithm (GA) under a constrained LLM evaluation budget.

The goal is to maximize answer quality (measured via cosine similarity) on a legal/compliance dataset by tuning key RAG parameters.

🧠 Problem Statement

RAG performance heavily depends on hyperparameters such as:

Chunk size
Chunk overlap
Temperature
Top-k retrieval

This project uses a Genetic Algorithm to automatically discover the best configuration within a limited LLM call budget (50 calls).

🏗️ Project Architecture
main.py
   │
   ├── rag_pipeline.py     → Document chunking + FAISS + LLM query
   ├── fitness.py          → Fitness evaluation (cosine similarity)
   ├── ga.py               → Genetic Algorithm implementation
   ├── visualize.py        → Convergence plots + comparison table
   └── gold_qa.py          → Ground truth Q&A pairs
⚙️ Tech Stack
LLM: Ollama (phi3)
Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
Vector DB: FAISS
Framework: LangChain
Optimization: Genetic Algorithm (custom implementation)
Language: Python
🔬 RAG Pipeline Workflow
Load Markdown document (legal/compliance dataset)
Split text using RecursiveCharacterTextSplitter
Generate embeddings using SentenceTransformers
Store embeddings in FAISS vector database
Retrieve top-k relevant chunks
Inject context into prompt
Generate answer using LLM (phi3)
🧪 Fitness Function

Fitness is computed as:

Mean cosine similarity between:
Generated answers
Gold standard answers
Constraints:
❌ Invalid if chunk_overlap >= chunk_size
🔁 Caching used to avoid repeated LLM calls
⛔ Budget-aware evaluation (max 50 calls)
🧬 Genetic Algorithm Details
Component	Description
Representation	[chunk_size, chunk_overlap, temperature, top_k]
Population	8 individuals
Selection	Tournament selection (k=3)
Crossover	Single-point (p=0.8)
Mutation	Gaussian (float), uniform (int)
Elitism	Best individual preserved
Termination	Budget exhausted (~2 generations)
📊 Results
✅ Best Configuration Found
chunk_size   = 656
chunk_overlap= 175
temperature  = 0.09
top_k        = 10
fitness      = 0.8497
📈 Key Observations
🎯 Fitness > 0.8 achieved in Generation 1
⚡ Early convergence observed
💸 Budget exhausted quickly (50 LLM calls)
📚 Legal documents require:
Larger chunk sizes
Higher overlap (~27%)
Higher top_k
📉 Insights
🔍 Domain Impact
Legal text requires high overlap to preserve clause boundaries
Multiple chunks are needed → higher top_k
⚠️ Limitations
Very limited search due to LLM cost
Premature convergence occurred
Results depend on random initialization
🔄 Future Improvements
Increase LLM budget (150–200 calls)
Use Bayesian Optimization or Differential Evolution
Add adaptive mutation rates
Parallelize fitness evaluation
Use better embedding models
▶️ How to Run
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
📁 Dataset
Domain: Legal / Compliance
Format: Markdown (privacy_policy.md)
Evaluation: 5 Gold Q&A pairs
📌 Key Takeaway

Optimizing RAG pipelines is highly budget-sensitive — even strong algorithms like GA can converge prematurely when LLM evaluations are expensive.

📚 References
Lewis et al. (2020) — RAG (NeurIPS)
Holland (1975) — Genetic Algorithms
Reimers & Gurevych (2019) — Sentence-BERT
LangChain Docs
Ollama Docs
