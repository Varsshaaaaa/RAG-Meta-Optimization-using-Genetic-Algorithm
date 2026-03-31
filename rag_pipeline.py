"""
rag_pipeline.py
Handles document loading, text chunking, FAISS vector store creation,
and querying Ollama with retrieved context.
"""

import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model — loaded once globally to avoid reloading on every call
_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_document(filepath: str) -> str:
    """
    Loads a document from disk.
    Supports .txt and .md files directly.
    For PDF, install pypdf: pip install pypdf
    """
    if filepath.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        return "\n\n".join([d.page_content for d in docs])
    else:
        # Handles .txt and .md
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


def build_vectorstore(text: str, chunk_size: int, chunk_overlap: int):
    """
    Splits text into chunks and indexes them into a FAISS vector store.

    Args:
        text:          Raw document text
        chunk_size:    Number of characters per chunk (GA parameter G1)
        chunk_overlap: Overlap between adjacent chunks (GA parameter G2)

    Returns:
        FAISS vectorstore ready for similarity search
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, _embeddings)
    return vectorstore


def query_rag(vectorstore, question: str, top_k: int, temperature: float) -> str:
    """
    Retrieves the top_k most relevant chunks from the vector store,
    then sends them as context to Ollama along with the question.

    Args:
        vectorstore: FAISS index built from the document
        question:    The question to answer
        top_k:       Number of chunks to retrieve (GA parameter G4)
        temperature: LLM sampling temperature (GA parameter G3)

    Returns:
        LLM answer string
    """
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
Be concise and factual. Do not add information not present in the context.

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
