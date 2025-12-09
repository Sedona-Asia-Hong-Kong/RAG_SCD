# rag_pdf_chroma.py - Complete RAG with Local LLM (Ollama/DeepSeek)
# VS Code + Vibe Coding ready
# pip install chromadb pypdf sentence-transformers requests

import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Tuple

# CONFIG
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_rag"
CHROMA_PATH = "./chroma_db"
PDF_DIR = "Data Extractor./pdfs"  # Put your PDFs here
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"  # or mistral, phi3, etc.

# DEEPSEEK CONFIG - uncomment and adjust if switching from Ollama
# DEEPSEEK_URL = "http://localhost:YOUR_DEEPSEEK_PORT/api/generate"  # e.g. 8080 if using DeepSeek local server
# DEEPSEEK_MODEL = "deepseek-r1"  # adjust to your DeepSeek model name
# USE_DEEPSEEK = True  # Set to True to use DeepSeek instead of Ollama

# Initialize
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def load_pdfs(pdf_dir: str = PDF_DIR) -> List[Tuple[str, str]]:
    """Load all PDFs from directory"""
    docs = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        try:
            reader = PdfReader(path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            docs.append((fname, full_text))
            print(f"Loaded: {fname} ({len(full_text)} chars)")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return docs

def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

def index_pdfs(pdf_dir: str = PDF_DIR):
    """Build ChromaDB index from PDFs"""
    print("Indexing PDFs...")
    docs = load_pdfs(pdf_dir)
    
    ids, texts, metadatas = [], [], []
    idx = 0
    
    for fname, text in docs:
        chunks = split_into_chunks(text)
        for chunk in chunks:
            ids.append(f"doc_{idx}")
            texts.append(chunk)
            metadatas.append({"source": fname, "chunk_id": idx})
            idx += 1
    
    # Embed and store
    embeddings = embedder.encode(texts).tolist()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Indexed {len(ids)} chunks")

def retrieve(question: str, k: int = 5) -> List[Tuple[str, dict]]:
    """Retrieve top-k relevant chunks"""
    q_emb = embedder.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    return list(zip(results["documents"][0], results["metadatas"][0]))

def call_local_llm(question: str, contexts: List[str]) -> str:
    """Call Ollama local LLM (or DeepSeek if configured)"""
    context_block = "\n\n".join(contexts)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.

CONTEXT:
{context_block}

QUESTION: {question}

ANSWER:"""
    
    # SWITCH TO DEEPSEEK: uncomment USE_DEEPSEEK=True above, adjust URL/model
    # if USE_DEEPSEEK:
    #     url = DEEPSEEK_URL
    #     model = DEEPSEEK_MODEL
    # else:
    #     url = OLLAMA_URL
    #     model = OLLAMA_MODEL
    
    try:
        response = requests.post(OLLAMA_URL, json={
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'top_p': 0.9
            }
        }, timeout=60)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"LLM Error: {e}. Make sure Ollama is running with '{OLLAMA_MODEL}' model."

def answer_question(question: str) -> str:
    """Full RAG pipeline: retrieve + generate"""
    print("Retrieving relevant docs...")
    docs = retrieve(question)
    contexts = [doc for doc, meta in docs]
    
    print("Generating answer...")
    answer = call_local_llm(question, contexts)
    return answer

# CLI Interface
def main():
    print("=== PDF RAG with Local LLM ===\n")
    print("1. Index PDFs: python rag_pdf_chroma.py index")
    print("2. Query: python rag_pdf_chroma.py query")
    
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "index":
            index_pdfs()
            print("✅ Indexing complete!")
        elif cmd == "query":
            if len(sys.argv) > 2:
                question = " ".join(sys.argv[2:])
                print(f"Q: {question}")
                answer = answer_question(question)
                print(f"\nA: {answer}")
            else:
                print("Usage: python rag_pdf_chroma.py query 'your question'")
    else:
        # Interactive mode
        while True:
            question = input("\nAsk about your PDFs (or 'quit'): ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question.strip():
                answer = answer_question(question)
                print(f"\n🤖 Answer: {answer}\n")

if __name__ == "__main__":
    main()
