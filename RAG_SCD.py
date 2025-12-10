import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # Free local embeddings
from langchain.vectorstores import Chroma
 
# DeepSeek API configuration
DEEPSEEK_API_KEY = "sk-7559c296c39c4cd6bba910a6e7c5c0d0"  # Replace with your actual DeepSeek key
DOC_PATH = "Data Extractor.pdf"
CHROMA_PATH = "db_name"
 
# Load your pdf doc
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()
 
# Split the doc into smaller chunks i.e. chunk_size=500
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
chunks = text_splitter.split_documents(pages)
 
# Use FREE local embeddings instead of OpenAI
embeddings = HuggingFaceEmbeddings(
	model_name="sentence-transformers/all-MiniLM-L6-v2"
)
 
# Embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
 
print(f"Hello World! PDF processed with DeepSeek-compatible system!")
 
# Optional: Add a function to query DeepSeek
def query_deepseek(question, context_text):
    """Query DeepSeek API with your question and labeled context; ask to synthesize across chunks."""
    prompt = f"""You are an experienced Simcorp Dimension consultant.

The information is provided as separate labeled chunks (e.g. [CHUNK 1], [CHUNK 2]). Pieces of the answer may be scattered across multiple chunks. Synthesize the information across the chunks, give a one-line summary, then a short detailed answer. Do NOT include chunk references like [CHUNK 1] or [CHUNK 2] in your final answer. If the answer cannot be found in the provided chunks, reply: "Answer not found in provided chunks."

Context:
{context_text}

Question: {question}

Answer:"""

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an experienced Simcorp Dimension consultant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling DeepSeek: {str(e)}"
 
def understand_question(question):
    """Use DeepSeek to understand what the question is really asking"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    prompt = f"""Analyze this question and extract key concepts and related terms that might appear in documentation.
Question: {question}

Provide 3-4 alternative ways to phrase this question or related search terms. Return them as a simple list."""

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return question
    except:
        return question

def _page_number_of_doc(doc):
    """Try common metadata keys to return an int page number, or None."""
    for k in ("page", "page_number", "source"):
        v = doc.metadata.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            # try to extract first number from string
            import re
            m = re.search(r"(\d+)", str(v))
            if m:
                return int(m.group(1))
    return None

def _lexical_page_scores(question, top_n=5):
    """Quick lexical scan over raw pages to find pages with obvious keyword matches."""
    import re
    terms_text = (question + " " + (understand_question(question) or "")).lower()
    # simple tokenization, drop short tokens
    tokens = [t for t in re.findall(r"\w{3,}", terms_text) if not t.isdigit()]
    if not tokens:
        return []

    scores = {}
    for i, p in enumerate(pages):
        text = p.page_content.lower()
        score = 0
        for t in tokens:
            # count occurrences (simple signal)
            score += text.count(t)
        if score > 0:
            # prefer exact phrase matches a bit more
            phrase_matches = sum(1 for t in tokens if t in text)
            scores[_page_number_of_doc(p) or i] = score + phrase_matches * 2

    # return top_n page numbers sorted by score desc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [p for p, s in ranked[:top_n]]

def _embedding_page_candidates(question, top_k=20):
    """Get candidate docs from vector DB and return their page numbers (preserving order)."""
    try:
        results = db_chroma.similarity_search_with_score(question, k=top_k)
    except Exception:
        # fallback
        docs = db_chroma.similarity_search(question, k=top_k)
        results = [(d, 999.0) for d in docs]

    pages_found = []
    for item in results:
        doc = item[0] if isinstance(item, tuple) else item
        pnum = _page_number_of_doc(doc)
        if pnum is None:
            # try to parse page from source file metadata if present
            pnum = _page_number_of_doc(doc)
        if pnum is None:
            continue
        if pnum not in pages_found:
            pages_found.append(pnum)
    return pages_found

def retrieve_candidates_by_page(question, lexical_top=3, embed_top=12, neighbor_radius=1):
    """Combine lexical and embedding signals to produce a prioritized list of chunks (documents).
    Returns a list of docs ordered by priority."""
    # 1) lexical signals
    lexical_pages = _lexical_page_scores(question, top_n=lexical_top)

    # 2) embedding signals
    embed_pages = _embedding_page_candidates(question, top_k=embed_top)

    # 3) combine and prioritize: lexical pages first (they are 'obvious'), then embedding pages
    combined_pages = []
    for p in lexical_pages + embed_pages:
        if p not in combined_pages:
            combined_pages.append(p)

    # 4) if still empty, fall back to top embedding pages
    if not combined_pages:
        combined_pages = embed_pages[:3]

    # 5) expand with neighbors (to capture context spanning chunk boundaries)
    expanded_pages = []
    for p in combined_pages:
        for n in range(p - neighbor_radius, p + neighbor_radius + 1):
            if n > 0 and n not in expanded_pages:
                expanded_pages.append(n)

    # 6) collect chunk docs that belong to these pages (preserve the expanded_pages ordering)
    page_to_docs = {}
    for doc in chunks:
        pnum = _page_number_of_doc(doc)
        if pnum is None:
            continue
        page_to_docs.setdefault(pnum, []).append(doc)

    result_docs = []
    added = set()
    for p in expanded_pages:
        docs = page_to_docs.get(p, [])
        for d in docs:
            h = hash(d.page_content)
            if h not in added:
                result_docs.append(d)
                added.add(h)
        # optional: limit how many docs we add per page to avoid huge context
        if len(result_docs) >= 20:
            break

    # 7) If still no docs, fall back to top embedding docs (no page grouping)
    if not result_docs:
        try:
            embs = db_chroma.similarity_search(question, k=6)
            for d in embs:
                h = hash(d.page_content)
                if h not in added:
                    result_docs.append(d)
                    added.add(h)
        except Exception:
            pass

    return result_docs

# Replace the body of answer_question to use retrieve_candidates_by_page
def answer_question(question):
    """Ask a question about your PDF document with smart understanding"""
    print(f"\n🔍 Understanding your question...")
    related_terms = understand_question(question)
    print(f"Related search terms found:\n{related_terms}")

    print(f"\n📚 Retrieving candidate pages (lexical + embedding)...")
    candidate_docs = retrieve_candidates_by_page(question, lexical_top=3, embed_top=20, neighbor_radius=1)

    # Build labeled context so DeepSeek can cite and synthesize across chunks
    chunks_text = []
    source_pages = set()
    for i, doc in enumerate(candidate_docs, start=1):
        page_meta = doc.metadata.get("page") or doc.metadata.get("page_number") or doc.metadata.get("source") or "?"
        label = f"[CHUNK {i}] (page: {page_meta})"
        chunks_text.append(f"{label}\n{doc.page_content.strip()}")
        
        # Extract page number for sources
        try:
            page_num = int(page_meta)
            source_pages.add(page_num)
        except (ValueError, TypeError):
            pass

    context = "\n\n".join(chunks_text) if chunks_text else ""
    print(f"Context includes pages: {[doc.metadata.get('page') or doc.metadata.get('page_number') for doc in candidate_docs]}")

    print(f"\n💡 Generating answer...")
    answer = query_deepseek(question, context)

    # Format sources
    sources_text = ""
    if source_pages:
        sorted_pages = sorted(list(source_pages))
        sources_text = f"\n\n**Sources:** Page(s) {', '.join(map(str, sorted_pages))}"

    final_answer = answer + sources_text

    print(f"\nQuestion: {question}")
    print(f"Answer: {final_answer}")
    print(f"Sources found: {len(candidate_docs)}")

    return final_answer

if __name__ == "__main__":
    # Simple test call
    question = "How do I link my extracted results to Communicaiton Server?"
    answer_question(question)