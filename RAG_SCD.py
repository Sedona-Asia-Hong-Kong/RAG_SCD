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
    """Query DeepSeek API with your question and context"""
    prompt = f"""You are an experienced Simcorp Dimension consultant with deep expertise in setup, configuration, and best practices.

Based on the following document context, please answer the question about Simcorp Dimension setup and configuration. If the answer cannot be found in the context, say so.

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
            {"role": "system", "content": "You are an experienced Simcorp Dimension consultant with deep expertise in setup, configuration, and best practices."},
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
 
# Example of how to use the system:
def answer_question(question):
	"""Ask a question about your PDF document"""
	# Search for relevant chunks in the database
	relevant_docs = db_chroma.similarity_search(question, k=3)

	# Combine the context from relevant documents
	context = "\n\n".join([doc.page_content for doc in relevant_docs])

	# Get answer from DeepSeek
	answer = query_deepseek(question, context)

	print(f"\nQuestion: {question}")
	print(f"Answer: {answer}")
	print(f"Sources found: {len(relevant_docs)}")

	return answer

if __name__ == "__main__":
    # Simple test call
    question = "How to sort fields in Extraction Definition?"
    answer_question(question)