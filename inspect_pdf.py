from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOC_PATH = "Data Extractor.pdf"

loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

print(f"Total pages: {len(pages)}")

# Inspect raw page text
for i in range(min(3, len(pages))):
    print(f"\n=== PAGE {i+1} RAW TEXT (first 800 chars) ===")
    print(pages[i].page_content[:800])

# Try a basic splitter and inspect chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = text_splitter.split_documents(pages)

print(f"\nTotal chunks: {len(chunks)}")

for i in range(min(8, len(chunks))):
    print(f"\n--- CHUNK {i} (len={len(chunks[i].page_content)}) ---")
    print(chunks[i].page_content[:800])