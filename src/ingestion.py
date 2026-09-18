import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/"
DB_PATH = "vectorstore/"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Strip null bytes and non-printable control characters
    text = text.replace("\x00", " ")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()

def build_vector_db():
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"Error: No PDFs found in '{DATA_PATH}'. Check your data directory.")
        return

    print("1. Loading PDFs from data/ directory...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"   Loaded {len(raw_documents)} raw pages.")

    print("2. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"   Split into {len(chunks)} raw chunks.")

    print("3. Sanitizing chunk contents...")
    clean_chunks = []
    for doc in chunks:
        sanitized = clean_text(str(doc.page_content))
        if len(sanitized) > 15:
            doc.page_content = sanitized
            clean_chunks.append(doc)

    print(f"   Retained {len(clean_chunks)} clean chunks.")

    print("4. Initializing Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("5. Processing vector embeddings in safe batches...")
    
    # Initialize Chroma Vector Store
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    BATCH_SIZE = 32
    total_chunks = len(clean_chunks)
    added_count = 0
    skipped_count = 0

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = clean_chunks[i : i + BATCH_SIZE]
        try:
            # Try adding batch directly
            vector_db.add_documents(batch)
            added_count += len(batch)
        except Exception as e:
            # Fallback: Isolate individual problematic chunks inside the failing batch
            print(f"\n   Notice: Batch [{i}-{i+len(batch)}] triggered a parsing error. Isolating bad chunk...")
            for single_doc in batch:
                try:
                    vector_db.add_documents([single_doc])
                    added_count += 1
                except Exception:
                    skipped_count += 1

        print(f"   Progress: {min(i + BATCH_SIZE, total_chunks)}/{total_chunks} chunks processed...", end="\r")

    print(f"\n\nSuccess! Vector database stored in '{DB_PATH}'.")
    print(f"   Total Chunks Embedded: {added_count}")
    print(f"   Malformed Chunks Skipped: {skipped_count}\n")

if __name__ == "__main__":
    build_vector_db()
