import os
import re
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/"
DB_PATH = "vectorstore/"

def clean_text(text):
    if not isinstance(text, str):
        return ""
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

    print("2. Splitting text into chunks (1000 size, 200 overlap)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"   Split into {len(chunks)} raw chunks.")

    print("3. Sanitizing and Prepending Metadata...")
    clean_chunks = []
    for doc in chunks:
        sanitized = clean_text(str(doc.page_content))
        if len(sanitized) > 20:
            filename = os.path.basename(doc.metadata.get('source', 'Document'))
            doc.page_content = f"Document: {filename}\nContent:\n{sanitized}"
            clean_chunks.append(doc)

    print(f"   Retained {len(clean_chunks)} enriched chunks.")

    print("4. Loading Embedding Model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )

    # Clean old vectorstore if present
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("5. Embedding into ChromaDB in safe batches...")
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
            vector_db.add_documents(batch)
            added_count += len(batch)
        except Exception:
            for single_doc in batch:
                try:
                    vector_db.add_documents([single_doc])
                    added_count += 1
                except Exception:
                    skipped_count += 1

        print(f"   Progress: {min(i + BATCH_SIZE, total_chunks)}/{total_chunks} chunks processed...", end="\r")

    print(f"\n\nSuccess! High-precision Vector DB stored in '{DB_PATH}'.")
    print(f"   Total Chunks Embedded: {added_count}")
    print(f"   Malformed Chunks Skipped: {skipped_count}\n")

if __name__ == "__main__":
    build_vector_db()
