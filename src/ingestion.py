import os
from transformers import Auto
from transformers import AutoModelForSequenceClassification

def ingestion(filename, sanitized):
    try:
        doc = {}
        doc['page_content'] = f"Document: {filename}\nContent:\n{sanitized}"
        chunk_size = 1000
        chunk_overlap = 200
        splitter = AutoModelForSequenceClassification(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split(sanitized)
        for i, chunk in enumerate(chunks):
            chunk['metadata'] = doc['page_content']
        if os.path.exists('vectorstore/'):
            os.system('rm -rf vectorstore/')
        embeddings = AutoModelForSequenceClassification(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={'normalize_embeddings': True})
        # ... rest of code ...
    except Exception as e:
        print(f"Error: {e}")
