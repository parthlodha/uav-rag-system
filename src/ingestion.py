import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def ingestion(filename, sanitized):
    try:
        doc = {}
        doc['page_content'] = f"Document: {filename}\nContent:\n{sanitized}"
        chunk_size = 1000
        chunk_overlap = 200
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(sanitized)
        for i, chunk in enumerate(chunks):
            chunk['metadata'] = doc['page_content']
        
        if os.path.exists('vectorstore/'):
            os.system('rm -rf vectorstore/')
        
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={'normalize_embeddings': True})
        
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local('vectorstore')
        
    except Exception as e:
        print(f"Error: {e}")
