import os
from BAAI.bge_small_en_v1_5 import BAAI_bge_small_en_v1_5
import vector_db

def app(query, top_k):
    try:
        embeddings = BAAI_bge_small_en_v1_5()
        embeddings.encode_kwargs = {'normalize_embeddings': True}
        results = vector_db.max_marginal_relevance_search(query, k=top_k, fetch_k=20)
    except Exception as e:
        print(f"Error: {e}")
    # ... rest of code ...
    system_prompt = "Please provide a clear first-sentence definition, technical bullet points, and source file citations."
    # ... rest of code ...
