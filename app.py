import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def app(query, top_k):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={'normalize_embeddings': True})
        vectorstore = FAISS.load_local('vectorstore', embeddings)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": 20})
        
        qa_chain = RetrievalQA.from_chain_type(llm=None, retriever=retriever)
        results = qa_chain({"query": query})
        
        system_prompt = "Please provide a clear first-sentence definition, technical bullet points, and source file citations."
        if not results['result']:
            results['result'] = "Information not available in the UAV domain knowledge base."
        
    except Exception as e:
        print(f"Error: {e}")
        results = {"result": "An error occurred while processing your request."}
    
    return results, system_prompt
