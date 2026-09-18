import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

DB_PATH = "vectorstore/"

st.set_page_config(
    page_title="UAV Domain QA System", 
    page_icon="🚁", 
    layout="wide"
)

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Streamlit Header
st.title("🚁 UAV & Aerial Robotics Intelligence System")
st.caption("Domain-Specific RAG QA System over Flight Dynamics, Trajectory Optimization, and Spraying Payload Research")

# Check if vector DB exists
if not os.path.exists(DB_PATH):
    st.error("Vector database missing! Please run 'python3 src/ingestion.py' first to build ChromaDB.")
    st.stop()

vector_db = load_vector_db()

# Query Layout
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Ask a technical question regarding UAV technology:")
with col2:
    top_k = st.slider("Retrieval Chunks (Top-K):", min_value=1, max_value=5, value=3)

if st.button("Submit & Generate Answer") and query:
    with st.spinner("Retrieving relevant context from ChromaDB and generating Llama 3 response..."):
        # 1. Similarity Search
        retrieved_docs = vector_db.similarity_search(query, k=top_k)
        
        # 2. Format Context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 3. Domain-Specific System Prompt
        prompt = f"""
You are an expert AI specialized in Unmanned Aerial Vehicles (UAVs) and Aerial Robotics.
Answer the user's question using ONLY the provided context below.
If the context does not contain enough information to answer, state explicitly:
"Information not available in the UAV domain knowledge base."

Context:
{context_text}

Question: {query}
Answer:
"""
        # 4. Local LLM Execution via Ollama
        try:
            llm = Ollama(model="llama3")
            response = llm.invoke(prompt)
            
            st.markdown("### 💡 Answer:")
            st.write(response)
            
            # Display Citations & Context Passages
            st.divider()
            st.markdown("### 📚 Source Citations & Context Passages")
            
            for i, doc in enumerate(retrieved_docs):
                source_path = doc.metadata.get('source', 'Unknown File')
                filename = os.path.basename(source_path)
                page_num = doc.metadata.get('page', 'N/A')
                
                with st.expander(f"Source [{i+1}]: {filename} | Page {page_num}"):
                    st.write(doc.page_content)
                    
        except Exception as e:
            st.error(f"Error connecting to Ollama. Make sure Ollama service is running (`ollama serve`). Details: {e}")
