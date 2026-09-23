import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

DB_PATH = "vectorstore/"

st.set_page_config(
    page_title="UAV Domain QA & Vector Explorer", 
    page_icon="🚁", 
    layout="wide"
)

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

st.title("🚁 UAV Domain RAG Intelligence & Vector Visualizer")

if not os.path.exists(DB_PATH):
    st.error("Vector database missing! Please run 'python3 src/ingestion.py' first.")
    st.stop()

vector_db = load_vector_db()

# Create Tabs
tab_qa, tab_viz = st.tabs(["💬 Precision Q&A System", "📊 Vector Space & Embedding Visualizer"])

# ---------------------------------------------------------
# TAB 1: QA SYSTEM
# ---------------------------------------------------------
with tab_qa:
    st.caption("Domain-Specific RAG QA over Flight Dynamics, Trajectory Optimization, and Spraying Payload Research")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Ask a technical question regarding UAV technology:", key="qa_query")
    with col2:
        top_k = st.slider("Diverse Chunks (MMR Top-K):", min_value=2, max_value=8, value=5, key="qa_k")

    if st.button("Submit & Generate Answer") and query:
        with st.spinner("Fetching diverse context via MMR and synthesizing technical response..."):
            retrieved_docs = vector_db.max_marginal_relevance_search(query, k=top_k, fetch_k=20)
            
            context_passages = []
            for i, doc in enumerate(retrieved_docs):
                src = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page = doc.metadata.get('page', 'N/A')
                context_passages.append(f"[Source {i+1} | File: {src} | Page {page}]:\n{doc.page_content}")
                
            full_context = "\n\n---\n\n".join(context_passages)
            
            prompt = f"""
You are a Lead Autonomous Systems & UAV Systems Engineer.
Analyze the provided technical research excerpts and answer the user's question with high technical accuracy, clarity, and depth.

STRICT RULES:
1. State a clear, direct definition or primary answer in the first sentence.
2. Structure technical specs, components, control loops, or dynamic models using bullet points.
3. Reference source files in your explanation (e.g., [File: tesi.pdf]).
4. If the provided context is insufficient to answer the query, state explicitly: "Information not available in the UAV domain knowledge base."

TECHNICAL CONTEXT:
{full_context}

USER QUESTION:
{query}

ENGINEERING RESPONSE:
"""
            try:
                llm = Ollama(model="qwen2.5-coder:7b")
                response = llm.invoke(prompt)
                
                st.markdown("### 💡 Technical Answer:")
                st.write(response)
                
                st.divider()
                st.markdown("### 📚 Source Citations & Context Passages")
                
                for i, doc in enumerate(retrieved_docs):
                    source_path = doc.metadata.get('source', 'Unknown File')
                    filename = os.path.basename(source_path)
                    page_num = doc.metadata.get('page', 'N/A')
                    
                    with st.expander(f"Source [{i+1}]: {filename} | Page {page_num}"):
                        st.write(doc.page_content)
                        
            except Exception as e:
                st.error(f"Error connecting to Ollama: {e}")

# ---------------------------------------------------------
# TAB 2: VECTOR VISUALIZER
# ---------------------------------------------------------
with tab_viz:
    st.subheader("2D Projection of High-Dimensional ChromaDB Embeddings")
    st.write("This tab projects 384-dimensional vector embeddings down to 2D space using **PCA (Principal Component Analysis)** to demonstrate semantic clustering and vector retrieval.")
    
    viz_query = st.text_input("Enter a test query to visualize vector distance matching:", value="flight controller firmware", key="viz_query")
    
    if st.button("Generate Vector Map"):
        with st.spinner("Extracting embeddings from ChromaDB and computing 2D PCA projection..."):
            # Extract raw database vectors
            raw_data = vector_db._collection.get(include=['embeddings', 'metadatas', 'documents'])
            embeddings_matrix = np.array(raw_data['embeddings'])
            metadatas = raw_data['metadatas']
            documents = raw_data['documents']
            
            if len(embeddings_matrix) == 0:
                st.warning("No vectors found in database!")
            else:
                # Get Query Vector
                query_vector = np.array(vector_db.embeddings.embed_query(viz_query)).reshape(1, -1)
                
                # Combine Query Vector + DB Vectors for PCA
                all_vectors = np.vstack([query_vector, embeddings_matrix])
                
                # Reduce 384D -> 2D via PCA
                pca = PCA(n_components=2)
                reduced_2d = pca.fit_transform(all_vectors)
                
                # Retrieve Top Matching Docs for query
                retrieved_docs = vector_db.max_marginal_relevance_search(viz_query, k=5, fetch_k=20)
                retrieved_contents = set([doc.page_content for doc in retrieved_docs])
                
                # Build Dataframe for Plotly
                points = []
                # Query Point
                points.append({
                    "x": reduced_2d[0, 0],
                    "y": reduced_2d[0, 1],
                    "Category": "🔍 User Query Vector",
                    "Source": "Query",
                    "Snippet": viz_query
                })
                
                # DB Points
                for idx in range(len(embeddings_matrix)):
                    doc_text = documents[idx]
                    src = os.path.basename(metadatas[idx].get('source', 'Unknown'))
                    
                    category = "🎯 Retrieved Chunks (Top Match)" if doc_text in retrieved_contents else "📄 Document Chunks in DB"
                    
                    points.append({
                        "x": reduced_2d[idx + 1, 0],
                        "y": reduced_2d[idx + 1, 1],
                        "Category": category,
                        "Source": src,
                        "Snippet": doc_text[:120] + "..."
                    })
                
                df = pd.DataFrame(points)
                
                # Plotly Scatter Diagram
                fig = px.scatter(
                    df, x="x", y="y", 
                    color="Category",
                    hover_data=["Source", "Snippet"],
                    title=f"Vector Space Map for Query: '{viz_query}'",
                    color_discrete_map={
                        "🔍 User Query Vector": "#FF0000",
                        "🎯 Retrieved Chunks (Top Match)": "#00FF00",
                        "📄 Document Chunks in DB": "#636EFA"
                    }
                )
                fig.update_traces(marker=dict(size=10, opacity=0.8))
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 ** Understanding this PLOT:** Every dot represents a 1,000-character chunk from your PDFs transformed into a 384-dimensional vector. The red dot is the query vector. Notice how the green dots (retrieved chunks) cluster closest to the red query vector in high-dimensional semantic space!")
