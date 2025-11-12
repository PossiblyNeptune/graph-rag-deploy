"""
Streamlit Chatbot UI for Premier League Football Knowledge
Enhanced debugging, metadata display, and general football theming.
Based on **The Mixer** (Michael Cox - Tactics) and **The Club** (Robinson & Clegg - Business).
Uses COHERE (Cloud) for all LLM and Embedding tasks.

**MERGED CODE:** Functionality from Code 2, UI (Sidebar + CSS) from Code 1.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import pickle
import traceback
import json  # Added from Code 1 for sidebar connectivity report

from langchain_cohere import ChatCohere
from create_embeddings import DocumentProcessor
from create_database import KnowledgeGraph
from graph_rag import QueryEngine
from metadata_logger import MetadataLogger

# Load environment variables
load_dotenv()

# Check for required environment variables
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PGVECTOR_CONNECTION_STRING = os.environ.get("PGVECTOR_CONNECTION_STRING")

# Handle both .env and Streamlit secrets
if not COHERE_API_KEY:
    try:
        COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    except:
        pass

if not PGVECTOR_CONNECTION_STRING:
    try:
        PGVECTOR_CONNECTION_STRING = st.secrets["PGVECTOR_CONNECTION_STRING"]
    except:
        pass


@st.cache_resource
def load_chatbot():
    """
    Load chatbot with cloud-based Cohere services 
    and the local .pkl graph file.
    (Functionality from Code 2)
    """
    
    if not COHERE_API_KEY:
        st.error("⚠️ COHERE_API_KEY not found!")
        st.info("Please set it in Streamlit Cloud settings under 'Secrets'")
        st.stop()
    
    if not PGVECTOR_CONNECTION_STRING:
        st.error("⚠️ PGVECTOR_CONNECTION_STRING not found!")
        st.info("Please set it in Streamlit Cloud settings under 'Secrets'")
        st.stop()
    
    print("Connecting to Cohere (Chat)...")
    llm = ChatCohere(
        model="command-a-03-2025",
        cohere_api_key=COHERE_API_KEY,
        temperature=0
    )
    print("✓ Connected to Cohere.")
    
    print("Connecting to cloud vector store (PGVector)...")
    processor = DocumentProcessor(collection_name="football_docs")
    vector_store = processor.load_vector_store()
    print("✓ Connected to PGVector.")
    
    print("Loading knowledge graph from 'knowledge_graph.pkl'...")
    knowledge_graph = KnowledgeGraph(build_mode=False)
    kg_path = "knowledge_graph.pkl"
    
    if not os.path.exists(kg_path):
        st.error(f"⚠️ Knowledge graph file '{kg_path}' not found!")
        st.info("Please ensure 'knowledge_graph.pkl' is committed to your repository.")
        st.stop()
    
    knowledge_graph.load_graph(kg_path)
    print("✓ Knowledge graph loaded.")

    logger = MetadataLogger(graph=knowledge_graph.graph)
    query_engine = QueryEngine(vector_store, knowledge_graph, llm)
    
    return query_engine, vector_store, knowledge_graph


def main():
    st.set_page_config(
        page_title="⚽ Premier League Knowledge Bot",
        page_icon="⚽",
        layout="wide"
    )
    
    # Custom CSS - FROM CODE 1
    st.markdown("""
        <style>
        .chunk-box {
            background-color: #f4f4f4;
            border-left: 4px solid #2e7d32;
            padding: 12px;
            margin: 8px 0;
            border-radius: 5px;
            font-size: 13px;
        }
        .pdf-label {
            background-color: #c8e6c9;
            color: #1b5e20;
            padding: 4px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
            margin-right: 8px;
        }
        .traversal-path {
            background-color: #fff9c4;
            border-left: 4px solid #f57f17;
            padding: 12px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .node-info {
            background-color: #e0f2f1;
            border: 1px solid #00897b;
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            font-size: 12px;
        }
        .response-container {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .query-type-badge {
            display: inline-block;
            padding: 4px 12px;
            background-color: #1565c0;
            color: white;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 8px;
        }
        /* Added from Code 2's CSS block, just in case */
        .stAlert {
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header (Identical in both)
    st.title("⚽ Premier League Knowledge Bot")
    st.markdown("""
    Explore Premier League history, business, tactics, players, managers, and teams. 
    Powered by **The Mixer** (Michael Cox - Tactics) and **The Club** (Robinson & Clegg - Business)
    """)
    
    # Sidebar - FROM CODE 1 (with functional modifications)
    with st.sidebar:
        st.header("📖 About This Bot")
        st.markdown("""
        This chatbot provides insights into:
        - **Tactics & Strategy** - Formation evolution, pressing systems, tactical philosophy
        - **Club History** - How Premier League clubs developed, ownership, culture
        - **Players** - Legends, their impact, playing style
        - **Managers** - Coaching philosophies, achievements, influence
        - **Business & Money** - Club finances, transfers, foreign investment
        - **Team Performance** - Historic seasons, achievements
        
        ### How to use:
        1. This bot is cloud-hosted and reads a pre-built knowledge base.
        2. Ask any question about Premier League football.
        3. Get context-aware answers.
        
        ### Example Questions:
        - "What was Arsene Wenger's impact on Arsenal?"
        - "How did Manchester United dominate the 90s?"
        - "Describe the 4-3-3 formation"
        - "Who were key players in Liverpool's success?"
        - "How did pressing tactics change English football?"
        - "Compare the tactics of Pep Guardiola and Sir Alex Ferguson"
        - "What was the influence of Roman Abramovich on the Premier League"
        """)
        
        st.divider()
        
        # Database info - Layout from Code 1, Logic from Code 2
        kg_path = "knowledge_graph.pkl"
        if os.path.exists(kg_path):
            st.success("✅ Knowledge Base Found")
            try:
                # Use Code 2's method for loading stats
                kg_info = KnowledgeGraph(build_mode=False)
                kg_info.load_graph(kg_path)
                num_nodes = len(kg_info.graph.nodes)
                num_edges = len(kg_info.graph.edges)
                
                st.metric("Knowledge Concepts (Nodes)", num_nodes)
                st.metric("Connections (Edges)", num_edges)
                # Add density metric from Code 1
                st.metric("Graph Density", f"{num_edges / max(num_nodes * (num_nodes - 1) / 2, 1):.3f}")
            except Exception as e:
                st.warning(f"Could not read graph stats: {e}")
        else:
            st.error("❌ Knowledge Base Not Found")
            st.info("Please ensure 'knowledge_graph.pkl' is in your repository root.")
        
        # Use Code 2's button text ("Reload Models")
        if st.button("🔄 Reload Models"):
            st.cache_resource.clear()
            st.rerun()
        
        # Debug toggle (Identical in both)
        debug_mode = st.checkbox("🔍 Show Retrieval Details", value=False)
        
        st.divider()

        st.subheader("📊 Graph Connectivity")

        # Display connectivity metrics (From Code 1)
        if os.path.exists("graph_visualizations/connectivity_report.json"):
            with open("graph_visualizations/connectivity_report.json", 'r') as f:
                connectivity = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Is Connected", "✅ Yes" if connectivity['is_connected'] else "❌ No")
                st.metric("Components", connectivity['num_components'])
            with col2:
                st.metric("Largest Comp.", f"{connectivity['largest_component_size']} nodes")
                st.metric("Isolated Nodes", connectivity['isolated_nodes'])
            
            st.divider()
            
            if st.button("📂 Open Graph Dashboard"):
                st.info("Open `graph_visualizations/dashboard.html` in your browser")
        
        # Use Code 2's caption text
        st.caption("Made with ⚽ and knowledge graphs | Powered by Cohere")
    
    # Check for graph file - FROM CODE 2
    if not os.path.exists("knowledge_graph.pkl"):
        st.error("⚠️ Knowledge graph file not found!")
        st.info("Please run your build script locally and commit 'knowledge_graph.pkl' to your repository.")
        return
    
    # Load chatbot - FROM CODE 2
    try:
        with st.spinner("Connecting to knowledge base..."):
            query_engine, vector_store, knowledge_graph = load_chatbot()
        st.success("✅ Bot ready to answer your questions!")
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return
    
    # Initialize chat history - FROM CODE 2
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history - FROM CODE 2
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show debug info if available (Code 2's logic)
            if debug_mode and "debug_info" in message:
                with st.expander("🔍 Retrieval Details"):
                    debug_info = message["debug_info"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Query Type", debug_info.get('query_type', 'N/A'))
                    with col2:
                        st.metric("Nodes Traversed", len(debug_info.get('traversal_path', [])))
                    with col3:
                        st.metric("Chunks Used", len(debug_info.get('chunks', [])))
                    
                    if debug_info.get('chunks'):
                        st.write("**Sources Used:**")
                        source_counts = {}
                        for chunk in debug_info['chunks']:
                            source = chunk['source']
                            source_counts[source] = source_counts.get(source, 0) + 1
                        
                        source_cols = st.columns(min(len(source_counts), 4))
                        for col, (source, count) in zip(source_cols, list(source_counts.items())[:4]):
                            with col:
                                st.metric(source, f"{count} refs")
                        
                        st.write("**Retrieved Content:**")
                        for i, chunk in enumerate(debug_info['chunks'][:5]):
                            with st.container():
                                # This block will now use the new CSS from Code 1
                                st.markdown(f"""
                                <div class="chunk-box">
                                    <span class="pdf-label">{chunk['source']}</span>
                                    <span style="color: #666;">Page {chunk['page']}</span>
                                    <p style="margin-top: 8px;">{chunk['content'][:300]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
    
    # Chat input - FROM CODE 2
    if prompt := st.chat_input("Ask about Premier League football..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response - FROM CODE 2
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    response, traversal_path, filtered_content = query_engine.query(prompt)
                    
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                    
                    st.markdown(response_text)
                    
                    # Prepare debug info (Code 2's logic)
                    debug_info = {
                        'traversal_path': traversal_path,
                        'query_type': query_engine._classify_query(prompt),
                        'chunks': []
                    }
                    
                    # Extract chunk information from graph (Code 2's logic)
                    for node_id in traversal_path:
                        if node_id in filtered_content:
                            try:
                                node_data = knowledge_graph.graph.nodes[node_id]
                                content = node_data.get('content', 'Content not found')
                                metadata = node_data.get('metadata', {})
                                source = metadata.get('source_file', 'Unknown')
                                page = metadata.get('page_number', 'Unknown')
                                
                                debug_info['chunks'].append({
                                    'node_id': node_id,
                                    'source': os.path.basename(source),
                                    'page': page,
                                    'content': content
                                })
                            except Exception as e:
                                print(f"Error retrieving metadata for node {node_id}: {e}")
                    
                    # Add assistant message (Code 2's logic)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "debug_info": debug_info
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    if debug_mode:
                        with st.expander("🔧 Error Details"):
                            st.code(traceback.format_exc(), language="python")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "error": traceback.format_exc()
                    })
    
    # Footer controls - FROM CODE 2
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("🔴 Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()

