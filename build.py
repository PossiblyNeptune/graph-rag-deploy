import os
import glob
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# Import our modified, cloud-ready classes
from create_embeddings import DocumentProcessor
from create_database import KnowledgeGraph

load_dotenv()

def run_build_process():
    print("="*60)
    print("🏈 BUILDING CLOUD DATABASE (Cohere + PGVector)")
    print("="*60)
    
    # 1. Check for API Keys (from .env)
    cohere_key = os.environ.get("COHERE_API_KEY")
    pgvector_conn = os.environ.get("PGVECTOR_CONNECTION_STRING")
    
    if not cohere_key:
        print("❌ ERROR: COHERE_API_KEY not found.")
        print("Please add it to your .env file.")
        sys.exit(1)
    
    if not pgvector_conn:
        print("❌ ERROR: PGVECTOR_CONNECTION_STRING not found.")
        print("Please add it to your .env file.")
        print("Format: postgresql+psycopg://user:password@host:port/dbname")
        sys.exit(1)

    # 2. Find and Load PDFs
    data_folder = "data"
    pdf_pattern = os.path.join(data_folder, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in {data_folder}/")
        print(f"Please add your PDF files to the '{data_folder}' folder.")
        sys.exit(1)
        
    print(f"\n📄 Found {len(pdf_files)} PDF file(s):")
    all_documents = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"    Loading: {filename}...")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"    ✓ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"    ✗ Error loading {filename}: {e}")
            
    if not all_documents:
        print("\n❌ No documents could be loaded!")
        sys.exit(1)
        
    print(f"\n    Total pages loaded: {len(all_documents)}")

    # 3. Initialize DocumentProcessor (with Cohere Embeddings)
    print("\n1. Initializing Document Processor (Cohere)...")
    try:
        doc_processor = DocumentProcessor(collection_name="football_docs")
    except Exception as e:
        print(f"❌ Error initializing DocumentProcessor: {e}")
        sys.exit(1)
    
    # 4. Process and UPLOAD to PGVector
    print("\n2. Processing & Uploading to PGVector...")
    try:
        splits, vector_store = doc_processor.process_documents(all_documents)
        print("✓ Cloud Vector Database (PGVector) is populated.")
    except Exception as e:
        print(f"❌ Error during processing/upload: {e}")
        sys.exit(1)
    
    # 5. Build Knowledge Graph
    print("\n3. Building Knowledge Graph...")
    try:
        kg = KnowledgeGraph(build_mode=True)
        kg.build_graph(
            splits=splits, 
            embedding_model=doc_processor.embeddings
        )
    except Exception as e:
        print(f"❌ Error building knowledge graph: {e}")
        sys.exit(1)
    
    # 6. Save the final graph file
    graph_path = "knowledge_graph.pkl"
    print(f"\n4. Saving Knowledge Graph to {graph_path}...")
    try:
        kg.save_graph(graph_path)
    except Exception as e:
        print(f"❌ Error saving knowledge graph: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ BUILD COMPLETE!")
    print("="*60)
    kg.print_graph_info()
    print("\nYour cloud PGVector database is populated.")
    print(f"Your '{graph_path}' file is created.")
    print("\n🚀 Next Steps:")
    print("1. Commit 'knowledge_graph.pkl' to your Git repository")
    print("2. Push to GitHub")
    print("3. Deploy to Streamlit Cloud")
    print("4. Add environment variables in Streamlit Cloud settings:")
    print("    - COHERE_API_KEY")
    print("    - PGVECTOR_CONNECTION_STRING")
    print("="*60)

if __name__ == "__main__":
    run_build_process()
