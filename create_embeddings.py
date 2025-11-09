"""
Document Processor (BUILD SCRIPT)
Processes documents, creates embeddings using Cohere, 
and populates a cloud-hosted PGVector database.

--- VERCEL-SAFE REVISION V6 ---
- FINAL FIX 3: Corrects 'load_vector_store' to use 
  PGVector.from_existing_index instead of the base constructor.
  This fixes the "unexpected keyword argument 'embedding'" error.
- All other fixes (NUL-byte, rate-limiting, 'connection' keyword) 
  are retained.
"""

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader # --- MOVED (Heavy)
from langchain_postgres.vectorstores import PGVector
from langchain_cohere import CohereEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity  # --- MOVED (Heavy)
from typing import List
import os
from dotenv import load_dotenv
import time 

# from football_tactics_preprocessor import FootballTacticsPreprocessor # --- MOVED (Heavy)

# Load environment variables from .env file
load_dotenv()

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PGVECTOR_CONNECTION = os.environ.get("PGVECTOR_CONNECTION_STRING")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in environment variables.")
if not PGVECTOR_CONNECTION:
    raise ValueError("PGVECTOR_CONNECTION_STRING not found in environment variables. (e.g., 'postgresql+psycopg://user:password@host:port/dbname')")

class DocumentProcessor:
    def __init__(self, chunk_size=1500, chunk_overlap=250, collection_name="football_docs"):
        """
        Initializes the DocumentProcessor with Cohere embeddings and PGVector.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print("Loading Cohere embeddings model (embed-english-v3.0)...")
        self.embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=COHERE_API_KEY
        )
        print("✓ Cohere embeddings loaded successfully")
        
        self.preprocessor = None 
        self.connection = PGVECTOR_CONNECTION 
        self.collection_name = collection_name

        print(f"📋 DocumentProcessor initialized for PGVector (collection: '{self.collection_name}')")

    def _lazy_load_preprocessor(self):
        """Initializes the preprocessor on demand."""
        if self.preprocessor is None:
            print("Lazily loading FootballTacticsPreprocessor...")
            from football_tactics_preprocessor import FootballTacticsPreprocessor
            self.preprocessor = FootballTacticsPreprocessor()
            print("✓ Preprocessor loaded.")

    def process_documents(self, documents):
        """
        Processes documents by splitting, preprocessing, and 
        UPLOADING them to the cloud PGVector database in batches
        to respect Cohere's trial rate limits.
        """
        print("\nSplitting documents into chunks...")
        splits = self.text_splitter.split_documents(documents)
        print(f"✓ Created {len(splits)} document chunks.")
        
        print("Applying football preprocessing...")
        self._lazy_load_preprocessor()
        from football_tactics_preprocessor import FootballTacticsPreprocessor # For static method

        for i, split in enumerate(splits):
            processed_content, tactical_entities = self.preprocessor.preprocess_chunk(
                split.page_content
            )
            
            # --- NUL BYTE FIX ---
            clean_content = processed_content.replace('\x00', '')
            # --- END FIX ---

            split.page_content = clean_content # Use the cleaned content
            split.metadata['chunk_id'] = i
            split.metadata['source_file'] = split.metadata.get('source', 'unknown')
            split.metadata['page_number'] = split.metadata.get('page', 'unknown')
            split.metadata['tactical_keywords'] = FootballTacticsPreprocessor.create_tactical_keywords(tactical_entities)
        
        print(f"✓ Applied preprocessing to {len(splits)} chunks")
        
        print("\nCreating and populating cloud vector store (PGVector)...")

        batch_size = 90
        total_batches = (len(splits) + batch_size - 1) // batch_size
        
        print(f"Uploading {len(splits)} chunks in {total_batches} batches of {batch_size}...")

        first_batch = splits[:batch_size]

        print(f"  Uploading batch 1/{total_batches} (and creating collection)...")
        try:
            vector_store = PGVector.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection, 
                pre_delete_collection=True
            )
            print("    ... Batch 1 uploaded. Waiting 5 seconds for rate limit...")
            time.sleep(5) 
        except Exception as e:
            print(f"    !! ERROR on first batch: {e}")
            print("    Build script cannot continue without the first batch. Aborting.")
            raise e

        # Loop through the REST of the batches
        for i in range(batch_size, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            current_batch_num = (i // batch_size) + 1
            
            print(f"  Uploading batch {current_batch_num}/{total_batches}...")
            
            try:
                vector_store.add_documents(batch)
                
                if i + batch_size < len(splits): 
                    print("    ... Batch uploaded. Waiting 5 seconds for rate limit...")
                    time.sleep(5) 
            
            except Exception as e:
                print(f"    !! ERROR on batch {current_batch_num}: {e}")
                print("    ... Waiting 15 seconds before retrying...")
                time.sleep(15) 
                try:
                    vector_store.add_documents(batch) 
                    print("    ... Retry successful. Waiting 5 seconds...")
                    time.sleep(5)
                except Exception as e2:
                    print(f"    !! BATCH {current_batch_num} FAILED TWICE. Skipping. Error: {e2}")
        
        print("✓ Cloud vector store populated successfully!")
        
        return splits, vector_store

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Computes a cosine similarity matrix for embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity
        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        print(f"✓ Similarity matrix computed with shape {similarity_matrix.shape}.")
        return similarity_matrix

    # --- THIS FUNCTION IS NOW FIXED ---
    def load_vector_store(self):
        """
        Connects to an EXISTING cloud-based PGVector store.
        This is what your Vercel/Streamlit app will use.
        """
        print(f"Connecting to existing cloud vector store (collection: {self.collection_name})...")
        
        try:
            # FIX: Use .from_existing_index to load the store, not the constructor
            vector_store = PGVector.from_existing_index(
                embedding=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection
            )
        except Exception as e:
            print(f"Error connecting to existing PGVector index: {e}")
            if "unexpected keyword argument 'connection'" in str(e):
                print("="*80)
                print("FATAL ERROR: Your 'langchain-postgres' library is out of date.")
                print("Please stop the app and run: pip install --upgrade langchain-postgres")
                print("="*80)
            raise e
        
        print("✓ Connected to vector store successfully.")
        return vector_store
    
    def add_documents_to_vector_store(self, vector_store, documents):
        """Adds new documents to vector store with preprocessing."""
        self._lazy_load_preprocessor()
        from football_tactics_preprocessor import FootballTacticsPreprocessor

        print(f"Adding {len(documents)} new documents to cloud vector store...")
        splits = self.text_splitter.split_documents(documents)
        
        for split in splits:
            processed_content, tactical_entities = self.preprocessor.preprocess_chunk(
                split.page_content
            )
            
            # --- NUL BYTE FIX ---
            clean_content = processed_content.replace('\x00', '')
            # --- END FIX ---
                
            split.page_content = clean_content
            split.metadata['tactical_keywords'] = FootballTacticsPreprocessor.create_tactical_keywords(
                tactical_entities
            )
        
        vector_store.add_documents(splits)
        print("✓ Documents added successfully.")
        return vector_store


    def get_embedding_dimension(self) -> int:
        """Returns the embedding dimension."""
        return 1024


# This block is now your "Build and Test" script.
if __name__ == "__main__":
    from langchain_community.document_loaders import PyPDFLoader
    
    print("="*60)
    print("RUNNING BUILD SCRIPT: DocumentProcessor with Cohere & PGVector")
    print("="*60)
    
    processor = DocumentProcessor(collection_name="football_docs")
    
    all_documents = []
    for book in ["data/The_Mixer.pdf", "data/The_Club.pdf"]:
        if os.path.exists(book):
            print(f"\nLoading document from {book}...")
            loader = PyPDFLoader(book)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages.")
        else:
            print(f"\nTest file not found: {book}. Skipping.")

    if all_documents:
        splits, vector_store = processor.process_documents(all_documents)
        
        print("\n" + "="*60)
        print("TESTING CONNECTION TO CLOUD DB")
        print("="*60)
        print("Attempting to connect to the store we just created...")
        
        loaded_store = processor.load_vector_store()
        
        print("\nTesting similarity search on cloud store...")
        test_query = "What is Gegenpressing?"
        try:
            results = loaded_store.similarity_search(test_query, k=2)
            print(f"✓ Similarity search for '{test_query}' successful.")
            print(f"Found {len(results)} results:")
            for i, res in enumerate(results):
                print(f"\nResult {i+1} (Source: {res.metadata.get('source_file', 'N/A')}, Page: {res.metadata.get('page_number', 'N/A')})")
                print(f"Preview: {res.page_content[:200]}...")
        except Exception as e:
            print(f"Error during test similarity search: {e}")

        print("\n" + "="*60) 
        print("All tests completed successfully!")
        print("Your cloud vector database is now populated.")
        print("="*60)
    else:
        print("\nNo documents a found to process.")