"""
Knowledge Graph Construction with Football Tactics Domain Optimization

--- VERCEL-SAFE REVISION ---
This file is now Vercel-safe. All heavy build-time libraries (spaCy, NLTK, etc.)
are "lazy-loaded" inside the 'build_mode=True' functions.
The main class can be safely imported by a Vercel app (like chatbot.py)
in 'build_mode=False' without loading any heavy dependencies.
"""

import networkx as nx
import pickle
import os
import time
# import matplotlib.pyplot as plt  # --- REMOVED (Heavy & Unused)
# import nltk                      # --- MOVED (Heavy)
# import spacy                     # --- MOVED (Heavy)
# from sklearn.metrics.pairwise import cosine_similarity # --- MOVED (Heavy)
from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm            # --- MOVED (Heavy)
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import json
# from football_tactics_preprocessor import FootballTacticsPreprocessor # --- MOVED (Heavy)

load_dotenv()


class KnowledgeGraph:
    def __init__(self, build_mode: bool = False, edges_threshold=0.7, 
                 save_progress=True, progress_dir="graph_progress"):
        """
        Initializes the KnowledgeGraph.

        Args:
        - build_mode (bool): If True, initializes all build tools (spaCy, NLTK).
                             If False, initializes as a lightweight loader (for Vercel).
        - edges_threshold (float): The threshold for adding edges based on similarity. 
        - save_progress (bool): Whether to save progress visualizations (build_mode only).
        - progress_dir (str): Directory to save progress (build_mode only).
        """
        self.graph = nx.Graph()
        self.edges_threshold = edges_threshold
        self.save_progress = save_progress
        self.progress_dir = progress_dir

        # Build-time-only dependencies
        self.lemmatizer = None
        self.nlp = None
        self.preprocessor = None
        self.concept_cache = {}
        self.extraction_metadata = []
        
        if build_mode:
            self._initialize_build_tools()
        
        print(f"🏈 KnowledgeGraph initialized (Build Mode: {build_mode})")

    def _initialize_build_tools(self):
        """Lazily loads all heavy build-time dependencies."""
        print("Build mode enabled. Initializing NLTK, spaCy, and preprocessor...")
        
        # --- LAZY IMPORTS ---
        import nltk
        from nltk.stem import WordNetLemmatizer
        from football_tactics_preprocessor import FootballTacticsPreprocessor
        # --- END LAZY IMPORTS ---

        # Download and import NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        self.lemmatizer = WordNetLemmatizer()

        # Load spaCy
        self.nlp = self._load_spacy_model()
        
        # Initialize preprocessor
        self.preprocessor = FootballTacticsPreprocessor()
        
        # Create progress directory
        if self.save_progress:
            os.makedirs(self.progress_dir, exist_ok=True)
            print(f"📊 Progress visualizations will be saved to: {self.progress_dir}/")
        
        print("✓ Build tools initialized successfully.")

    def _load_spacy_model(self):
        """
        Loads the spaCy NLP model, downloading it if necessary.
        """
        # --- LAZY IMPORTS ---
        import spacy
        from spacy.cli import download
        # --- END LAZY IMPORTS ---

        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content):
        """
        Extracts concepts using spaCy + football preprocessing (NO LLM needed).
        (This method is only called in build_mode)
        """
        if self.nlp is None or self.preprocessor is None:
            raise Exception("Cannot extract concepts: Build tools were not initialized. Set build_mode=True.")

        if content in self.concept_cache:
            return self.concept_cache[content]

        # Extract named entities using spaCy
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]

        # Extract football tactics entities
        tactical_entities = self.preprocessor.extract_tactical_entities(content)
        
        # Combine named entities with tactical entities
        tactical_concepts = (
            tactical_entities.get('formations', []) +
            tactical_entities.get('tactical_roles', []) +
            tactical_entities.get('defensive_concepts', []) +
            tactical_entities.get('offensive_concepts', [])
        )

        all_concepts = list(set(named_entities + tactical_concepts))[:15]
        
        self.extraction_metadata.append({
            'content_hash': hash(content[:100]),
            'total_concepts': len(all_concepts),
            'extraction_success': len(all_concepts) > 0
        })

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _compute_similarities(self, embeddings):
        """Computes the cosine similarity matrix for the embeddings."""
        # --- LAZY IMPORT ---
        from sklearn.metrics.pairwise import cosine_similarity
        # --- END LAZY IMPORT ---
        return cosine_similarity(embeddings)

    def build_graph(self, splits, embedding_model):
        """
        Builds the knowledge graph with football tactics optimizations.

        Args:
        - splits (list): A list of document splits.
        - embedding_model: An instance of an embedding model (e.g., CohereEmbeddings).
        """
        if not self.nlp:
            print("ERROR: Graph cannot be built. Please initialize class with build_mode=True.")
            return

        print("\n" + "="*60)
        print("🏈 Building Football Tactics Knowledge Graph")
        print("="*60)
        
        # Step 1: Add nodes
        print("\nStep 1: Adding nodes to graph...")
        self._add_nodes(splits)
        
        # Step 2: Create embeddings
        print("\nStep 2: Creating embeddings...")
        embeddings = self._create_embeddings(splits, embedding_model)
        
        # Step 3: Extract concepts
        print("\nStep 3: Extracting football tactical concepts and entities...")
        self._extract_concepts(splits)
        
        # Step 4: Add edges
        print("\nStep 4: Adding edges based on similarity and shared concepts...")
        self._add_edges(embeddings)
        
        self._save_extraction_metadata()
        
        print("\n" + "="*60)
        print(f"✅ Knowledge Graph Built Successfully!")
        print(f"   Total Nodes: {len(self.graph.nodes)}")
        print(f"   Total Edges: {len(self.graph.edges)}")
        print(f"   Extraction Success Rate: {self._get_extraction_success_rate():.1%}")
        print("="*60)

    def _add_nodes(self, splits):
        """Adds nodes to the graph from the document splits."""
        for i, split in enumerate(splits):
            self.graph.add_node(
                i,
                content=split.page_content,
                metadata=split.metadata,
                source_file=split.metadata.get('source_file', 'unknown'),
                page_number=split.metadata.get('page_number', 'unknown'),
                chunk_id=split.metadata.get('chunk_id', i)
            )
        print(f"✓ Added {len(splits)} nodes to the graph")

    def _create_embeddings(self, splits, embedding_model):
        """Creates embeddings for the document splits."""
        texts = [split.page_content for split in splits]
        print(f"Creating embeddings for {len(texts)} text chunks...")
        
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            print(f"   Processing embedding batch {batch_num}/{total_batches}...")
            
            try:
                batch_embeddings = embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"   ⚠️  Error creating embeddings: {e}")
                import numpy as np
                # Dimension for Cohere (embed-english-v3.0)
                all_embeddings.extend([np.zeros(1024).tolist() for _ in batch])
        
        print(f"✓ Created {len(all_embeddings)} embeddings")
        return all_embeddings

    def _extract_concepts(self, splits):
        """Extracts concepts for all document splits."""
        # --- LAZY IMPORT ---
        from tqdm import tqdm
        # --- END LAZY IMPORT ---
        print(f"Processing {len(splits)} chunks for concept extraction...")
        
        for i, split in enumerate(tqdm(splits, desc="Extracting concepts")):
            try:
                concepts = self._extract_concepts_and_entities(split.page_content)
                self.graph.nodes[i]['concepts'] = concepts
            except Exception as e:
                print(f"Error processing node {i}: {e}")
                self.graph.nodes[i]['concepts'] = []

    def _add_edges(self, embeddings):
        """Adds edges to the graph based on similarity and shared concepts."""
        # --- LAZY IMPORT ---
        from tqdm import tqdm
        # --- END LAZY IMPORT ---

        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)
        edges_added = 0
        
        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                
                if similarity_score > self.edges_threshold:
                    concepts1 = set(self.graph.nodes[node1].get('concepts', []))
                    concepts2 = set(self.graph.nodes[node2].get('concepts', []))
                    shared_concepts = concepts1 & concepts2
                    
                    edge_weight = self._calculate_edge_weight(
                        node1, node2,
                        similarity_score,
                        shared_concepts
                    )
                    
                    self.graph.add_edge(
                        node1, node2,
                        weight=edge_weight,
                        similarity=similarity_score,
                        shared_concepts=list(shared_concepts),
                        confidence=min(similarity_score, 1.0),
                        edge_type='semantic'
                    )
                    edges_added += 1
        
        print(f"✓ Added {edges_added} edges to the graph")

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """Calculates the weight of an edge."""
        max_possible_shared = min(
            len(self.graph.nodes[node1].get('concepts', [])),
            len(self.graph.nodes[node2].get('concepts', []))
        )
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        
        return alpha * similarity_score + beta * normalized_shared_concepts

    def save_graph(self, path: str = "knowledge_graph.pkl"):
        """
        Saves ONLY the NetworkX graph object to disk using pickle.
        """
        print(f"Saving knowledge graph object to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
        print("✓ Knowledge graph saved successfully.")

    def load_graph(self, path: str = "knowledge_graph.pkl"):
        """
        Loads the NetworkX graph object from disk.
        This method is Vercel-safe as it doesn't load build-time dependencies.
        """
        print(f"Loading knowledge graph from {path}...")
        if not os.path.exists(path):
            print(f"❌ ERROR: Graph file not found at {path}")
            print("Please run the build script first to generate the graph.")
            raise FileNotFoundError(f"Graph file not found: {path}")
            
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
        
        print(f"✓ Knowledge graph loaded successfully.")
        print(f"   Nodes: {len(self.graph.nodes)}, Edges: {len(self.graph.edges)}")

    def get_graph_statistics(self) -> Dict:
        """Returns statistics about the knowledge graph."""
        stats = {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if len(self.graph.nodes) > 0 else 0,
            'density': nx.density(self.graph) if len(self.graph.nodes) > 0 else 0,
            'is_connected': nx.is_connected(self.graph) if len(self.graph.nodes) > 0 else False,
            'num_connected_components': nx.number_connected_components(self.graph) if len(self.graph.nodes) > 0 else 0
        }
        return stats

    def print_graph_info(self):
        """Prints detailed information about the knowledge graph."""
        stats = self.get_graph_statistics()
        
        print("\n" + "="*60)
        print("🏈 Football Tactics Knowledge Graph Statistics")
        print("="*60)
        print(f"Number of Nodes: {stats['num_nodes']}")
        print(f"Number of Edges: {stats['num_edges']}")
        print(f"Average Degree: {stats['average_degree']:.2f}")
        print(f"Graph Density: {stats['density']:.4f}")
        print(f"Is Connected: {stats['is_connected']}")
        print(f"Number of Connected Components: {stats['num_connected_components']}")
        print("="*60)
    
    def _save_extraction_metadata(self):
        """Save extraction metadata to JSON for analysis."""
        if not self.extraction_metadata:
            return
        
        metadata_file = os.path.join(self.progress_dir, "extraction_metadata.json")
        total_extractions = len(self.extraction_metadata)
        successful = sum(1 for m in self.extraction_metadata if m['extraction_success'])
        
        summary = {
            'total_extractions': total_extractions,
            'successful_extractions': successful,
            'success_rate': successful / total_extractions if total_extractions > 0 else 0,
        }
        
        with open(metadata_file, 'w') as f:
            json.dump({'summary': summary, 'metadata': self.extraction_metadata}, f, indent=2)
        
        print(f"✓ Extraction metadata saved to {metadata_file}")
    
    def _get_extraction_success_rate(self) -> float:
        """Get extraction success rate."""
        if not self.extraction_metadata:
            return 0
        
        successful = sum(1 for m in self.extraction_metadata if m['extraction_success'])
        return successful / len(self.extraction_metadata)