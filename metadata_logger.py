import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

class MetadataLogger:
    
    def __init__(self, graph=None):
        self.graph = graph
        print("📊Metadata logging initialized.")
    
    def log_query_session(self, 
                          query: str,
                          query_type: str,
                          retrieved_docs: List[Any], 
                          traversal_path: List[int],
                          traversal_decisions: List[Dict],
                          filtered_content: Dict[int, str],
                          final_answer: str,
                          response_time: float,
                          answer_confidence: float = None,
                          is_complete: bool = None,
                          missing_elements: List[str] = None) -> str:
        """
        Logs a complete query session to stdout as a JSON string.
        """
        session_id = hashlib.md5(
            (query + datetime.now().isoformat()).encode()
        ).hexdigest()[:12]
        
        log_entry = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_type': query_type,
            
            'initial_retrieval': {
                'num_docs_retrieved': len(retrieved_docs),
            },          
            'graph_traversal': {
                'nodes_visited': traversal_path,
                'num_nodes': len(traversal_path),
            },
            'context_quality': {
                'total_context_length': sum(len(content) for content in filtered_content.values()),
                'num_chunks': len(filtered_content),
            },
            'response_quality': {
                'answer_length': len(final_answer),
                'response_time_seconds': response_time,
                'confidence_score': answer_confidence,
            },
        }
        
        try:
            print(json.dumps(log_entry))
            print(f"📊 Session logged: {session_id}")
        except Exception as e:
            print(f"⚠️  Could not serialize or print log: {e}")
            print(f"Fallback log: {{'session_id': '{session_id}', 'query': '{query}'}}")

        return session_id
    
    def log_traversal_decision(self,
                                 current_node: int,
                                 target_node: int,
                                 edge_weight: float,
                                 shared_concepts: List[str],
                                 concept_relevance: float,
                                 reason_selected: str,
                                 accumulated_context_length: int,
                                 traversal_depth: int) -> Dict:
        """Returns the decision dict to be aggregated."""
        decision = {
            'current_node': current_node,
            'target_node': target_node,
            'edge_weight': edge_weight,
            'num_shared_concepts': len(shared_concepts),
            'concept_relevance': concept_relevance,
            'reason_selected': reason_selected,
        }
        return decision

    # --- ALL FILE-READING METHODS ARE REMOVED ---
    # Methods like get_session_logs, generate_retrieval_report,
    # and get_statistics_summary are removed as they

    # depend on reading files that no longer exist.
