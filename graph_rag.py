import os
import heapq
import time
from dotenv import load_dotenv
from typing import List, Tuple, Dict

from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate

from metadata_logger import MetadataLogger

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm: ChatCohere):
        """
        Initializes the QueryEngine with vector store, knowledge graph, and LLM.

        Args:
        - vector_store: A PGVector store (or any LangChain vector store)
        - knowledge_graph: A KnowledgeGraph instance (e.g., NetworkX graph)
        - llm: A language model instance (ChatCohere)
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_context_length = 4000
        self.metadata_logger = MetadataLogger()
        self.traversal_decisions = []
        
        # Create node lookup cache for fast O(1) access
        self.content_to_node_id = {}
        for node_id in self.knowledge_graph.graph.nodes:
            content = self.knowledge_graph.graph.nodes[node_id]['content']
            content_hash = hash(content[:100])
            self.content_to_node_id[content_hash] = node_id
        
        print("⚽ QueryEngine initialized for Premier League football knowledge")

    def _classify_query(self, query: str) -> str:
        """
        Classify the query type for optimized processing.
        """
        query_lower = query.lower()
        
        # CHECK MANAGER FIRST (before player)
        if any(kw in query_lower for kw in ['manager', 'coach', 'managed', 'coached', 
                                            'wenger', 'ferguson', 'clough', 'revie',
                                            'who managed', 'who coached', 'did coach',
                                            'did manage', 'under coach', 'under manager']):
            return 'manager'
        
        # Business/ownership queries
        elif any(kw in query_lower for kw in ['owner', 'ownership', 'investment', 'finance',
                                              'money', 'abramovich', 'glazer', 'kroenke',
                                              'takeover', 'business', 'revenue', 'commercial']):
            return 'business'
        
        # Player queries
        elif any(kw in query_lower for kw in ['player', 'striker', 'goalkeeper', 'defender', 
                                              'midfielder', 'footballer', 'how did']):
            return 'player'
        
        # Tactics queries
        elif any(kw in query_lower for kw in ['formation', 'setup', 'system', 'tactic', 'pressing',
                                              '4-3-3', '5-2-3', '3-5-2', 'how to', 'explain']):
            return 'tactics'
        
        # Team performance queries
        elif any(kw in query_lower for kw in ['team', 'performance', 'season', 'won', 'title', 
                                              'league', 'trophy', 'arsenal', 'united', 'liverpool']):
            return 'team_performance'
        
        # Comparison queries
        elif any(kw in query_lower for kw in ['vs', 'compare', 'comparison', 'difference', 
                                              'better', 'advantage']):
            return 'comparison'
        
        # Historical queries
        elif any(kw in query_lower for kw in ['why', 'how', 'reason', 'caused', 'evolved', 'changed']):
            return 'historical'
        
        # Cultural queries
        elif any(kw in query_lower for kw in ['culture', 'style', 'approach', 'philosophy', 'mentality']):
            return 'cultural'
        
        else:
            return 'general'

    def _expand_context(self, query: str, query_type: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str]]:
        """
        Expands the context by traversing the knowledge graph.
        """
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        self.traversal_decisions = []

        priority_queue = []
        distances = {}

        print("\n🧭 Traversing the knowledge graph:")

        for doc in relevant_docs:
            try:
                content_hash = hash(doc.page_content[:100])
                node_id = self.content_to_node_id.get(content_hash)
                
                if node_id is not None:
                    priority = 1.0
                    heapq.heappush(priority_queue, (priority, node_id))
                    distances[node_id] = priority
                
            except Exception as e:
                print(f"    ⚠️  Error initializing traversal: {e}")
                continue

        step = 0
        while priority_queue and len(traversal_path) < 20:
            current_priority, current_node = heapq.heappop(priority_queue)

            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node].get('concepts', [])

                new_context = expanded_context + "\n" + node_content if expanded_context else node_content
                
                if len(new_context) > self.max_context_length:
                    print(f"    ⚠️  Context limit reached ({len(new_context)} chars)")
                    break
                
                filtered_content[current_node] = node_content
                expanded_context = new_context

                print(f"\n    Step {step} - Node {current_node}:")
                print(f"    Concepts: {', '.join(node_concepts[:5]) if node_concepts else 'None'}")
                print(f"    Content length: {len(node_content)} chars")

                node_concepts_set = set(node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        if neighbor in traversal_path:
                            continue
                            
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data.get('weight', 0.5)
                        shared_concepts = edge_data.get('shared_concepts', [])

                        neighbor_concepts = set(self.knowledge_graph.graph.nodes[neighbor].get('concepts', []))
                        concept_relevance = len(set(shared_concepts)) / max(len(neighbor_concepts), 1)

                        adjusted_weight = (0.7 * edge_weight) + (0.3 * concept_relevance)
                        depth_penalty = 1.0 + (0.05 * len(traversal_path))
                        distance = (current_priority + (1 / adjusted_weight if adjusted_weight > 0 else float('inf'))) * depth_penalty

                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))
                            
                            decision = self.metadata_logger.log_traversal_decision(
                                current_node=current_node,
                                target_node=neighbor,
                                edge_weight=edge_weight,
                                shared_concepts=shared_concepts,
                                concept_relevance=concept_relevance,
                                reason_selected=f"query_type={query_type}",
                                accumulated_context_length=len(expanded_context),
                                traversal_depth=len(traversal_path)
                            )
                            self.traversal_decisions.append(decision)

        return expanded_context, traversal_path, filtered_content

    def _generate_answer(self, query: str, context: str, query_type: str) -> Tuple[str, float]:
        """
        Generate answer using LLM.
        """
        guidance_map = {
            'manager': "Focus on: philosophy, key decisions, achievements, team influence, legacy",
            'business': "Focus on: ownership, financial impact, strategic decisions, club development",
            'player': "Focus on: playing style, achievements, team impact, background, career evolution",
            'tactics': "Focus on: how it works, who used it, key players, performance impact, variations",
            'team_performance': "Focus on: results, contributing factors, key personnel, challenges, context",
            'comparison': "Focus on: key differences, strengths/weaknesses, context, outcomes, influence",
            'historical': "Focus on: emergence, key figures, drivers, impact, present connection",
            'cultural': "Focus on: underlying approach, team/player shaping, examples, performance, evolution",
            'general': "Provide relevant football context, players, teams, performance, and insights"
        }
        
        guidance = guidance_map.get(query_type, guidance_map['general'])
        
        response_prompt = PromptTemplate(
            input_variables=["query", "context", "guidance"],
            template="""Answer this football question using the provided knowledge base:

QUESTION: {query}

KNOWLEDGE BASE:
{context}

GUIDANCE: {guidance}

Provide a clear, direct answer based on the context. Be specific and reference relevant details:"""
        )
        
        try:
            print(f"\n🔧 Generating answer with Cohere...") 
            
            response_chain = response_prompt | self.llm
            response = response_chain.invoke({
                "query": query,
                "context": context,
                "guidance": guidance
            })
            
            if hasattr(response, 'content'):
                answer_text = response.content
            elif isinstance(response, str):
                answer_text = response
            else:
                answer_text = str(response)
            
            answer_length = len(answer_text)
            has_structure = any(phrase in answer_text.lower() for phrase in 
                                ['because', 'however', 'therefore', 'example', 'specifically', 'particularly'])
            
            if answer_length > 300 and has_structure:
                confidence = 0.85
            elif answer_length > 150:
                confidence = 0.75
            else:
                confidence = 0.6
            
            return answer_text, confidence
            
        except Exception as e:
            print(f"\n❌ ERROR in _generate_answer: {e}")
            
            if context:
                first_sentences = context.split('\n')[0:3]
                fallback_answer = " ".join(first_sentences)[:500]
                return fallback_answer, 0.5
            else:
                return f"Unable to generate answer: {str(e)}", 0.0

    def _retrieve_relevant_documents(self, query: str):
        """
        Retrieves relevant documents with query-type tuning.
        """
        print("\n🔍 Retrieving relevant documents...")
        
        enriched_query = self._enrich_query(query)
        print(f"    Enriched query: {enriched_query}")
        
        query_type = self._classify_query(query)
        
        if query_type in ['manager', 'player', 'business']:
            k = 12
        else:
            k = 8
        
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        results = base_retriever.invoke(enriched_query)
        print(f"    ✓ Retrieved {len(results)} relevant documents (k={k})")
        
        return results

    def _enrich_query(self, query: str) -> str:
        """
        Enrich query with related football terms.
        """
        enriched = query
        
        expansions = {
            'wenger': 'Arsene Wenger Arsenal manager biography history philosophy achievements',
            'ferguson': 'Alex Ferguson Sir Alex Manchester United biography history manager',
            'owner': 'ownership investment finance club business strategy',
            'abramovich': 'Roman Abramovich Chelsea owner investment',
            'glazer': 'Glazer family Manchester United ownership',
            'arsenal': 'Arsenal Gunners invincibles history',
            'united': 'Manchester United history manager achievements',
            'tactics': 'tactics formation system approach strategy',
        }
        
        for term, expansion in expansions.items():
            if term.lower() in query.lower():
                enriched += f" {expansion}"
        
        return enriched

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """
        Processes a query by retrieving documents, expanding context, and generating answer.
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"⚽ Processing Query: {query}")
        print(f"{'='*60}")
        
        query_type = self._classify_query(query)
        print(f"Query Type: {query_type}")
        
        relevant_docs = self._retrieve_relevant_documents(query)
        
        expanded_context, traversal_path, filtered_content = self._expand_context(
            query, query_type, relevant_docs
        )
        
        print("\n📝 Generating answer from accumulated context...")
        final_answer, confidence = self._generate_answer(query, expanded_context, query_type)
        
        response_time = time.time() - start_time
        
        session_id = self.metadata_logger.log_query_session(
            query=query,
            query_type=query_type,
            retrieved_docs=relevant_docs,
            traversal_path=traversal_path,
            traversal_decisions=self.traversal_decisions,
            filtered_content=filtered_content,
            final_answer=final_answer,
            response_time=response_time,
            answer_confidence=confidence
        )
        
        print(f"\n✅ Response generated in {response_time:.2f}s")
        print(f"📊 Session ID: {session_id}")
        

        return final_answer, traversal_path, filtered_content
