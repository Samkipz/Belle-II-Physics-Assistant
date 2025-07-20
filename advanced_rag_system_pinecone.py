import os
import json
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from pathlib import Path
import hashlib
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
    LLAMA_2 = "meta-llama/Llama-2-7b-chat-hf"
    VICUNA = "lmsys/vicuna-7b-v1.5"
    GROK_1 = "xai-org/grok-1"  # Placeholder for future integration


@dataclass
class RetrievalResult:
    """Represents a retrieval result with metadata"""
    content: str
    chunk_id: str
    document: str
    chunk_type: str
    page_number: int
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Represents a complete RAG response"""
    answer: str
    sources: List[RetrievalResult]
    model_used: str
    processing_time: float
    confidence_score: float
    chunk_types_used: List[str]


class AdvancedRAGSystem:
    """Advanced RAG system with multi-model support and sophisticated retrieval using Pinecone"""

    def __init__(self,
                 index_name: str = "belle2-advanced",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 default_model: ModelType = ModelType.MISTRAL_7B,
                 use_integrated_embeddings: bool = False):  # Changed default to False
        """
        Initialize the RAG system with Pinecone

        Args:
            index_name: Name of the Pinecone index
            embedding_model: Local embedding model (used if use_integrated_embeddings=False)
            default_model: Default LLM for answer generation
            use_integrated_embeddings: Whether to use Pinecone's integrated embeddings
        """

        self.index_name = index_name
        self.embedding_model_name = embedding_model
        self.default_model = default_model
        self.use_integrated_embeddings = use_integrated_embeddings

        # Initialize Pinecone
        self._init_pinecone()

        # Initialize local embedding model (always needed for now)
        self.embedding_model = SentenceTransformer(embedding_model)

        # Model configurations
        self.model_configs = {
            ModelType.MISTRAL_7B: {
                "api_url": "https://router.huggingface.co/v1/chat/completions",
                "temperature": 0.1,
                "max_tokens": 1500,
                "context_window": 8192
            },
            ModelType.LLAMA_2: {
                "api_url": "https://router.huggingface.co/v1/chat/completions",
                "temperature": 0.1,
                "max_tokens": 1500,
                "context_window": 4096
            },
            ModelType.VICUNA: {
                "api_url": "https://router.huggingface.co/v1/chat/completions",
                "temperature": 0.1,
                "max_tokens": 1500,
                "context_window": 4096
            }
        }

        # Retrieval strategies
        self.retrieval_strategies = {
            "semantic": self._semantic_retrieval,
            "hybrid": self._hybrid_retrieval,
            "multi_vector": self._multi_vector_retrieval
        }

    def _init_pinecone(self):
        """Initialize Pinecone client and index"""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)

        # Get or create index
        if self.index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,  # For BAAI/bge-large-en-v1.5 (was 768)
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            time.sleep(10)

        # Connect to index
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")

    def ingest_documents(self, jsonl_file: str):
        """Ingest processed documents into Pinecone"""
        logger.info(f"Ingesting documents from {jsonl_file}")

        vectors = []
        metadatas = []
        ids = []

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)

                    # Create document ID
                    doc_id = f"{data['document']}_{data['chunk_id']}"

                    # Prepare metadata
                    metadata = {
                        "document": data["document"],
                        "chunk_type": data["chunk_type"],
                        "page_number": data["page_number"],
                        # Store content in metadata
                        "content": data["content"],
                        **data["metadata"]
                    }

                    # Prepare vector data
                    if self.use_integrated_embeddings:
                        # For now, we'll use local embeddings since integrated embeddings
                        # require a different setup in the new Pinecone API
                        embedding = self.embedding_model.encode(
                            [data["content"]], convert_to_numpy=True)[0]
                        vector_data = {
                            "id": doc_id,
                            "values": embedding.tolist(),
                            "metadata": metadata
                        }
                    else:
                        # Use local embeddings
                        embedding = self.embedding_model.encode(
                            [data["content"]], convert_to_numpy=True)[0]
                        vector_data = {
                            "id": doc_id,
                            "values": embedding.tolist(),
                            "metadata": metadata
                        }

                    vectors.append(vector_data)
                    metadatas.append(metadata)
                    ids.append(doc_id)

                    if (line_num + 1) % 1000 == 0:
                        logger.info(f"Processed {line_num + 1} chunks")

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num + 1}: {e}")
                    continue

        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]

            # Upsert to Pinecone
            self.index.upsert(vectors=batch_vectors)

            logger.info(
                f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")

        logger.info(f"Ingestion complete! Total documents: {len(vectors)}")

    def retrieve(self,
                 query: str,
                 strategy: str = "hybrid",
                 top_k: int = 5,
                 chunk_type_filter: Optional[str] = None,
                 document_filter: Optional[str] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents using specified strategy"""

        if strategy not in self.retrieval_strategies:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        return self.retrieval_strategies[strategy](query, top_k, chunk_type_filter, document_filter)

    def _semantic_retrieval(self, query: str, top_k: int, chunk_type_filter: str = None, document_filter: str = None) -> List[RetrievalResult]:
        """Pure semantic retrieval using embeddings"""

        # Prepare filter
        filter_dict = {}
        if chunk_type_filter:
            filter_dict["chunk_type"] = chunk_type_filter
        if document_filter:
            filter_dict["document"] = document_filter

        # Use local embeddings (since integrated embeddings require different setup)
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True)[0]
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )

        return self._format_retrieval_results(results)

    def _hybrid_retrieval(self, query: str, top_k: int, chunk_type_filter: str = None, document_filter: str = None) -> List[RetrievalResult]:
        """Hybrid retrieval combining semantic and keyword search"""
        # Get semantic results
        semantic_results = self._semantic_retrieval(
            query, top_k * 2, chunk_type_filter, document_filter)

        # Get keyword results (simple implementation)
        keyword_results = self._keyword_retrieval(
            query, top_k * 2, chunk_type_filter, document_filter)

        # Combine and rerank
        combined_results = self._combine_and_rerank(
            semantic_results, keyword_results, top_k)

        return combined_results

    def _multi_vector_retrieval(self, query: str, top_k: int, chunk_type_filter: str = None, document_filter: str = None) -> List[RetrievalResult]:
        """Multi-vector retrieval using different embedding strategies and query expansion"""

        # Strategy 1: Original query embedding
        original_results = self._semantic_retrieval(
            query, top_k * 2, chunk_type_filter, document_filter)

        # Strategy 2: Query expansion with synonyms and related terms
        expanded_query = self._expand_query(query)
        expanded_results = self._semantic_retrieval(
            expanded_query, top_k * 2, chunk_type_filter, document_filter)

        # Strategy 3: Chunk-type specific retrieval
        type_specific_results = self._chunk_type_specific_retrieval(
            query, top_k, chunk_type_filter, document_filter)

        # Combine all strategies with sophisticated reranking
        all_results = original_results + expanded_results + type_specific_results

        # Remove duplicates and rerank
        unique_results = self._deduplicate_and_rerank(all_results, top_k)

        return unique_results

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms for physics concepts"""

        # Physics-specific query expansion
        physics_terms = {
            "luminosity": ["integrated luminosity", "instantaneous luminosity", "beam luminosity"],
            "detector": ["detection system", "measurement apparatus", "sensor"],
            "background": ["background processes", "noise", "interference"],
            "efficiency": ["detection efficiency", "reconstruction efficiency", "selection efficiency"],
            "resolution": ["energy resolution", "momentum resolution", "spatial resolution"],
            "calibration": ["calibration procedure", "calibration method", "calibration system"],
            "trigger": ["trigger system", "trigger logic", "trigger efficiency"],
            "reconstruction": ["particle reconstruction", "track reconstruction", "event reconstruction"],
            "selection": ["event selection", "particle selection", "cut selection"],
            "systematic": ["systematic uncertainty", "systematic error", "systematic effect"]
        }

        expanded_terms = []
        query_lower = query.lower()

        for term, synonyms in physics_terms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)

        # Add original query terms
        expanded_terms.extend(query.split())

        # Create expanded query (limit to avoid too long queries)
        expanded_query = " ".join(list(set(expanded_terms))[:10])

        return expanded_query if expanded_query != query else query

    def _chunk_type_specific_retrieval(self, query: str, top_k: int, chunk_type_filter: str = None, document_filter: str = None) -> List[RetrievalResult]:
        """Retrieve results optimized for different chunk types"""

        results = []

        # If no specific chunk type filter, try to optimize for different types
        if not chunk_type_filter:
            chunk_types = ["text", "table", "equation", "figure"]

            for chunk_type in chunk_types:
                # Adjust query for specific chunk types
                adjusted_query = self._adjust_query_for_chunk_type(
                    query, chunk_type)

                type_results = self._semantic_retrieval(
                    adjusted_query, top_k // 2, chunk_type, document_filter)
                results.extend(type_results)
        else:
            # Use the specified chunk type filter
            adjusted_query = self._adjust_query_for_chunk_type(
                query, chunk_type_filter)
            results = self._semantic_retrieval(
                adjusted_query, top_k, chunk_type_filter, document_filter)

        return results

    def _adjust_query_for_chunk_type(self, query: str, chunk_type: str) -> str:
        """Adjust query to be more suitable for specific chunk types"""

        if chunk_type == "table":
            # Add terms that suggest tabular data
            table_indicators = ["table", "data", "values",
                                "numbers", "statistics", "measurements"]
            return f"{query} {' '.join(table_indicators)}"

        elif chunk_type == "equation":
            # Add terms that suggest mathematical content
            equation_indicators = ["equation", "formula",
                                   "calculation", "mathematical", "theoretical"]
            return f"{query} {' '.join(equation_indicators)}"

        elif chunk_type == "figure":
            # Add terms that suggest visual content
            figure_indicators = ["figure", "plot",
                                 "graph", "diagram", "visualization", "image"]
            return f"{query} {' '.join(figure_indicators)}"

        else:  # text
            return query

    def _deduplicate_and_rerank(self, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """Remove duplicates and rerank results using multiple factors"""

        # Group by chunk_id to remove duplicates
        unique_results = {}
        for result in results:
            if result.chunk_id not in unique_results:
                unique_results[result.chunk_id] = result
            else:
                # Keep the one with higher similarity score
                if result.similarity_score > unique_results[result.chunk_id].similarity_score:
                    unique_results[result.chunk_id] = result

        # Convert back to list and calculate enhanced scores
        enhanced_results = []
        for result in unique_results.values():
            # Calculate enhanced score based on multiple factors
            enhanced_score = self._calculate_enhanced_score(result)
            enhanced_results.append((result, enhanced_score))

        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x[1], reverse=True)

        return [result for result, _ in enhanced_results[:top_k]]

    def _calculate_enhanced_score(self, result: RetrievalResult) -> float:
        """Calculate enhanced score based on multiple factors"""

        base_score = result.similarity_score

        # Factor 1: Content length (prefer longer, more detailed content)
        length_factor = min(len(result.content) / 1000,
                            1.0)  # Normalize to 0-1

        # Factor 2: Chunk type preference (prefer text and equations for physics)
        type_weights = {
            "text": 1.0,
            "equation": 0.9,
            "table": 0.8,
            "figure": 0.7
        }
        type_factor = type_weights.get(result.chunk_type, 0.5)

        # Factor 3: Document diversity (prefer results from different documents)
        # This would require tracking document distribution, simplified for now
        diversity_factor = 1.0

        # Combine factors with weights
        enhanced_score = (
            base_score * 0.6 +
            length_factor * 0.2 +
            type_factor * 0.15 +
            diversity_factor * 0.05
        )

        return enhanced_score

    def _keyword_retrieval(self, query: str, top_k: int, chunk_type_filter: str = None, document_filter: str = None) -> List[RetrievalResult]:
        """Simple keyword-based retrieval using Pinecone's sparse vectors"""
        # Extract keywords from query
        keywords = query.lower().split()

        # Get all documents and filter by keywords
        # Note: This is a simplified implementation
        # In production, you might want to use sparse vectors or hybrid search

        # For now, we'll use semantic search with a broader scope
        results = self._semantic_retrieval(
            query, top_k * 3, chunk_type_filter, document_filter)

        # Score based on keyword presence
        scored_results = []
        for result in results:
            score = sum(
                1 for keyword in keywords if keyword in result.content.lower())
            if score > 0:
                scored_results.append({
                    'result': result,
                    'score': score
                })

        # Sort by score and return top_k
        scored_results.sort(key=lambda x: x['score'], reverse=True)

        return [item['result'] for item in scored_results[:top_k]]

    def _combine_and_rerank(self, semantic_results: List[RetrievalResult], keyword_results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """Combine and rerank results from different retrieval methods"""
        # Create a combined list with scores
        combined = {}

        # Add semantic results with weight 0.7
        for result in semantic_results:
            combined[result.chunk_id] = {
                'result': result,
                'score': result.similarity_score * 0.7
            }

        # Add keyword results with weight 0.3
        for result in keyword_results:
            if result.chunk_id in combined:
                combined[result.chunk_id]['score'] += result.similarity_score * 0.3
            else:
                combined[result.chunk_id] = {
                    'result': result,
                    'score': result.similarity_score * 0.3
                }

        # Sort by combined score
        sorted_results = sorted(
            combined.values(), key=lambda x: x['score'], reverse=True)

        return [item['result'] for item in sorted_results[:top_k]]

    def _format_retrieval_results(self, pinecone_results) -> List[RetrievalResult]:
        """Format Pinecone results into RetrievalResult objects"""
        results = []

        for match in pinecone_results.matches:
            metadata = match.metadata

            results.append(RetrievalResult(
                content=metadata.get('content', ''),
                chunk_id=match.id,
                document=metadata.get('document', ''),
                chunk_type=metadata.get('chunk_type', ''),
                page_number=metadata.get('page_number', 0),
                similarity_score=match.score,
                metadata=metadata
            ))

        return results

    def generate_answer(self,
                        query: str,
                        retrieval_results: List[RetrievalResult],
                        model_type: ModelType = None,
                        temperature: float = None) -> RAGResponse:
        """Generate answer using specified model"""

        try:
            if model_type is None:
                model_type = self.default_model

            # Debug logging
            logger.info(f"Generating answer with model_type: {model_type}")
            logger.info(f"Model_type type: {type(model_type)}")
            logger.info(
                f"Model_type value: {model_type.value if hasattr(model_type, 'value') else model_type}")

            start_time = time.time()

            # Compose prompt
            prompt = self._compose_advanced_prompt(query, retrieval_results)

            # Get model configuration
            if model_type not in self.model_configs:
                raise ValueError(
                    f"Unknown model type: {model_type}. Available models: {list(self.model_configs.keys())}")

            config = self.model_configs[model_type]
            if temperature is not None:
                config = config.copy()
                config['temperature'] = temperature

            # Generate answer
            answer = self._query_model(prompt, model_type, config)

            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                retrieval_results, answer)

            processing_time = time.time() - start_time

            return RAGResponse(
                answer=answer,
                sources=retrieval_results,
                model_used=model_type.value,
                processing_time=processing_time,
                confidence_score=confidence_score,
                chunk_types_used=list(
                    set(r.chunk_type for r in retrieval_results))
            )
        except Exception as e:
            logger.error(f"Error in generate_answer: {e}")
            logger.error(f"Model type: {model_type}")
            logger.error(f"Model type class: {type(model_type)}")
            raise

    def _compose_advanced_prompt(self, query: str, retrieval_results: List[RetrievalResult]) -> str:
        """Compose advanced prompt with structured context"""

        # Group results by chunk type
        text_chunks = [r for r in retrieval_results if r.chunk_type == "text"]
        table_chunks = [
            r for r in retrieval_results if r.chunk_type == "table"]
        equation_chunks = [
            r for r in retrieval_results if r.chunk_type == "equation"]
        figure_chunks = [
            r for r in retrieval_results if r.chunk_type == "figure"]

        context_parts = []

        # Add text chunks
        if text_chunks:
            context_parts.append("TEXT CONTENT:")
            for i, chunk in enumerate(text_chunks[:3], 1):  # Limit to top 3
                context_parts.append(
                    f"Text {i} (from {chunk.document}, page {chunk.page_number}):\n{chunk.content}")

        # Add table chunks
        if table_chunks:
            context_parts.append("\nTABLE DATA:")
            for i, chunk in enumerate(table_chunks[:2], 1):  # Limit to top 2
                context_parts.append(
                    f"Table {i} (from {chunk.document}, page {chunk.page_number}):\n{chunk.content}")

        # Add equation chunks
        if equation_chunks:
            context_parts.append("\nEQUATIONS:")
            # Limit to top 2
            for i, chunk in enumerate(equation_chunks[:2], 1):
                context_parts.append(
                    f"Equation {i} (from {chunk.document}, page {chunk.page_number}):\n{chunk.content}")

        # Add figure chunks
        if figure_chunks:
            context_parts.append("\nFIGURES:")
            for i, chunk in enumerate(figure_chunks[:2], 1):  # Limit to top 2
                context_parts.append(
                    f"Figure {i} (from {chunk.document}, page {chunk.page_number}):\n{chunk.content}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are an expert Belle II physics assistant. Your task is to provide a comprehensive answer to the user's question based on the scientific documentation provided below.

CRITICAL INSTRUCTIONS:
1. DO NOT copy text directly from the context. Instead, synthesize and explain the information in your own words.
2. Base your answer EXCLUSIVELY on the provided context - do not use external knowledge.
3. If the context doesn't contain enough information, clearly state what information is missing.
4. Be precise, scientific, and well-structured in your response.
5. When referencing information, mention the source document and page number.
6. If you see equations, explain their meaning and significance.
7. If you see tables, interpret the data and explain its relevance.
8. If you see figures, describe what they show and their importance.

CONTEXT INFORMATION:
{context}

USER QUESTION: {query}

Please provide a comprehensive, well-structured answer that synthesizes the information from the context above. Write in a clear, scientific style that would be appropriate for a physics researcher:"""

        return prompt

    def _query_model(self, prompt: str, model_type: ModelType, config: Dict[str, Any]) -> str:
        """Query the specified model"""

        try:
            api_key = os.getenv("HF_API_KEY")
            if not api_key:
                raise ValueError("HF_API_KEY environment variable not set")

            # Validate model_type
            if not isinstance(model_type, ModelType):
                raise ValueError(
                    f"Invalid model_type: {model_type}. Expected ModelType enum, got {type(model_type)}")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model_type.value,
                "stream": False,
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"]
            }

            logger.info(f"Querying model: {model_type.value}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info(f"Temperature: {config['temperature']}")
            logger.info(f"Max tokens: {config['max_tokens']}")

            response = requests.post(
                config["api_url"], headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            logger.info(f"Model response status: {response.status_code}")
            logger.info(f"Response keys: {list(result.keys())}")

            if "choices" in result and len(result["choices"]) > 0:
                model_response = result["choices"][0]["message"]["content"]
                logger.info(
                    f"Model response length: {len(model_response)} characters")
                logger.info(
                    f"Model response preview: {model_response[:200]}...")
                return model_response
            else:
                error_msg = f"[Error from model API]: {result}"
                logger.error(error_msg)
                return error_msg

        except Exception as e:
            logger.error(
                f"Error querying model {model_type.value if hasattr(model_type, 'value') else model_type}: {e}")
            return f"[Error]: Failed to generate response from {model_type.value if hasattr(model_type, 'value') else model_type}"

    def _calculate_confidence(self, retrieval_results: List[RetrievalResult], answer: str) -> float:
        """Calculate confidence score based on retrieval quality and answer coherence"""

        if not retrieval_results:
            return 0.0

        # Average similarity score of retrieved documents
        avg_similarity = np.mean(
            [r.similarity_score for r in retrieval_results])

        # Answer length factor (longer answers might be more comprehensive)
        length_factor = min(len(answer) / 500, 1.0)  # Normalize to 0-1

        # Source diversity factor
        unique_docs = len(set(r.document for r in retrieval_results))
        diversity_factor = min(unique_docs / 3, 1.0)  # Normalize to 0-1

        # Combine factors
        confidence = (avg_similarity * 0.6 + length_factor *
                      0.2 + diversity_factor * 0.2)

        return min(confidence, 1.0)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Get index stats from Pinecone
            index_stats = self.index.describe_index_stats()

            # Extract relevant stats
            total_vectors = index_stats.total_vector_count

            # Analyze chunk types from a sample query
            try:
                sample_results = self.index.query(
                    # Fixed: Use 1024 dimensions for BAAI/bge-large-en-v1.5
                    vector=[0] * 1024,
                    top_k=1000,
                    include_metadata=True
                )

                chunk_types = {}
                documents = set()

                for match in sample_results.matches:
                    metadata = match.metadata
                    chunk_type = metadata.get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(
                        chunk_type, 0) + 1
                    documents.add(metadata.get('document', 'unknown'))

                return {
                    "total_chunks": total_vectors,
                    "chunk_type_distribution": chunk_types,
                    "unique_documents": len(documents),
                    "embedding_model": self.embedding_model_name,
                    "available_models": [model.value for model in ModelType],
                    "index_name": self.index_name,
                    "use_integrated_embeddings": self.use_integrated_embeddings
                }
            except Exception as query_error:
                # If the dummy query fails, return basic stats
                logger.warning(f"Could not get detailed stats: {query_error}")
                return {
                    "total_chunks": total_vectors,
                    "chunk_type_distribution": {},
                    "unique_documents": 0,
                    "embedding_model": self.embedding_model_name,
                    "available_models": [model.value for model in ModelType],
                    "index_name": self.index_name,
                    "use_integrated_embeddings": self.use_integrated_embeddings
                }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}


def main():
    """Example usage of the advanced RAG system with Pinecone"""

    # Initialize system
    rag_system = AdvancedRAGSystem(
        index_name="belle2-advanced",
        use_integrated_embeddings=True
    )

    # Ingest documents (run this once)
    # rag_system.ingest_documents("belle2_processed_chunks.jsonl")

    # Example query
    query = "How is integrated luminosity measured in Belle II experiments?"

    # Retrieve relevant documents
    results = rag_system.retrieve(query, strategy="hybrid", top_k=5)

    # Generate answer
    response = rag_system.generate_answer(query, results, ModelType.MISTRAL_7B)

    print(f"Query: {query}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Processing time: {response.processing_time:.2f}s")
    print(f"Sources used: {len(response.sources)}")
    print(f"Chunk types: {response.chunk_types_used}")


if __name__ == "__main__":
    main()
