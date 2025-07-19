import os
import json
import chromadb
from chromadb.config import Settings
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
    """Advanced RAG system with multi-model support and sophisticated retrieval"""

    def __init__(self,
                 chroma_db_dir: str = "./chroma_db",
                 collection_name: str = "belle2_advanced",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 default_model: ModelType = ModelType.MISTRAL_7B):

        self.chroma_db_dir = chroma_db_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.default_model = default_model

        # Initialize components
        self.chroma_client = chromadb.Client(
            Settings(persist_directory=chroma_db_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            collection_name)
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

    def ingest_documents(self, jsonl_file: str):
        """Ingest processed documents into ChromaDB"""
        logger.info(f"Ingesting documents from {jsonl_file}")

        documents = []
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
                        **data["metadata"]
                    }

                    documents.append(data["content"])
                    metadatas.append(metadata)
                    ids.append(doc_id)

                    if (line_num + 1) % 1000 == 0:
                        logger.info(f"Processed {line_num + 1} chunks")

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num + 1}: {e}")
                    continue

        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                batch_docs, convert_to_numpy=True)

            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            logger.info(
                f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")

        logger.info(f"Ingestion complete! Total documents: {len(documents)}")

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
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True)[0]

        # Build where clause for filtering
        where_clause = {}
        if chunk_type_filter:
            where_clause["chunk_type"] = chunk_type_filter
        if document_filter:
            where_clause["document"] = document_filter

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
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
        """Multi-vector retrieval using different embedding strategies"""
        # This would implement more sophisticated multi-vector retrieval
        # For now, return semantic retrieval
        return self._semantic_retrieval(query, top_k, chunk_type_filter, document_filter)

    def _keyword_retrieval(self, query: str, top_k: int, chunk_type_filter: str = None, document_filter: str = None) -> List[RetrievalResult]:
        """Simple keyword-based retrieval"""
        # Extract keywords from query
        keywords = query.lower().split()

        # Get all documents and filter by keywords
        all_results = self.collection.get()

        scored_results = []
        for i, doc in enumerate(all_results['documents']):
            score = sum(1 for keyword in keywords if keyword in doc.lower())
            if score > 0:
                scored_results.append({
                    'index': i,
                    'score': score,
                    'document': doc,
                    'metadata': all_results['metadatas'][i],
                    'id': all_results['ids'][i]
                })

        # Sort by score and return top_k
        scored_results.sort(key=lambda x: x['score'], reverse=True)

        return [RetrievalResult(
            content=result['document'],
            chunk_id=result['id'],
            document=result['metadata']['document'],
            chunk_type=result['metadata']['chunk_type'],
            page_number=result['metadata']['page_number'],
            similarity_score=result['score'],
            metadata=result['metadata']
        ) for result in scored_results[:top_k]]

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

    def _format_retrieval_results(self, chroma_results) -> List[RetrievalResult]:
        """Format ChromaDB results into RetrievalResult objects"""
        results = []

        for i, (doc, metadata, distance) in enumerate(zip(
            chroma_results['documents'][0],
            chroma_results['metadatas'][0],
            chroma_results['distances'][0]
        )):
            # Convert distance to similarity score (1 - normalized distance)
            similarity_score = 1 - \
                (distance / max(chroma_results['distances'][0]))

            results.append(RetrievalResult(
                content=doc,
                chunk_id=chroma_results['ids'][0][i],
                document=metadata['document'],
                chunk_type=metadata['chunk_type'],
                page_number=metadata['page_number'],
                similarity_score=similarity_score,
                metadata=metadata
            ))

        return results

    def generate_answer(self,
                        query: str,
                        retrieval_results: List[RetrievalResult],
                        model_type: ModelType = None,
                        temperature: float = None) -> RAGResponse:
        """Generate answer using specified model"""

        if model_type is None:
            model_type = self.default_model

        start_time = time.time()

        # Compose prompt
        prompt = self._compose_advanced_prompt(query, retrieval_results)

        # Get model configuration
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
            chunk_types_used=list(set(r.chunk_type for r in retrieval_results))
        )

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

        prompt = f"""You are an expert Belle II physics assistant with access to comprehensive scientific documentation. Answer the user's question using ONLY the information provided in the context below.

IMPORTANT INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the provided context
2. If the context doesn't contain enough information, clearly state what information is missing
3. Do not add any external knowledge or make assumptions
4. Be precise and scientific in your response
5. Reference specific sources when possible (document names, page numbers)
6. If you see equations, explain them clearly
7. If you see tables, interpret the data accurately
8. If you see figures, describe what they show

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a comprehensive, accurate answer based on the context above:"""

        return prompt

    def _query_model(self, prompt: str, model_type: ModelType, config: Dict[str, Any]) -> str:
        """Query the specified model"""

        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY environment variable not set")

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

        try:
            response = requests.post(
                config["api_url"], headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"[Error from model API]: {result}"

        except Exception as e:
            logger.error(f"Error querying model {model_type.value}: {e}")
            return f"[Error]: Failed to generate response from {model_type.value}"

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
            count = self.collection.count()

            # Get sample documents for analysis
            sample_results = self.collection.get(limit=1000)

            # Analyze chunk types
            chunk_types = {}
            documents = set()

            for metadata in sample_results['metadatas']:
                chunk_type = metadata.get('chunk_type', 'unknown')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                documents.add(metadata.get('document', 'unknown'))

            return {
                "total_chunks": count,
                "chunk_type_distribution": chunk_types,
                "unique_documents": len(documents),
                "embedding_model": self.embedding_model_name,
                "available_models": [model.value for model in ModelType]
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}


def main():
    """Example usage of the advanced RAG system"""

    # Initialize system
    rag_system = AdvancedRAGSystem()

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
