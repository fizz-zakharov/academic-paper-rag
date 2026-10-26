import os
import re
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

load_dotenv()


def simple_tokenize(text: str) -> List[str]:
    """Tokenize and lower-case text for BM25 keyword matching."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """
    Hybrid Search Engine combining:
    1. Dense Semantic Search via ChromaDB (gemini-embedding-001)
    2. Sparse Keyword Search via in-memory BM25
    3. Fusion via Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        documents: List[Document],
        collection_name: str = "academic_paper",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        # 1. Structure-aware chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = self.splitter.split_documents(documents)

        # 2. Dense Vector Store Setup
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

        print(f"Indexing {len(self.chunks)} chunks into ChromaDB...")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        # Ingest with strict per-minute pacing (Free Tier allows max 100 items/min)
        existing_count = self.vector_store._collection.count()
        if existing_count < len(self.chunks):
            batch_size = 40  # 40 chunks per batch (well below 100 limit)
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i : i + batch_size]
                self.vector_store.add_documents(batch)
                print(f"  ✓ Embedded chunks {i+1} to {min(i+batch_size, len(self.chunks))} of {len(self.chunks)}")
                
                # If more chunks remain, sleep 30s to keep rolling requests safely under 100/min
                if i + batch_size < len(self.chunks):
                    print("  ⏳ Cooling down 30s to respect Google Free Tier 100 RPM quota...")
                    time.sleep(30)
            print("  ✓ All chunks successfully indexed into ChromaDB!")
        else:
            print(f"  ✓ Found existing ChromaDB index with {existing_count} chunks. Skipping re-embedding.")

        # 3. Sparse Keyword Store (In-Memory BM25)
        self.tokenized_corpus = [
            simple_tokenize(chunk.page_content) for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search_dense(self, query: str, top_k: int = 5) -> List[Document]:
        """Dense semantic search using cosine similarity in ChromaDB."""
        return self.vector_store.similarity_search(query, k=top_k)

    def search_sparse(self, query: str, top_k: int = 5) -> List[Document]:
        """Sparse lexical search using BM25Okapi."""
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]

    def hybrid_search(
        self, query: str, top_k: int = 4, rrf_k: int = 60
    ) -> List[Document]:
        """
        Merges dense and sparse ranks via Reciprocal Rank Fusion (RRF):
        Score(d) = sum(1 / (rrf_k + rank(d)))
        """
        dense_results = self.search_dense(query, top_k=top_k * 2)
        sparse_results = self.search_sparse(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 1. Rank Dense Results
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Rank Sparse Results
        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # 3. Sort by fused score
        sorted_doc_ids = sorted(
            rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True
        )

        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:top_k]]


if __name__ == "__main__":
    import sys
    from parser import parse_academic_pdf

    sample_pdf = "data/sample_papers/demo_paper.pdf"
    if len(sys.argv) > 1:
        sample_pdf = sys.argv[1]

    print(f"1. Parsing paper: {sample_pdf}")
    docs = parse_academic_pdf(sample_pdf)
    print(f"Extracted {len(docs)} pages.")

    print("\n2. Initializing Hybrid Search Engine...")
    retriever = HybridRetriever(docs)

    test_query = "What was the average AOD550 and PM2.5 concentration in 2019?"
    print(f"\n3. Querying: '{test_query}'")

    results = retriever.hybrid_search(test_query, top_k=3)
    for i, res in enumerate(results, start=1):
        print(f"\n--- Result #{i} [Page {res.metadata.get('page')}] ---")
        print(res.page_content[:350].strip())