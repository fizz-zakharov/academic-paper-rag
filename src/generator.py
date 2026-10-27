import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def format_context_with_citations(documents: List[Document]) -> str:
    """
    Formats retrieved candidate chunks with unambiguous XML-like boundaries
    and page citations so the LLM knows the precise origin of each fact.
    """
    formatted_chunks = []
    for idx, doc in enumerate(documents, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        source = os.path.basename(doc.metadata.get("source", "Unknown Document"))
        
        chunk_entry = (
            f"--- [CHUNK {idx} | Source: {source} | Page: {page_num}] ---\n"
            f"{doc.page_content.strip()}\n"
        )
        formatted_chunks.append(chunk_entry)

    return "\n\n".join(formatted_chunks)


class AcademicGenerator:
    """Citation-grounded response generator powered by Gemini & LangChain LCEL."""

    def __init__(
        self, model_name: str = "models/gemini-3.8-flash", temperature: float = 0.1
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

        # Low temperature (0.1) suppresses hallucinations and random sampling
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

        # Strict citation-grounded prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert academic research assistant. Your sole job is to answer "
                "the user's question based STRICTLY and ONLY on the provided research paper chunks below.\n\n"
                "CRITICAL CITATION RULES:\n"
                "1. Every factual assertion, numeric figure, or finding MUST be immediately followed by a citation "
                "indicating its source page, formatted exactly like: [Page X].\n"
                "2. When discussing specific quantitative data or primary conclusions, quote key excerpts directly.\n"
                "3. If the provided chunks do NOT contain sufficient information to answer the question, state: "
                "'The provided paper context does not contain sufficient information to answer this question.' "
                "Do NOT speculate, assume, or pull from external training knowledge.\n"
                "4. Structure your response clearly using bullet points and concise technical summaries."
            ),
            (
                "human",
                "PAPER CONTEXT CHUNKS:\n\n{context}\n\n"
                "RESEARCH QUESTION: {question}\n\n"
                "GROUNDED ANSWER (with [Page X] citations):"
            )
        ])

        # LCEL declarative pipeline
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, question: str, retrieved_docs: List[Document]) -> str:
        """Executes the LCEL chain synchronously."""
        context_str = format_context_with_citations(retrieved_docs)
        return self.chain.invoke({"context": context_str, "question": question})

    def stream_response(self, question: str, retrieved_docs: List[Document]):
        """
        Streams generated response token-by-token.
        Essential for Streamlit real-time UI streaming in Milestone 4.
        """
        context_str = format_context_with_citations(retrieved_docs)
        for chunk in self.chain.stream({"context": context_str, "question": question}):
            yield chunk


if __name__ == "__main__":
    import sys
    from parser import parse_academic_pdf
    from retrieval import HybridRetriever

    sample_pdf = "data/sample_papers/demo_paper.pdf"
    if len(sys.argv) > 1:
        sample_pdf = sys.argv[1]

    print(f"1. Loading & Parsing Paper: {sample_pdf}")
    docs = parse_academic_pdf(sample_pdf)

    print("\n2. Initializing Retriever...")
    retriever = HybridRetriever(docs)

    print("\n3. Initializing Gemini Generator (LCEL Chain)...")
    generator = AcademicGenerator()

    # Academic test query requiring exact metrics
    query = "What was the AOD550 and PM2.5 in 2019, and how much did PM2.5 exceed the WHO guideline?"
    print(f"\n4. User Question: '{query}'")

    print("\n5. Running Hybrid Retrieval (ChromaDB + BM25 via RRF)...")
    retrieved_chunks = retriever.hybrid_search(query, top_k=3)

    print("\n6. Grounded Gemini Response:")
    print("=" * 60)
    # Testing streaming output
    for token in generator.stream_response(query, retrieved_chunks):
        print(token, end="", flush=True)
    print("\n" + "=" * 60)