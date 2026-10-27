import os
import tempfile
import streamlit as st

from src.parser import parse_academic_pdf
from src.retrieval import HybridRetriever
from src.generator import AcademicGenerator

st.set_page_config(
    page_title="Academic Paper Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Academic Paper Research Assistant")
st.caption("Hybrid Search (ChromaDB + BM25) with Citation Grounding via Gemini")


@st.cache_resource(show_spinner=False)
def initialize_pipeline(pdf_path: str):
    """
    Parses PDF, builds HybridRetriever (ChromaDB + BM25),
    and initializes AcademicGenerator. Cached across reruns.
    """
    docs = parse_academic_pdf(pdf_path)
    retriever = HybridRetriever(docs)
    # Explicitly enforce the working, active Gemini model
    generator = AcademicGenerator(model_name="models/gemini-3.8-flash", temperature=0.1)
    return docs, retriever, generator


# --- Sidebar Setup ---
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader(
        "Upload a Research PDF",
        type=["pdf"],
        help="Upload any single or two-column scientific paper.",
    )

    st.markdown("---")
    st.header("⚙️ Retrieval Parameters")
    top_k = st.slider("Top Context Chunks (k)", min_value=1, max_value=8, value=3)

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Determine active PDF source
active_pdf_path = None

if uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    active_pdf_path = temp_pdf_path
else:
    default_paper = "data/sample_papers/demo_paper.pdf"
    if os.path.exists(default_paper):
        active_pdf_path = default_paper

if not active_pdf_path:
    st.info("👈 Please upload an academic paper PDF from the sidebar to begin.")
    st.stop()

# --- Initialize Pipeline ---
with st.spinner("Indexing paper (Parsing layout, building ChromaDB & BM25)..."):
    docs, retriever, generator = initialize_pipeline(active_pdf_path)

with st.sidebar:
    st.success(f"Loaded: `{os.path.basename(active_pdf_path)}`")
    st.metric("Total Pages", len(docs))
    st.metric("Total Indexed Chunks", len(retriever.chunks))

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I have indexed your paper. Ask me anything about its methodology, metrics, datasets, or conclusions, and I will provide cited answers.",
        }
    ]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
if prompt := st.chat_input("Ask a question about this research paper..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Retrieval Step with status feedback
        with st.status("Searching document chunks via Hybrid RRF...", expanded=False) as status:
            retrieved_chunks = retriever.hybrid_search(prompt, top_k=top_k)
            status.update(label=f"Retrieved {len(retrieved_chunks)} relevant chunks", state="complete")

        # 2. Token Streaming directly into UI
        response = st.write_stream(
            generator.stream_response(prompt, retrieved_chunks)
        )

        # 3. Source drawer
        with st.expander("🔍 View Retrieved Source Chunks"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                page_no = chunk.metadata.get("page", "N/A")
                st.markdown(f"**Chunk #{i} (Page {page_no})**")
                st.text(chunk.page_content.strip())
                st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": response})