# 📚 Academic Paper Research Assistant (Hybrid RAG)

An end-to-end Retrieval-Augmented Generation (RAG) system engineered specifically for scientific and academic literature. The platform resolves multi-column text reading errors via coordinate-aware extraction, pairs dense semantic embeddings with sparse keyword search via Reciprocal Rank Fusion (RRF), and enforces strict citation grounding back to source pages and sections.

---

## ⚡ Key Highlights

- **Multi-Column Layout Preservation:** Overcomes standard text-stitching bugs across two-column scientific papers using block-level bounding box coordinate extraction (`PyMuPDF`).
- **Hybrid Retrieval Engine (Dense + Sparse):** Merges semantic vector search with lexical BM25 retrieval using **Reciprocal Rank Fusion (RRF)** to accurately capture both conceptual themes and exact scientific entities (e.g., formulas, metrics, dataset names).
- **Citation-Grounded Generation:** Enforces strict provenance by binding synthesized findings to explicit source page tags (`[Page X]`) and direct quotes, preventing hallucinations.
- **Interactive Web Interface:** Streamlit dashboard featuring multi-turn conversation memory, real-time token streaming, and an inspection drawer for retrieved source passages.

---

## 🏗️ System Architecture

```text
                              [ Academic Research PDF ]
                                          │
                                          ▼
                      [ Coordinate-Aware Parser (PyMuPDF) ]
                        ├─ Bounding box coordinate sorting
                        └─ Column partitioning & header recovery
                                          │
                                          ▼
                      [ Structure-Aware Text Splitter ]
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
           [ Dense Vector Index ]                    [ Sparse Lexical Index ]
          Google Gemini Embeddings                    In-Memory BM25Okapi
             (ChromaDB Storage)                      (Term Frequencies & IDF)
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          ▼
                          [ Reciprocal Rank Fusion (RRF) ]
                              Score(d) = Σ 1 / (60 + rank)
                                          │
                                          ▼
                             [ Top-k Context Chunks ]
                                          │
                                          ▼
                      [ Citation-Grounded LCEL Chain ]
                        ├─ Strict Prompt Grounding Rules
                        └─ Google Gemini LLM (Low Temperature)
                                          │
                                          ▼
                      [ Streamlit UI / Token Streaming ]
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM Orchestration | LangChain (LCEL) |
| Large Language Model | Google Gemini (`gemini-3.8-flash`) |
| Embeddings | Google Gemini Embeddings (`gemini-embedding-001`) |
| Vector Database | ChromaDB (Local Persistence) |
| Lexical Search | `rank-bm25` (BM25Okapi) |
| PDF Extraction | PyMuPDF (`fitz`) |
| Frontend UI | Streamlit |

---

## 📁 Repository Structure

```text
academic-paper-rag/
├── data/
│   └── sample_papers/       # Directory for sample research PDFs
├── chroma_db/               # Persistent local vector storage
├── src/
│   ├── __init__.py
│   ├── parser.py            # Coordinate-based multi-column PDF parser
│   ├── retrieval.py         # Dual-engine search (ChromaDB + BM25 + RRF)
│   └── generator.py         # Citation-grounded generator via LangChain LCEL
├── app.py                   # Streamlit web application
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variable template
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ (Python 3.10 or 3.11 recommended)
- A Google AI Studio API Key ([Get an API key here](https://aistudio.google.com/app/apikey))

### 2. Clone and Set Up the Environment

```bash
git clone https://github.com/<your-username>/academic-paper-rag.git
cd academic-paper-rag

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure API Credentials

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Add your Gemini API key inside `.env`:

```text
GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

### 4. Launch the Web Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Upload any single or two-column scientific paper (or query the pre-loaded sample paper).

---

## 🧪 Modular Testing via CLI

Each pipeline layer can be executed independently from the terminal:

```bash
# 1. Test multi-column coordinate parsing
python src/parser.py data/sample_papers/demo_paper.pdf

# 2. Test Hybrid Search (ChromaDB + BM25 via RRF)
python src/retrieval.py data/sample_papers/demo_paper.pdf

# 3. Test Citation Grounding & Token Streaming
python src/generator.py data/sample_papers/demo_paper.pdf
```

---

## ⚙️ How It Works

### 1. Multi-Column Reading Order Parsing

Standard PDF extractors scan strictly along horizontal lines ($y$-axis), interleaving text between columns into garbled sentences. This parser uses PyMuPDF to inspect the bounding boxes `(x0, y0, x1, y1)` of each text block:

- Blocks spanning $\ge 80\%$ of page width are classified as full-width headers/abstracts.
- Remaining blocks are partitioned along the page horizontal midpoint into Column 1 and Column 2.
- Blocks are sorted top-to-bottom ($y_0$) within Column 1 first, then Column 2, preserving reading order.

### 2. Reciprocal Rank Fusion (RRF)

Cosine similarity ($0.0$ to $1.0$) and BM25 scores ($0$ to $\infty$) have vastly different score distributions. Instead of using arbitrary score normalization weights, RRF merges results purely on their relative rank positions:

$$
\text{RRF\_Score}(d) = \sum_{m \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{k + \text{rank}_m(d)}
$$

(Using smoothing constant $k = 60$.) Chunks surfacing near the top of both search methods receive the highest priority.

### 3. Citation-Grounded Synthesis

Retrieved context chunks are formatted with explicit source boundaries (`--- [CHUNK i | Source | Page] ---`). The generation chain uses a low temperature ($T = 0.1$) and prompt rules that:

- mandate claim-level citations (`[Page X]`),
- require quoting quantitative findings directly, and
- reject speculation when context is insufficient.

---
