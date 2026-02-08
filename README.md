# DHL Logistics RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with official DHL documentation, featuring semantic chunking, LangGraph agents, and comprehensive evaluation.

## Features

- **Semantic Chunking**: Intelligent document splitting based on embedding similarity, preserving context boundaries
- **LangGraph ReAct Agent**: Multi-tool orchestration for knowledge retrieval, shipment tracking, and cost estimation
- **Vector Search**: ChromaDB with cosine similarity for efficient document retrieval
- **Local Deployment**: Zero API cost using Ollama (Mistral) for embeddings and generation
- **Evaluation Framework**: Comprehensive metrics including Recall@K, MRR, and LLM-as-Judge scoring

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain, LangGraph |
| Vector DB | ChromaDB |
| LLM | Ollama (Mistral) |
| Embeddings | Ollama (Mistral) |
| Data | Official DHL PDFs |

## Project Structure

```
RAG/
├── dhl-rag-v2.ipynb          # Main notebook with full pipeline
├── dhl-rag-project.ipynb     # Original version (reference)
├── README.md
├── requirements.txt
└── data/
    ├── dhl_express_terms.pdf
    ├── dhl_customs_guide.pdf
    └── dhl_ecommerce_terms.pdf
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OFFLINE PHASE (Indexing)                   │
├─────────────────────────────────────────────────────────────┤
│  DHL PDFs → Load → Semantic Split → Embed → Store (ChromaDB)│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  ONLINE PHASE (Querying)                    │
├─────────────────────────────────────────────────────────────┤
│  User Question                                              │
│      ↓                                                      │
│  LangGraph ReAct Agent                                      │
│      ├── search_dhl_knowledge (Vector Retrieval)            │
│      ├── check_shipment (Status Query)                      │
│      └── estimate_shipping_cost (Cost Calculator)           │
│      ↓                                                      │
│  Mistral LLM Generates Response                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Install Ollama**
   ```bash
   # macOS
   brew install ollama

   # Or download from https://ollama.ai/download
   ```

2. **Pull Mistral model**
   ```bash
   ollama pull mistral
   ```

3. **Start Ollama server**
   ```bash
   ollama serve
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dhl-rag-system.git
cd dhl-rag-system

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook dhl-rag-v2.ipynb
```

## Usage

### Basic Query
```python
ask_dhl("What items are prohibited from shipping with DHL?")
```

### Shipment Tracking
```python
ask_dhl("What is the status of shipment DHL001?")
```

### Cost Estimation
```python
ask_dhl("How much to ship 5kg from New York to London?")
```

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Recall@10 | Source document in top 10 results | > 85% |
| MRR | Mean Reciprocal Rank | > 0.8 |
| LLM Score | Generation quality (1-5) | > 4.0 |

## Key Implementation Details

### Semantic Chunking
Unlike fixed-size chunking that may cut sentences mid-way, semantic chunking:
- Calculates embedding similarity between adjacent sentences
- Splits when similarity drops below threshold (85th percentile)
- Preserves topical coherence within chunks

```python
SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85
)
```

### ReAct Agent
The agent uses a Reasoning + Acting loop:
1. Receives user question
2. Decides which tool to use
3. Executes tool and observes result
4. Generates final answer based on observations

## License

MIT License

## Acknowledgments

- DHL for providing public documentation
- LangChain & LangGraph teams for the excellent framework
- Ollama for local LLM deployment
