# DHL Logistics RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for DHL logistics documentation.

## Features

- **Semantic Chunking**: Intelligent document splitting based on embedding similarity
- **LangGraph ReAct Agent**: Multi-tool orchestration with reasoning capabilities
- **Vector Search**: ChromaDB with cosine similarity for efficient retrieval
- **Local Deployment**: Zero API cost using Ollama (Mistral 7B)
- **Comprehensive Evaluation**: Recall@K, MRR, and LLM-as-Judge metrics
- **Docker Support**: Full containerization with docker-compose

## Project Structure

```
DHL-RAG-System/
├── dhl_logistics_rag/           # Main package
│   ├── common/                  # Config, constants
│   ├── core/                    # RAG pipeline components
│   │   ├── document_loader.py   # PDF loading
│   │   ├── chunker.py          # Semantic/fixed chunking
│   │   ├── embedder.py         # Embedding service
│   │   ├── vector_store.py     # ChromaDB manager
│   │   └── rag_pipeline.py     # Pipeline orchestration
│   ├── agents/                  # LangGraph agents
│   │   └── rag_agent.py        # ReAct agent
│   ├── tools/                   # Agent tools
│   │   ├── knowledge_search.py # Vector search tool
│   │   ├── shipment_tracker.py # Tracking tool
│   │   └── cost_estimator.py   # Cost calculation tool
│   ├── evaluation/             # Evaluation framework
│   └── utils/                  # Utilities
├── tests/                      # Unit tests
├── scripts/                    # Entry points
├── config/                     # Configuration files
├── data/                       # PDF documents
├── Makefile                    # Build commands
├── docker-compose.yml          # Container orchestration
├── Dockerfile.backend          # Backend container
└── Dockerfile.vectordb         # ChromaDB container
```

## Quick Start

### Prerequisites

```bash
# Install Ollama
brew install ollama

# Pull Mistral model
ollama pull mistral

# Start Ollama
ollama serve
```

### Installation

```bash
# Clone repository
git clone https://github.com/Zayn407/DHL-Logistics-RAG-System.git
cd DHL-Logistics-RAG-System

# Install dependencies
make install

# Copy PDF files to data/
cp /path/to/pdfs/* data/
```

### Run

```bash
# Run interactively
make run

# Or with Python
python -m scripts.main
```

### Docker

```bash
# Build and start all services
make docker-build
make docker-up

# View logs
make docker-logs

# Stop
make docker-down
```

## Usage

```python
from dhl_logistics_rag import RAGPipeline, DHLRagAgent

# Initialize pipeline
pipeline = RAGPipeline()
retriever = pipeline.run()

# Create agent
agent = DHLRagAgent(pipeline=pipeline)
agent.initialize(retriever)

# Ask questions
agent.chat("What items are prohibited from shipping?")
agent.chat("Check status of DHL001")
agent.chat("How much to ship 5kg to London?")
```

## Evaluation

```bash
make evaluate
```

Results:
- Recall@10: 100%
- MRR: 0.87
- LLM Score: 4.5/5

## Testing

```bash
# Run all tests
make test

# Quick test
make test-fast
```

## License

MIT License
