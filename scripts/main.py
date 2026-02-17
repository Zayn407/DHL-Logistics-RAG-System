#!/usr/bin/env python3
"""Main entry point for DHL RAG System."""

import logging
from dhl_logistics_rag.common.config import Config
from dhl_logistics_rag.core.rag_pipeline import RAGPipeline
from dhl_logistics_rag.agents.rag_agent import DHLRagAgent
from dhl_logistics_rag.utils.helpers import setup_logging, check_ollama_status


def main():
    # Setup
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    # Initialize config
    config = Config()

    # Check Ollama if using local provider
    if config.llm.provider == "ollama":
        status = check_ollama_status()
        if status["status"] != "running":
            logger.error(f"Ollama not running: {status}")
            return
        logger.info(f"Ollama running with models: {status['models']}")

    # Run pipeline
    logger.info("Starting RAG pipeline...")
    pipeline = RAGPipeline(config)
    retriever = pipeline.run()

    # Create agent
    agent = DHLRagAgent(config, pipeline)
    agent.initialize(retriever)

    # Interactive mode
    print("\n" + "=" * 60)
    print("🚀 DHL RAG System Ready!")
    print("=" * 60)
    print("\nExample queries:")
    print("  - What items are prohibited from shipping?")
    print("  - Check status of DHL001")
    print("  - How much to ship 5kg to London?")
    print("\nType 'quit' to exit.\n")

    while True:
        try:
            question = input("❓ Ask: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not question:
                continue

            agent.chat(question)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
