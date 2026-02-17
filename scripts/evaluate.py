#!/usr/bin/env python3
"""Evaluation script for DHL RAG System."""

import logging
from dhl_logistics_rag.common.config import Config
from dhl_logistics_rag.common.llm_factory import LLMFactory
from dhl_logistics_rag.core.rag_pipeline import RAGPipeline
from dhl_logistics_rag.agents.rag_agent import DHLRagAgent
from dhl_logistics_rag.evaluation.evaluator import RAGEvaluator
from dhl_logistics_rag.utils.helpers import setup_logging, check_ollama_status


def main():
    # Setup
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    # Initialize
    config = Config()

    # Check Ollama if using local provider
    if config.llm.provider == "ollama":
        status = check_ollama_status()
        if status["status"] != "running":
            logger.error(f"Ollama not running: {status}")
            return

    # Run pipeline
    logger.info("Starting RAG pipeline...")
    pipeline = RAGPipeline(config)
    retriever = pipeline.run()

    # Create agent
    agent = DHLRagAgent(config, pipeline)
    agent.initialize(retriever)

    # Create LLM for evaluation (uses factory for provider flexibility)
    llm = LLMFactory.create(
        provider=config.llm.provider,
        model=config.llm.model,
        temperature=config.llm.temperature,
        base_url=config.llm.base_url
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    evaluator = RAGEvaluator(llm, retriever, agent)
    results = evaluator.run_full_evaluation(num_samples=5)

    # Print report
    evaluator.print_report(results)


if __name__ == "__main__":
    main()
