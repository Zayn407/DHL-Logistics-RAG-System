"""RAG system evaluator."""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from .metrics import RetrievalMetrics, GenerationMetrics, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Test case for evaluation."""
    id: int
    question: str
    expected_source: str
    category: str


class RAGEvaluator:
    """Evaluator for RAG system performance."""

    # Default test dataset
    DEFAULT_TEST_CASES = [
        TestCase(1, "What items are prohibited from shipping with DHL?", "dhl_express_terms", "prohibited_items"),
        TestCase(2, "What is DHL's liability limit for lost packages?", "dhl_express_terms", "liability"),
        TestCase(3, "How do I file a claim for a damaged shipment?", "dhl_express_terms", "claims"),
        TestCase(4, "What documents are required for customs clearance?", "dhl_customs_guide", "customs"),
        TestCase(5, "What is the process for importing goods to the US?", "dhl_customs_guide", "customs"),
        TestCase(6, "What are the delivery time guarantees for DHL Express?", "dhl_express_terms", "delivery"),
        TestCase(7, "How does DHL handle shipments that cannot be delivered?", "dhl_express_terms", "delivery"),
        TestCase(8, "What are the packaging requirements for DHL shipments?", "dhl_express_terms", "packaging"),
        TestCase(9, "What duties and taxes apply to international shipments?", "dhl_customs_guide", "customs"),
        TestCase(10, "What is DHL's privacy policy regarding shipment data?", "dhl_express_terms", "privacy"),
    ]

    def __init__(self, llm, retriever, agent):
        self.llm = llm
        self.retriever = retriever
        self.agent = agent
        self.test_cases = self.DEFAULT_TEST_CASES.copy()

    def evaluate_retrieval(self, k: int = 10) -> RetrievalMetrics:
        """Evaluate retrieval performance.

        Args:
            k: Number of documents to retrieve

        Returns:
            RetrievalMetrics
        """
        results = []

        for test_case in self.test_cases:
            docs = self.retriever.invoke(test_case.question)
            sources = [doc.metadata.get("source_name", "") for doc in docs]

            source_hit = test_case.expected_source in sources

            source_rank = None
            for i, source in enumerate(sources):
                if source == test_case.expected_source:
                    source_rank = i + 1
                    break

            results.append({
                "source_hit": source_hit,
                "source_rank": source_rank
            })

        recall_at_k = sum(1 for r in results if r["source_hit"]) / len(results)
        mrr = sum(1/r["source_rank"] for r in results if r["source_rank"]) / len(results)

        return RetrievalMetrics(recall_at_k=recall_at_k, mrr=mrr, k=k)

    def evaluate_generation(self, question: str, answer: str) -> GenerationMetrics:
        """Evaluate answer quality using LLM-as-Judge.

        Args:
            question: Original question
            answer: Generated answer

        Returns:
            GenerationMetrics
        """
        eval_prompt = f"""Rate the following answer on a scale of 1-5 for each criterion.

Question: {question}

Answer: {answer}

Rate (respond with ONLY three numbers separated by commas, e.g., "4,3,5"):
1. Relevance (1-5): Is the answer relevant?
2. Completeness (1-5): Does it fully address the question?
3. Accuracy (1-5): Does it seem factually correct?

Scores:"""

        try:
            response = self.llm.invoke(eval_prompt)
            scores_text = response.content.strip()

            numbers = re.findall(r'\d+', scores_text)
            if len(numbers) >= 3:
                relevance = min(int(numbers[0]), 5)
                completeness = min(int(numbers[1]), 5)
                accuracy = min(int(numbers[2]), 5)
            else:
                relevance = completeness = accuracy = 3

            return GenerationMetrics(
                relevance=relevance,
                completeness=completeness,
                accuracy=accuracy
            )

        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}")
            return GenerationMetrics(relevance=0, completeness=0, accuracy=0)

    def run_full_evaluation(self, num_samples: int = 5) -> Dict:
        """Run complete evaluation.

        Args:
            num_samples: Number of test cases to evaluate

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Running evaluation on {num_samples} samples...")

        # Retrieval evaluation
        retrieval_metrics = self.evaluate_retrieval()

        # Generation evaluation
        generation_results = []

        for test_case in self.test_cases[:num_samples]:
            logger.info(f"Evaluating: {test_case.question[:50]}...")

            # Get agent answer
            try:
                answer = self.agent.ask(test_case.question)
            except Exception as e:
                answer = f"Error: {e}"

            # Evaluate generation
            gen_metrics = self.evaluate_generation(test_case.question, answer)

            # Check retrieval
            docs = self.retriever.invoke(test_case.question)
            sources = [doc.metadata.get("source_name", "") for doc in docs]
            source_hit = test_case.expected_source in sources

            generation_results.append(EvaluationResult(
                question_id=test_case.id,
                question=test_case.question,
                source_hit=source_hit,
                source_rank=None,
                generation_metrics=gen_metrics,
                category=test_case.category
            ))

        # Calculate averages
        avg_relevance = sum(r.generation_metrics.relevance for r in generation_results) / len(generation_results)
        avg_completeness = sum(r.generation_metrics.completeness for r in generation_results) / len(generation_results)
        avg_accuracy = sum(r.generation_metrics.accuracy for r in generation_results) / len(generation_results)
        avg_overall = (avg_relevance + avg_completeness + avg_accuracy) / 3

        return {
            "retrieval": retrieval_metrics.to_dict(),
            "generation": {
                "relevance": avg_relevance,
                "completeness": avg_completeness,
                "accuracy": avg_accuracy,
                "overall": avg_overall
            },
            "results": [r.to_dict() for r in generation_results],
            "overall_score": (retrieval_metrics.recall_at_k + avg_overall / 5) / 2 * 100
        }

    def print_report(self, results: Dict):
        """Print evaluation report."""
        print("\n" + "=" * 60)
        print("📊 RAG SYSTEM EVALUATION REPORT")
        print("=" * 60)

        print("\n【1. Retrieval Evaluation】")
        print(f"   Recall@{results['retrieval']['k']}: {results['retrieval']['recall_at_k']:.1%}")
        print(f"   MRR: {results['retrieval']['mrr']:.3f}")

        print("\n【2. Generation Evaluation (LLM-as-Judge)】")
        print(f"   Relevance: {results['generation']['relevance']:.1f}/5")
        print(f"   Completeness: {results['generation']['completeness']:.1f}/5")
        print(f"   Accuracy: {results['generation']['accuracy']:.1f}/5")
        print(f"   Overall: {results['generation']['overall']:.1f}/5")

        print("\n【3. Summary】")
        print("=" * 60)
        score = results['overall_score']
        grade = "A" if score >= 80 else "B" if score >= 70 else "C" if score >= 60 else "D"
        print(f"   Overall Score: {score:.0f}/100 - {grade}")
