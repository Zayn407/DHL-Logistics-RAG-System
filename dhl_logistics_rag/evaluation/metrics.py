"""Evaluation metrics for RAG system."""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation."""
    recall_at_k: float
    mrr: float
    k: int

    def to_dict(self) -> Dict:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "k": self.k
        }

    def __str__(self) -> str:
        return f"Recall@{self.k}: {self.recall_at_k:.1%}, MRR: {self.mrr:.3f}"


@dataclass
class GenerationMetrics:
    """Metrics for generation evaluation (LLM-as-Judge)."""
    relevance: float
    completeness: float
    accuracy: float

    @property
    def average(self) -> float:
        return (self.relevance + self.completeness + self.accuracy) / 3

    def to_dict(self) -> Dict:
        return {
            "relevance": self.relevance,
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "average": self.average
        }

    def __str__(self) -> str:
        return (f"Relevance: {self.relevance:.1f}/5, "
                f"Completeness: {self.completeness:.1f}/5, "
                f"Accuracy: {self.accuracy:.1f}/5, "
                f"Average: {self.average:.1f}/5")


@dataclass
class EvaluationResult:
    """Combined evaluation result."""
    question_id: int
    question: str
    source_hit: bool
    source_rank: Optional[int]
    generation_metrics: GenerationMetrics
    category: str

    def to_dict(self) -> Dict:
        return {
            "id": self.question_id,
            "question": self.question,
            "source_hit": self.source_hit,
            "source_rank": self.source_rank,
            "generation": self.generation_metrics.to_dict(),
            "category": self.category
        }
