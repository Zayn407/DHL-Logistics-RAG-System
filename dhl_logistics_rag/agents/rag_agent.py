"""DHL RAG Agent implementation."""

import logging
from typing import List, Optional, Dict, Any

from langgraph.prebuilt import create_react_agent

from ..common.config import Config
from ..common.llm_factory import LLMFactory
from ..core.rag_pipeline import RAGPipeline
from ..tools.knowledge_search import KnowledgeSearchTool
from ..tools.shipment_tracker import ShipmentTracker
from ..tools.cost_estimator import CostEstimator

logger = logging.getLogger(__name__)


class DHLRagAgent:
    """ReAct Agent for DHL logistics queries.

    The agent can:
    - Search the DHL knowledge base
    - Track shipments
    - Estimate shipping costs

    Supports multiple LLM providers: ollama, openai, anthropic
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        pipeline: Optional[RAGPipeline] = None,
        llm: Optional[Any] = None
    ):
        self.config = config or Config()
        self.pipeline = pipeline

        # Initialize LLM (use provided or create from config)
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMFactory.create(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                base_url=self.config.llm.base_url
            )

        logger.info(f"Agent using LLM: provider={self.config.llm.provider}, model={self.config.llm.model}")

        # Initialize tools
        self.shipment_tracker = ShipmentTracker()
        self.cost_estimator = CostEstimator()
        self.knowledge_search: Optional[KnowledgeSearchTool] = None

        self._agent = None
        self._tools: List = []

    def initialize(self, retriever=None):
        """Initialize the agent with tools.

        Args:
            retriever: Optional retriever. If not provided, uses pipeline retriever.
        """
        # Get retriever
        if retriever is not None:
            ret = retriever
        elif self.pipeline is not None:
            ret = self.pipeline.get_retriever()
        else:
            raise ValueError("Either retriever or pipeline must be provided")

        # Create knowledge search tool
        self.knowledge_search = KnowledgeSearchTool(ret)

        # Collect tools
        self._tools = [
            self.knowledge_search.get_tool(),
            self.shipment_tracker.get_tool(),
            self.cost_estimator.get_tool()
        ]

        # Create ReAct agent
        self._agent = create_react_agent(self.llm, self._tools)

        logger.info(f"Agent initialized with {len(self._tools)} tools")

    def invoke(self, question: str) -> Dict[str, Any]:
        """Invoke the agent with a question.

        Args:
            question: User question

        Returns:
            Agent response dictionary
        """
        if self._agent is None:
            raise ValueError("Agent not initialized. Call initialize() first.")

        response = self._agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        return response

    def ask(self, question: str) -> str:
        """Ask a question and get the answer.

        Args:
            question: User question

        Returns:
            Answer string
        """
        response = self.invoke(question)
        return response["messages"][-1].content

    def chat(self, question: str) -> str:
        """Interactive chat interface.

        Args:
            question: User question

        Returns:
            Formatted answer with question
        """
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")

        answer = self.ask(question)

        print(f"\n🤖 Answer:\n{answer}")
        return answer

    @property
    def tools(self) -> List:
        """Get list of available tools."""
        return self._tools

    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get descriptions of all tools."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools
        ]
