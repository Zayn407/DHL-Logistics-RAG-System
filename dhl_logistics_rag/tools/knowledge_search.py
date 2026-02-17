"""Knowledge base search tool."""

import logging
from typing import List, Optional

from langchain_core.tools import tool
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class KnowledgeSearchTool:
    """Tool for searching the DHL knowledge base."""

    def __init__(self, retriever, max_results: int = 3):
        self.retriever = retriever
        self.max_results = max_results

    def search(self, query: str) -> str:
        """Search the knowledge base.

        Args:
            query: Search query

        Returns:
            Formatted search results
        """
        docs = self.retriever.invoke(query)

        if not docs:
            return "No relevant information found in the knowledge base."

        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source_name', 'unknown')}]\n{doc.page_content}"
            for doc in docs[:self.max_results]
        ])

        return f"DHL Documentation:\n{context}"

    def get_tool(self):
        """Get LangChain tool wrapper."""
        retriever = self.retriever
        max_results = self.max_results

        @tool
        def search_dhl_knowledge(query: str) -> str:
            """USE THIS TOOL to answer questions about DHL policies, shipping rules, prohibited items, customs, terms and conditions. Always call this tool for any DHL-related information questions."""
            docs = retriever.invoke(query)

            if not docs:
                return "No relevant information found."

            context = "\n\n".join([
                f"[Source: {doc.metadata.get('source_name', 'unknown')}]\n{doc.page_content}"
                for doc in docs[:max_results]
            ])

            return f"DHL Documentation:\n{context}"

        return search_dhl_knowledge
