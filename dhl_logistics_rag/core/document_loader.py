"""Document loading utilities."""

import os
import logging
from typing import List, Dict
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Container for loaded document with metadata."""
    content: str
    source_name: str
    page_number: int
    metadata: Dict


class DocumentLoader:
    """Handles loading documents from various sources."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._documents: List[Document] = []

    def load_pdf(self, file_path: str, source_name: str) -> List[Document]:
        """Load a single PDF file."""
        full_path = os.path.join(self.data_dir, file_path) if not os.path.isabs(file_path) else file_path

        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            return []

        loader = PyPDFLoader(full_path)
        docs = loader.load()

        # Add source metadata
        for doc in docs:
            doc.metadata["source_name"] = source_name

        logger.info(f"Loaded {len(docs)} pages from {source_name}")
        return docs

    def load_multiple_pdfs(self, pdf_files: Dict[str, str]) -> List[Document]:
        """Load multiple PDF files.

        Args:
            pdf_files: Dict mapping source_name to file_path

        Returns:
            List of all loaded documents
        """
        all_documents = []

        for source_name, file_path in pdf_files.items():
            docs = self.load_pdf(file_path, source_name)
            all_documents.extend(docs)

        self._documents = all_documents
        logger.info(f"Total documents loaded: {len(all_documents)} pages")
        return all_documents

    @property
    def documents(self) -> List[Document]:
        """Get all loaded documents."""
        return self._documents

    def get_document_stats(self) -> Dict:
        """Get statistics about loaded documents."""
        if not self._documents:
            return {"total_pages": 0, "sources": {}}

        sources = {}
        for doc in self._documents:
            source = doc.metadata.get("source_name", "unknown")
            sources[source] = sources.get(source, 0) + 1

        return {
            "total_pages": len(self._documents),
            "sources": sources,
            "total_characters": sum(len(doc.page_content) for doc in self._documents)
        }
