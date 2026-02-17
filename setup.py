"""Setup script for DHL RAG System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dhl-logistics-rag",
    version="1.0.0",
    author="Zayn",
    description="A production-ready RAG system for DHL logistics documentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zayn407/DHL-Logistics-RAG-System",
    packages=find_packages(exclude=["tests", "scripts"]),
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-experimental>=0.0.40",
        "langgraph>=0.0.20",
        "langchain-ollama>=0.0.1",
        "langchain-chroma>=0.1.0",
        "chromadb>=0.4.0",
        "pypdf>=3.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
