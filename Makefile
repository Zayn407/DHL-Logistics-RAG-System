.PHONY: install dev test lint format clean docker-build docker-up docker-down run evaluate help

# Variables
PYTHON := python3
PIP := pip3
PROJECT := dhl_logistics_rag

# Default target
help:
	@echo "DHL RAG System - Available Commands"
	@echo "===================================="
	@echo "make install      - Install dependencies"
	@echo "make dev          - Install dev dependencies"
	@echo "make test         - Run tests"
	@echo "make lint         - Run linter"
	@echo "make format       - Format code"
	@echo "make clean        - Clean cache files"
	@echo "make docker-build - Build Docker images"
	@echo "make docker-up    - Start Docker containers"
	@echo "make docker-down  - Stop Docker containers"
	@echo "make run          - Run the RAG system"
	@echo "make evaluate     - Run evaluation"

# Installation
install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v --cov=$(PROJECT) --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --ff

# Code quality
lint:
	flake8 $(PROJECT) tests/
	mypy $(PROJECT)

format:
	black $(PROJECT) tests/
	isort $(PROJECT) tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf chroma_db/ 2>/dev/null || true
	rm -rf *.egg-info/ 2>/dev/null || true

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Run application
run:
	$(PYTHON) -m scripts.main

evaluate:
	$(PYTHON) -m scripts.evaluate

# Ollama
ollama-pull:
	ollama pull mistral

ollama-serve:
	ollama serve

# Data
copy-data:
	mkdir -p data/
	@echo "Copy PDF files to data/ directory"
