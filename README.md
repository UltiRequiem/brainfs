# Brainfs

> AI Document Search & RAG

[![codecov](https://codecov.io/github/UltiRequiem/brainfs/graph/badge.svg?token=3nfAbcCKUc)](https://codecov.io/github/UltiRequiem/brainfs)

A modern CLI tool for document indexing and querying using vector search and
LLM-powered answer generation.

## Features

- **Multi-format Support**: PDF, Markdown, TXT, DOCX parsing
- **Smart Chunking**: Sentence-based, paragraph-based, or word-based strategies
- **Vector Search**: Efficient similarity search with NLTK stemming and stopword
  removal
- **LLM Integration**: OpenAI API for intelligent answer generation
- **Persistent Storage**: SQLite database with document deduplication
- **Rich CLI**: Beautiful terminal interface with progress bars and interactive
  mode

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Documents │───▶│   Parsing   │───▶│   Chunking  │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌───────▼─────┐
│ LLM Answers │◀───│Vector Search│◀───│   Storage   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <your-repo>
cd brainfs
uv sync

# Set up OpenAI API key (optional, for answer generation)
echo "OPENAI_API_KEY=your-api-key" > .env
```

## Usage

### Basic Commands

```bash
# Index documents
brainfs index /path/to/docs/ --recursive

# Query documents
brainfs query "What is machine learning?"

# Interactive mode
brainfs query --interactive

# Generate AI answers
brainfs query "How does Python support OOP?" --generate

# List indexed documents
brainfs list

# Show configuration
brainfs config

# Clear all indexed documents
brainfs clear
```

### Advanced Chunking

```bash
# Sentence-based chunking (recommended)
brainfs index docs/ --chunk-method sentences --chunk-size 3

# Paragraph-based chunking
brainfs index docs/ --chunk-method paragraphs

# Word-based chunking
brainfs index docs/ --chunk-method words --chunk-size 100
```

## Examples

### Index Technical Documentation

```bash
# Index all markdown files recursively
brainfs index ./technical-docs/ --recursive --chunk-method sentences

# Query with AI answers
brainfs query "How do I set up authentication?" --generate
```

### Interactive Research Session

```bash
# Start interactive mode
brainfs query --interactive --generate

# Query: "What are the main features?"
# Query: "How does the API work?"
# Query: quit
```

## Development

### Code Quality Tools

This project uses modern Python tooling:

- **[uv](https://github.com/astral-sh/uv)**: Fast Python package manager
- **[ruff](https://github.com/astral-sh/ruff)**: Lightning-fast linter and
  formatter
- **[ty](https://docs.astral.sh/ty/)**: Astral's fast type checker
- **[pre-commit](https://pre-commit.com/)**: Git hooks for code quality

### Commands

```bash
# Linting and formatting
uv run ruff check .          # Check for issues
uv run ruff check --fix .    # Fix auto-fixable issues
uv run ruff format .         # Format code

# Type checking
uv run ty check .

# Run all checks
uv run pre-commit run --all-files
```

### Project Structure

```
├── brainfs/
│   ├── __init__.py          # Package initialization
│   ├── cli.py              # CLI interface and commands
│   ├── database.py         # SQLite document storage
│   ├── parsers.py          # Document format parsers
│   ├── text.py             # Text chunking strategies
│   ├── tokenizer.py        # Vector tokenization with NLTK
│   ├── vector_store.py     # Vector storage and search
│   ├── vector.py           # Vector math operations
│   └── llm.py              # OpenAI LLM integration
├── tests/                  # Test suite
├── pyproject.toml          # Project configuration
└── .github/workflows/      # CI/CD pipelines
```

## Configuration

Brainfs stores its configuration and database in:

- **Database**: `~/.brainfs/documents.db`
- **Config**: Environment variables via `.env`

### Environment Variables

```bash
# Required for LLM features
OPENAI_API_KEY=sk-...
```

## Supported Formats

- **Text**: `.txt`, `.text`
- **Markdown**: `.md`, `.markdown`
- **PDF**: `.pdf`
- **Word**: `.docx`

## Core Components

### Tokenizer (`tokenizer.py`)

- NLTK-powered stemming and stopword removal
- Bag-of-words vectorization with fallbacks
- Automatic NLTK data downloads

### Vector Store (`vector_store.py`)

- Normalized vector storage
- Efficient top-k search using heaps
- Cosine similarity scoring

### Database (`database.py`)

- SQLite document storage with metadata
- Automatic deduplication by file hash
- Chunked text storage with vectors

### LLM Integration (`llm.py`)

- OpenAI API integration
- Context-aware answer generation
- Graceful fallbacks when unavailable

## CI/CD

GitHub Actions automatically runs:

- Code quality checks (ruff)
- Type checking (ty)
- Multi-Python version testing (3.9-3.12)
- Functionality tests

## Contributing

1. Install development dependencies: `uv sync --group dev`
2. Set up pre-commit hooks: `uv run pre-commit install`
3. Make changes and commit (hooks run automatically)
4. All CI checks must pass

## Performance

- **Indexing**: ~1000 documents/minute (varies by size)
- **Search**: Sub-second query response times
- **Storage**: ~1MB per 100 documents (compressed vectors)

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**: Automatically downloads on first run
2. **OpenAI Quota**: LLM features gracefully degrade without API access
3. **Large Files**: PDFs >10MB may take longer to process
4. **Permission Errors**: Ensure write access to `~/.brainfs/`

### Getting Help

```bash
brainfs --help
brainfs <command> --help
```
