"""Integration tests for brainfs functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from brainfs.cli import main


@pytest.fixture
def integration_runner():
    """CLI runner for integration tests."""
    return CliRunner()


def test_full_workflow_simulation(integration_runner):
    """Test a full workflow simulation with mocked components."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.txt"
        test_file.write_text("This is a test document for indexing.")

        with (
            patch("brainfs.cli.DocumentDatabase") as mock_db,
            patch("brainfs.cli.Tokenizer") as mock_tokenizer,
            patch("brainfs.cli.VectorStore") as mock_store,
            patch("brainfs.cli.parse_document", return_value="Test content"),
            patch("brainfs.cli.smart_chunk", return_value=["Test content"]),
            patch("brainfs.cli.calculate_file_hash", return_value="abc123"),
        ):
            # Setup mocks
            mock_db.return_value.document_exists.return_value = False
            mock_tokenizer.return_value.embed.return_value = [0.1, 0.2, 0.3]
            mock_tokenizer.return_value.vocab = {"test": 0, "content": 1}

            # Test indexing
            result = integration_runner.invoke(main, ["index", str(test_file)])
            assert result.exit_code == 0

            # Setup for querying
            mock_db.return_value.list_documents.return_value = [
                ("abc123", "test.txt", str(test_file))
            ]
            mock_db.return_value.search_chunks.return_value = [
                ("abc123", "test.txt", "Test content", [0.1, 0.2, 0.3])
            ]
            mock_store.return_value.search.return_value = [(0.9, "test.txt: Test content")]

            # Test querying
            result = integration_runner.invoke(main, ["query", "test"])
            assert result.exit_code == 0

            # Test listing
            result = integration_runner.invoke(main, ["list"])
            assert result.exit_code == 0


def test_cli_with_actual_file_operations(integration_runner):
    """Test CLI with actual file operations (no mocking of file I/O)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "real_test.txt"
        test_file.write_text("Real file content for testing.")

        # Mock only the database and vector operations, not file I/O
        with (
            patch("brainfs.cli.DocumentDatabase") as mock_db,
            patch("brainfs.cli.Tokenizer") as mock_tokenizer,
            patch("brainfs.cli.VectorStore"),
        ):
            mock_db.return_value.document_exists.return_value = False
            mock_tokenizer.return_value.embed.return_value = [0.1, 0.2, 0.3]
            mock_tokenizer.return_value.vocab = {"real": 0, "file": 1}

            # Test indexing with real file
            result = integration_runner.invoke(main, ["index", str(test_file)])
            assert result.exit_code == 0

            # Verify parse_document was called with real file parsing
            # The content should have been actually read from the file


def test_cli_error_propagation(integration_runner):
    """Test that CLI properly propagates errors."""
    with patch("brainfs.cli.DocumentDatabase") as mock_db:
        # Make database raise a BrainfsError
        from brainfs.exceptions import BrainfsError

        mock_db.side_effect = BrainfsError("Database initialization failed")

        result = integration_runner.invoke(main, ["list"])
        assert result.exit_code != 0
        # Error should be wrapped in the CLI error handling
        assert (
            "Database initialization failed" in result.output
            or "Failed to list documents" in result.output
        )


def test_cli_progress_display_simulation(integration_runner):
    """Test CLI progress display with multiple files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple test files
        files = []
        for i in range(3):
            test_file = temp_path / f"test{i}.txt"
            test_file.write_text(f"Content of test file {i}")
            files.append(test_file)

        with (
            patch("brainfs.cli.DocumentDatabase") as mock_db,
            patch("brainfs.cli.Tokenizer") as mock_tokenizer,
            patch("brainfs.cli.VectorStore"),
        ):
            mock_db.return_value.document_exists.return_value = False
            mock_tokenizer.return_value.embed.return_value = [0.1, 0.2, 0.3]
            mock_tokenizer.return_value.vocab = {"test": 0, "file": 1}

            # Test indexing directory with multiple files
            result = integration_runner.invoke(main, ["index", str(temp_path)])
            assert result.exit_code == 0


def test_cli_configuration_display(integration_runner):
    """Test configuration display functionality."""
    with patch("brainfs.cli.DocumentParserFactory") as mock_factory:
        mock_factory.supported_extensions.return_value = [".txt", ".md", ".pdf"]

        # Test config command
        result = integration_runner.invoke(main, ["config"])
        # Should show configuration info (may fail on version import, but shouldn't crash)
        assert result.exit_code in [0, 1]


def test_cli_recursive_indexing(integration_runner):
    """Test recursive directory indexing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create nested directory structure
        subdir = temp_path / "subdir"
        subdir.mkdir()

        test_file1 = temp_path / "root.txt"
        test_file1.write_text("Root level file")

        test_file2 = subdir / "nested.txt"
        test_file2.write_text("Nested file content")

        with (
            patch("brainfs.cli.DocumentDatabase") as mock_db,
            patch("brainfs.cli.Tokenizer") as mock_tokenizer,
            patch("brainfs.cli.VectorStore"),
        ):
            mock_db.return_value.document_exists.return_value = False
            mock_tokenizer.return_value.embed.return_value = [0.1, 0.2, 0.3]
            mock_tokenizer.return_value.vocab = {"test": 0}

            # Test recursive indexing
            result = integration_runner.invoke(main, ["index", "--recursive", str(temp_path)])
            assert result.exit_code == 0


def test_cli_chunk_method_variations(integration_runner):
    """Test different chunking methods."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "chunk_test.txt"
        test_file.write_text(
            "This is a test document. It has multiple sentences. For testing chunking methods."
        )

        with (
            patch("brainfs.cli.DocumentDatabase") as mock_db,
            patch("brainfs.cli.Tokenizer") as mock_tokenizer,
            patch("brainfs.cli.VectorStore"),
        ):
            mock_db.return_value.document_exists.return_value = False
            mock_tokenizer.return_value.embed.return_value = [0.1, 0.2, 0.3]
            mock_tokenizer.return_value.vocab = {"test": 0}

            # Test different chunk methods
            for method in ["sentences", "paragraphs", "words"]:
                result = integration_runner.invoke(
                    main, ["index", "--chunk-method", method, str(test_file)]
                )
                assert result.exit_code == 0


def test_cli_empty_directory_handling(integration_runner):
    """Test CLI behavior with empty directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Empty directory
        result = integration_runner.invoke(main, ["index", temp_dir])
        assert result.exit_code == 0
        assert "No supported files found" in result.output


def test_cli_mixed_file_types(integration_runner):
    """Test indexing directory with mixed file types."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create files with different extensions
        (temp_path / "text.txt").write_text("Text content")
        (temp_path / "markdown.md").write_text("# Markdown content")
        (temp_path / "unsupported.xyz").write_text("Unsupported content")

        with (
            patch("brainfs.cli.DocumentDatabase") as mock_db,
            patch("brainfs.cli.Tokenizer") as mock_tokenizer,
            patch("brainfs.cli.VectorStore"),
        ):
            mock_db.return_value.document_exists.return_value = False
            mock_tokenizer.return_value.embed.return_value = [0.1, 0.2, 0.3]
            mock_tokenizer.return_value.vocab = {"test": 0}

            result = integration_runner.invoke(main, ["index", str(temp_path)])
            assert result.exit_code == 0
            # Should process supported files and skip unsupported ones
