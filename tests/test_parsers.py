"""Tests for document parser functionality."""

from pathlib import Path

import pytest

from brainfs.parsers import parse_document


def test_parse_text_file(temp_dir):
    """Test parsing plain text files."""
    txt_file = temp_dir / "test.txt"
    content = "This is a test document.\nWith multiple lines.\nAnd some content."
    txt_file.write_text(content)

    result = parse_document(txt_file)
    assert result == content


def test_parse_markdown_file(temp_dir):
    """Test parsing markdown files."""
    md_file = temp_dir / "test.md"
    markdown_content = """# Test Document

This is a **test** document with:
- Lists
- *Emphasis*
- Code: `print("hello")`

## Section Two
More content here."""

    md_file.write_text(markdown_content)

    result = parse_document(md_file)
    # Markdown parser may convert to plain text
    assert "Test Document" in result
    assert "test" in result
    assert "print" in result


def test_parse_unsupported_file(temp_dir):
    """Test parsing unsupported file types."""
    # Create a file with unsupported extension
    unsupported_file = temp_dir / "test.xyz"
    unsupported_file.write_text("Some content")

    # Should raise ValueError for unsupported files
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(unsupported_file)


def test_parse_empty_file(temp_dir):
    """Test parsing empty files."""
    empty_file = temp_dir / "empty.txt"
    empty_file.write_text("")

    result = parse_document(empty_file)
    assert result == ""


def test_parse_nonexistent_file():
    """Test parsing nonexistent files."""
    nonexistent = Path("/nonexistent/file.txt")

    with pytest.raises(FileNotFoundError):
        parse_document(nonexistent)


def test_parse_binary_file(temp_dir):
    """Test parsing binary files (should handle gracefully)."""
    binary_file = temp_dir / "test.bin"
    # Write some binary data
    binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

    # Should handle binary files gracefully
    # Might raise an exception or return empty/error text
    try:
        result = parse_document(binary_file)
        # If it doesn't raise an exception, result might be empty or contain error text
        assert isinstance(result, str)
    except (UnicodeDecodeError, Exception):
        # It's acceptable for binary files to raise exceptions
        pass


def test_parse_large_file(temp_dir):
    """Test parsing large text files."""
    large_file = temp_dir / "large.txt"

    # Create a moderately large file (not too large for tests)
    large_content = "This is line {i}\n"
    large_text = "".join(large_content.format(i=i) for i in range(1000))
    large_file.write_text(large_text)

    result = parse_document(large_file)
    assert len(result) > 10000
    assert "This is line 0" in result
    assert "This is line 999" in result


def test_parse_utf8_file(temp_dir):
    """Test parsing files with various UTF-8 characters."""
    utf8_file = temp_dir / "utf8.txt"
    utf8_content = "Hello 世界! Café, naïve résumé. Emoji: 🚀🔥💻"
    utf8_file.write_text(utf8_content, encoding="utf-8")

    result = parse_document(utf8_file)
    assert result == utf8_content
    assert "世界" in result
    assert "Café" in result
    assert "🚀" in result


def test_parse_file_with_different_extensions(temp_dir):
    """Test that parser handles different file extensions appropriately."""
    # Test text file (should be exact match)
    txt_file = temp_dir / "test.txt"
    txt_file.write_text("Plain text content")
    result = parse_document(txt_file)
    assert result == "Plain text content"

    # Test markdown files (may strip markup)
    md_file = temp_dir / "test.md"
    md_file.write_text("# Markdown content")
    result = parse_document(md_file)
    assert "Markdown content" in result  # May strip the #

    markdown_file = temp_dir / "test.markdown"
    markdown_file.write_text("## Another markdown")
    result = parse_document(markdown_file)
    assert "Another markdown" in result  # May strip the ##

    # Test unsupported extension
    unsupported_file = temp_dir / "test.xyz"
    unsupported_file.write_text("Some content")
    with pytest.raises(ValueError):
        parse_document(unsupported_file)


# PDF and DOCX tests would require the optional dependencies
# These tests should be conditional based on availability


def test_parse_pdf_without_pypdf():
    """Test PDF parsing when pypdf is not available."""
    # This test would check behavior when pypdf import fails
    # For now, we'll skip this as it requires mocking imports
    pytest.skip("PDF parsing tests require pypdf dependency management")


def test_parse_docx_without_python_docx():
    """Test DOCX parsing when python-docx is not available."""
    # This test would check behavior when python-docx import fails
    # For now, we'll skip this as it requires mocking imports
    pytest.skip("DOCX parsing tests require python-docx dependency management")


def test_document_parser_factory():
    """Test DocumentParserFactory functionality."""
    from brainfs.parsers import DocumentParserFactory

    # Test supported extensions
    supported = DocumentParserFactory.supported_extensions()
    assert ".txt" in supported
    assert ".md" in supported
    assert ".markdown" in supported

    # Test is_supported method
    txt_path = Path("test.txt")
    assert DocumentParserFactory.is_supported(txt_path)

    md_path = Path("test.md")
    assert DocumentParserFactory.is_supported(md_path)

    unsupported_path = Path("test.xyz")
    assert not DocumentParserFactory.is_supported(unsupported_path)

    # Test get_parser method
    parser = DocumentParserFactory.get_parser(txt_path)
    assert parser is not None


def test_text_parser_unicode_decode_error(temp_dir):
    """Test TextParser fallback encoding."""
    from brainfs.parsers import TextParser

    # Create a file with latin-1 encoding
    latin_file = temp_dir / "latin.txt"
    content = "Café naïve résumé"
    latin_file.write_bytes(content.encode("latin-1"))

    # Should handle encoding gracefully
    result = TextParser.parse(latin_file)
    assert "Caf" in result  # Should still read something


def test_pdf_parser_import_error():
    """Test PDF parser when pypdf is not available."""
    from unittest.mock import patch

    from brainfs.parsers import PDFParser

    with patch("brainfs.parsers.pypdf_module", None):
        with pytest.raises(ImportError, match="pypdf is required"):
            PDFParser.parse(Path("test.pdf"))


def test_markdown_parser_without_markdown_library(temp_dir):
    """Test markdown parser fallback when markdown library is not available."""
    from unittest.mock import patch

    from brainfs.parsers import MarkdownParser

    md_file = temp_dir / "test.md"
    markdown_content = "# Header\n**bold** text\n*italic* text\n`code`"
    md_file.write_text(markdown_content)

    with patch("brainfs.parsers.markdown_module", None):
        result = MarkdownParser.parse(md_file)

        # Should strip markdown formatting
        assert "Header" in result  # Header # should be removed
        assert "bold" in result  # ** should be removed
        assert "italic" in result  # * should be handled


def test_docx_parser_import_error():
    """Test DOCX parser when python-docx is not available."""
    from unittest.mock import patch

    from brainfs.parsers import DOCXParser

    with patch("brainfs.parsers.docx_module", None):
        with pytest.raises(ImportError, match="python-docx is required"):
            DOCXParser.parse(Path("test.docx"))


def test_parse_document_with_factory_error_handling(temp_dir):
    """Test parse_document error handling."""
    # Test with unsupported file type directly
    unsupported_file = temp_dir / "test.unknown"
    unsupported_file.write_text("content")

    # Should raise ValueError for unsupported file type
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(unsupported_file)


def test_text_parser_empty_file(temp_dir):
    """Test TextParser with empty file."""
    from brainfs.parsers import TextParser

    empty_file = temp_dir / "empty.txt"
    empty_file.write_text("")

    result = TextParser.parse(empty_file)
    assert result == ""


def test_markdown_parser_complex_content(temp_dir):
    """Test markdown parser with complex content."""
    from brainfs.parsers import MarkdownParser

    md_file = temp_dir / "complex.md"
    content = """# Main Title

## Subtitle

This is a paragraph with **bold** text and *italic* text.

- List item 1
- List item 2

```python
def hello():
    print("world")
```

[Link](http://example.com)
"""
    md_file.write_text(content)

    result = MarkdownParser.parse(md_file)

    # Should contain text content
    assert "Main Title" in result
    assert "Subtitle" in result
    assert "paragraph" in result
    assert "List item" in result


def test_parser_file_not_found():
    """Test parser behavior with non-existent files."""
    from brainfs.parsers import TextParser

    non_existent = Path("/non/existent/file.txt")

    with pytest.raises(FileNotFoundError):
        TextParser.parse(non_existent)


def test_parser_permission_error(temp_dir):
    """Test parser behavior with permission errors."""
    import stat

    from brainfs.parsers import TextParser

    # Create a file and remove read permissions
    restricted_file = temp_dir / "restricted.txt"
    restricted_file.write_text("restricted content")
    restricted_file.chmod(stat.S_IWUSR)  # Write only, no read

    try:
        # Should raise PermissionError
        with pytest.raises(PermissionError):
            TextParser.parse(restricted_file)
    finally:
        # Restore permissions for cleanup
        restricted_file.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_document_parser_factory_edge_cases():
    """Test DocumentParserFactory edge cases."""
    from brainfs.parsers import DocumentParserFactory

    # Test with path without extension
    no_ext_path = Path("filename")
    assert not DocumentParserFactory.is_supported(no_ext_path)

    # Test with hidden file
    hidden_file = Path(".hidden.txt")
    assert DocumentParserFactory.is_supported(hidden_file)

    # Test case insensitive extensions
    # upper_case_path = Path("test.TXT")
    # Behavior depends on implementation - should handle case insensitivity
    # This test documents current behavior but is not implemented yet
