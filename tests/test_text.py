"""Tests for text chunking functionality."""

from brainfs.text import chunk_by_paragraphs, chunk_by_sentences, chunk_text, smart_chunk


def test_chunk_text_basic():
    """Test basic word-based chunking."""
    text = "This is a simple test document with some words"
    chunks = chunk_text(text, chunk_size=3, overlap=1)

    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert chunks[0] == "This is a"
    assert chunks[1] == "a simple test"


def test_chunk_text_no_overlap():
    """Test chunking without overlap."""
    text = "one two three four five six"
    chunks = chunk_text(text, chunk_size=2, overlap=0)

    expected = ["one two", "three four", "five six"]
    assert chunks == expected


def test_chunk_text_large_overlap():
    """Test chunking with large overlap."""
    text = "one two three four five"
    chunks = chunk_text(text, chunk_size=3, overlap=2)

    # Should have significant overlap
    assert len(chunks) >= 3
    assert "two" in chunks[0] and "two" in chunks[1]


def test_chunk_by_sentences():
    """Test sentence-based chunking."""
    text = (
        "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    )
    chunks = chunk_by_sentences(text, sentences_per_chunk=2, overlap=1, min_chunk_length=10)

    assert len(chunks) > 1
    # Each chunk should contain complete sentences
    assert all("." in chunk for chunk in chunks if chunk)


def test_chunk_by_sentences_min_length():
    """Test sentence chunking with minimum length filter."""
    text = "Short. This is a much longer sentence that should definitely be included."
    chunks = chunk_by_sentences(text, sentences_per_chunk=1, overlap=0, min_chunk_length=20)

    # Only chunks meeting minimum length should be included
    assert all(len(chunk) >= 20 for chunk in chunks)
    # Should have at least one chunk (the longer sentence)
    assert len(chunks) >= 1


def test_chunk_by_paragraphs():
    """Test paragraph-based chunking."""
    text = """Paragraph one content here.
More content in paragraph one.

Paragraph two content here.
More content in paragraph two.

Paragraph three content here."""

    chunks = chunk_by_paragraphs(text, max_chunk_size=100, overlap=20)

    assert len(chunks) > 0
    # Should respect paragraph boundaries
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_smart_chunk_auto_detect():
    """Test smart chunking auto-detection."""
    # Text with paragraphs should use paragraph chunking
    paragraph_text = """First paragraph here.

Second paragraph here.

Third paragraph here."""

    chunks = smart_chunk(paragraph_text, method="auto")
    assert len(chunks) > 0

    # Text with sentences should use sentence chunking
    sentence_text = "This is first sentence with enough content. This is second sentence with enough content. This is third sentence with enough content."
    chunks = smart_chunk(sentence_text, method="auto")
    assert len(chunks) > 0

    # Short text should use word chunking
    word_text = "short text"
    chunks = smart_chunk(word_text, method="auto")
    assert len(chunks) > 0


def test_smart_chunk_explicit_methods():
    """Test explicit chunking methods."""
    text = "This is a test sentence with enough content. Another sentence with more content. Third sentence with additional content."

    # Test explicit sentence chunking
    sentence_chunks = smart_chunk(
        text, method="sentences", sentences_per_chunk=2, min_chunk_length=10
    )
    assert len(sentence_chunks) > 0

    # Test explicit word chunking
    word_chunks = smart_chunk(text, method="words", chunk_size=3)
    assert len(word_chunks) > 0

    # Test explicit paragraph chunking
    paragraph_chunks = smart_chunk(text, method="paragraphs")
    assert len(paragraph_chunks) > 0


def test_empty_text():
    """Test chunking with empty text."""
    assert chunk_text("", chunk_size=3) == []
    assert chunk_by_sentences("", sentences_per_chunk=2) == []
    assert chunk_by_paragraphs("", max_chunk_size=100) == []
    assert smart_chunk("", method="sentences") == []


def test_chunk_edge_cases():
    """Test edge cases in chunking."""
    # Single word
    assert chunk_text("word", chunk_size=3) == ["word"]

    # Text shorter than chunk size
    assert chunk_text("one two", chunk_size=5) == ["one two"]

    # Large chunk size
    text = "one two three"
    assert chunk_text(text, chunk_size=100) == [text]


def test_smart_chunk_fallback_to_sentences():
    """Test smart_chunk fallback when paragraph chunking returns no chunks."""
    from unittest.mock import patch

    # Text that would trigger paragraph chunking but return empty
    text = "Single line of text without double newlines."

    with patch("brainfs.text.chunk_by_paragraphs", return_value=[]):
        # Should fallback to sentence chunking
        chunks = smart_chunk(text, method="auto")
        assert len(chunks) > 0


def test_chunk_by_paragraphs_fallback():
    """Test chunk_by_paragraphs fallback to sentence chunking."""
    # Text without paragraphs (no double newlines)
    text = "Single line text that has no paragraph breaks at all."

    chunks = chunk_by_paragraphs(text, max_chunk_size=1000)

    # Should fallback and return chunks
    assert len(chunks) > 0
