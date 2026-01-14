"""Tests for tokenizer functionality."""

from brainfs.tokenizer import Tokenizer


def test_tokenizer_basic():
    """Test basic tokenizer functionality."""
    tokenizer = Tokenizer()
    documents = ["hello world", "world peace", "hello peace"]

    # Test fitting
    tokenizer.fit(documents)
    assert len(tokenizer.vocab) > 0

    # Test that common words are in vocab
    expected_words = {"hello", "world", "peac"}  # peac due to stemming
    assert all(word in tokenizer.vocab for word in expected_words)


def test_tokenizer_embed():
    """Test text embedding."""
    tokenizer = Tokenizer()
    documents = ["hello world", "world peace"]
    tokenizer.fit(documents)

    # Test embedding
    vector = tokenizer.embed("hello world")
    assert len(vector) == len(tokenizer.vocab)
    assert sum(vector) > 0  # Should have some non-zero values

    # Test empty string
    empty_vector = tokenizer.embed("")
    assert all(v == 0 for v in empty_vector)

    # Test unknown words
    unknown_vector = tokenizer.embed("xyz123")
    assert all(v == 0 for v in unknown_vector)


def test_tokenizer_stemming():
    """Test that stemming is working."""
    tokenizer = Tokenizer()

    # Test stemming with different word forms
    words = tokenizer._clean("running runs runner")
    # All should be stemmed to similar forms
    assert len(set(words)) <= 2  # Should be fewer unique words due to stemming


def test_tokenizer_cleaning():
    """Test text cleaning functionality."""
    tokenizer = Tokenizer()

    # Test punctuation removal
    cleaned = tokenizer._clean("Hello, world! How are you?")
    assert "," not in " ".join(cleaned)
    assert "!" not in " ".join(cleaned)
    assert "?" not in " ".join(cleaned)

    # Test lowercase conversion
    cleaned = tokenizer._clean("HELLO World")
    assert all(word.islower() or not word.isalpha() for word in cleaned)


def test_tokenizer_empty_input():
    """Test tokenizer with empty or None input."""
    tokenizer = Tokenizer()

    # Test with empty documents
    tokenizer.fit([])
    assert len(tokenizer.vocab) == 0

    # Test with empty string documents
    tokenizer.fit(["", "  ", "\n"])
    assert len(tokenizer.vocab) == 0

    # Test embedding with no vocabulary
    vector = tokenizer.embed("hello")
    assert len(vector) == 0


def test_tokenizer_consistency():
    """Test that tokenizer produces consistent results."""
    tokenizer1 = Tokenizer()
    tokenizer2 = Tokenizer()

    documents = ["hello world", "world peace", "hello peace"]

    tokenizer1.fit(documents)
    tokenizer2.fit(documents)

    # Both should produce the same vocabulary
    assert tokenizer1.vocab == tokenizer2.vocab

    # Both should produce the same embeddings
    text = "hello world"
    vector1 = tokenizer1.embed(text)
    vector2 = tokenizer2.embed(text)
    assert vector1 == vector2


def test_tokenizer_nltk_import_error():
    """Test tokenizer behavior when NLTK is not available."""
    from unittest.mock import patch

    with patch("brainfs.tokenizer.NLTK_AVAILABLE", False):
        tokenizer = Tokenizer()
        documents = ["hello world", "testing fallback"]

        # Should use fallback stemmer
        tokenizer.fit(documents)
        assert len(tokenizer.vocab) > 0

        # Should still work with simple stemming
        words = tokenizer._clean("running runs runner")
        assert len(words) > 0


def test_tokenizer_large_vocabulary():
    """Test tokenizer with large vocabulary."""
    tokenizer = Tokenizer()

    # Create documents with many unique words
    documents = []
    for i in range(50):
        doc = f"document {i} contains unique words like word{i} and term{i}"
        documents.append(doc)

    tokenizer.fit(documents)

    # Should handle large vocabulary
    assert len(tokenizer.vocab) > 20  # Should have many unique terms

    # Embedding should work with large vocab
    vector = tokenizer.embed("document unique word")
    assert len(vector) == len(tokenizer.vocab)
    assert sum(vector) > 0


def test_tokenizer_special_characters():
    """Test tokenizer with special characters and edge cases."""
    tokenizer = Tokenizer()

    documents = [
        "hello@example.com",
        "test-case_with_underscores",
        "numbers123mixed456",
        "punctuation... lots!!! of??? it.",
    ]

    tokenizer.fit(documents)
    assert len(tokenizer.vocab) > 0

    # Should handle special characters in embedding
    vector = tokenizer.embed("test punctuation")
    assert len(vector) == len(tokenizer.vocab)


def test_tokenizer_very_long_text():
    """Test tokenizer with very long text."""
    tokenizer = Tokenizer()

    # Create long document
    long_text = " ".join([f"word{i}" for i in range(100)])
    documents = [long_text]

    tokenizer.fit(documents)

    # Should handle long text
    vector = tokenizer.embed("word1 word2 word3")
    assert len(vector) == len(tokenizer.vocab)


def test_tokenizer_duplicate_documents():
    """Test tokenizer with duplicate documents."""
    tokenizer = Tokenizer()

    # Same document repeated
    documents = ["hello world"] * 10

    tokenizer.fit(documents)

    # Should handle duplicates gracefully
    assert len(tokenizer.vocab) > 0


def test_tokenizer_numeric_content():
    """Test tokenizer with numeric content."""
    tokenizer = Tokenizer()

    documents = [
        "123 456 789",
        "price 29",
        "version 1",
        "year 2023",
    ]

    tokenizer.fit(documents)

    # Should handle numeric content
    vector = tokenizer.embed("123 price version")
    assert len(vector) == len(tokenizer.vocab)


def test_tokenizer_embedding_vector_properties():
    """Test properties of embedding vectors."""
    tokenizer = Tokenizer()

    documents = ["hello world peace", "world peace love", "love happiness"]
    tokenizer.fit(documents)

    vector = tokenizer.embed("hello peace")

    # Vector should have correct dimensions
    assert len(vector) == len(tokenizer.vocab)

    # Should be numeric
    assert all(isinstance(x, (int, float)) for x in vector)

    # Should be non-negative (bag of words)
    assert all(x >= 0 for x in vector)


def test_tokenizer_word_frequency_impact():
    """Test that word frequency impacts embeddings correctly."""
    tokenizer = Tokenizer()

    documents = [
        "apple apple apple",  # High frequency
        "banana",  # Low frequency
        "apple banana",
    ]

    tokenizer.fit(documents)

    # Get embeddings
    apple_heavy = tokenizer.embed("apple apple")
    banana_single = tokenizer.embed("banana")

    # Both should have non-zero elements
    assert sum(apple_heavy) > 0
    assert sum(banana_single) > 0
