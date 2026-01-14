"""Tests for vector store functionality."""

import pytest

from brainfs.vector_store import VectorStore


def test_vector_store_init():
    """Test vector store initialization."""
    store = VectorStore()
    assert store.database == []


def test_add_item_single():
    """Test adding a single item to vector store."""
    store = VectorStore()
    text = "This is a test document"
    vector = [1.0, 2.0, 3.0]

    store.add_item(text, vector)

    assert len(store.database) == 1
    assert store.database[0]["text"] == text
    # Vector should be normalized
    assert len(store.database[0]["vector"]) == 3
    assert isinstance(store.database[0]["vector"], list)


def test_add_multiple_items():
    """Test adding multiple items to vector store."""
    store = VectorStore()
    items = [
        ("Document one", [1.0, 0.0, 0.0]),
        ("Document two", [0.0, 1.0, 0.0]),
        ("Document three", [0.0, 0.0, 1.0]),
    ]

    for text, vector in items:
        store.add_item(text, vector)

    assert len(store.database) == 3
    texts = [item["text"] for item in store.database]
    assert "Document one" in texts
    assert "Document two" in texts
    assert "Document three" in texts


def test_search_basic():
    """Test basic vector search functionality."""
    store = VectorStore()

    # Add some test vectors
    store.add_item("Document about cats", [1.0, 0.0, 0.0])
    store.add_item("Document about dogs", [0.0, 1.0, 0.0])
    store.add_item("Document about cats and dogs", [0.5, 0.5, 0.0])

    # Search for cat-like vector
    query_vector = [1.0, 0.0, 0.0]
    results = store.search(query_vector, top_k=2)

    assert len(results) == 2
    # Results should be sorted by score (highest first)
    assert results[0][0] >= results[1][0]
    # First result should be the cat document (exact match)
    assert "cats" in results[0][1]


def test_search_with_top_k():
    """Test search with different top_k values."""
    store = VectorStore()

    # Add 5 documents
    for i in range(5):
        store.add_item(f"Document {i}", [float(i), 1.0, 0.0])

    query_vector = [0.0, 1.0, 0.0]

    # Test top_k = 3
    results = store.search(query_vector, top_k=3)
    assert len(results) == 3

    # Test top_k = 1
    results = store.search(query_vector, top_k=1)
    assert len(results) == 1

    # Test top_k = 10 (more than available)
    results = store.search(query_vector, top_k=10)
    assert len(results) == 5  # Should return all available


def test_search_zero_top_k():
    """Test search with top_k = 0."""
    store = VectorStore()
    store.add_item("Document", [1.0, 0.0, 0.0])

    results = store.search([1.0, 0.0, 0.0], top_k=0)
    assert results == []


def test_search_negative_top_k():
    """Test search with negative top_k."""
    store = VectorStore()
    store.add_item("Document", [1.0, 0.0, 0.0])

    results = store.search([1.0, 0.0, 0.0], top_k=-1)
    assert results == []


def test_search_empty_store():
    """Test search on empty vector store."""
    store = VectorStore()
    results = store.search([1.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_search_zero_query_vector():
    """Test search with zero query vector."""
    store = VectorStore()
    store.add_item("Document", [1.0, 0.0, 0.0])

    # Zero vector should return empty results
    results = store.search([0.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_search_all_zero_query_vector():
    """Test search with all-zero query vector."""
    store = VectorStore()
    store.add_item("Document 1", [1.0, 0.0, 0.0])
    store.add_item("Document 2", [0.0, 1.0, 0.0])

    # All zeros should return empty results
    results = store.search([0, 0, 0], top_k=5)
    assert results == []


def test_full_search():
    """Test full search functionality."""
    store = VectorStore()

    # Add some documents
    store.add_item("High similarity", [1.0, 0.0, 0.0])
    store.add_item("Medium similarity", [0.5, 0.5, 0.0])
    store.add_item("Low similarity", [0.0, 0.0, 1.0])

    query_vector = [1.0, 0.0, 0.0]
    results = store.full_search(query_vector)

    # Should return all results sorted by score
    assert len(results) == 3
    assert results[0][0] >= results[1][0] >= results[2][0]
    assert "High similarity" in results[0][1]


def test_full_search_empty_store():
    """Test full search on empty store."""
    store = VectorStore()
    results = store.full_search([1.0, 0.0, 0.0])
    assert results == []


def test_vector_normalization():
    """Test that vectors are properly normalized when added."""
    store = VectorStore()

    # Add a vector that needs normalization
    original_vector = [3.0, 4.0, 0.0]  # Magnitude = 5
    store.add_item("Test document", original_vector)

    stored_vector = store.database[0]["vector"]

    # Check that the vector is normalized (magnitude should be 1)
    magnitude = sum(x**2 for x in stored_vector) ** 0.5
    assert abs(magnitude - 1.0) < 1e-10


def test_search_result_sorting():
    """Test that search results are correctly sorted by score."""
    store = VectorStore()

    # Add vectors with different similarities to [1, 0, 0]
    store.add_item("Perfect match", [1.0, 0.0, 0.0])  # score = 1.0
    store.add_item("Good match", [0.8, 0.6, 0.0])  # score ≈ 0.8 after normalization
    store.add_item("Poor match", [0.0, 1.0, 0.0])  # score = 0.0

    query_vector = [1.0, 0.0, 0.0]
    results = store.search(query_vector, top_k=3)

    assert len(results) == 3
    # Verify descending order
    assert results[0][0] >= results[1][0] >= results[2][0]
    # Perfect match should be first
    assert "Perfect match" in results[0][1]
    # Poor match should be last
    assert "Poor match" in results[2][1]


def test_heap_behavior_with_large_k():
    """Test that search works correctly when top_k is larger than database size."""
    store = VectorStore()

    store.add_item("Doc 1", [1.0, 0.0, 0.0])
    store.add_item("Doc 2", [0.0, 1.0, 0.0])

    # Request more results than available
    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 2  # Only 2 documents available
    assert all(isinstance(score, float) for score, _ in results)


def test_search_with_identical_scores():
    """Test search behavior when multiple documents have identical scores."""
    store = VectorStore()

    # Add multiple documents with identical vectors
    identical_vector = [1.0, 0.0, 0.0]
    store.add_item("Doc A", identical_vector)
    store.add_item("Doc B", identical_vector)
    store.add_item("Doc C", identical_vector)

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    # All should have the same score (perfect match)
    assert abs(results[0][0] - results[1][0]) < 1e-10


def test_edge_case_single_dimension_vectors():
    """Test with single-dimension vectors."""
    store = VectorStore()

    store.add_item("Positive", [5.0])
    store.add_item("Negative", [-3.0])

    # Don't add zero vector as it would cause normalization error
    # store.add_item("Zero", [0.0])

    # Search for positive direction
    results = store.search([1.0], top_k=3)

    assert len(results) == 2  # Should have two valid results
    # Positive should score higher than negative for positive query
    positive_result = next(r for r in results if "Positive" in r[1])
    assert positive_result[0] > 0


def test_zero_vector_handling():
    """Test that zero vectors are handled properly."""
    store = VectorStore()

    # Adding a zero vector should raise an error during normalization
    with pytest.raises(ValueError, match="Cannot normalize a zero-length vector"):
        store.add_item("Zero vector", [0.0, 0.0, 0.0])
