"""Tests for database functionality."""

import sqlite3

from brainfs.database import Document, DocumentDatabase


def test_database_init(temp_dir):
    """Test database initialization."""
    db_path = temp_dir / "test.db"
    DocumentDatabase(db_path)

    # Database file should be created
    assert db_path.exists()

    # Tables should be created
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "documents" in tables


def test_add_document(test_db):
    """Test adding documents to database."""
    document = Document(
        id="test-doc-1",
        path="/test/doc.txt",
        filename="doc.txt",
        content="Test content",
        file_hash="abc123",
        chunks=["Test content"],
        vectors=[[1.0, 2.0, 3.0]],
    )

    test_db.add_document(document)

    # Verify document was added
    retrieved = test_db.get_document("test-doc-1")
    assert retrieved is not None
    assert retrieved.path == "/test/doc.txt"
    assert retrieved.content == "Test content"


def test_add_document_with_file_hash(test_db, sample_files):
    """Test adding document with file hash for deduplication."""
    txt_file = sample_files["txt"]

    document1 = Document(
        id="test-doc-1",
        path=str(txt_file),
        filename="sample.txt",
        content="Test content",
        file_hash="abc123",
        chunks=["Test content"],
        vectors=[[1.0, 2.0, 3.0]],
    )

    document2 = Document(
        id="test-doc-2",
        path=str(txt_file),
        filename="sample.txt",
        content="Different content",
        file_hash="abc123",
        chunks=["Different content"],
        vectors=[[4.0, 5.0, 6.0]],
    )

    test_db.add_document(document1)

    # Check if document exists by hash
    assert test_db.document_exists("abc123")

    # Add second document with same hash (will replace)
    test_db.add_document(document2)

    # Should still exist
    docs = test_db.list_documents()
    assert len(docs) >= 1


def test_document_exists(test_db):
    """Test checking if document exists."""
    # Initially should not exist
    assert not test_db.document_exists("abc123")

    # Add document
    document = Document(
        id="test-doc-1",
        path="/test/doc.txt",
        filename="doc.txt",
        content="Test content",
        file_hash="abc123",
        chunks=["Test content"],
        vectors=[[1.0, 2.0, 3.0]],
    )
    test_db.add_document(document)

    # Now should exist
    assert test_db.document_exists("abc123")


def test_list_documents(test_db):
    """Test retrieving all documents."""
    # Initially empty
    docs = test_db.list_documents()
    assert len(docs) == 0

    # Add multiple documents
    for i in range(3):
        document = Document(
            id=f"test-doc-{i}",
            path=f"/doc{i}.txt",
            filename=f"doc{i}.txt",
            content=f"Content {i}",
            file_hash=f"hash{i}",
            chunks=[f"Content {i}"],
            vectors=[[float(i), float(i + 1), float(i + 2)]],
        )
        test_db.add_document(document)

    # Should retrieve all
    docs = test_db.list_documents()
    assert len(docs) == 3

    filenames = {doc[1] for doc in docs}  # doc[1] is filename
    assert filenames == {"doc0.txt", "doc1.txt", "doc2.txt"}


def test_clear_documents(test_db):
    """Test clearing all documents."""
    # Add some documents
    for i in range(2):
        document = Document(
            id=f"test-doc-{i}",
            path=f"/doc{i}.txt",
            filename=f"doc{i}.txt",
            content=f"Content {i}",
            file_hash=f"hash{i}",
            chunks=[f"Content {i}"],
            vectors=[[float(i), float(i + 1), float(i + 2)]],
        )
        test_db.add_document(document)

    # Verify they exist
    assert len(test_db.list_documents()) == 2

    # Clear all
    count = test_db.clear_all()
    assert count == 2

    # Should be empty
    assert len(test_db.list_documents()) == 0


def test_search_chunks(test_db):
    """Test searching chunks functionality."""
    document = Document(
        id="test-doc-1",
        path="/test/doc.txt",
        filename="doc.txt",
        content="Test content with multiple chunks",
        file_hash="abc123",
        chunks=["Test content", "with multiple", "chunks"],
        vectors=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    )

    test_db.add_document(document)

    # Search chunks
    chunks = test_db.search_chunks(limit=10)
    assert len(chunks) == 3
    assert chunks[0][1] == "doc.txt"  # filename
    assert chunks[0][2] == "Test content"  # text
    assert chunks[0][3] == [1.0, 2.0, 3.0]  # vector


def test_database_concurrent_access(test_db):
    """Test that database handles concurrent-like access."""
    # Simulate rapid insertions
    for i in range(10):
        document = Document(
            id=f"test-doc-{i}",
            path=f"/doc{i}.txt",
            filename=f"doc{i}.txt",
            content=f"Content {i}",
            file_hash=f"hash{i}",
            chunks=[f"Content {i}"],
            vectors=[[float(i), float(i + 1), float(i + 2)]],
        )
        test_db.add_document(document)

    # All should be stored
    docs = test_db.list_documents()
    assert len(docs) == 10


def test_database_error_handling(temp_dir):
    """Test database error handling."""
    # Create database in non-existent directory should work
    # (database will create necessary directories)
    nested_path = temp_dir / "nested" / "deep" / "test.db"
    db = DocumentDatabase(nested_path)

    # Should still work
    document = Document(
        id="test-doc-1",
        path="/test.txt",
        filename="test.txt",
        content="Test",
        file_hash="abc123",
        chunks=["Test"],
        vectors=[[1.0, 2.0, 3.0]],
    )
    db.add_document(document)

    retrieved = db.get_document("test-doc-1")
    assert retrieved is not None


def test_database_home_directory_default(temp_dir):
    """Test database with default home directory path."""
    from unittest.mock import patch

    # Mock Path.home() to use temp directory
    with patch("pathlib.Path.home", return_value=temp_dir):
        db = DocumentDatabase()
        # Should create database in mocked home/.brainfs/
        expected_path = temp_dir / ".brainfs" / "documents.db"
        assert str(db.db_path) == str(expected_path)


def test_database_document_not_found(test_db):
    """Test get_document with non-existent document."""
    result = test_db.get_document("non-existent-id")
    assert result is None


def test_database_search_chunks_with_limit(test_db):
    """Test search_chunks with custom limit."""
    # Add multiple documents
    for i in range(5):
        document = Document(
            id=f"doc-{i}",
            path=f"/test/doc{i}.txt",
            filename=f"doc{i}.txt",
            content=f"Content {i}",
            file_hash=f"hash{i}",
            chunks=[f"Content {i}", f"More content {i}"],
            vectors=[[float(i), 0.0, 0.0], [0.0, float(i), 0.0]],
        )
        test_db.add_document(document)

    # Search with limit
    chunks = test_db.search_chunks(limit=5)
    assert len(chunks) == 5  # Should respect limit

    # Search with no limit (uses default)
    all_chunks = test_db.search_chunks()
    assert len(all_chunks) == 10  # 5 docs * 2 chunks each
