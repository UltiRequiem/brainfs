"""Tests for exception handling."""

import pytest

from brainfs.exceptions import BrainfsError


def test_brainfs_error_creation():
    """Test BrainfsError can be created and raised."""
    with pytest.raises(BrainfsError, match="Test error"):
        raise BrainfsError("Test error")


def test_brainfs_error_inheritance():
    """Test BrainfsError inherits from Exception."""
    error = BrainfsError("Test")
    assert isinstance(error, Exception)


def test_brainfs_error_with_cause():
    """Test BrainfsError can be raised with a cause."""
    original_error = ValueError("Original error")

    with pytest.raises(BrainfsError) as exc_info:
        try:
            raise original_error
        except ValueError as e:
            raise BrainfsError("Wrapped error") from e

    assert exc_info.value.__cause__ == original_error


def test_brainfs_error_str_representation():
    """Test string representation of BrainfsError."""
    error = BrainfsError("Test error message")
    assert str(error) == "Test error message"


def test_brainfs_error_empty_message():
    """Test BrainfsError with empty message."""
    error = BrainfsError("")
    assert str(error) == ""


def test_brainfs_error_multiple_args():
    """Test BrainfsError with multiple arguments."""
    error = BrainfsError("Error", "details", 123)
    # Should handle multiple args like standard exceptions
    assert len(error.args) == 3
