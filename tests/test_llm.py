"""Tests for LLM functionality."""

from unittest.mock import MagicMock, patch

import pytest

from brainfs.llm import LLMClient


def test_llm_import_error():
    """Test LLMClient when OpenAI is not available."""
    with patch("brainfs.llm.OPENAI_AVAILABLE", False):
        with pytest.raises(ImportError, match="OpenAI library not installed"):
            LLMClient()


def test_llm_missing_api_key():
    """Test LLMClient when API key is not set."""
    with (
        patch("brainfs.llm.OPENAI_AVAILABLE", True),
        patch("brainfs.llm.os.getenv", return_value=None),
    ):
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
            LLMClient()


@patch("brainfs.llm.OPENAI_AVAILABLE", True)
@patch("brainfs.llm.os.getenv")
@patch("brainfs.llm.openai.OpenAI")
def test_llm_initialization_success(mock_openai, mock_getenv):
    """Test successful LLMClient initialization."""
    mock_getenv.return_value = "test-api-key"
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    llm = LLMClient()

    assert llm.api_key == "test-api-key"
    assert llm.client == mock_client
    mock_openai.assert_called_once_with(api_key="test-api-key")


@patch("brainfs.llm.OPENAI_AVAILABLE", True)
@patch("brainfs.llm.os.getenv")
@patch("brainfs.llm.openai.OpenAI")
def test_generate_answer_success(mock_openai, mock_getenv):
    """Test successful answer generation."""
    mock_getenv.return_value = "test-api-key"
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  This is a generated answer.  "
    mock_client.chat.completions.create.return_value = mock_response

    llm = LLMClient()
    contexts = ["Context 1 content", "Context 2 content"]
    query = "What is the answer?"

    result = llm.generate_answer(query, contexts)

    assert result == "This is a generated answer."
    mock_client.chat.completions.create.assert_called_once()

    # Check the call arguments
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-3.5-turbo"
    assert call_args.kwargs["max_tokens"] == 500
    assert call_args.kwargs["temperature"] == 0.1

    # Check that contexts are combined
    prompt_content = call_args.kwargs["messages"][1]["content"]
    assert "Context 1 content" in prompt_content
    assert "Context 2 content" in prompt_content
    assert "---" in prompt_content  # Context separator
    assert query in prompt_content


@patch("brainfs.llm.OPENAI_AVAILABLE", True)
@patch("brainfs.llm.os.getenv")
@patch("brainfs.llm.openai.OpenAI")
def test_generate_answer_with_custom_params(mock_openai, mock_getenv):
    """Test answer generation with custom parameters."""
    mock_getenv.return_value = "test-api-key"
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Custom answer"
    mock_client.chat.completions.create.return_value = mock_response

    llm = LLMClient()
    contexts = ["Context content"]
    query = "Test query?"

    result = llm.generate_answer(query, contexts, model="gpt-4", max_tokens=1000)

    assert result == "Custom answer"

    # Check custom parameters were used
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4"
    assert call_args.kwargs["max_tokens"] == 1000


@patch("brainfs.llm.OPENAI_AVAILABLE", True)
@patch("brainfs.llm.os.getenv")
@patch("brainfs.llm.openai.OpenAI")
def test_generate_answer_api_error(mock_openai, mock_getenv):
    """Test answer generation when API call fails."""
    mock_getenv.return_value = "test-api-key"
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock API error
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    llm = LLMClient()
    contexts = ["Context content"]
    query = "Test query?"

    result = llm.generate_answer(query, contexts)

    assert "Error generating answer: API Error" in result


@patch("brainfs.llm.OPENAI_AVAILABLE", True)
@patch("brainfs.llm.os.getenv")
def test_is_available_with_api_key(mock_getenv):
    """Test is_available when API key is set."""
    mock_getenv.return_value = "test-api-key"

    with patch("brainfs.llm.openai.OpenAI"):
        llm = LLMClient()
        assert llm.is_available() is True


@patch("brainfs.llm.OPENAI_AVAILABLE", True)
@patch("brainfs.llm.os.getenv")
def test_is_available_without_api_key(mock_getenv):
    """Test is_available when API key is not set."""
    mock_getenv.return_value = None

    with pytest.raises(ValueError):
        LLMClient()


@patch("brainfs.llm.OPENAI_AVAILABLE", False)
def test_is_available_without_openai():
    """Test is_available when OpenAI is not available."""
    with pytest.raises(ImportError):
        LLMClient()


def test_generate_answer_empty_contexts():
    """Test answer generation with empty contexts."""
    with (
        patch("brainfs.llm.OPENAI_AVAILABLE", True),
        patch("brainfs.llm.os.getenv", return_value="test-key"),
        patch("brainfs.llm.openai.OpenAI") as mock_openai,
    ):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "No context answer"
        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMClient()
        result = llm.generate_answer("Test query?", [])

        assert result == "No context answer"

        # Should still make API call even with empty contexts
        mock_client.chat.completions.create.assert_called_once()


def test_prompt_construction():
    """Test that the prompt is constructed correctly."""
    with (
        patch("brainfs.llm.OPENAI_AVAILABLE", True),
        patch("brainfs.llm.os.getenv", return_value="test-key"),
        patch("brainfs.llm.openai.OpenAI") as mock_openai,
    ):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Answer"
        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMClient()
        contexts = ["First context", "Second context"]
        query = "What is this about?"

        llm.generate_answer(query, contexts)

        # Check the prompt structure
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        # Should have system and user messages
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # User message should contain the prompt with contexts and query
        user_content = messages[1]["content"]
        assert "Context:" in user_content
        assert "First context" in user_content
        assert "Second context" in user_content
        assert "Question: What is this about?" in user_content
        assert "Answer:" in user_content
        assert "---" in user_content  # Context separator
