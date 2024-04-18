from .config_messages import (
    generate_openai_context_message,
    generate_mistral_context_message,
    generate_openai_summarize_transcript_message,
    generate_mistral_summarize_transcript_message,
    generate_openai_summarize_summaries_message,
    generate_mistral_summarize_summaries_message,
)

class Config:
    """
    This allows to configure the app:
    - Qdrant vector storage DB behavior
    - LLM models properties

    Model names are mapped to various model properties and
    message (prompts) generation functions.
    """

    QDRANT_COLLECTION_NAME = "qdrant"

    MODEL_TO_MODEL_REF_NAME = {
        "chat-gpt": "gpt-3.5-turbo",
        "mistral": "mistral",
    }

    MODEL_TO_EMBEDDING_MODEL_NAME = {
        "chat-gpt": "text-embedding-3-small",  # small embedding
        "mistral": None,  # not required by ollama models
    }

    MODEL_TO_EMBEDDING_VECTOR_SIZE = {
        "chat-gpt": 1536,  # small embedding
        "mistral": 4096,
    }

    MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE = {
        "chat-gpt": 4096,
        "mistral": 4096,
    }

    MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC = {
        "chat-gpt": generate_openai_context_message,
        "mistral": generate_mistral_context_message,
    }

    MODEL_TO_GENERATE_SUMMARIZE_TRANSCRIPT_MESSAGES_FUNC = {
        "chat-gpt": generate_openai_summarize_transcript_message,
        "mistral": generate_mistral_summarize_transcript_message,
    }

    MODEL_TO_GENERATE_SUMMARIZE_SUMMARIES_MESSAGES_FUNC = {
        "chat-gpt": generate_openai_summarize_summaries_message,
        "mistral": generate_mistral_summarize_summaries_message,
    }
