from .config_messages import (
    generate_openai_context_message,
    generate_mistral_context_message,
    generate_openai_hypothetical_message,
    generate_mistral_hypothetical_message,
    generate_openai_summarize_transcript_message,
    generate_mistral_summarize_transcript_message,
    generate_openai_summarize_summaries_message,
    generate_mistral_summarize_summaries_message,
)

class Config:
    """
    This allows to configure the app:
    - Qdrant DB vector storage and retrieval behavior
    - LLM models properties

    Model names are mapped to various model properties and
    message (prompts) generation functions.
    """

    QDRANT_COLLECTION_NAME = "qdrant"

    QDRANT_HOST = ":memory:" # can also be "https://<myaddress>:<myport>" (default Qdrant port is 6333)

    RETRIEVAL_USE_HYPOTHETICAL = True

    RETRIEVAL_TOP_K = 5

    MODEL_TO_MODEL_REF_NAME = {
        "chatgpt": "gpt-3.5-turbo",
        "mistral": "mistral",
    }

    MODEL_TO_EMBEDDING_MODEL_NAME = {
        "chatgpt": "text-embedding-3-small",  # small embedding
        "mistral": None,  # not required by ollama models
    }

    MODEL_TO_EMBEDDING_VECTOR_SIZE = {
        "chatgpt": 1536,  # small embedding
        "mistral": 4096,
    }

    MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE = {
        "chatgpt": 4096,
        "mistral": 4096,
    }

    MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC = {
        "chatgpt": generate_openai_context_message,
        "mistral": generate_mistral_context_message,
    }

    MODEL_TO_GENERATE_HYPOTHETICAL_MESSAGES_FUNC = {
        "chatgpt": generate_openai_hypothetical_message,
        "mistral": generate_mistral_hypothetical_message,
    }

    MODEL_TO_GENERATE_SUMMARIZE_TRANSCRIPT_MESSAGES_FUNC = {
        "chatgpt": generate_openai_summarize_transcript_message,
        "mistral": generate_mistral_summarize_transcript_message,
    }

    MODEL_TO_GENERATE_SUMMARIZE_SUMMARIES_MESSAGES_FUNC = {
        "chatgpt": generate_openai_summarize_summaries_message,
        "mistral": generate_mistral_summarize_summaries_message,
    }
