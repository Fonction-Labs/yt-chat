from .config_models_messages import (
    generate_openai_context_message,
    generate_mistral_context_message,
    generate_openai_summarize_transcript_message,
    generate_mistral_summarize_transcript_message,
    generate_openai_summarize_summaries_message,
    generate_mistral_summarize_summaries_message,
)

class ConfigModels:

    MODEL_TO_MODEL_TYPE = {
        "gpt-3.5-turbo": "openai",
        "mistral": "ollama",
    }

    MODEL_TO_EMBEDDING_MODEL_NAME = {
        "gpt-3.5-turbo": "text-embedding-3-small",  # small embedding
        "mistral": None,  # not required by ollama models
    }

    MODEL_TO_EMBEDDING_VECTOR_SIZE = {
        "gpt-3.5-turbo": 1536,  # small embedding
        "mistral": 4096,
    }

    MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE = {
        "gpt-3.5-turbo": 4096,
        "mistral": 4096,
    }

    MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC = {
        "gpt-3.5-turbo": generate_openai_context_message,
        "mistral": generate_mistral_context_message,
    }

    MODEL_TO_GENERATE_SUMMARIZE_TRANSCRIPT_MESSAGES_FUNC = {
        "gpt-3.5-turbo": generate_openai_summarize_transcript_message,
        "mistral": generate_mistral_summarize_transcript_message,
    }

    MODEL_TO_GENERATE_SUMMARIZE_SUMMARIES_MESSAGES_FUNC = {
        "gpt-3.5-turbo": generate_openai_summarize_summaries_message,
        "mistral": generate_mistral_summarize_summaries_message,
    }
