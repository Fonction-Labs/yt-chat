from .config_prompts import (
    generate_openai_context_prompt,
    generate_mistral_context_prompt,
    generate_openai_hypothetical_prompt,
    generate_mistral_hypothetical_prompt,
    generate_openai_summarize_transcript_prompt,
    generate_mistral_summarize_transcript_prompt,
    generate_openai_summarize_summaries_prompt,
    generate_mistral_summarize_summaries_prompt,
)

class Config:
    """
    This allows to configure the app:
    - Qdrant DB vector storage and retrieval behavior
    - LLM models properties

    Model names are mapped to various model properties and
    prompt generation functions.
    """

    LIMIT_RATE = "5/day"

    QDRANT_COLLECTION_NAME = "qdrant"

    QDRANT_HOST = ":memory:" # can also be "https://<myaddress>:<myport>" (default Qdrant port is 6333)

    RETRIEVAL_USE_HYPOTHETICAL = True

    RETRIEVAL_TOP_K = 5

    HYPOTHETICAL_PROMPT_FUNC = {
        "gpt-3.5-turbo": generate_openai_hypothetical_prompt,
        "gpt-4-turbo": generate_openai_hypothetical_prompt,
        "mistral": generate_mistral_hypothetical_prompt,
    }

    CONTEXT_PROMPT_FUNC = {
        "gpt-3.5-turbo": generate_openai_context_prompt,
        "gpt-4-turbo": generate_openai_context_prompt,
        "mistral": generate_mistral_context_prompt,
    }

    SUMMARIZE_TRANSCRIPT_PROMPT_FUNC = {
        "gpt-3.5-turbo": generate_openai_summarize_transcript_prompt,
        "gpt-4-turbo": generate_openai_summarize_transcript_prompt,
        "mistral": generate_mistral_summarize_transcript_prompt,
    }

    SUMMARIZE_SUMMARIES_PROMPT_FUNC = {
        "gpt-3.5-turbo": generate_openai_summarize_summaries_prompt,
        "gpt-4-turbo": generate_openai_summarize_summaries_prompt,
        "mistral": generate_mistral_summarize_summaries_prompt,
    }
