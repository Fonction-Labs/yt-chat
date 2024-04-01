from pydantic import BaseModel

class AgentSettings(BaseModel):
    token_context_size: int = 4096
    safety_token_prct: int = 0.7
    n_vectors: int = 5
    base_prompt_transcript_summary: str
    base_prompt_summaries_summary: str

class ModelSettings(BaseModel):
    n_threads: int = 8
    n_gpu: int = -1
    path_model: str = "./mistral-7b-instruct-v0.2.Q5_K_S.gguf"


DEFAULT_AGENT_SETTING_OPENAI = AgentSettings(
    **{
        "base_prompt_transcript_summary": """
            Write a detailed summary of the following youtube video transcript.
                "{docs}".
            CONCISE SUMMARY:
        """,
        "base_prompt_summaries_summary": """
            The following is set of summaries:
            {docs}
            Take these and distill it into a final, consolidated detailed summary of 3 to 5 paragraphs.
            Helpful Answer:
        """
    }
)

DEFAULT_AGENT_SETTING_MISTRAL = AgentSettings(
    **{
        "base_prompt_transcript_summary": """
            Write a detailed summary of the following youtube video transcript.
                "{docs}".
            CONCISE SUMMARY:
        """,
        "base_prompt_summaries_summary": """
            The following is set of summaries:
            {docs}
            Take these and distill it into a final, consolidated detailed summary of 3 to 5 paragraphs.
            Helpful Answer:
        """
    }
)

DEFAULT_MODEL_SETTINGS_MISTRAL = ModelSettings()
