from yt_chat.llm.models import OpenAILLM, OllamaLLM

class Config:
    QDRANT_COLLECTION_NAME = "qdrant"

    # For model specific config, see config_models.py
    # For model messages (prompts) specific config, see config_models_messages.py
    MODELS = {
       "gpt-3.5-turbo": OpenAILLM("gpt-3.5-turbo"),
       "mistral": OllamaLLM("mistral"),
    }
