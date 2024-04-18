from yt_chat.llm.models import OpenAILLM, OllamaLLM

class Config:
    QDRANT_COLLECTION_NAME = "qdrant"

    MODELS = {
       "chat-gpt": OpenAILLM(model_ref_name="gpt-3.5-turbo"),
       "mistral": OllamaLLM(model_ref_name="mistral"),
    }
