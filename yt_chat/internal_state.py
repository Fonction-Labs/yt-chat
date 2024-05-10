from typing import Optional

from yt_chat.utils.chunk_text import ChunkSettings
from yt_chat.utils.qdrant import SingleCollectionQdrantClient, create_qdrant_client

from yt_chat.config import Config
from yt_chat.llm.models import OpenAILLM, OllamaLLM

from yt_chat.llm.answer import embed_and_store_text, answer_query

# Could not put this in Config due to circular import
MODEL_TO_MODEL_CLASS = {
    "chatgpt": OpenAILLM,
    "chatgpt4": OpenAILLM,
    "mistral": OllamaLLM,
}

class InternalState:
    def __init__(self, model_name: str, api_key: str):
        model_class = MODEL_TO_MODEL_CLASS[model_name]
        self._model = model_class(model_name, api_key)
        self._chunk_settings = ChunkSettings(token_context_size=self._model.context_window_token_size)
        self._qdrant_client = create_qdrant_client(
            host=Config.QDRANT_HOST,
            collection_name=Config.QDRANT_COLLECTION_NAME,
            embedding_vector_size=self._model.embedding_vector_size
        )

    @property
    def model(self) -> Optional[str]:
        return self._model

    @property
    def chunk_settings(self) -> Optional[ChunkSettings]:
        return self._chunk_settings

    @property
    def qdrant_client(self) -> Optional[SingleCollectionQdrantClient]:
        return self._qdrant_client
