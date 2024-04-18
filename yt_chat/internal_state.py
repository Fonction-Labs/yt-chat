from typing import Optional

from qdrant_client import QdrantClient
from yt_chat.utils.chunk_text import ChunkSettings

from yt_chat.utils.qdrant import create_qdrant_collection

from yt_chat.settings import (
    QDRANT_COLLECTION_NAME,
    MODELS,
    MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE,
    MODEL_TO_EMBEDDING_VECTOR_SIZE,
    MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC,
)

class InternalState:
    def __init__(self, model_name: str):
        self._model = MODELS[model_name]
        self._chunk_settings = ChunkSettings(token_context_size=MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE[model_name])
        self._qdrant_client = create_qdrant_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            embedding_vector_size=MODEL_TO_EMBEDDING_VECTOR_SIZE[model_name]
        )

    @property
    def model(self) -> Optional[str]:
        return self._model

    @property
    def chunk_settings(self) -> Optional[ChunkSettings]:
        return self._chunk_settings

    @property
    def qdrant_client(self) -> Optional[QdrantClient]:
        return self._qdrant_client
