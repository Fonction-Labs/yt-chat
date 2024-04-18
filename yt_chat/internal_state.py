from typing import Optional

from qdrant_client import QdrantClient
from yt_chat.utils.chunk_text import ChunkSettings

from yt_chat.utils.qdrant import create_qdrant_collection

from yt_chat.config import Config

class InternalState:
    def __init__(self, model_name: str):
        self._model = Config.MODELS[model_name]
        self._chunk_settings = ChunkSettings(token_context_size=self._model.context_window_token_size)
        self._qdrant_client = create_qdrant_collection(
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
    def qdrant_client(self) -> Optional[QdrantClient]:
        return self._qdrant_client
