from typing import Optional

from flib.utils.chunk_text import ChunkSettings
from flib.utils.qdrant import SingleCollectionQdrantClient, create_qdrant_client

from yt_chat.config import Config

class InternalState:
    def __init__(self, model, embedding_model):
        self._model = model
        self._embedding_model = embedding_model

        self._chunk_settings = ChunkSettings(token_context_size=self._model.context_window_token_size)
        self._qdrant_client = create_qdrant_client(
            host=Config.QDRANT_HOST,
            collection_name=Config.QDRANT_COLLECTION_NAME,
            embedding_vector_size=self._embedding_model.embedding_vector_size
        )

        self._generate_hypothetical_prompt = Config.HYPOTHETICAL_PROMPT_FUNC[self._model.model_name]
        self._generate_context_prompt = Config.CONTEXT_PROMPT_FUNC[self._model.model_name]

    @property
    def model(self):
        return self._model

    @property
    def embedding_model(self):
        return self._embedding_model

    @property
    def chunk_settings(self) -> Optional[ChunkSettings]:
        return self._chunk_settings

    @property
    def qdrant_client(self) -> Optional[SingleCollectionQdrantClient]:
        return self._qdrant_client

    @property
    def generate_hypothetical_prompt(self):
        return self._generate_hypothetical_prompt

    @property
    def generate_context_prompt(self):
        return self._generate_context_prompt
