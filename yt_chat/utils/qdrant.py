from qdrant_client.http.models import Distance, VectorParams

from qdrant_client import QdrantClient

def create_qdrant_collection(collection_name: str, embedding_vector_size: int) -> QdrantClient:
    qdrant_client = QdrantClient(":memory:")
    # qdrant_client = QdrantClient(path="path/to/db")  # Persists changes to disk
    # qdrant_client = QdrantClient(url="http://localhost:6333")
    # qdrant_client = QdrantClient(host="localhost", port=6333)
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_vector_size,
                                    distance=Distance.COSINE),
    )
    return qdrant_client
