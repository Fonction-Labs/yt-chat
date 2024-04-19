from qdrant_client.http.models import Distance, VectorParams

from qdrant_client import QdrantClient

class SingleCollectionQdrantClient(QdrantClient):
    """
    host (str): either ':memory:', or 'http://<myadress>:<myport>' (default Qdrant port is 6333)
    """
    def __init__(self, host: str, collection_name: str):
        super().__init__(host)
        self.collection_name = collection_name

def create_qdrant_client(host: str, collection_name: str, embedding_vector_size: int) -> SingleCollectionQdrantClient:
    qdrant_client = SingleCollectionQdrantClient(host, collection_name)
    qdrant_client.recreate_collection(
        collection_name=qdrant_client.collection_name,
        vectors_config=VectorParams(size=embedding_vector_size,
                                    distance=Distance.COSINE),
    )
    return qdrant_client
