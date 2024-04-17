import uuid

from joblib import Parallel, delayed
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

def embed_and_store_chunks(model, chunks: list[str], qdrant_client: QdrantClient, collection_name: str) -> None:
    # Embed chunks
    embeddings = Parallel(n_jobs=8, prefer="threads")(
        delayed(model.embed)(chunk) for chunk in chunks
    )

    # Store embedded chunks
    qdrant_client.upsert(
        collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"meta": "info", "text": chunk},
            )
            for chunk, embedding in zip(chunks, embeddings)
        ],
    )


def retrieve_top_k_chunks_for_query(
        model, query: str, qdrant_client: QdrantClient, collection_name: str, top_k: int
) -> list[str]:
    """
    Embeds a query with a given model, and return top_k chunks in the collection
    with highest embedded vector similarity.
    """
    # Embed the query
    query_embedding = model.embed(query)
    # Retrieve the top_k chunks with highest similarity in the collection
    query_results = qdrant_client.search(collection_name, query_vector=query_embedding, limit=top_k)
    return [result.payload["text"] for result in query_results]
