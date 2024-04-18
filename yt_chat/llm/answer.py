import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from yt_chat.utils.chunk_text import ChunkSettings, get_text_chunks
from yt_chat.settings import QDRANT_COLLECTION_NAME, MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC

def embed_and_store_chunks(model, chunks: list[str], qdrant_client: QdrantClient, collection_name: str) -> None:
    # Embed chunks
    embeddings = model.embed_batch_parallel(chunks)
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

def embed_and_store_text(text: str, model, chunk_settings: ChunkSettings, qdrant_client: QdrantClient):
    # Cut transcript text into chunks
    chunks = get_text_chunks(
        text, chunk_settings.chunk_size, chunk_settings.chunk_overlap
    )
    # Embed and store chunks
    embed_and_store_chunks(model, chunks, qdrant_client, collection_name=QDRANT_COLLECTION_NAME)

def retrieve_top_k_chunks_for_query(
        model, query: str, qdrant_client: QdrantClient, collection_name: str, top_k: int
) -> list[str]:
    """
    Embeds a query with a given model, and return top_k chunks in the collection
    with highest embedded vector similarity.
    """
    # Embed query
    query_embedding = model.embed(query)
    # Retrieve top_k chunks with highest similarity in the collection
    query_results = qdrant_client.search(collection_name, query_vector=query_embedding, limit=top_k)
    return [result.payload["text"] for result in query_results]

def answer_query(query: str, model, qdrant_client: QdrantClient):
    # Retrieve top_k chunks for query
    top_k_chunks_for_query = retrieve_top_k_chunks_for_query(
        model, query, qdrant_client, collection_name=QDRANT_COLLECTION_NAME, top_k=5
    )
    # Merge top_k chunks into a single string for context
    context = " ".join(top_k_chunks_for_query)
    # Generate LLM "context"-type message
    messages_with_context = MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC[
        model.model_name
    ](query, context=context)
    # Get LLM response
    bot_response = model.predict_messages(messages_with_context, temperature=0.0)
    return bot_response

