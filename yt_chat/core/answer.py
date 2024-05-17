import uuid

from qdrant_client.http.models import PointStruct

from flib.utils.chunk_text import ChunkSettings, get_text_chunks
from flib.utils.qdrant import SingleCollectionQdrantClient

def embed_and_store_chunks(embedding_model, chunks: list[str], qdrant_client: SingleCollectionQdrantClient) -> None:
    # Embed chunks
    embeddings = embedding_model.run_batch(chunks, parallel=True)

    # Store embedded chunks
    qdrant_client.upsert(
        qdrant_client.collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"meta": "info", "text": chunk},
            )
            for chunk, embedding in zip(chunks, embeddings)
        ],
    )

def embed_and_store_text(text: str, embedding_model, chunk_settings: ChunkSettings, qdrant_client: SingleCollectionQdrantClient):
    # Cut transcript text into chunks
    chunks = get_text_chunks(
        text, chunk_settings.chunk_size, chunk_settings.chunk_overlap
    )

    # Embed and store chunks
    embed_and_store_chunks(embedding_model, chunks, qdrant_client)

def retrieve_top_k_chunks_for_query(
        embedding_model, query: str, qdrant_client: SingleCollectionQdrantClient, top_k: int
) -> list[str]:
    """
    Embeds a query with a given model, and return top_k chunks in the collection
    with highest embedded vector similarity.
    """
    # Embed query
    query_embedding = embedding_model.run(query)

    # Retrieve top_k chunks with highest similarity in the collection
    query_results = qdrant_client.search(qdrant_client.collection_name, query_vector=query_embedding, limit=top_k)
    return [result.payload["text"] for result in query_results]

def answer_query(query: str, model, embedding_model,
                 qdrant_client: SingleCollectionQdrantClient,
                 generate_hypothetical_prompt,
                 generate_context_prompt,
                 top_k: int, use_hypothetical: bool = True):
    """
    Answers a question (query), using a model and embeddings stored in a Qdrant database.

    If use_hypothetical is set to True, this will formulate and use a fake,
    hypothetical answer to the question in order to perform the embeddings retrieval
    (this can improve the quality of retrievals, at the cost of one additional LLM request per query).
    """
    # Saves the original query into a new variable
    original_query = query

    # If use_hypothetical, formulate a fake answer to the question to improve embeddings retrieval
    if use_hypothetical:
        prompt = generate_hypothetical_prompt(question=query)
        query = model.run(prompt, temperature=0.7)

    # Retrieve top_k chunks for query
    top_k_chunks_for_query = retrieve_top_k_chunks_for_query(
        embedding_model, query, qdrant_client, top_k=top_k
    )

    # Merge top_k chunks into a single string for context
    context = " ".join(top_k_chunks_for_query)

    # Generate LLM "context"-type message
    prompt = generate_context_prompt(question=original_query, context=context)

    # Get LLM response
    bot_response = model.run(prompt=prompt, temperature=0.)

    return bot_response
