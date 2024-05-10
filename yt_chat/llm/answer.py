import uuid

from qdrant_client.http.models import PointStruct

from yt_chat.utils.chunk_text import ChunkSettings, get_text_chunks
from yt_chat.utils.qdrant import SingleCollectionQdrantClient
from yt_chat.utils.images import load_image

def embed_and_store_chunks(model, chunks: list[str], metas: list[dict], qdrant_client: SingleCollectionQdrantClient) -> None:
    # Embed chunks
    embeddings = model.embed_batch_parallel(chunks)

    # Store embedded chunks
    qdrant_client.upsert(
        qdrant_client.collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"meta": meta, "text": chunk},
            )
            for chunk, meta, embedding in zip(chunks, metas, embeddings)
        ],
    )

def embed_and_store_text(text: str, meta: dict, model, chunk_settings: ChunkSettings, qdrant_client: SingleCollectionQdrantClient):
    # Cut transcript text into chunks
    chunks = get_text_chunks(
        text, chunk_settings.chunk_size, chunk_settings.chunk_overlap
    )
    metas = [meta for i in range(len(chunks))] # TODO: fix? We copy the same meta for all chunks of the same text

    # Embed and store chunks
    embed_and_store_chunks(model, chunks, metas, qdrant_client)

def retrieve_top_k_chunks_for_query(
        model, query: str, qdrant_client: SingleCollectionQdrantClient, top_k: int
) -> list[str, dict]:
    """
    Embeds a query with a given model, and return top_k chunks ((text, meta) tuples) in the collection
    with highest embedded vector similarity.
    """
    # Embed query
    query_embedding = model.embed(query)

    # Retrieve top_k chunks with highest similarity in the collection
    query_results = qdrant_client.search(qdrant_client.collection_name, query_vector=query_embedding, limit=top_k)
    return [(result.payload["text"], result.payload["meta"]) for result in query_results]

def answer_query(query: str, model,
                 qdrant_client: SingleCollectionQdrantClient,
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
        messages_hypothetical = model.generate_hypothetical_messages_func(question=query)
        query = model.predict_messages(messages_hypothetical, temperature=0.7)

    # Retrieve top_k chunks for query
    top_k_chunks_for_query = retrieve_top_k_chunks_for_query(
        model, query, qdrant_client, top_k=top_k
    )
    top_k_texts_for_query, top_k_metas_for_query = zip(*top_k_chunks_for_query)

    # Merge top_k chunks into a single string for context
    context = " ".join(top_k_texts_for_query)

    images = None
    if top_k_metas_for_query:
        images = [load_image('temp_images/' + 'page'+ str(meta["pdf_page"]) +'.jpg') for meta in top_k_metas_for_query]

    # Generate LLM "context"-type message
    messages_with_context = model.generate_context_messages_func(question=original_query, context=context, images=images)

    # Get LLM response
    bot_response = model.predict_messages(messages_with_context, temperature=0.)

    return bot_response, top_k_metas_for_query
