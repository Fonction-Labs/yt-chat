import re
import chainlit as cl
from qdrant_client import QdrantClient

from yt_chat.llm.summarize import summarize_transcript
from yt_chat.utils.transcript import get_video_transcript
from yt_chat.utils.chunk_text import ChunkSettings, get_text_chunks
from yt_chat.utils.qdrant import create_qdrant_collection
from yt_chat.llm.summarize import summarize_transcript
from yt_chat.llm.retriever import (
    embed_and_store_chunks,
    retrieve_top_k_chunks_for_query,
)

from yt_chat.settings import (
    QDRANT_COLLECTION_NAME,
    MODELS,
    MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE,
    MODEL_TO_EMBEDDING_VECTOR_SIZE,
    MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC,
)


CHAT_PROFILE_TO_MODEL_NAME = {"ChatGPT (3.5)": "gpt-3.5-turbo",
                              "Mistral (7B)": "mistral"}

@cl.on_chat_start
async def main():
    model, chunk_settings, qdrant_client = set_internal_state()
    await chainlit_summarize_video(model, chunk_settings, qdrant_client)


async def chainlit_summarize_video(model, chunk_settings, qdrant_client):
    response = await cl.AskUserMessage(content="Please specify the youtube URL you want to summarize.").send()

    if response:
        video_url = response["output"]
        if is_valid_youtube_url(video_url):
            summary = get_summary(video_url, model, chunk_settings, qdrant_client)
            await cl.Message(
                content=f"Here is the summary of the youtube video: \n {summary}",
            ).send()

            await chainlit_ask_if_new_video(model, chunk_settings, qdrant_client)
        else:
            await cl.Message(
                content="You did not provide a valid youtube URL",
            ).send()
            await chainlit_summarize_video(model, chunk_settings, qdrant_client)

async def chainlit_ask_if_new_video(model, chunk_settings, qdrant_client):
    action = await cl.AskActionMessage(
        content="Do you want to summarize a new video?",
        actions=[
            cl.Action(name="New video?", value="new_video", label="✅ Yes"),
            cl.Action(name="Continue chatting", value="continue", label="❌ No"),
        ],
    ).send()

    if action:
        if action.get("value") == "new_video":
            model, chunk_settings, qdrant_client = set_internal_state()
            await chainlit_summarize_video(model, chunk_settings, qdrant_client)

# TODO: use hypothetical answer
# TODO (optional): use automatic rephrasing of query (perf / quality)
# TODO: use chat history as context

@cl.on_message
async def on_message(message: cl.Message):
    model, chunk_settings, qdrant_client = get_internal_state()

    response = get_bot_response(
        query=message.content,
        model=model,
        qdrant_client=qdrant_client,
    )

    await cl.Message(
            content=response,
        ).send()

    await chainlit_ask_if_new_video(model, chunk_settings, qdrant_client)

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="ChatGPT (3.5)",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="Mistral (7B)",
            markdown_description="The underlying LLM model is **Mistral**.",
            icon="https://picsum.photos/250",
        ),
    ]

def get_summary(video_url: str, model: str, chunk_settings: ChunkSettings, qdrant_client: QdrantClient):
    # Get video transcript
    transcript = get_video_transcript(video_url)
    # Cut transcript text into chunks
    chunks = get_text_chunks(
        transcript, chunk_settings.chunk_size, chunk_settings.chunk_overlap
    )
    # Embed and store chunks
    embed_and_store_chunks(model, chunks, qdrant_client, collection_name=QDRANT_COLLECTION_NAME)
    # Chunk transcript and summarize
    # TODO: have summarize_transcript take only chunks and only summarize
    return summarize_transcript(
        model=model,
        transcript=transcript,
        chunk_size=chunk_settings.chunk_size,
        chunk_overlap=chunk_settings.chunk_overlap,
    )

def get_bot_response(query: str, model, qdrant_client: QdrantClient):
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

### ---------- ###
### Setters and getters for chainlit internal state variables

def set_model(model_name: str):
    model = MODELS[model_name]
    cl.user_session.set("model", model)
    return model

def get_model():
    return cl.user_session.get("model")

def set_chunk_settings(model_name: str) -> ChunkSettings:
    chunk_settings = ChunkSettings(token_context_size=MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE[model_name])
    cl.user_session.set("chunk_settings", chunk_settings)
    return chunk_settings

def get_chunk_settings() -> ChunkSettings:
    return cl.user_session.get("chunk_settings")

def set_qdrant_client(model_name: str) -> QdrantClient:
    qdrant_client = create_qdrant_collection(collection_name=QDRANT_COLLECTION_NAME,
                                             embedding_vector_size=MODEL_TO_EMBEDDING_VECTOR_SIZE[model_name])
    cl.user_session.set("qdrant_client", qdrant_client)
    return qdrant_client

def get_qdrant_client() -> QdrantClient:
    return cl.user_session.get("qdrant_client")

### ---------- ###

def set_internal_state():
    model_name = CHAT_PROFILE_TO_MODEL_NAME[cl.user_session.get("chat_profile")]
    model = set_model(model_name)
    chunk_settings = set_chunk_settings(model_name)
    qdrant_client = set_qdrant_client(model_name)
    return model, chunk_settings, qdrant_client

def get_internal_state():
    model = cl.user_session.get("model")
    chunk_settings = cl.user_session.get("chunk_settings")
    qdrant_client = cl.user_session.get("qdrant_client")
    return model, chunk_settings, qdrant_client

### ---------- ###

def is_valid_youtube_url(url):
    """
    Check if the given string is a valid YouTube URL.

    Args:
    url (str): The string to be checked.

    Returns:
    bool: True if the string is a valid YouTube URL, False otherwise.
    """
    # Regular expression pattern for YouTube URL
    youtube_pattern = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    # Match the pattern
    match = re.match(youtube_pattern, url)
    return bool(match)
