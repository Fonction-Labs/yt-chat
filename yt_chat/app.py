import os
import chainlit as cl
from chainlit import make_async
from openai import OpenAIError
from typing import Optional

from flib.models.openai import OpenAIGPTModel, OpenAIEmbeddingModel
from flib.models.ollama import OllamaModel, OllamaEmbeddingModel

from flib.utils.stream import stream_string
from yt_chat.utils.youtube import (
    is_valid_youtube_url,
    get_video_transcript_and_duration,
)

from yt_chat.internal_state import InternalState
from yt_chat.llm.summarize import summarize_transcript
from yt_chat.llm.answer import embed_and_store_text, answer_query
from yt_chat.config import Config

# TODO (optional): use automatic rephrasing of query (perf / quality)
# TODO: use chat history as context
# TODO: tests

# ------ CHAINLIT CHAT PROFILES AND INTERNAL STATE ------


def set_internal_state() -> InternalState:
    chat_profile = cl.user_session.get("chat_profile")
    api_key = cl.user_session.get("env")["OPENAI_API_KEY"]

    # Cannot do a dic config, or else all models will be loaded in memory which is not desirable.
    if chat_profile == "ChatGPT":
        model = OpenAIGPTModel("gpt-3.5-turbo", api_key)
        embedding_model = OpenAIEmbeddingModel("text-embedding-3-small", api_key)
    elif chat_profile == "ChatGPT-4":
        model = OpenAIGPTModel("gpt-4-turbo", api_key)
        embedding_model = OpenAIEmbeddingModel("text-embedding-3-small", api_key)
    elif chat_profile == "Mistral":
        model = OllamaModel("mistral")
        embedding_model = OllamaEmbeddingModel("mistral")
    else:
        raise ValueError(f"Cannot find model for profile {chat_profile}. Make sure that you selected a correct profile.")

    internal_state = InternalState(model, embedding_model)
    cl.user_session.set("internal_state", internal_state)
    return internal_state


def get_internal_state() -> Optional[InternalState]:
    return cl.user_session.get("internal_state")


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="ChatGPT",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1024px-ChatGPT_logo.svg.png",
        ),
        cl.ChatProfile(
            name="ChatGPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1024px-ChatGPT_logo.svg.png",
        ),
        cl.ChatProfile(
            name="Mistral",
            markdown_description="The underlying LLM model is **Mistral 7B**.",
            icon="https://cdn.jaimelesstartups.fr/wp-content/uploads/2024/02/announcing-mistral.png",
        ),
    ]


# ------------

# ------ CHAINLIT PROCESSES ------


@cl.on_chat_start
async def main():
    """
    Executed on every new chat start (change profile)
    """
    internal_state = set_internal_state()
    await cl.Avatar(
        name="yt-chat",
        url="./public/avatar.png",
    ).send()
    await chainlit_summarize_video(internal_state)


async def chainlit_summarize_video(internal_state):
    response = await cl.AskUserMessage(
        content="Please specify a YouTube URL for a video you want to summarize."
    ).send()

    if response and is_valid_youtube_url(response["output"]):
        video_url = response["output"]

        # Get video transcript
        transcript, duration = await make_async(get_video_transcript_and_duration)(
            video_url
        )
        if transcript is None:
            await cl.Message(
                content="Unfortunately, no transcript exists for the provided YouTube video."
            ).send()
            await chainlit_summarize_video(internal_state)

        # Prepare output chat message
        output = cl.Message(content="")
        await output.send()

        # Summarize transcript
        try:
            summary = await make_async(summarize_transcript)(
                transcript, internal_state.model, internal_state.chunk_settings,
                internal_state.generate_summarize_transcript_prompt, internal_state.generate_summarize_summaries_prompt,
            )
            summary = f"Here is the summary of the YouTube video:\n{summary}\n\n**yt-chat** just saved you **{duration} minutes** of your life! 🕰️ "
        except OpenAIError as e:
            await cl.Message(
                content=f"Error authenticating to the OpenAI API.\n\nMake sure the API key you provided is correct (click on your avatar, and then on **API Keys** to set your key in **yt-chat**).\n\n{e}"
            ).send()
            await chainlit_summarize_video(internal_state)

        # Send (streaming) output message
        summary = stream_string(summary)
        async for part in summary:
            await output.stream_token(part)
        #output.content = summary # alternative if not streaming
        await output.update()

        # Compute and store embeddings for later chatting
        await make_async(embed_and_store_text)(
            transcript,
            internal_state.embedding_model,
            internal_state.chunk_settings,
            internal_state.qdrant_client,
        )
        await chainlit_ask_if_new_video(internal_state)
    else:
        await cl.Message(content="You did not provide a valid YouTube URL.").send()
        await chainlit_summarize_video(internal_state)


async def chainlit_ask_if_new_video(internal_state):
    action = await cl.AskActionMessage(
        content="Do you want to summarize a new video?",
        actions=[
            cl.Action(
                name="New video?",
                value="new_video",
                label="🔁 Yes, summarize a new video",
            ),
            cl.Action(
                name="Continue chatting",
                value="continue",
                label="➡️  No, continue chatting",
            ),
        ],
    ).send()

    if action and action.get("value") == "new_video":
        # When we work with a new video, we need to start with a new internal state (new vector database)
        internal_state = set_internal_state()
        await chainlit_summarize_video(internal_state)


@cl.on_message
async def on_message(message: cl.Message):
    internal_state = get_internal_state()

    output = cl.Message(content="")
    await output.send()

    answer = await make_async(answer_query)(
        query=message.content,
        model=internal_state.model,
        embedding_model=internal_state.embedding_model,
        qdrant_client=internal_state.qdrant_client,
        generate_hypothetical_prompt=internal_state.generate_hypothetical_prompt,
        generate_context_prompt=internal_state.generate_context_prompt,
        top_k=Config.RETRIEVAL_TOP_K,
        use_hypothetical=Config.RETRIEVAL_USE_HYPOTHETICAL,
    )

    answer = stream_string(answer)
    async for part in answer:
        await output.stream_token(part)
    #output.content = answer # alternative if not streaming
    await output.update()

    await chainlit_ask_if_new_video(internal_state)


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


# ------------
