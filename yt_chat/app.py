import os
import chainlit as cl
from chainlit import make_async
from typing import Optional

from yt_chat.internal_state import InternalState

from yt_chat.utils.youtube import is_valid_youtube_url
from yt_chat.utils.transcript import get_video_transcript
from yt_chat.llm.summarize import summarize_transcript
from yt_chat.llm.answer import embed_and_store_text, answer_query

from yt_chat.config import Config

# TODO: use hypothetical answer
# TODO (optional): use automatic rephrasing of query (perf / quality)
# TODO: use chat history as context

# TODO: send chat message if youtube transcript does not exist for video
# TODO: handle error (send chat message) if rate limit or connection error on OpenAI

# TODO: tests
# TODO: docker

# ------ CHAINLIT CHAT PROFILES AND INTERNAL STATE ------

CHAT_PROFILE_TO_MODEL_NAME = {"ChatGPT": "chat-gpt", "Mistral": "mistral"}

def set_internal_state() -> InternalState:
    chat_profile = cl.user_session.get("chat_profile")
    api_key = cl.user_session.get("env")["OPENAI_API_KEY"]

    model_name = CHAT_PROFILE_TO_MODEL_NAME.get(chat_profile)
    if model_name:
        internal_state = InternalState(model_name, api_key)
        cl.user_session.set("internal_state", internal_state)
        return internal_state
    else:
        raise ValueError("Unknown chat profile or model name")


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
        content="Please specify the YouTube URL you want to summarize."
    ).send()

    if response and is_valid_youtube_url(response["output"]):
        video_url = response["output"]

        # Get video transcript
        transcript = await make_async(get_video_transcript)(video_url)

        output_message = cl.Message(content="")
        await output_message.send()

        # Summarize transcript
        summary = await make_async(summarize_transcript)(
            transcript, internal_state.model, internal_state.chunk_settings
        )

        output_message.content = f"Here is the summary of the YouTube video:\n{summary}"
        await output_message.update()

        # Compute and store embeddings for later chatting
        await make_async(embed_and_store_text)(
            transcript,
            internal_state.model,
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
                name="New video?", value="new_video", label="🔁 Yes, summarize new video"
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
        qdrant_client=internal_state.qdrant_client,
    )

    output.content = answer
    await output.update()
    await chainlit_ask_if_new_video(internal_state)


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


# ------------
