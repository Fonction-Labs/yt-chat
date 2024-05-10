import os
import chainlit as cl
from chainlit import make_async
from openai import OpenAIError
from typing import Optional

from yt_chat.internal_state import InternalState

from yt_chat.utils.stream import stream_string
from yt_chat.utils.youtube import (
    is_valid_youtube_url,
    get_video_transcript_and_duration,
)
from yt_chat.llm.summarize import summarize_transcript
from yt_chat.llm.answer import embed_and_store_text, answer_query

from yt_chat.config import Config

# TODO (optional): use automatic rephrasing of query (perf / quality)
# TODO: use chat history as context
# TODO: tests

# ------ CHAINLIT CHAT PROFILES AND INTERNAL STATE ------

CHAT_PROFILE_TO_MODEL_NAME = {"ChatGPT": "chatgpt4"} #, "Mistral": "mistral"}


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
    await ask_for_file(internal_state)


async def ask_for_file(internal_state):
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a document to begin!",
            accept=["text/plain", "application/pdf", "text/csv"],
            max_size_mb=20,
            timeout=180,
        ).send()
    file = files[0]
    # TODO: write temp pdf file and open it

    from pypdf import PdfReader
    filename = "file.pdf"
    reader = PdfReader(filename)
    for i, page in enumerate(reader.pages[:20]):
        text = page.extract_text()
        embed_and_store_text(text, # text
                             {"pdf_page": i, "pdf_filename": filename}, # meta
                             internal_state.model,
                             internal_state.chunk_settings,
                             internal_state.qdrant_client)

    # Convert PDF to images
    if False: # Set this to true when running for first time
        from pdf2image import convert_from_path
        from tqdm import tqdm
        images = convert_from_path(filename)
        for i in tqdm(range(len(images))):
            images[i].save('temp_images/' + 'page'+ str(i) +'.jpg', 'JPEG')

    await answer_question(internal_state)

async def answer_question(internal_state):
    question = await cl.AskUserMessage(
        content="Please ask a question on the document."
    ).send()

    if question:
        question = question["output"]
        output = cl.Message(content="")
        await output.send()

        response, top_k_metas = await make_async(answer_query)(
            query=question,
            model=internal_state.model,
            qdrant_client=internal_state.qdrant_client,
            top_k=Config.RETRIEVAL_TOP_K,
            use_hypothetical=Config.RETRIEVAL_USE_HYPOTHETICAL,
        )

        print("TOP K METAS", top_k_metas)

        for meta in top_k_metas:
            image_path = 'temp_images/page' + str(meta["pdf_page"]) + '.jpg'
            image = cl.Image(path=image_path, name="image", display="inline")
            await cl.Message(
                content=f"Source material (page {meta['pdf_page']})",
                elements=[image],
            ).send()

        # Send (streaming) output message
        response = stream_string(response)
        async for part in response:
            await output.stream_token(part)
        await output.update()

        await answer_question(internal_state)

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


# ------------
