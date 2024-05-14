import os
import chainlit as cl
from chainlit import make_async
from openai import OpenAIError
from typing import Optional

from flib.models.openai import OpenAIGPTModel, OpenAIEmbeddingModel
from flib.models.ollama import OllamaModel, OllamaEmbeddingModel
from flib.utils.stream import stream_string

from yt_chat.internal_state import InternalState
from yt_chat.core.answer import embed_and_store_text, answer_query
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
            name="ChatGPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1024px-ChatGPT_logo.svg.png",
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
        name="doc-chat",
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
    filepath = file.path
    filename = filepath.split("/")[-1]

    output = cl.Message(content="")
    await output.send()

    from pypdf import PdfReader
    from tqdm import tqdm
    from pdf2image import convert_from_path
    import tempfile
    # Create temp dir for image-converted PDF pages
    temp_dir = tempfile.mkdtemp()
    cl.user_session.set("temp_dir", temp_dir)
    # Convert PDF to images
    images = convert_from_path(filepath)
    for i in tqdm(range(len(images))):
        images[i].save(os.path.join(temp_dir, f"{filename}_page{str(i)}.jpg"), "JPEG")

    # Extract text
    reader = PdfReader(filepath)
    pages_text = [page.extract_text() for page in tqdm(reader.pages)]
    # Embed text pages
    # /var/folders/ss/750ds0zs2sj6r9gvwhtbxsgc0000gn/T/tmph_zcq8fj/page5.jp
    print("FILEPATH", filepath)
    print("FILENAME", filename)
    for i, text in enumerate(pages_text):
        await make_async(embed_and_store_text)(text, # text for page i
                                               {"pdf_page": i, "jpg_path": os.path.join(temp_dir, f"{filename}_page{str(i)}.jpg")}, # meta
                                               internal_state.embedding_model,
                                               internal_state.chunk_settings,
                                               internal_state.qdrant_client)

    await output.update()

    await on_message(internal_state)

# @cl.on_message
async def on_message(internal_state):
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
            embedding_model=internal_state.embedding_model,
            qdrant_client=internal_state.qdrant_client,
            generate_hypothetical_prompt=internal_state.generate_hypothetical_prompt,
            generate_context_prompt=internal_state.generate_context_prompt,
            top_k=Config.RETRIEVAL_TOP_K,
            cosine_threshold=Config.RETRIEVAL_COSINE_THRESHOLD,
            use_hypothetical=Config.RETRIEVAL_USE_HYPOTHETICAL,
        )

        print("TOP K METAS", top_k_metas)
        temp_dir = cl.user_session.get("temp_dir")
        metas = sorted(top_k_metas, key=lambda meta : meta["pdf_page"])
        for meta in metas:
            image = cl.Image(path=meta["jpg_path"], name="image", display="inline")
            await cl.Message(
                content=f"Source material (page {str(meta['pdf_page'])})",
                elements=[image],
            ).send()

        # Send (streaming) output message
        response = stream_string(response)
        async for part in response:
            await output.stream_token(part)
        await output.update()

        await on_message(internal_state)
    #else:
    #    await on_message(internal_state)

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


# ------------
