import os
import argparse
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from tqdm import tqdm
from functools import reduce, cache

from yt_chat.llm.summarize import summarize_transcript
from yt_chat.llm.retriever import retrieve_top_k_chunks_for_query
from yt_chat.utils.transcript import get_video_transcript
from yt_chat.utils.chunk_text import get_text_chunks
from yt_chat.settings import MODELS, MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE, MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC

class ChunkSettings:
    def __init__(self, model):
        token_context_size = MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE[model.model_name]
        safety_percentage = 0.7 # "Hard-coded" - this is to avoid going over the window context size of the model when chunking text
        characters_per_token = 4 # "Hard-coded" - this is the average value for the numbers of characters per token
        # We choose to chunk our input text so that each chunk takes half of the context window size of the model
        self.chunk_size = int(token_context_size * characters_per_token * safety_percentage)
        # We choose a chunk overlap of 10% of the chunk size
        self.chunk_overlap = int(self.chunk_size * 0.1)

# TODO: use chat history as context
def get_app(model_name):
    model = MODELS[model_name]
    chunk_settings = ChunkSettings(model)

    app = FastAPI()
    templates = Jinja2Templates(directory="yt_chat/templates")
    chat_history = {}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request, "video_url": None, "summary": None})

    @app.post("/summarize", response_class=HTMLResponse)
    async def summarize(request: Request, video_url: str = Form(...)):
        transcript = get_video_transcript(video_url)
        summary = summarize_transcript(
            model=model,
            transcript=transcript,
            chunk_size=chunk_settings.chunk_size,
            chunk_overlap=chunk_settings.chunk_overlap,
        )
        chat_history[video_url] = []  # Initialize chat history for the video
        return templates.TemplateResponse("index.html", {"request": request, "video_url": video_url, "summary": summary, "chat_history": chat_history.get(video_url, [])})

    @app.post("/chat", response_model=dict)
    async def chat(request: Request, video_url: str = Form(...), user_message: str = Form(...)):
        if video_url not in chat_history:
            raise HTTPException(status_code=404, detail="Video chat history not found")

        chat_history[video_url].append({"role": "user", "content": user_message})
        transcript = get_video_transcript(video_url)
        chunks = get_text_chunks(transcript, chunk_settings.chunk_size, chunk_settings.chunk_overlap)
        query = user_message # TODO: query should be hypothetical answer rather than question
        context = " ".join(retrieve_top_k_chunks_for_query(model, query, chunks, top_k=5))
        messages_with_context = MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC[model.model_name](query, context=context)
        bot_response = model.predict_messages(messages_with_context, temperature=0.)
        chat_history[video_url].append({"role": "bot", "content": bot_answer})
        return {"bot_response": bot_response, "chat_history": chat_history[video_url]}
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI server with model options")
    parser.add_argument("--model_name", choices=MODELS.keys(), default=list(MODELS.keys())[0], help="Choose the model to use for chat and summarize functions")
    args = parser.parse_args()
    app = get_app(args.model_name)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
