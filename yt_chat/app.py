import os
import argparse
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from tqdm import tqdm
from functools import reduce, cache

from yt_chat.llm.chat import answer_question_video
from yt_chat.llm.summarize import summarize_transcript
from yt_chat.utils.transcript import get_video_transcript
from yt_chat.utils.chunk_text import get_text_chunks
from yt_chat.llm.models import ChatOpenAILLM, LlamaLLM
from yt_chat.settings import DEFAULT_MODEL_SETTINGS_MISTRAL, DEFAULT_AGENT_SETTING_OPENAI, DEFAULT_AGENT_SETTING_MISTRAL


def get_model(model_name):
    if model_name == "gpt3-api":
        llm = ChatOpenAILLM(model_name="gpt-3.5-turbo", temperature=0)
    elif model_name == "mistral-local":
        llm = LlamaLLM(**DEFAULT_MODEL_SETTINGS_MISTRAL.dict())
    else:
        raise ValueError(
            f"Incorrect value '{model_name}' for MODEL_CHOICE. Must be either 'GPT' (for gpt-3.5's API) or 'MISTRAL' (for MistralAI's local mistral-7b)"
        )
    return llm

def get_agent_settings(model_name):
    if model_name == "gpt3-api":
        return DEFAULT_AGENT_SETTING_OPENAI
    elif model_name == "mistral-local":
        return DEFAULT_AGENT_SETTING_MISTRAL
    else:
        raise ValueError(
            f"Incorrect value '{model_name}' for MODEL_CHOICE. Must be either 'GPT' (for gpt-3.5's API) or 'MISTRAL' (for MistralAI's local mistral-7b)"
        )

def get_app(model_name):

    llm = get_model(model_name)
    agent_settings = get_agent_settings(model_name)

    app = FastAPI()
    templates = Jinja2Templates(directory="yt_chat/templates")
    chat_history = {}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request, "video_url": None, "summary": None})

    @app.post("/summarize", response_class=HTMLResponse)
    async def summarize(request: Request, video_url: str = Form(...)):
        transcript_text = get_video_transcript(video_url)
        summary = summarize_transcript(
            transcript_text,
            llm=llm,
            agent_settings=agent_settings
        )
        chat_history[video_url] = []  # Initialize chat history for the video
        return templates.TemplateResponse("index.html", {"request": request, "video_url": video_url, "summary": summary, "chat_history": chat_history.get(video_url, [])})

    @app.post("/chat", response_model=dict)
    async def chat(request: Request, video_url: str = Form(...), user_message: str = Form(...)):
        if video_url not in chat_history:
            raise HTTPException(status_code=404, detail="Video chat history not found")
        
        chat_history[video_url].append({"role": "user", "content": user_message})
        transcript_text = get_video_transcript(video_url)
        docs = get_text_chunks(transcript_text, int(agent_settings.token_context_size * agent_settings.safety_token_prct), int(agent_settings.token_context_size * 0.1))
        bot_response = answer_question_video(docs, user_message, n=agent_settings.n_vectors)
        chat_history[video_url].append({"role": "bot", "content": bot_response})
        return {"bot_response": bot_response, "chat_history": chat_history[video_url]}
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI server with model options")
    parser.add_argument("--model_name", choices=["gpt3-api", "mistral-local"], default="chatgpt3", help="Choose the model to use for chat and summarize functions")
    args = parser.parse_args()
    app = get_app(args.model_name)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
