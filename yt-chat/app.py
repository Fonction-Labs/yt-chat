import os
from tqdm import tqdm
from flask import Flask, render_template, request, jsonify
from functools import reduce, cache
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain.schema.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter

from models import ChatOpenAILLM, LlamaLLM

app = Flask(__name__)
chat_history = {}

MODEL_CHOICE = "GPT"  # OpenAI's API for gpt-3.5
#MODEL_CHOICE = "MISTRAL"  # MistralAI's local mistral-7b

if MODEL_CHOICE == "GPT":
    MODEL_TOKENS_CONTEXT = 4096
    SAFETY_TOKEN_PERCENTAGE = 0.7
    BASE_PROMPT_TRANSCRIPT_SUMMARY = """Write a detailed summary of the following youtube video transcript.
    "{docs}".
    CONCISE SUMMARY:"""
    BASE_PROMPT_SUMMARIES_SUMMARY = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated detailed summary of 3 to 5 paragraphs.
    Helpful Answer:"""
elif MODEL_CHOICE == "MISTRAL":
    MODEL_TOKENS_CONTEXT = 4096
    SAFETY_TOKEN_PERCENTAGE = 0.7
    N_THREADS = 8
    N_GPU_LAYERS = -1
    BASE_PROMPT_TRANSCRIPT_SUMMARY = """I want you to write a concise summary of an extract of a youtube video transcript. Do NOT mention things you are not sure or do not know. If you don't know or lack context, do not make up things. Only use information provided by the transcript to write your concise summary. TRANSCRIPT: '{docs}'. ### CONCISE SUMMARY:"""
    BASE_PROMPT_SUMMARIES_SUMMARY = """The following will be a set of summaries from a transcript. Take these and distill it into a final, consolidated summary of the main points. Construct it as a well organized summary of the main points and should be between 3 and 5 paragraphs. SUMMARIES: "{docs}". ### DISTILLED ANSWER:"""
    MODEL_PATH = "./mistral-7b-instruct-v0.2.Q5_K_S.gguf"

if MODEL_CHOICE == "GPT":
    LLM = ChatOpenAILLM(model_name="gpt-3.5-turbo", temperature=0)
elif MODEL_CHOICE == "MISTRAL":
    LLM = LlamaLLM(
        model_path=MODEL_PATH,
        n_ctx=MODEL_TOKENS_CONTEXT,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
    )
else:
    raise ValueError(
        f"Incorrect value '{MODEL_CHOCIE}' for MODEL_CHOICE. Must be either 'GPT' (for gpt-3.5's API) or 'MISTRAL' (for MistralAI's local mistral-7b)"
    )
print("Model loaded")


@app.route("/")
def index():
    return render_template(
        "index.html",
        video_url=None,
        summary=None,
    )


@app.route("/summarize", methods=["GET", "POST"])
def summarize():
    video_url = None
    summary = None
    if request.method == "POST":
        video_url = request.form["video_url"]
        transcript_text = get_video_transcript(video_url)
        summary = summarize_video(transcript_text)
        chat_history[video_url] = []  # Initialize chat history for the video
    return render_template(
       "index.html",
       video_url=video_url,
       summary=summary,
       chat_history=chat_history.get(video_url, []),
    )


# @cache
def get_video_transcript(video_url):
    """
    Get the transcript of a YouTube video.

    Parameters:
        video_url (str): URL of the YouTube video.

    Returns:
        str: The concatenated text of the video transcript.
    """
    try:
        video_id = video_url.split("v=")[-1]
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "fr"])
        text = reduce(lambda x, y: x + " " + y, map(lambda x: x["text"], srt))
        return text
    except Exception as e:
        print(f"Error retrieving video transcript: {e}")
        return None


def get_text_chunks_langchain(text):
    """
    Split text into chunks for processing with LangChain.

    Parameters:
        text (str): The text to be split.

    Returns:
        list: List of LangChain Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MODEL_TOKENS_CONTEXT * 4 * SAFETY_TOKEN_PERCENTAGE,
        chunk_overlap=200 * 4,
    )  # A token is roughly 4-characters, on average
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def summarize_video(transcript_text):
    """
    Summarize the video transcript using LangChain.

    Parameters:
        transcript_text (str): The text of the video transcript.

    Returns:
        str: The generated summary.
    """
    docs = get_text_chunks_langchain(transcript_text)
    transcripts = [d.page_content for d in docs]

    summaries = []
    for transcript in tqdm(transcripts):
        #break
        prompt = BASE_PROMPT_TRANSCRIPT_SUMMARY.replace("{docs}", transcript)
        answer = LLM.predict(prompt, temperature=0) # Temperature has no impact for ChatOpenAILLM (set at initialization)
        print(answer)
        summaries.append(answer)
    total_summary = " ".join(summaries)

    while (
        True
    ):  # While the total summary length is over 90% of the theoretical token limit input of the model, continue splitting the total summary
        new_summaries = []
        split_total_summary = [
            d.page_content for d in get_text_chunks_langchain(total_summary)
        ]
        print(f"Split all transcript summaries into {len(split_total_summary)}")
        for split_summary in split_total_summary:
            prompt = BASE_PROMPT_SUMMARIES_SUMMARY.replace("{docs}", split_summary)
            answer = LLM.predict(prompt, temperature=0.7) # Temperature has no impact for ChatOpenAILLM (set at initialization)
            print(answer)
            new_summaries.append(answer)
        total_summary = " ".join(new_summaries)

        if len(total_summary) < MODEL_TOKENS_CONTEXT * 4 * SAFETY_TOKEN_PERCENTAGE:
            break

    return total_summary.replace("\n", "<br>")


def answer_question_video(docs, question):
    """
    Generate an answer to a question about the video using LangChain.

    Parameters:
        docs (list): List of LangChain Document objects representing the video transcript.
        question (str): The question asked by the user.

    Returns:
        str: The generated answer.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        # model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_file="mistral-7b-instruct-v0.1.Q2_K.gguf",
        config={"max_new_tokens": 4096, "temperature": 0.7, "context_length": 4096},
        threads=os.cpu_count(),
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Process user messages and generate bot responses.

    Returns:
        str: JSON response containing the bot's response and chat history.
    """
    video_url = request.form["video_url"]
    user_message = request.form["user_message"]

    # Add user message to chat history
    chat_history[video_url].append({"role": "user", "content": user_message})

    # Get the video transcript
    transcript_text = get_video_transcript(video_url)
    docs = get_text_chunks_langchain(transcript_text)

    # Get the bot's response using the LangChain function
    bot_response = answer_question_video(docs, user_message)

    # Add bot response to chat history
    chat_history[video_url].append({"role": "bot", "content": bot_response})

    return jsonify(
        {"bot_response": bot_response, "chat_history": chat_history[video_url]}
    )


if __name__ == "__main__":
    app.run(debug=True, port=9000)
