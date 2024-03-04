import os
from flask import Flask, render_template, request, jsonify
from functools import reduce, cache
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.llms import CTransformers
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter


app = Flask(__name__)
chat_history = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Render the index page and process the form submission.

    Returns:
        str: Rendered HTML content.
    """
    video_url = None
    summary = None
    if request.method == 'POST':
        video_url = request.form['video_url']
        transcript_text = get_video_transcript(video_url)
        summary = summarize_video(transcript_text)
        chat_history[video_url] = []  # Initialize chat history for the video
    return render_template('index.html', video_url=video_url, summary=summary, chat_history=chat_history.get(video_url, []))

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
        video_id = video_url.split('v=')[-1]
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'fr'])
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
    #text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    # splits = text_splitter.split_documents(docs)
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

    #llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                        #model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                        model_file="mistral-7b-instruct-v0.1.Q2_K.gguf",
                        config={'max_new_tokens': 4096, 'temperature': 0.7, 'context_length': 4096},
                        threads=os.cpu_count())

    map_template = """Write a detailed summary of the following youtube video transcript.
    "{docs}".
    # CONCISE SUMMARY:"""
    map_template="Give me a list of the colors of the rainbow."
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    generated_text = llm("AI is going to")
    print(generated_text)

    if False:
        # llm_chain = LLMChain(llm=llm, prompt=prompt_langchain)
        # stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        # final_summary = stuff_chain.run(docs)
        # return final_summary
        reduce_template = """The following is set of summaries:
        {docs}
        Take these and distill it into a final, consolidated detailed summary.
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        # Run chain
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=4096, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(docs)
    #return map_reduce_chain.run(split_docs)
    return map_chain.run(docs)

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
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                        #model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                        model_file="mistral-7b-instruct-v0.1.Q2_K.gguf",
                        config={'max_new_tokens': 4096, 'temperature': 0.7, 'context_length': 4096},
                        threads=os.cpu_count())

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Process user messages and generate bot responses.

    Returns:
        str: JSON response containing the bot's response and chat history.
    """
    video_url = request.form['video_url']
    user_message = request.form['user_message']

    # Add user message to chat history
    chat_history[video_url].append({'role': 'user', 'content': user_message})

    # Get the video transcript
    transcript_text = get_video_transcript(video_url)
    docs = get_text_chunks_langchain(transcript_text)

    # Get the bot's response using the LangChain function
    bot_response = answer_question_video(docs, user_message)

    # Add bot response to chat history
    chat_history[video_url].append({'role': 'bot', 'content': bot_response})

    return jsonify({'bot_response': bot_response, 'chat_history': chat_history[video_url]})

if __name__ == '__main__':
    app.run(debug=True)
