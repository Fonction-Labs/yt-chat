<div align="center">
  <picture align="center" with="200">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/mcordier/yt-chat/blob/5bbae54e1c9f46f11af9090a83089786d9832e6f/public/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/mcordier/yt-chat/blob/5bbae54e1c9f46f11af9090a83089786d9832e6f/public/logo_light.png">
  <img alt="yt-chat logo" src="https://github.com/mcordier/yt-chat/blob/5bbae54e1c9f46f11af9090a83089786d9832e6f/public/logo_light.png" width="400"/>
  </picture>
</div>

<h3 align="center">yt-chat is a tool designed to help you summarize any Youtube video.</h3>
<h4 align="center">Once a video is summarized, you can also ask more precise questions about the video in question.</h4>

<div align="center">
<!-- <img alt="All workflows" src="https://github.com/Fonction-Labs/jade/actions/workflows/all.yml/badge.svg"/> -->
<!-- <a href="https://github.com/Fonction-Labs/jade/actions/workflows/all.yml?query=branch%3Amain"> -->
<img alt="Python version" src="https://img.shields.io/badge/python-3.10-blue"/>
<!-- </a> -->
<a href="https://yt.chat">
<img alt="Try live!" src="https://img.shields.io/static/v1?color=green&label=&message=Try live!"/>
</a>
</div>

<div align="center">
<img src="https://github.com/mcordier/yt-chat/assets/40168022/daa1f7b3-0cf8-414c-9200-429142b4e251"></img>
</div>

---

[Try yt-chat!](https://yt.chat)
------------
You can play with yt-chat right away by visiting [yt.chat](https://yt.chat). You can summarize up to 5 videos per day.

Installation
------------
This section is useful if you want to install yt-chat on your machine.

After cloning the repository, and with [`poetry`](https://python-poetry.org/) installed, run the following command from the repository root:
```
poetry install
```

To run `yt-chat`, simply do:
```
poetry run chainlit run yt_chat/app.py -w
```

If you don't want to bother, you can also use try the [online version](https://yt.chat) (only handles OpenAI models, i.e. GPT-3.5 and GPT-4).


### Using ChatGPT-3.5
If you wish to use an `OpenAI` model, for example `gpt-3.5`, you will need your [OpenAI API key](https://platform.openai.com/api-keys).

Once you've input your OpenAI API key requested by `yt-chat`, select the `ChatGPT` chat profile in the UI.


### Using Mistral-7B
If you wish to use a local `ollama` model, for example `mistral-7b`, you will need to install [ollama](https://ollama.com/) on your machine.

First, make sure your `ollama` server is running. Then, run `yt-chat` (when running `yt-chat` for the first time, you will asked for an OpenAI API key; this is irrelevant for local models, enter anything to continue).

Once `yt-chat` is running, simply select the `Mistral` chat profile in the UI.


Configuration
------------
Check out `yt_chat/config.py` and `yt_chat/config_prompts.py` for configuring the app parameters and prompts.


Docker
------------
We provide Docker support:
```
docker-compose up -d --build
```


Acknowledgments
------------
**yt-chat** is powered by **[chainlit](https://github.com/Chainlit/chainlit)**, **[qdrant](https://github.com/qdrant/qdrant)**, and **[ollama](https://github.com/ollama/ollama-python)**.
