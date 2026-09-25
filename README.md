<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="public/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="public/logo_light.png">
    <img alt="yt-chat logo" src="public/logo_light.png" width="400">
  </picture>
</div>

<h3 align="center">yt-chat is now an AI skill.</h3>
<h4 align="center">Paste a YouTube link. Codex or Claude Code can summarize the video and answer your questions.</h4>

<div align="center">
  <img alt="Python version" src="https://img.shields.io/badge/python-3.10-blue">
</div>

## What changed

yt-chat now does two things: a small Python script fetches a video's public captions without a YouTube API key, and your AI assistant uses the timestamped transcript to summarize the video or answer your questions. The Chainlit app, model configuration, Qdrant, Docker, and Poetry have been removed. There is no mandatory summary before you ask questions.

## Install

Clone or copy this repository to `~/.codex/skills/yt-chat` for Codex, or `~/.claude/skills/yt-chat` for Claude Code. Then install its only Python dependency:

```bash
cd ~/.codex/skills/yt-chat # or ~/.claude/skills/yt-chat
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Invoke `$yt-chat` or simply share a YouTube link and ask for a summary or pose a question. The skill uses the assistant's existing model; it needs no separate OpenAI API key.

## Run the script directly

From the skill directory:

```bash
.venv/bin/python scripts/get_transcript.py 'https://www.youtube.com/watch?v=jNQXAC9IVRw' --output /tmp/yt-chat-transcript.md
```

The script also accepts a video ID or a `youtu.be`, `shorts`, or `live` URL. It prefers French captions, then English, and falls back to an available original-language track. Use `--languages en,fr` to change the preference. Its Markdown output includes timestamps linked to the corresponding moments in the video.

Caption retrieval depends on what YouTube makes publicly available and whether YouTube is reachable from your machine. Videos without accessible captions, private videos, and blocked videos may fail. The script does not transcribe audio itself.

---

## Before the skill: the chat application

The material below preserves the original chat application's README for historical context. Its installation commands and file paths refer to the retired version and do not work in this skill-only repository.

<h3 align="center">yt-chat is a tool designed to help you summarize any Youtube video.</h3>
<h4 align="center">Once a video is summarized, you can also ask more precise questions about the video in question.</h4>

<div align="center">
  <img alt="Original yt-chat app demo" src="https://github.com/mcordier/yt-chat/assets/40168022/daa1f7b3-0cf8-414c-9200-429142b4e251">
</div>

### Installation

This section is useful if you want to install the original yt-chat app on your machine.

After cloning the original repository, and with [`poetry`](https://python-poetry.org/) installed, run the following command from the repository root:

```
poetry install
```

To run `yt-chat`, simply do:

```
poetry run chainlit run yt_chat/app.py -w
```

### Using ChatGPT-3.5

If you wish to use an `OpenAI` model, for example `gpt-3.5`, you will need your [OpenAI API key](https://platform.openai.com/api-keys).

Once you've input your OpenAI API key requested by `yt-chat`, select the `ChatGPT` chat profile in the UI.

### Using Mistral-7B

If you wish to use a local `ollama` model, for example `mistral-7b`, you will need to install [ollama](https://ollama.com/) on your machine.

First, make sure your `ollama` server is running. Then, run `yt-chat` (when running `yt-chat` for the first time, you will be asked for an OpenAI API key; this is irrelevant for local models, enter anything to continue).

Once `yt-chat` is running, simply select the `Mistral` chat profile in the UI.

### Configuration

Check out `yt_chat/config.py` and `yt_chat/config_prompts.py` for configuring the original app parameters and prompts.

### Docker

The original app provided Docker support:

```
docker-compose up -d --build
```

### Acknowledgments

The original **yt-chat** app was powered by **[chainlit](https://github.com/Chainlit/chainlit)**, **[qdrant](https://github.com/qdrant/qdrant)**, and **[ollama](https://github.com/ollama/ollama-python)**.
