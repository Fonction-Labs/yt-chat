# YT Chat

YT Chat is now an AI skill.
Paste a YouTube link. Codex or Claude Code answers your questions.

## What changed

YT Chat now does two things: a small Python script fetches a video's public captions without a YouTube API key, and your AI assistant answers questions using the timestamped transcript. The Chainlit app, model configuration, Qdrant, Docker, and Poetry have been removed. There is no mandatory summary before you ask questions.

## Install

Clone or copy this repository to `~/.codex/skills/yt-chat` for Codex, or `~/.claude/skills/yt-chat` for Claude Code. Then install its only Python dependency:

```bash
cd ~/.codex/skills/yt-chat # or ~/.claude/skills/yt-chat
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Invoke `$yt-chat` or simply share a YouTube link and ask a question. The skill uses the assistant's existing model; it needs no separate OpenAI API key.

## Run the script directly

From the skill directory:

```bash
.venv/bin/python scripts/get_transcript.py 'https://www.youtube.com/watch?v=jNQXAC9IVRw' --output /tmp/yt-chat-transcript.md
```

The script also accepts a video ID or a `youtu.be`, `shorts`, or `live` URL. It prefers French captions, then English, and falls back to an available original-language track. Use `--languages en,fr` to change the preference. Its Markdown output includes timestamps linked to the corresponding moments in the video.

Caption retrieval depends on what YouTube makes publicly available and whether YouTube is reachable from your machine. Videos without accessible captions, private videos, and blocked videos may fail. The script does not transcribe audio itself.
