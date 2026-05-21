---
name: "youtube-video-doc"
description: "Use when the user wants to read, search, quote, summarize, or inspect a YouTube video via its transcript. Treats a YouTube URL as a timestamped document and is a good fit for prompts like 'summarize this video', 'find where they talk about X', 'quote the section around 12:30', or 'compare these two YouTube videos'."
---

# YouTube Video Doc

Use this skill to treat a YouTube video like a readable document.

The bundled wrapper is:

```bash
./scripts/yt-video-doc
```

It will:

- use local `yt-video` if it is already installed
- otherwise run `yt-video` from `mcordier/yt-chat` through `uvx`

## Default workflow

1. Fetch or inspect metadata first if the video context is still unclear.
2. If the video is short enough, read the full transcript.
3. If the video is long, search for the relevant topic and then read one or more time slices.
4. Only after reading enough transcript, summarize or answer questions.

## Commands

Fetch and cache:

```bash
./scripts/yt-video-doc fetch "YOUTUBE_URL"
```

Read the full transcript:

```bash
./scripts/yt-video-doc read "YOUTUBE_URL"
```

Read a specific time range:

```bash
./scripts/yt-video-doc read "YOUTUBE_URL" --from 12:30 --to 18:00
```

Search inside the transcript:

```bash
./scripts/yt-video-doc search "YOUTUBE_URL" "topic or quote"
```

Get structured output:

```bash
./scripts/yt-video-doc read "YOUTUBE_URL" --format json
```

## Notes

- `--lang` sets transcript language preference order. It does not translate.
- `TRANSCRIPT_API_KEY` is optional and only used as a managed fallback.
- Prefer transcript-grounded summaries, quotes, and timestamps. If the transcript is incomplete, say so clearly.
