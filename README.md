# yt-video-doc

Treat a YouTube video as a timestamped text document.

This project replaces the old Chainlit prototype with a smaller agent-facing tool:

- a CLI for fetching, reading, slicing, and searching transcripts
- an MCP server exposing the same behavior through one tool
- a local cache so videos become reusable documents

The tool does not summarize by itself. It gives the agent reliable document access, then the agent can summarize, quote, compare, or inspect the transcript.

## Install

```bash
uv sync
```

## Skill

A Codex skill is bundled in:

```text
skills/youtube-video-doc
```

If your skill manager supports installing from a GitHub repo path, target:

```text
mcordier/yt-chat -> skills/youtube-video-doc
```

The skill uses the local `yt-video` CLI when available, and otherwise falls back to:

```bash
uvx --from "git+https://github.com/mcordier/yt-chat.git@main" yt-video
```

Once installed, it is a good fit for prompts like:

- `Summarize this YouTube video`
- `Find where they talk about X in this video`
- `Quote the section around 12:30`
- `Compare these two YouTube videos`

## CLI

Fetch and cache a transcript:

```bash
uv run yt-video fetch "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Read the transcript:

```bash
uv run yt-video read "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Read a time slice:

```bash
uv run yt-video read "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --from 00:00 --to 01:00
```

Search inside the transcript:

```bash
uv run yt-video search "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "never gonna"
```

Output JSON when another tool needs structured data:

```bash
uv run yt-video read "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --format json
```

## MCP

Run the server over stdio:

```bash
uv run yt-video-mcp
```

The server exposes a single tool named `youtube_video_doc`.

Supported modes:

- `fetch`: fetch and cache the transcript
- `read`: read the transcript, optionally with `start`, `end`, and `max_chars`
- `search`: search words or phrases in the transcript with timestamped context
- `metadata`: return video and cache metadata

Example MCP call shape:

```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "mode": "search",
  "query": "never gonna",
  "context_seconds": 20
}
```

## Transcript sources

Fetch order:

1. `youtube-transcript-api` for public YouTube captions and auto captions.
2. `yt-dlp` caption metadata and JSON3 caption tracks.
3. TranscriptAPI if `TRANSCRIPT_API_KEY` is available.

Google's official YouTube Data API can list and download caption tracks only with the right OAuth permissions for the video owner. It is not enough for arbitrary public videos, so the default path uses public caption surfaces instead.

## Cache

Default cache directory:

```text
~/.cache/yt-video-doc
```

Override it with:

```bash
YT_VIDEO_DOC_CACHE_DIR=/path/to/cache uv run yt-video read VIDEO_URL
```

Each cached video has:

- `VIDEO_ID.json`: structured transcript document
- `VIDEO_ID.md`: readable timestamped transcript

## Legacy

The previous Chainlit application has been moved to:

```text
legacy/chainlit-app
```
