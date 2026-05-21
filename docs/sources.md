# Transcript source notes

## Google and YouTube

The official YouTube Data API `captions.download` endpoint requires OAuth scopes and permissions on the video. That makes it useful for owned channels, but not for arbitrary public YouTube URLs.

For arbitrary public videos, this project uses the same public caption surfaces that browser playback and common transcript tools rely on:

- `youtube-transcript-api`
- `yt-dlp` caption metadata and JSON3 subtitle URLs

Both can break when YouTube changes clients, rate limits, or proof-of-origin enforcement. The code keeps source diagnostics explicit so a caller can distinguish "no captions" from "caption fetch blocked".

## Optional managed fallback

If `TRANSCRIPT_API_KEY` is set, the tool can call TranscriptAPI as a last fallback. This is intentionally optional because the default tool should work locally when YouTube exposes public captions.
