#!/usr/bin/env python3
"""Fetch public YouTube captions and render timestamped Markdown."""

import argparse
import html
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi, YouTubeTranscriptApiException


VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(value: str) -> str:
    if VIDEO_ID.fullmatch(value):
        return value

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com"}:
        parts = parsed.path.strip("/").split("/")
        if parts[0] == "watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif parts[0] in {"shorts", "live", "embed"} and len(parts) > 1:
            candidate = parts[1]
        else:
            candidate = ""
    else:
        candidate = ""

    if not VIDEO_ID.fullmatch(candidate):
        raise ValueError("Provide a valid YouTube URL or 11-character video ID.")
    return candidate


def choose_track(tracks, languages):
    for language in languages:
        matches = [track for track in tracks if track.language_code == language]
        if matches:
            return next((track for track in matches if not track.is_generated), matches[0])
    return next((track for track in tracks if not track.is_generated), tracks[0])


def timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def render(video_id, track, snippets) -> str:
    kind = "automatic" if track.is_generated else "manual"
    lines = [
        f"# YouTube transcript: {video_id}",
        "",
        f"Source: https://www.youtube.com/watch?v={video_id}",
        f"Language: {track.language_code} ({kind})",
        "",
    ]
    for snippet in snippets:
        content = " ".join(html.unescape(snippet.text).split())
        if content:
            lines.append(f"[{timestamp(snippet.start)}](https://youtu.be/{video_id}?t={int(snippet.start)}) {content}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="YouTube URL or video ID")
    parser.add_argument("--languages", default="fr,en", help="Preferred language codes, comma-separated (default: fr,en)")
    parser.add_argument("--output", type=Path, help="Write Markdown to this file instead of stdout")
    args = parser.parse_args(argv)

    try:
        video_id = extract_video_id(args.video)
    except ValueError as exc:
        parser.error(str(exc))

    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    if not languages:
        parser.error("--languages must contain at least one language code")

    try:
        tracks = list(YouTubeTranscriptApi().list(video_id))
        if not tracks:
            raise RuntimeError("No captions are available for this video.")
        track = choose_track(tracks, languages)
        result = render(video_id, track, track.fetch())
    except (YouTubeTranscriptApiException, RuntimeError) as exc:
        print(f"Could not retrieve captions: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(args.output)
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
