#!/usr/bin/env python3
"""Fetch public YouTube captions and render timestamped Markdown."""

import argparse
import html
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi


VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(frozen=True)
class Caption:
    start: float
    duration: float
    text: str


@dataclass(frozen=True)
class CaptionTrack:
    language_code: str
    is_generated: bool


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


def language_rank(code: str, languages: list[str]) -> tuple[int, int]:
    code = code.casefold()
    for index, preferred in enumerate(languages):
        preferred = preferred.casefold()
        if code == preferred:
            return index, 0
        if code.startswith(preferred + "-") or preferred.startswith(code + "-"):
            return index, 1
    return len(languages), 0


def choose_track(tracks, languages):
    return min(
        tracks,
        key=lambda track: (
            *language_rank(track.language_code, languages),
            track.is_generated,
        ),
    )


def fetch_with_transcript_api(video_id: str, languages: list[str]):
    tracks = list(YouTubeTranscriptApi().list(video_id))
    if not tracks:
        raise RuntimeError("No captions are available for this video.")
    track = choose_track(tracks, languages)
    captions = list(track.fetch())
    if not captions:
        raise RuntimeError("The selected caption track is empty.")
    return track, captions


def choose_ytdlp_track(info: dict, languages: list[str]) -> dict:
    candidates = []
    for kind in ("subtitles", "automatic_captions"):
        for language, formats in (info.get(kind) or {}).items():
            for item in formats:
                if item.get("url") and item.get("ext") in {"json3", "vtt"}:
                    candidates.append({**item, "language": language, "generated": kind == "automatic_captions"})
    if not candidates:
        raise RuntimeError("yt-dlp found no usable caption track.")
    return min(
        candidates,
        key=lambda track: (
            *language_rank(track["language"], languages),
            track["generated"],
            track["ext"] != "json3",
        ),
    )


def parse_json3(raw: str) -> list[Caption]:
    captions = []
    for event in json.loads(raw).get("events", []):
        content = "".join(str(part.get("utf8") or "") for part in event.get("segs") or [])
        if content.strip():
            captions.append(Caption(
                start=float(event.get("tStartMs", 0)) / 1000,
                duration=float(event.get("dDurationMs", 0)) / 1000,
                text=content,
            ))
    return captions


def parse_vtt(raw: str) -> list[Caption]:
    captions = []
    for block in re.split(r"\r?\n\s*\r?\n", raw):
        lines = block.splitlines()
        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start_raw, end_raw = lines[timing_index].split("-->", 1)
        start = parse_timecode(start_raw.strip())
        end = parse_timecode(end_raw.strip().split()[0])
        content = re.sub(r"<[^>]+>", "", " ".join(lines[timing_index + 1:]))
        if content.strip():
            captions.append(Caption(start, max(0.0, end - start), content))
    return captions


def fetch_with_ytdlp(video_id: str, languages: list[str]):
    from yt_dlp import YoutubeDL
    from yt_dlp.networking.common import Request

    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        if not info:
            raise RuntimeError("yt-dlp could not read the video metadata.")
        track = choose_ytdlp_track(info, languages)
        request = Request(track["url"], headers=info.get("http_headers") or {})
        with ydl.urlopen(request) as response:
            raw = response.read().decode("utf-8", errors="replace")
    captions = parse_json3(raw) if track["ext"] == "json3" else parse_vtt(raw)
    if not captions:
        raise RuntimeError("yt-dlp returned an empty caption track.")
    return CaptionTrack(track["language"], track["generated"]), captions


def parse_timecode(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid timecode: {value}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timecode: {value}") from exc
    if (any(not math.isfinite(number) or number < 0 for number in numbers)
            or any(number >= 60 for number in numbers[1:])
            or (len(parts) > 1 and any(not part.isdigit() for part in parts[:-1]))):
        raise ValueError(f"Invalid timecode: {value}")
    return sum(number * 60 ** index for index, number in enumerate(reversed(numbers)))


def timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def select_captions(captions, start: float | None, end: float | None):
    return [
        caption for caption in captions
        if (start is None or caption.start >= start or caption.start + caption.duration > start)
        and (end is None or caption.start < end)
    ]


def render(video_id, track, captions, start: float | None = None, end: float | None = None) -> str:
    kind = "automatic" if track.is_generated else "manual"
    lines = [
        f"# YouTube transcript: {video_id}",
        "",
        f"Source: https://www.youtube.com/watch?v={video_id}",
        f"Language: {track.language_code} ({kind})",
    ]
    if start is not None or end is not None:
        lines.append(f"Range: {timestamp(start or 0)}–{timestamp(end) if end is not None else 'end'}")
    lines.append("")
    for caption in captions:
        content = " ".join(html.unescape(caption.text).split())
        if content:
            lines.append(f"[{timestamp(caption.start)}](https://youtu.be/{video_id}?t={int(caption.start)}) {content}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="YouTube URL or video ID")
    parser.add_argument("--languages", default="fr,en", help="Preferred language codes, comma-separated (default: fr,en)")
    parser.add_argument("--from", dest="start", help="Start time in seconds, MM:SS, or HH:MM:SS")
    parser.add_argument("--to", dest="end", help="End time in seconds, MM:SS, or HH:MM:SS")
    parser.add_argument("--output", type=Path, help="Write Markdown to this file instead of stdout")
    args = parser.parse_args(argv)

    try:
        video_id = extract_video_id(args.video)
        start = parse_timecode(args.start) if args.start is not None else None
        end = parse_timecode(args.end) if args.end is not None else None
    except ValueError as exc:
        parser.error(str(exc))
    if start is not None and end is not None and start >= end:
        parser.error("--from must be earlier than --to")

    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    if not languages:
        parser.error("--languages must contain at least one language code")

    try:
        track, captions = fetch_with_transcript_api(video_id, languages)
    except Exception as primary_error:
        try:
            track, captions = fetch_with_ytdlp(video_id, languages)
        except Exception as fallback_error:
            print(f"Could not retrieve captions: {primary_error}; yt-dlp: {fallback_error}", file=sys.stderr)
            return 1

    selected = select_captions(captions, start, end)
    if not selected:
        print("No captions in the requested time range.", file=sys.stderr)
        return 1
    result = render(video_id, track, selected, start, end)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(args.output)
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
