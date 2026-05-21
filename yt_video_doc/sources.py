from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from yt_video_doc.models import TranscriptSegment, VideoDocument, VideoMetadata
from yt_video_doc.video import canonical_url, extract_video_id


class TranscriptSourceError(RuntimeError):
    pass


class TranscriptUnavailable(TranscriptSourceError):
    pass


@dataclass(frozen=True)
class SourceFailure:
    source: str
    error: str


def normalize_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_languages(languages: str | Iterable[str] | None) -> list[str]:
    if languages is None:
        return ["fr", "en"]
    if isinstance(languages, str):
        values = re.split(r"[, ]+", languages)
    else:
        values = list(languages)
    parsed = [item.strip() for item in values if item and item.strip()]
    return parsed or ["fr", "en"]


def _segments_from_raw(raw_items: Iterable[dict[str, Any]]) -> list[TranscriptSegment]:
    segments = []
    for item in raw_items:
        text = normalize_text(str(item.get("text", "")))
        if not text:
            continue
        start = float(item.get("start", 0.0))
        duration = float(item.get("duration", 0.0))
        segments.append(TranscriptSegment(text=text, start=start, duration=duration))
    return segments


def fetch_with_youtube_transcript_api(video_url: str, languages: list[str]) -> VideoDocument:
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = extract_video_id(video_url)
    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=languages)
    except Exception:
        fetched = _fetch_best_transcript(api, video_id, languages)

    raw_items = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
    segments = _segments_from_raw(raw_items)
    if not segments:
        raise TranscriptUnavailable("youtube-transcript-api returned an empty transcript")

    metadata = VideoMetadata(
        video_id=video_id,
        url=canonical_url(video_id),
        language=getattr(fetched, "language", None),
        language_code=getattr(fetched, "language_code", None),
        is_generated=getattr(fetched, "is_generated", None),
        source="youtube-transcript-api",
    ).with_source("youtube-transcript-api")
    return VideoDocument(metadata=metadata, segments=segments)


def _fetch_best_transcript(api: Any, video_id: str, languages: list[str]) -> Any:
    transcript_list = api.list(video_id)
    transcripts = list(transcript_list)
    if not transcripts:
        raise TranscriptUnavailable("no transcripts listed by youtube-transcript-api")

    def find_by_language(pool: list[Any], language_code: str) -> Any | None:
        for transcript in pool:
            if getattr(transcript, "language_code", None) == language_code:
                return transcript
        return None

    manual = [item for item in transcripts if not getattr(item, "is_generated", False)]
    generated = [item for item in transcripts if getattr(item, "is_generated", False)]

    for language_code in languages:
        for pool in (manual, generated):
            transcript = find_by_language(pool, language_code)
            if transcript is not None:
                return transcript.fetch()

    return (manual or generated or transcripts)[0].fetch()


def fetch_with_ytdlp(video_url: str, languages: list[str]) -> VideoDocument:
    from yt_dlp import YoutubeDL

    video_id = extract_video_id(video_url)
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(canonical_url(video_id), download=False)

    metadata = VideoMetadata(
        video_id=video_id,
        url=canonical_url(video_id),
        title=info.get("title"),
        author=info.get("channel") or info.get("uploader"),
        duration=info.get("duration"),
        source="yt-dlp",
    ).with_source("yt-dlp")

    track = _select_ytdlp_caption_track(info, languages)
    if track is None:
        raise TranscriptUnavailable("yt-dlp found no usable caption track")

    raw = _download_text(track["url"])
    ext = track.get("ext")
    if ext == "json3" or raw.lstrip().startswith("{"):
        segments = parse_json3_segments(raw)
    else:
        segments = parse_vtt_segments(raw)

    if not segments:
        raise TranscriptUnavailable("yt-dlp returned an empty caption track")

    metadata = metadata.merge(
        language=track.get("name") or track.get("language"),
        language_code=track.get("language"),
        is_generated=track.get("kind") == "automatic_captions",
    )
    return VideoDocument(metadata=metadata, segments=segments)


def _select_ytdlp_caption_track(info: dict[str, Any], languages: list[str]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for kind in ("subtitles", "automatic_captions"):
        tracks = info.get(kind) or {}
        for language, formats in tracks.items():
            for item in formats:
                candidate = dict(item)
                candidate["language"] = language
                candidate["kind"] = kind
                candidates.append(candidate)

    if not candidates:
        return None

    def candidate_rank(candidate: dict[str, Any]) -> tuple[int, int, int]:
        language = candidate.get("language", "")
        try:
            language_rank = languages.index(language)
        except ValueError:
            language_rank = len(languages) + 1
        ext_rank = 0 if candidate.get("ext") == "json3" else 1
        kind_rank = 0 if candidate.get("kind") == "subtitles" else 1
        return (language_rank, kind_rank, ext_rank)

    usable = [item for item in candidates if item.get("url") and item.get("ext") in {"json3", "vtt"}]
    return sorted(usable, key=candidate_rank)[0] if usable else None


def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_json3_segments(raw: str) -> list[TranscriptSegment]:
    payload = json.loads(raw)
    segments = []
    for event in payload.get("events", []):
        parts = event.get("segs") or []
        text = normalize_text("".join(part.get("utf8", "") for part in parts))
        if not text:
            continue
        start = float(event.get("tStartMs", 0)) / 1000.0
        duration = float(event.get("dDurationMs", 0)) / 1000.0
        segments.append(TranscriptSegment(text=text, start=start, duration=duration))
    return segments


def parse_vtt_segments(raw: str) -> list[TranscriptSegment]:
    segments = []
    block: list[str] = []
    for line in raw.splitlines():
        if line.strip():
            block.append(line)
            continue
        _append_vtt_block(segments, block)
        block = []
    _append_vtt_block(segments, block)
    return segments


def _append_vtt_block(segments: list[TranscriptSegment], block: list[str]) -> None:
    timing_index = next((index for index, line in enumerate(block) if "-->" in line), None)
    if timing_index is None:
        return
    timing = block[timing_index]
    start_raw, end_raw = [part.strip() for part in timing.split("-->", 1)]
    start = _parse_vtt_time(start_raw)
    end = _parse_vtt_time(end_raw.split()[0])
    text = normalize_text(" ".join(block[timing_index + 1 :]))
    if text:
        segments.append(TranscriptSegment(text=text, start=start, duration=max(0.0, end - start)))


def _parse_vtt_time(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    seconds = 0.0
    multiplier = 1
    for part in reversed(parts):
        seconds += float(part) * multiplier
        multiplier *= 60
    return seconds


def fetch_with_transcript_api(video_url: str, languages: list[str]) -> VideoDocument:
    api_key = os.environ.get("TRANSCRIPT_API_KEY")
    if not api_key:
        raise TranscriptUnavailable("TRANSCRIPT_API_KEY is not set")

    params = urllib.parse.urlencode(
        {
            "video_url": video_url,
            "format": "json",
            "include_timestamp": "true",
            "send_metadata": "true",
        }
    )
    request = urllib.request.Request(
        f"https://transcriptapi.com/api/v2/youtube/transcript?{params}",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    raw_segments = payload.get("transcript") or []
    segments = _segments_from_raw(raw_segments)
    if not segments:
        raise TranscriptUnavailable("TranscriptAPI returned an empty transcript")

    metadata_payload = payload.get("metadata") or {}
    video_id = payload.get("video_id") or extract_video_id(video_url)
    metadata = VideoMetadata(
        video_id=video_id,
        url=canonical_url(video_id),
        title=metadata_payload.get("title"),
        author=metadata_payload.get("author_name") or metadata_payload.get("author"),
        language=payload.get("language"),
        language_code=payload.get("language"),
        source="transcriptapi",
    ).with_source("transcriptapi")
    return VideoDocument(metadata=metadata, segments=segments)


def fetch_from_sources(video_url: str, languages: list[str]) -> tuple[VideoDocument, list[SourceFailure]]:
    failures: list[SourceFailure] = []
    for source_name, fetcher in (
        ("youtube-transcript-api", fetch_with_youtube_transcript_api),
        ("yt-dlp", fetch_with_ytdlp),
        ("transcriptapi", fetch_with_transcript_api),
    ):
        try:
            return fetcher(video_url, languages), failures
        except Exception as exc:
            failures.append(SourceFailure(source=source_name, error=str(exc)))
    detail = "; ".join(f"{failure.source}: {failure.error}" for failure in failures)
    raise TranscriptUnavailable(f"could not fetch transcript. {detail}")
