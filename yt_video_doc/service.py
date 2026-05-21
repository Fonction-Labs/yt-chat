from __future__ import annotations

from typing import Any

from yt_video_doc.cache import TranscriptCache
from yt_video_doc.formatters import document_to_markdown, segments_to_text
from yt_video_doc.models import TranscriptSegment, VideoDocument
from yt_video_doc.sources import fetch_from_sources, parse_languages
from yt_video_doc.timecode import parse_timecode
from yt_video_doc.video import extract_video_id


def fetch_document(
    url: str,
    languages: str | list[str] | None = None,
    refresh: bool = False,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    video_id = extract_video_id(url)
    cache = TranscriptCache(cache_dir)
    failures = []

    if not refresh and cache.exists(video_id):
        document = cache.load(video_id)
        paths = {"json": str(cache.json_path(video_id)), "markdown": str(cache.markdown_path(video_id))}
        return _document_payload("fetch", document, paths, cache_hit=True, failures=failures)

    document, failures = fetch_from_sources(url, parse_languages(languages))
    paths = cache.save(document)
    return _document_payload("fetch", document, paths, cache_hit=False, failures=failures)


def load_or_fetch_document(
    url: str,
    languages: str | list[str] | None = None,
    refresh: bool = False,
    cache_dir: str | None = None,
) -> tuple[VideoDocument, dict[str, str], bool]:
    video_id = extract_video_id(url)
    cache = TranscriptCache(cache_dir)
    if not refresh and cache.exists(video_id):
        return (
            cache.load(video_id),
            {"json": str(cache.json_path(video_id)), "markdown": str(cache.markdown_path(video_id))},
            True,
        )
    payload = fetch_document(url, languages=languages, refresh=True, cache_dir=cache_dir)
    return (
        cache.load(video_id),
        payload.get("cache", {}),
        False,
    )


def read_document(
    url: str,
    start: str | int | float | None = None,
    end: str | int | float | None = None,
    max_chars: int | None = None,
    output_format: str = "markdown",
    languages: str | list[str] | None = None,
    refresh: bool = False,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    document, paths, cache_hit = load_or_fetch_document(url, languages, refresh, cache_dir)
    selected = slice_segments(document.segments, parse_timecode(start), parse_timecode(end))
    content = _format_content(document, selected, output_format)
    content, truncated = truncate_text(content, max_chars)
    payload = _document_payload("read", document, paths, cache_hit=cache_hit)
    payload.update(
        {
            "content": content,
            "format": output_format,
            "truncated": truncated,
            "range": {"start": parse_timecode(start), "end": parse_timecode(end)},
        }
    )
    return payload


def search_document(
    url: str,
    query: str,
    limit: int = 8,
    context_seconds: float = 20,
    languages: str | list[str] | None = None,
    refresh: bool = False,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    document, paths, cache_hit = load_or_fetch_document(url, languages, refresh, cache_dir)
    results = search_segments(document.segments, query, limit=limit, context_seconds=context_seconds)
    payload = _document_payload("search", document, paths, cache_hit=cache_hit)
    payload.update({"query": query, "results": results})
    return payload


def metadata_document(
    url: str,
    languages: str | list[str] | None = None,
    refresh: bool = False,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    document, paths, cache_hit = load_or_fetch_document(url, languages, refresh, cache_dir)
    return _document_payload("metadata", document, paths, cache_hit=cache_hit)


def slice_segments(
    segments: list[TranscriptSegment],
    start: float | None = None,
    end: float | None = None,
) -> list[TranscriptSegment]:
    selected = []
    for segment in segments:
        if start is not None and segment.end < start:
            continue
        if end is not None and segment.start > end:
            continue
        selected.append(segment)
    return selected


def search_segments(
    segments: list[TranscriptSegment],
    query: str,
    limit: int = 8,
    context_seconds: float = 20,
) -> list[dict[str, Any]]:
    normalized_query = query.casefold().strip()
    terms = [term for term in normalized_query.split() if term]
    scored = []
    for index, segment in enumerate(segments):
        haystack = segment.text.casefold()
        direct = normalized_query in haystack if normalized_query else False
        term_score = sum(haystack.count(term) for term in terms)
        score = (10 if direct else 0) + term_score
        if score:
            scored.append((score, segment.start, index, segment))

    scored.sort(key=lambda item: (-item[0], item[1]))
    results = []
    seen_windows: set[tuple[int, int]] = set()
    for score, _, index, segment in scored[: max(limit * 3, limit)]:
        start = max(0.0, segment.start - context_seconds)
        end = segment.end + context_seconds
        context = slice_segments(segments, start, end)
        key = (int(context[0].start), int(context[-1].end)) if context else (index, index)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        results.append(
            {
                "score": score,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "context": segments_to_text(context),
            }
        )
        if len(results) >= limit:
            break
    return results


def truncate_text(content: Any, max_chars: int | None) -> tuple[Any, bool]:
    if not isinstance(content, str):
        return content, False
    if max_chars is None or max_chars <= 0 or len(content) <= max_chars:
        return content, False
    marker = "\n\n[truncated]"
    return content[: max(0, max_chars - len(marker))].rstrip() + marker, True


def _format_content(document: VideoDocument, segments: list[TranscriptSegment], output_format: str) -> str | dict[str, Any]:
    if output_format == "json":
        return {
            "metadata": document.metadata.to_dict(),
            "segments": [segment.to_dict() for segment in segments],
        }
    if output_format == "text":
        return segments_to_text(segments)
    if output_format == "markdown":
        return document_to_markdown(document, segments)
    raise ValueError(f"Unsupported format: {output_format}")


def _document_payload(
    mode: str,
    document: VideoDocument,
    paths: dict[str, str],
    cache_hit: bool,
    failures: list[Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "mode": mode,
        "video_id": document.metadata.video_id,
        "metadata": document.metadata.to_dict(),
        "segment_count": len(document.segments),
        "cache_hit": cache_hit,
        "cache": paths,
    }
    if failures:
        payload["source_failures"] = [
            failure.__dict__ if hasattr(failure, "__dict__") else failure for failure in failures
        ]
    return payload
