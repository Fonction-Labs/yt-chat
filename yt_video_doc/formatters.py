from __future__ import annotations

import json
from typing import Any

from yt_video_doc.models import TranscriptSegment, VideoDocument
from yt_video_doc.timecode import format_timecode


def segments_to_text(segments: list[TranscriptSegment], include_timestamps: bool = True) -> str:
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if include_timestamps:
            lines.append(f"[{format_timecode(segment.start)}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def document_to_markdown(document: VideoDocument, segments: list[TranscriptSegment] | None = None) -> str:
    selected = segments if segments is not None else document.segments
    metadata = document.metadata
    title = metadata.title or metadata.video_id
    lines = [
        f"# {title}",
        "",
        f"Video: {metadata.url}",
        f"Video ID: {metadata.video_id}",
    ]
    if metadata.author:
        lines.append(f"Channel: {metadata.author}")
    if metadata.language_code:
        lines.append(f"Language: {metadata.language_code}")
    if metadata.source:
        lines.append(f"Transcript source: {metadata.source}")
    if metadata.fetched_at:
        lines.append(f"Fetched at: {metadata.fetched_at}")
    lines.extend(["", "## Transcript", "", segments_to_text(selected)])
    return "\n".join(lines).strip() + "\n"


def render_payload(payload: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if isinstance(payload, str):
        return payload
    if "content" in payload:
        return str(payload["content"])
    return json.dumps(payload, ensure_ascii=False, indent=2)
