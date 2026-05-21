from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(value: str) -> str:
    raw = value.strip()
    if VIDEO_ID_RE.match(raw):
        return raw

    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if host.endswith("youtu.be") and path_parts:
        candidate = path_parts[0]
        if VIDEO_ID_RE.match(candidate):
            return candidate

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id and VIDEO_ID_RE.match(query_id):
            return query_id

        for marker in ("shorts", "embed", "live"):
            if marker in path_parts:
                index = path_parts.index(marker)
                if len(path_parts) > index + 1 and VIDEO_ID_RE.match(path_parts[index + 1]):
                    return path_parts[index + 1]

    raise ValueError(f"Could not extract a YouTube video id from: {value}")


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"
