from __future__ import annotations

import json
import os
from pathlib import Path

from yt_video_doc.formatters import document_to_markdown
from yt_video_doc.models import VideoDocument


def default_cache_dir() -> Path:
    configured = os.environ.get("YT_VIDEO_DOC_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "yt-video-doc"


class TranscriptCache:
    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()

    def json_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}.json"

    def markdown_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}.md"

    def exists(self, video_id: str) -> bool:
        return self.json_path(video_id).exists()

    def load(self, video_id: str) -> VideoDocument:
        with self.json_path(video_id).open("r", encoding="utf-8") as handle:
            return VideoDocument.from_dict(json.load(handle))

    def save(self, document: VideoDocument) -> dict[str, str]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.json_path(document.metadata.video_id)
        md_path = self.markdown_path(document.metadata.video_id)

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(document.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        with md_path.open("w", encoding="utf-8") as handle:
            handle.write(document_to_markdown(document))

        return {"json": str(json_path), "markdown": str(md_path)}
