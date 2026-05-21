from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            text=str(data.get("text", "")),
            start=float(data.get("start", 0.0)),
            duration=float(data.get("duration", 0.0)),
        )


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    url: str
    title: str | None = None
    author: str | None = None
    duration: float | None = None
    language: str | None = None
    language_code: str | None = None
    is_generated: bool | None = None
    source: str | None = None
    fetched_at: str | None = None

    def with_source(self, source: str) -> "VideoMetadata":
        return VideoMetadata(
            video_id=self.video_id,
            url=self.url,
            title=self.title,
            author=self.author,
            duration=self.duration,
            language=self.language,
            language_code=self.language_code,
            is_generated=self.is_generated,
            source=source,
            fetched_at=self.fetched_at or datetime.now(timezone.utc).isoformat(),
        )

    def merge(self, **updates: Any) -> "VideoMetadata":
        values = self.to_dict()
        values.update({key: value for key, value in updates.items() if value is not None})
        return VideoMetadata.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "author": self.author,
            "duration": self.duration,
            "language": self.language,
            "language_code": self.language_code,
            "is_generated": self.is_generated,
            "source": self.source,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMetadata":
        return cls(
            video_id=str(data["video_id"]),
            url=str(data.get("url") or data["video_id"]),
            title=data.get("title"),
            author=data.get("author") or data.get("channel") or data.get("author_name"),
            duration=float(data["duration"]) if data.get("duration") is not None else None,
            language=data.get("language"),
            language_code=data.get("language_code"),
            is_generated=data.get("is_generated"),
            source=data.get("source"),
            fetched_at=data.get("fetched_at"),
        )


@dataclass(frozen=True)
class VideoDocument:
    metadata: VideoMetadata
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoDocument":
        return cls(
            metadata=VideoMetadata.from_dict(data["metadata"]),
            segments=[TranscriptSegment.from_dict(item) for item in data.get("segments", [])],
        )

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments if segment.text).strip()
