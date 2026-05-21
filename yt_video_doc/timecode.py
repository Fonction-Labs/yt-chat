from __future__ import annotations


def parse_timecode(value: str | int | float | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)

    raw = str(value).strip()
    if raw.isdigit():
        return float(raw)

    parts = raw.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid timecode: {value}")

    seconds = 0.0
    multiplier = 1
    for part in reversed(parts):
        if not part:
            raise ValueError(f"Invalid timecode: {value}")
        seconds += float(part) * multiplier
        multiplier *= 60
    return seconds


def format_timecode(seconds: float | int | None) -> str:
    if seconds is None:
        return "00:00"
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
