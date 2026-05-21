from yt_video_doc.models import TranscriptSegment
from yt_video_doc.service import search_segments, slice_segments, truncate_text


def test_slice_segments_by_time_range():
    segments = [
        TranscriptSegment("intro", 0, 2),
        TranscriptSegment("middle", 10, 3),
        TranscriptSegment("end", 20, 2),
    ]
    selected = slice_segments(segments, start=5, end=15)
    assert [segment.text for segment in selected] == ["middle"]


def test_search_segments_returns_context():
    segments = [
        TranscriptSegment("alpha intro", 0, 2),
        TranscriptSegment("beta target phrase", 10, 3),
        TranscriptSegment("gamma outro", 20, 2),
    ]
    results = search_segments(segments, "target", limit=1, context_seconds=20)
    assert len(results) == 1
    assert results[0]["text"] == "beta target phrase"
    assert "alpha intro" in results[0]["context"]
    assert "gamma outro" in results[0]["context"]


def test_truncate_text():
    content, truncated = truncate_text("abcdef", 4)
    assert truncated is True
    assert content.endswith("[truncated]")
