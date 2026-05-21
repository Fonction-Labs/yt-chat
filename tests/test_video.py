import pytest

from yt_video_doc.video import extract_video_id


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id(url, expected):
    assert extract_video_id(url) == expected


def test_extract_video_id_rejects_unknown_url():
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/nope")
