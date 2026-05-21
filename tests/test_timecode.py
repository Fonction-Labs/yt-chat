import pytest

from yt_video_doc.timecode import format_timecode, parse_timecode


def test_parse_timecode_accepts_common_shapes():
    assert parse_timecode("90") == 90.0
    assert parse_timecode("01:30") == 90.0
    assert parse_timecode("01:02:03") == 3723.0
    assert parse_timecode(None) is None


def test_parse_timecode_rejects_invalid_shape():
    with pytest.raises(ValueError):
        parse_timecode("01::03")


def test_format_timecode():
    assert format_timecode(90) == "01:30"
    assert format_timecode(3723) == "01:02:03"
