from yt_video_doc.sources import parse_json3_segments, parse_languages, parse_vtt_segments


def test_parse_languages():
    assert parse_languages("fr,en") == ["fr", "en"]
    assert parse_languages(["de", "en"]) == ["de", "en"]
    assert parse_languages(None) == ["fr", "en"]


def test_parse_json3_segments():
    raw = """
    {
      "events": [
        {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "hello"}, {"utf8": " world"}]},
        {"tStartMs": 3000, "dDurationMs": 1000, "segs": [{"utf8": "\\n"}]}
      ]
    }
    """
    segments = parse_json3_segments(raw)
    assert len(segments) == 1
    assert segments[0].text == "hello world"
    assert segments[0].start == 1.0
    assert segments[0].duration == 2.0


def test_parse_vtt_segments():
    raw = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:03.000 --> 00:00:04.000
Again
"""
    segments = parse_vtt_segments(raw)
    assert [segment.text for segment in segments] == ["Hello world", "Again"]
