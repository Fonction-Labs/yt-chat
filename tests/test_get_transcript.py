import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "get_transcript.py"
spec = importlib.util.spec_from_file_location("get_transcript", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TranscriptTest(unittest.TestCase):
    def test_video_url_forms_and_invalid_host(self):
        video_id = "jNQXAC9IVRw"
        for value in (
            video_id,
            f"https://www.youtube.com/watch?v={video_id}&t=12",
            f"https://youtu.be/{video_id}",
            f"https://www.youtube.com/shorts/{video_id}",
            f"https://www.youtube.com/live/{video_id}",
        ):
            self.assertEqual(module.extract_video_id(value), video_id)
        with self.assertRaises(ValueError):
            module.extract_video_id(f"https://notyoutube.com/watch?v={video_id}")

    def test_original_language_fallback_preserves_timestamp_link(self):
        track = SimpleNamespace(language_code="de", is_generated=False)
        self.assertIs(module.choose_track([track], ["fr", "en"]), track)
        snippet = SimpleNamespace(start=65.8, text="Hallo &amp; willkommen\nbei uns")
        output = module.render("jNQXAC9IVRw", track, [snippet])
        self.assertIn("[01:05](https://youtu.be/jNQXAC9IVRw?t=65) Hallo & willkommen bei uns", output)

    def test_fallback_and_time_range_keep_overlapping_captions(self):
        track = module.CaptionTrack("en", False)
        captions = [
            module.Caption(4, 2, "First"),
            module.Caption(8, 2, "Second"),
            module.Caption(12, 1, "Third"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "excerpt.md"
            with patch.object(module, "fetch_with_transcript_api", side_effect=RuntimeError("blocked")), patch.object(
                module, "fetch_with_ytdlp", return_value=(track, captions)
            ) as fallback:
                result = module.main(["jNQXAC9IVRw", "--from", "00:05", "--to", "00:12", "--output", str(output)])
            self.assertEqual(result, 0)
            fallback.assert_called_once_with("jNQXAC9IVRw", ["fr", "en"])
            markdown = output.read_text()
            self.assertIn("Range: 00:05–00:12", markdown)
            self.assertIn("First", markdown)
            self.assertIn("Second", markdown)
            self.assertNotIn("Third", markdown)

    def test_ytdlp_track_selection_prefers_requested_locale(self):
        info = {
            "subtitles": {"es": [{"ext": "json3", "url": "https://example.com/es"}],
                          "en-US": [{"ext": "vtt", "url": "https://example.com/en"}]},
            "automatic_captions": {"fr": [{"ext": "json3", "url": "https://example.com/fr"}]},
        }
        chosen = module.choose_ytdlp_track(info, ["en"])
        self.assertEqual(chosen["language"], "en-US")
        self.assertFalse(chosen["generated"])

    def test_caption_parsers_preserve_timing(self):
        json3 = '{"events":[{"tStartMs":1500,"dDurationMs":2000,"segs":[{"utf8":"Hello"},{"utf8":" world"}]}]}'
        self.assertEqual(module.parse_json3(json3), [module.Caption(1.5, 2, "Hello world")])
        vtt = "WEBVTT\n\n00:00:05.000 --> 00:00:07.000\n<c>Goodbye</c>\n"
        self.assertEqual(module.parse_vtt(vtt), [module.Caption(5, 2, "Goodbye")])

    def test_timecodes_reject_invalid_ranges(self):
        self.assertEqual(module.parse_timecode("01:02:03"), 3723)
        self.assertEqual(module.parse_timecode("90"), 90)
        for value in ("00:75", "nan", "1.5:02"):
            with self.assertRaises(ValueError):
                module.parse_timecode(value)


if __name__ == "__main__":
    unittest.main()
