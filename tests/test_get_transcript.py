import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


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


if __name__ == "__main__":
    unittest.main()
