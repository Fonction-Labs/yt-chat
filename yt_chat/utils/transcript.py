from functools import reduce
from youtube_transcript_api import YouTubeTranscriptApi

# @cache
def get_video_transcript(video_url):
    """
    Get the transcript of a YouTube video.

    Parameters:
        video_url (str): URL of the YouTube video.

    Returns:
        str: The concatenated text of the video transcript.
    """
    try:
        video_id = video_url.split("v=")[-1]
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "fr"])
        text = reduce(lambda x, y: x + " " + y, map(lambda x: x["text"], srt))
        return text
    except Exception as e:
        print(f"Error retrieving video transcript: {e}")
        return None