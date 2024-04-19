import re
from functools import reduce
from youtube_transcript_api import YouTubeTranscriptApi

def is_valid_youtube_url(url):
    """
    Check if the given string is a valid YouTube URL.

    Args:
    url (str): The string to be checked.

    Returns:
    bool: True if the string is a valid YouTube URL, False otherwise.
    """
    youtube_pattern = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    match = re.match(youtube_pattern, url)
    return bool(match)


def get_video_transcript_and_duration(video_url):
    """
    Get the transcript of a YouTube video, along with its duration.

    Parameters:
        video_url (str): URL of the YouTube video.

    Returns:
        str: the concatenated text of the video transcript.
        duration: the duration of the video in minutes.
    """
    try:
        video_id = video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "fr"])
        duration = int(transcript[-1]["start"] // 60)
        transcript = reduce(lambda x, y: x + " " + y, map(lambda x: x["text"], transcript))
        return transcript, duration
    except Exception as e:
        print(f"Error retrieving video transcript: {e}")
        return None, None
