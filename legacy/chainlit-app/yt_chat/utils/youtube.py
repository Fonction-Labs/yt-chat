import re
from functools import reduce
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url):
    # Regular expressions for different URL formats
    regex_patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:m\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in regex_patterns:
        match = re.match(pattern, url)
        if match:
            return match.group(1)
    return None

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
        video_id = extract_video_id(video_url)
        languages = YouTubeTranscriptApi.list_transcripts(video_id)._generated_transcripts.keys()
        if languages:
            lang = list(languages)[0]
        else:
            lang = 'en'
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        duration = int(transcript[-1]["start"] // 60)
        transcript = reduce(lambda x, y: x + " " + y, map(lambda x: x["text"], transcript))
        return transcript, duration
    except Exception as e:
        print(f"Error retrieving video transcript: {e}")
        return None, None
