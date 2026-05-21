import pytest
from app import get_video_transcript, get_text_chunks_langchain, summarize_video

@pytest.fixture
def valid_video_url():
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

@pytest.fixture
def invalid_video_url():
    return "https://www.youtube.com/watch?v=invalid"

def test_get_video_transcript_valid(valid_video_url):
    # Test for successful retrieval of video transcript
    transcript_text = get_video_transcript(valid_video_url)
    assert transcript_text is not None

def test_get_video_transcript_invalid(invalid_video_url):
    # Test for error handling when video does not exist
    transcript_text = get_video_transcript(invalid_video_url)
    assert transcript_text is None

def test_get_text_chunks_langchain():
    # Test for splitting text into chunks
    text = "This is a test text for splitting into chunks."
    docs = get_text_chunks_langchain(text)
    assert docs

def test_summarize_video(valid_video_url):
    # Test for summarizing video transcript
    transcript_text = get_video_transcript(valid_video_url)
    summary = summarize_video(transcript_text)
    assert summary
