import re

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
