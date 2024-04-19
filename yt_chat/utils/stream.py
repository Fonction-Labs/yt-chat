import time

async def stream_string(s, chunk_size=10):
    """
    "Fake" string streamer.
    Transforms an already-ready, regular string into
    an async iter object to be streamed by chainlit.
    """
    for i in range(0, len(s), chunk_size):
        time.sleep(0.01)
        yield s[i:i+chunk_size]
