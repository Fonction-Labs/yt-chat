from pydantic import BaseModel

class ChunkSettings(BaseModel):
    token_context_size: int
    safety_percentage: float = 0.7
    characters_per_token: int = 4

    @property
    def chunk_size(self):
        return int(self.token_context_size * self.characters_per_token * self.safety_percentage)

    @property
    def chunk_overlap(self):
        return int(self.chunk_size * 0.1)

def get_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> str:
    """
    Split text into chunks for processing.

    Parameters:
        text (str): The text to be split.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between adjacent chunks.

    Returns:
        list: List of text chunks (strings).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = max(end, end - chunk_overlap)
    return chunks
