from tqdm import tqdm
from functools import partial

from yt_chat.utils.chunk_text import ChunkSettings, get_text_chunks

def summarize_transcript(
    transcript: str,
    model,
    chunk_settings: ChunkSettings,
) -> str:

    total_summary = transcript
    # If transcript (or combination of summaries) is too long (longer than chunk_size)...
    while (len(total_summary) > chunk_settings.chunk_size):
        # Chunk the transcript (or combination of summaries) into multiple chunks
        transcripts = get_text_chunks(text=total_summary,
                                      chunk_size=chunk_settings.chunk_size,
                                      chunk_overlap=chunk_settings.chunk_overlap)
        print(f"Split all transcript summaries into {len(transcripts)}")
        # Generate summarization messages for the model
        # (the output is [messages1, messages2, ...])
        list_messages = [model.generate_summarize_transcript_messages_func(transcript) for transcript in transcripts]
        # Summarize with the model and generated message
        summaries = model.predict_messages_batch_parallel(list_messages, temperature=0.)
        # Combine summaries into one string
        total_summary = " ".join(summaries)
    # Summarize the final summary
    messages = model.generate_summarize_summaries_messages_func(total_summary)
    return model.predict_messages(messages=messages, temperature=0.)
