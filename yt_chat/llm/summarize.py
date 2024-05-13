from tqdm import tqdm
from functools import partial

from flib.utils.chunk_text import ChunkSettings, get_text_chunks

def summarize_transcript(
    transcript: str,
    model,
    chunk_settings: ChunkSettings,
    generate_summarize_transcript_prompt,
    generate_summarize_summaries_prompt,
) -> str:

    total_summary = transcript
    # If transcript (or combination of summaries) is too long (longer than chunk_size)...
    while (len(total_summary) > chunk_settings.chunk_size):
        # Chunk the transcript (or combination of summaries) into multiple chunks
        transcripts = get_text_chunks(text=total_summary,
                                      chunk_size=chunk_settings.chunk_size,
                                      chunk_overlap=chunk_settings.chunk_overlap)
        print(f"Split all transcript summaries into {len(transcripts)}")
        # Generate summarization prompts for the model
        # (the output is [messages1, messages2, ...])
        prompts = [generate_summarize_transcript_prompt(transcript) for transcript in transcripts]
        # Summarize with the model and generated prompt
        #summaries = model.run(prompt=prompts, temperature=0.)
        summaries = [model.run(prompt=prompt, temperature=0.) for prompt in prompts] #TODO!!! Parallelize
        # Combine summaries into one string
        total_summary = " ".join(summaries)
    # Summarize the final summary
    prompt = generate_summarize_summaries_prompt(total_summary)
    return model.run(prompt=prompt, temperature=0.)
