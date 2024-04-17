from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from ..utils.chunk_text import get_text_chunks
from ..utils.tqdm_joblib import tqdm_joblib

from ..settings  import MODEL_TO_GENERATE_SUMMARIZE_TRANSCRIPT_MESSAGES_FUNC, MODEL_TO_GENERATE_SUMMARIZE_SUMMARIES_MESSAGES_FUNC

def summarize_transcript(
    model,
    transcript: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:

    generate_summarize_transcript_messages_func = MODEL_TO_GENERATE_SUMMARIZE_TRANSCRIPT_MESSAGES_FUNC[model.model_name]
    generate_summarize_summaries_messages_func = MODEL_TO_GENERATE_SUMMARIZE_SUMMARIES_MESSAGES_FUNC[model.model_name]

    total_summary = transcript
    # If transcript (or combination of summaries) is too long (longer than chunk_size)...
    while (len(total_summary) > chunk_size):
        # Chunk the transcript (or combination of summaries) into multiple chunks
        transcripts = get_text_chunks(text=total_summary,
                                      chunk_size=chunk_size,
                                      chunk_overlap=chunk_overlap)
        print(f"Split all transcript summaries into {len(transcripts)}")
        # Generate summarization messages for the model
        # (the output is [messages1, messages2, ...])
        list_messages = [generate_summarize_transcript_messages_func(transcript) for transcript in transcripts]
        # Summarize with the model and generated message
        summarize_func = partial(model.predict_messages, temperature=0.)

        with tqdm_joblib(tqdm(desc="Summarization...", total=len(list_messages))) as progress_bar:
            summaries = Parallel(n_jobs=8, prefer="threads")(delayed(summarize_func)(messages) for messages in list_messages)
        print(summaries)

        # Combine summaries into one string
        total_summary = " ".join(summaries)
    # Summarize the final summary
    messages = generate_summarize_summaries_messages_func(total_summary)
    return model.predict_messages(messages=messages, temperature=0.).replace("\n", "<br>") # Hack for proper browser display
