from tqdm import tqdm
from functools import partial

from ..utils.chunk_text import get_text_chunks

def summarize_document(
    transcript_text: str,
    llm,
    base_prompt_transcript_summary: str
    ) -> str:
    prompt = base_prompt_transcript_summary.replace("{docs}", transcript_text)
    answer = llm.predict(prompt, temperature=0)
    return answer

def summarize_transcript(
    transcript_text: str,
    llm,
    base_prompt_transcript_summary: str,
    model_tokens_ctx: int = 4096,
    safety_token_prct: float = 0.7
) -> str:
    """
    Summarize the video..

    Parameters:
        transcript_text (str): The text of the video transcript.

    Returns:
        str: The generated summary.
    """
    total_summary = transcript_text
    if len(total_summary) < model_tokens_ctx * 4 * safety_token_prct:
        return total_summary
    else:
        transcripts = get_text_chunks(total_summary)
        print(f"Split all transcript summaries into {len(transcripts)}")
        sumup_func = partial(summarize_document, llm=llm, base_prompt_transcript_summary=base_prompt_transcript_summary)
        summaries = map(sumup_func, tqdm(transcripts))
        total_summary = " ".join(summaries)
    return total_summary.replace("\n", "<br>")
