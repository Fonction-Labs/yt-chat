from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from ..utils.chunk_text import get_text_chunks

def summarize_document(
    transcript_text: str,
    llm,
    prompt_template: str
    ) -> str:
    prompt = prompt_template.replace("{docs}", transcript_text)
    answer = llm.predict(prompt, temperature=0)
    return answer

def summarize_transcript(
    transcript_text: str,
    llm,
    agent_settings: dict
) -> str:
    """
    Summarize the video..

    Parameters:
        transcript_text (str): The text of the video transcript.

    Returns:
        str: The generated summary.
    """
    i = 0
    total_summary = transcript_text
    while (i == 0 or len(total_summary) > agent_settings.token_context_size * 4 * agent_settings.safety_token_prct):
        transcripts = get_text_chunks(total_summary, int(agent_settings.token_context_size * agent_settings.safety_token_prct), int(agent_settings.token_context_size * 0.1))
        print(f"Split all transcript summaries into {len(transcripts)}")
        sumup_func = partial(summarize_document, llm=llm, prompt_template=agent_settings.summarize_transcript_template)
        summaries = Parallel(n_jobs=8, prefer="threads")(delayed(sumup_func)(i) for i in tqdm(transcripts))
        total_summary = " ".join(summaries)
        i += 1
    return summarize_document(total_summary.replace("\n", "<br>"), llm=llm, prompt_template=agent_settings.summarize_summaries_template)
