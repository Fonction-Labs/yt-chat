# Copy-pasted from:
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution

import contextlib
import joblib.parallel
from joblib import Parallel, delayed
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as arguments
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def parallel(method):
    """
    Parallelization python decorator written for parallelizing LLM model objects
    (from models.py) methods like .embed, .predict, and .predict_messages().
    """
    def wrapper(self, *args, **kwargs):
        items = args[0]  # Assuming args[0] is the list of items to parallelize
        n_jobs = kwargs.get("n_jobs", 8)  # Default value of n_jobs is 8

        # Calculate total number of items to process
        total_items = len(items)

        # Create tqdm_joblib context manager for progress bar
        with tqdm_joblib(tqdm(desc="Parallelizing batches...", total=total_items)):
            # Call the original method with provided arguments
            result = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(method)(self, item, *args[1:], **kwargs) for item in items
            )
            return result

    return wrapper
