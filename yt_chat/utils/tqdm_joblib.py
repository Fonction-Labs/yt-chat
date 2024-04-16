# Copy-pasted from:
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution

import contextlib
from joblib.parallel import BatchCompletionCallBack
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = BatchCompletionCallBack
    BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
