# yt-chat
Summarize and chat with a bot about a youtube video

### Install
```
poetry install
poetry run python yt-chat/app.py
```

### If you wish to use `mistral-7b` locally:

Install Python for ARM64
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

Download the GGUF-format `mistral-7b-instruct-v0.2.Q5_K_S.gguf` weights from [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) and move them to the root of this repository.

Use this ARM64 version of Python for the poetry environment associated to this project
```
poetry env use /PATH/TO/MINIFORGE3/miniforge3/bin/python3
```

Install `llama-cpp-python` (with Metal support if you are using an Apple Silicon Mac)
```
CMAKE_ARGS="-DLLAMA_METAL=on" poetry run python -m pip install -U llama-cpp-python --no-cache-dir
```
