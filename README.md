<div align="center">
  <picture align="center" with="200">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/mcordier/yt-chat/blob/5bbae54e1c9f46f11af9090a83089786d9832e6f/public/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/mcordier/yt-chat/blob/5bbae54e1c9f46f11af9090a83089786d9832e6f/public/logo_light.png">
  <img alt="yt-chat logo" src="https://github.com/mcordier/yt-chat/blob/5bbae54e1c9f46f11af9090a83089786d9832e6f/public/logo_light.png" width="400"/>
  </picture>
</div>


<h1 align="center">yt-chat</h1>
<h3 align="center">yt-chat is a tool designed to help you summarize any Youtube video.</h3>
<h4 align="center">Once a video is summarized, you can also ask more precise questions about the video in question.<br>Enjoy!</h4>

---

<div align="center">
<a href="https://fonctionlabs.com/yt-chat">
<img alt="Try live!" src="https://img.shields.io/static/v1?label=&message=Try live!"/>
</a>
</div>

---

<div align="center">
<a href="https://github.com/Fonction-Labs/jade/actions/workflows/all.yml?query=branch%3Amain">
<!-- <img alt="All workflows" src="https://github.com/Fonction-Labs/jade/actions/workflows/all.yml/badge.svg"/> -->
<img alt="Python version" src="https://img.shields.io/badge/python-3.9-blue"/>
</a>
</div>

Installation
------------
```
poetry install
poetry run python yt-chat/app.py
```

### If you wish to use `gpt-3.5` (with your OpenAI API key):
```
export OPENAI_API_KEY='<MY-OPENAI-API-KEY>'
```

Don't forget to set `MODEL_CHOICE` in `app.py` to `"GPT"`.

### If you wish to use `mistral-7b` (local model):

1. Install Python for ARM64
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

2. Download the GGUF-format `mistral-7b-instruct-v0.2.Q5_K_S.gguf` weights from [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) and move them to the root of this repository.

3. Use this ARM64 version of Python for the poetry environment associated to this project
```
poetry env use /PATH/TO/MINIFORGE3/miniforge3/bin/python3
poetry install
```

4. Install `llama-cpp-python` (with Metal support if you are using an Apple Silicon Mac)
```
CMAKE_ARGS="-DLLAMA_METAL=on" poetry run python -m pip install -U llama-cpp-python --no-cache-dir
```

Don't forget to set `MODEL_CHOICE` in `app.py` to `"MISTRAL"`.
