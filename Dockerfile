# The builder image, used to build the virtual environment
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY . .

RUN poetry install --no-dev && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
# slim is required, alpine does not work since gcc is required by chainlit
FROM python:3.10-slim

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app

COPY . .

CMD ["chainlit", "run", "yt_chat/app.py"]
