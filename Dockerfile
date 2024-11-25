# Use the official Python image
FROM python:3.11.7
ARG POETRY_VERSION=1.8.2

# Set environment variables
ENV POETRY_VERSION=$POETRY_VERSION \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false
    # COHERE_API_KEY=".."

# Set the working directory in the container
WORKDIR /app

# Copy the poetry files
COPY pyproject.toml poetry.lock /app/
COPY pi_agent_core /app/pi_agent_core
COPY config /app/config
COPY knowledge_base /app/knowledge_base

# Install poetry and project dependencies
RUN pip install poetry==$POETRY_VERSION
RUN poetry install

# Command to start your FastAPI app
CMD ["poetry", "run", "uvicorn", "pi_agent_core.app:app", "--host", "0.0.0.0", "--port", "8000"]
