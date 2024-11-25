import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from contextlib import asynccontextmanager

from pi_agent_core.routers import agent
from pi_agent_core.infraestructure.ai_service import set_service_context


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logging.basicConfig(level=logging.INFO)
    # Set OpenAI SDK specific loggin level
    logging.getLogger("openai").setLevel(logging.DEBUG)
    # Set llama-index setting upper-level configuration
    set_service_context()

    yield


app = FastAPI(lifespan=lifespan)

# Include API routes
app.include_router(agent.router)
