import os
import logging

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import Settings

from dotenv import load_dotenv

from pi_agent_core.helpers.utils import load_config_file
from config.config import PI_AGENT_CONFIG

load_dotenv(override=True)


def set_service_context() -> None:
    """Configures the global Settings object with the appropriate language model (LLM)
    and embedding model based on the specified provider in the environment variables.

    This function dynamically selects and initializes the models depending on the
    value of the `LLM_PROVIDER` environment variable. It supports OpenAI, Azure OpenAI,
    and Cohere as providers, falling back to Cohere if no valid provider is specified.

    Process:
        1. Loads service context configuration from a configuration file.
        2. Reads the `LLM_PROVIDER` environment variable to determine the model provider.
        3. Initializes the appropriate LLM and embedding model for the provider.
        4. Updates the global `Settings` object with the configured models.

    Environment Variables:
        - LLM_PROVIDER: Specifies the provider to use (e.g., "COHERE", "AZURE", or "OPENAI").
        - COHERE_API_KEY: API key for Cohere services.
        - OPENAI_API_KEY: API key for OpenAI services.
        - AZURE_API_KEY: API key for Azure services.
        - Additional environment variables required for Azure configurations (e.g., endpoint, version).
    """
    # Load agent parameters for the service context configuration
    agent_params = load_config_file(PI_AGENT_CONFIG)["service_context"]

    llm = None
    embed_model = None

    # Fetch the LLM provider from environment variables
    llm_provider = os.getenv("LLM_PROVIDER")
    if llm_provider is None:
        llm_provider = "COHERE"

    logging.info(f"LLM_PROVIDER: {str(llm_provider)}")

    # Configure models for OpenAI
    if llm_provider.upper() == "OPENAI":
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=agent_params["llm"]["open_ai"]["model"],
            temperature=agent_params["llm"]["open_ai"]["temperature"],
        )
        embed_model = OpenAIEmbedding(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=agent_params["embedding"]["open_ai"]["model"],
        )

    # Configure models for Azure OpenAI
    elif llm_provider.upper() == "AZURE":
        llm = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            engine=os.getenv("AZURE_LLM_MODEL_DEPLOYMENT"),
            model=agent_params["llm"]["azure_open_ai"]["model"],
            temperature=agent_params["llm"]["azure_open_ai"]["temperature"],
        )
        embed_model = AzureOpenAIEmbedding(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_MODEL_DEPLOYMENT"),
            model=agent_params["embedding"]["azure_open_ai"]["model"],
        )

    # Defaults to Cohere models if no matching provider is found
    else:
        llm = Cohere(
            api_key=os.getenv("COHERE_API_KEY"),
            model=agent_params["llm"]["cohere"]["model"],
            temperature=agent_params["llm"]["cohere"]["temperature"],
        )
        embed_model = CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name=agent_params["embedding"]["cohere"]["model"],
        )

    # Update the Settings object with the configured LLM and embedding models
    Settings.llm = llm
    Settings.embed_model = embed_model

    return None
