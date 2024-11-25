import os
import time
import logging

from fastapi import APIRouter, Depends

from pi_agent_core.index_generation.index_generation_process import (
    create_index_from_knowleadge_base,
)
from pi_agent_core.models import CreateIndexResponse, SimpleResponse, RequestPrompt
from pi_agent_core.application.query_engine_creator_service import (
    CreateQueryEngineUseCase,
)
from pi_agent_core.application.chat_service import ChatService
from pi_agent_core.helpers.utils import (
    detect_language,
    check_and_translate_to_specific_language,
)
from config.config import PATH_KNOWLEDGE_BASE

router = APIRouter(prefix="/agent")


def get_create_query_engine_use_case() -> CreateQueryEngineUseCase:
    """Retrieves an instance of the CreateQueryEngineUseCase.

    This function ensures that the CreateQueryEngineUseCase singleton is used
    for creating query engine instances. It follows a dependency injection
    pattern to provide this instance wherever required.

    Returns:
        CreateQueryEngineUseCase: The singleton instance of the CreateQueryEngineUseCase.
    """
    return CreateQueryEngineUseCase.get_instance()


@router.post("/predict", tags=["pi"])
def predict(
    request: RequestPrompt,
    engine: CreateQueryEngineUseCase = Depends(get_create_query_engine_use_case),
) -> SimpleResponse:
    """Handles the predict endpoint to process user queries and generate model responses.

    This function:
    1. Initializes a query engine using the provided use case.
    2. Detects the language of the user's query.
    3. Generates a response using the chat service.
    4. Ensures the response is translated into the detected language, if necessary.
    5. Send the response.

    Args:
        request (RequestPrompt): The incoming request containing the user's query.
        engine (CreateQueryEngineUseCase): Dependency-injected query engine use case. Defaults to get_create_query_engine_use_case().

    Returns:
        SimpleResponse: A structured response containing the status code, response message,
                        elapsed time, and any errors that occurred.
    """
    try:
        start_time = time.time()

        # Generete query_engine
        query_engine = engine.execute()

        # Initialize the chat service with the query engine
        chat_service = ChatService(
            engine=query_engine,
        )

        # Detect the language of the input query
        language = detect_language(request.query)

        # Get the agent's response
        agent_response = chat_service.chat(request.query)

        # Translate the response to the user's language if necessary
        final_agent_response = check_and_translate_to_specific_language(
            model_response=agent_response, language=language
        ).final_model_output

        # Calculate elapsed time for performance tracking
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Create the response object
        predict_response = SimpleResponse(
            status_code=200,
            error=None,
            response=final_agent_response,
            elapsed_time=elapsed_time,
        )

    except Exception as e:
        logging.error(f"An unexpected error ocurred: {str(e)}")

        # Create default error Response
        predict_response = SimpleResponse(
            status_code=500,
            error="An unexpected error ocurred.",
            response="Lo siento, estamos experimentando dificultades técnicas en este momento. Por favor, vuelve a intentarlo en unos minutos ⏳",
            elapsed_time=-1.0,
        )

    logging.info("Response sent: %s", predict_response)

    return predict_response


@router.post("/create_index", tags=["pi"])
async def create_index() -> CreateIndexResponse:
    """Handles the ingestion endpoint to trigger the creation of an index.

    This function allows asynchronous ingestion of data into the system,
    potentially for building or updating indices for search or query operations.

    Returns:
        CreateIndexResponse: The response object containing details about the
                             ingestion operation.
    """
    try:
        # Get the list of files to be processed
        files_to_process = os.listdir(PATH_KNOWLEDGE_BASE)

        # Index generation
        create_index_from_knowleadge_base()

        # Create a success response
        response = CreateIndexResponse(
            status_code=200,
            message="Index generated successfully",
            processed_files=files_to_process,
        )

    # Create error response
    except Exception as e:
        logging.error(f"An error occurred during ingestion: {str(e)}")
        response = CreateIndexResponse(
            status_code=500,
            message=f"An error occurred during ingestion: {str(e)}",
            processed_files=[],
        )

    return response
