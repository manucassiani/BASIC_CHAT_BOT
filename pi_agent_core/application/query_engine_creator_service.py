from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import Settings, PromptTemplate

from pi_agent_core.helpers.utils import load_config_file
from config.config import PI_AGENT_CONFIG, INDEX_PATH, VECTOR_STORE
from pi_agent_core.infraestructure.index_managment import IndexManagment


class CreateQueryEngineUseCase:
    """Singleton class to manage the creation and configuration of a query engine.

    This class ensures only one instance of the query engine use case exists,
    providing consistent behavior and efficient resource management.
    """

    _instance = None

    @staticmethod
    def get_instance():
        """
        Static method to provide a singleton instance of CreateQueryEngineUseCase.
        """
        if CreateQueryEngineUseCase._instance is None:
            CreateQueryEngineUseCase._instance = CreateQueryEngineUseCase()
        return CreateQueryEngineUseCase._instance

    def __init__(
        self,
    ):
        """Initializes the CreateQueryEngineUseCase class."""
        if CreateQueryEngineUseCase._instance is not None:
            raise Exception("This class is a singleton! Use 'get_instance()' method.")

        self.agent_params = load_config_file(PI_AGENT_CONFIG)
        index_managment = IndexManagment()
        self.index = index_managment.load_index(
            index_path=INDEX_PATH, vector_store=VECTOR_STORE
        )

    def execute(self) -> BaseQueryEngine:
        """Configures and returns a query engine instance.

        The query engine is built using the loaded index and additional configuration
        parameters like similarity threshold, QA templates, and temperature.

        Returns:
            BaseQueryEngine: A configured query engine ready for processing queries.
        """

        return self.index.as_query_engine(
            llm=Settings.llm,
            similarity_top_k=self.agent_params["query_engine"]["similarity_top_k"],
            text_qa_template=PromptTemplate(
                self.agent_params["query_engine"]["qa_template"]
            ),
            temperature=self.agent_params["query_engine"]["temperature"],
        )
