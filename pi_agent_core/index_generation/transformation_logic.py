from pi_agent_core.index_generation.transformations.ParagraphChunking import (
    ParagraphChunking,
)

from llama_index.core.node_parser import TokenTextSplitter


def get_transformation_context(
    model_params: dict, paragraph_chunking_activate: bool = True
) -> list:
    """Creates a transformation context to preprocess documents for node generation.
    Depending on the configuration, it uses either ParagraphChunking or TokenTextSplitter
    to divide the text into manageable chunks.

    Args:
        model_params (dict): Parameters for configuring the transformation pipeline,
                             including chunk size and overlap for TokenTextSplitter.
        paragraph_chunking_activate (bool): If True, activates ParagraphChunking to split
                                            text by double line breaks. Otherwise, uses
                                            TokenTextSplitter based on the model parameters.

    Returns:
        list: A list of transformation components to be applied to the documents.
    """
    if paragraph_chunking_activate:
        paragraph_chunking = ParagraphChunking()

        transformations = [paragraph_chunking]

    else:
        text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=model_params["service_context"]["chunk_size"],
            chunk_overlap=model_params["service_context"]["chunk_overlap"],
        )
        transformations = [text_splitter]

    return transformations
