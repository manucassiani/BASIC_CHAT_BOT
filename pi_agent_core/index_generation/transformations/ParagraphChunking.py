import uuid

from typing import List, Any
from llama_index.core.schema import BaseNode, TransformComponent, TextNode


class ParagraphChunking(TransformComponent):
    """A class that splits text into smaller chunks using double line breaks as delimiters.
    The objective is to divide the text into node paragraphs.
    """

    def __call__(self, documents: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Processes a list of documents and splits their text content into smaller chunks.

        Args:
            documents (List[BaseNode]): A list of documents to be chunked.

        Returns:
            List[BaseNode]: A list of new BaseNode objects, each containing a chunk of text.
        """
        new_nodes = []
        for doc in documents:
            text = doc.text
            new_chunks = [
                chunk.strip() for chunk in text.split("\n\n") if chunk.strip()
            ]

            for chunk in new_chunks:
                node = TextNode(
                    id_=str(uuid.uuid4()),
                    text=chunk,
                )
                new_nodes.append(node)

        return new_nodes
