from pydantic import BaseModel, Field
from typing import List, Optional, Union


class CreateIndexResponse(BaseModel):
    """A model that defines the response structure for index generation"""

    status_code: int
    message: str
    processed_files: List[str] = Field(default=[])


class SimpleResponse(BaseModel):
    """A model defining simple responses from the agent response"""

    status_code: int
    error: Optional[str] = None
    response: str
    elapsed_time: float


class RequestPrompt(BaseModel):
    """A model structuring user requests"""

    user_name: str = Field(default="John Doe")
    query: str


class DetectLanguageOutput(BaseModel):
    """A model structuring the output for language detection"""

    language: str


class TranslateLanguageOutput(BaseModel):
    """A model defining the response for a language translation"""

    final_model_output: Union[str, dict[str, str]]
