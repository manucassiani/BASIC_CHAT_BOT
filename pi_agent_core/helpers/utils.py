import os
import shutil
import yaml

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.settings import Settings
from config.config import PI_AGENT_CONFIG
from pi_agent_core.models import DetectLanguageOutput, TranslateLanguageOutput


def load_config_file(path: str) -> dict:
    """
    loads config.yml file containing ServiceContext content: llm_predictor, embeddings, prompt_helper and system prompt.
    args:
        - path pointing to the config file
    returns:
        - dictionary with the config file content
    """
    with open(path, "rb") as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise Exception(exc)
    return params


def delete_tmp_files(directory: str) -> None:
    """Delete all files and folders that are inside the main folders of the directory folder.

    Args:
        directory (str): path of directory.
    """
    # Get the list of folders inside the directory
    folders = [
        folder
        for folder in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder))
    ]

    # Iterate through each folder and delete the files inside them
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        files = [file for file in os.listdir(folder_path)]

        # Delete the files inside the folder
        for file in files:
            file_path = os.path.join(folder_path, file)
            # delete directory
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            # delete file
            else:
                os.remove(file_path)


def detect_language(user_input: str) -> DetectLanguageOutput:
    """Detects the language of the provided user input.

    Args:
        user_input (str): The text input whose language needs to be detected.

    Returns:
        DetectLanguageOutput: An object containing the detected language.
    """
    agent_params = load_config_file(PI_AGENT_CONFIG)
    SP = agent_params["llm_simple_program"]["detect_language_prompt"]
    SP = SP.format(user_input=user_input)
    program = LLMTextCompletionProgram.from_defaults(
        llm=Settings.llm,
        output_cls=DetectLanguageOutput,
        prompt_template_str=SP,
        verbose=True,
    )
    output = program()
    return output


def check_and_translate_to_specific_language(
    model_response: str, language: str
) -> TranslateLanguageOutput:
    """Checks the content of a model's response and translates it into a specific language.

    Args:
        model_response (str): The text content to be translated.
        language (str): The target language for the translation.

    Returns:
        TranslateLanguageOutput: An object containing the translated text.
    """
    agent_params = load_config_file(PI_AGENT_CONFIG)
    SP = agent_params["llm_simple_program"]["check_and_translate_to_specific_language"]
    SP = SP.format(model_response=model_response, language=language)
    program = LLMTextCompletionProgram.from_defaults(
        llm=Settings.llm,
        output_cls=TranslateLanguageOutput,
        prompt_template_str=SP,
        verbose=True,
    )
    output = program()
    return output
