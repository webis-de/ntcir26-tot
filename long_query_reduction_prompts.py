# All prompts are from our 2023 long query reductions:
# https://github.com/webis-de/TREC23/blob/main/tomt-query-reduction/chat-gpt/query-variants-in-progress/query-expansion-prompts.json
from typing import List, Dict


def prompt_01():
    """Please reduce this search query to the most important details omitting unimportant points.

    Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.
    """

def prompt_02():
    """You are an expert searcher. Please reduce this search query to the most important details omitting unimportant points. The resulting query must return very good results on Google.

    Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.
    """

def prompt_03():
    """You are an expert searcher. I wanted to search the web for but I was not able to find relevant documents. Please reduce my query to the most important details so that the results returned by Google are relevant.

    Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.
    """

def prompt_04():
    """You are an expert searcher. I try to find a known item, but my search query does not yield my known item. Please reduce this search query to the most important details omitting unimportant points so that the query returns good results.

Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.

###

Query: {query}"""

def prompt_05():
    """You are an experienced librarian and expert in formulating good search queries. My query does not yield good results because I have included too many unimportant details. Can you please reduce this search query so that it yields relevant results.

    Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.
    """


PROMPT_REGISTRY = {
    "prompt_01": {"system": prompt_01},
    "prompt_02": {"system": prompt_02},
    "prompt_03": {"system": prompt_03},
    "prompt_04": {"system": prompt_04},
    "prompt_05": {"system": prompt_05},
}

def get_prompt_names() -> List[str]:
    return list(PROMPT_REGISTRY.keys())

def get_chat_messages(prompt_name: str, query: str) -> List[Dict[str, str]]:
    if prompt_name not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt {prompt_name} not found in registry. Available prompts: {list(PROMPT_REGISTRY.keys())}")

    system_instruction = PROMPT_REGISTRY[prompt_name]["system"]
    return [
        {"role": "system", "content": system_instruction()},
        {"role": "user", "content": query},
    ]