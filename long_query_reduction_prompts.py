from outlines import prompt

# All prompts are from our 2023 long query reductions:
# https://github.com/webis-de/TREC23/blob/main/tomt-query-reduction/chat-gpt/query-variants-in-progress/query-expansion-prompts.json

@prompt
def prompt_01(query: str):
    """Please reduce this search query to the most important details omitting unimportant points.

Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.

###

Query: {{query}}"""

@prompt
def prompt_02(query: str):
    """You are an expert searcher. Please reduce this search query to the most important details omitting unimportant points. The resulting query must return very good results on Google.

Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.

###

Query: {{query}}"""

@prompt
def prompt_03(query: str):
    """You are an expert searcher. I wanted to search the web for but I was not able to find relevant documents. Please reduce my query to the most important details so that the results returned by Google are relevant.

Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.

###

Query: {{query}}"""

@prompt
def prompt_04(query: str):
    """You are an expert searcher. I try to find a known item, but my search query does not yield my known item. Please reduce this search query to the most important details omitting unimportant points so that the query returns good results.

Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.

###

Query: {{query}}"""

@prompt
def prompt_05(query: str):
    """You are an experienced librarian and expert in formulating good search queries. My query does not yield good results because I have included too many unimportant details. Can you please reduce this search query so that it yields relevant results.

Instructions: Only provide the reduced query on the output line. Do not provide any further details or explanation.

###

Query: {{query}}"""


def prompt(query, prompt):
    import sys
    p = getattr(sys.modules[__name__], prompt)
    return p(query.default_text())
