#!/usr/bin/python3
import json
import os

import click
import ir_datasets
from openai import OpenAI

from long_query_reduction_prompts import prompt


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)


def process_query(query, model, request_prompt):
    print(f'Process Query: {query.query_id}')

    request = prompt(query, request_prompt)
    response = {'request': request, 'request_prompt': request_prompt}
    response[model] = client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request}
            ],
    ).choices[0].message.content
    return response


@click.command('main')
@click.option('--num', type=int, default=10)
@click.option('--model_name', type=str)
@click.option('--request_prompt', type=str)
def main(num, model_name, request_prompt):
    performed = 0
    os.makedirs('predictions/reduced-reasoning', exist_ok=True)
    target_file = f'predictions/reduced-reasoning/query-expansions-from-{model_name.replace('/', '-').replace('.', '-')}-raw-{request_prompt}.json'

    if not os.path.exists(target_file):
        with open(target_file, 'w') as f:
            json.dump({}, f)

    ret = json.load(open(target_file))

    for query in ir_datasets.load('trec-tot/2025/test').queries_iter():
        qid = str(query.query_id)
        if qid in ret.keys():
            print(f'Skip: {qid}')
            continue

        try:
            ret[qid] = process_query(query, model_name, request_prompt)
            performed += 1
        except Exception as e:
            print(e)
            break

        if performed >= num:
            break

    json.dump(ret, open(target_file, 'w'))


if __name__ == '__main__':
    main()