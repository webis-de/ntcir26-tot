#!/usr/bin/python3
import json
import logging
import os
import time
from typing import List, Any, Dict

import click
import ir_datasets
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from config import OUT_DIR, DATASET_NAME, GPU_UTIL, MAX_MODEL_LEN, TEMPERATURE, MAX_TOKENS, SEED, \
    CHUNK_SIZE, LOG_DIR
from long_query_reduction_prompts import get_chat_messages, get_prompt_names
from utils.vllm_utils import collect_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LongQueryReduction(BaseModel):
    query_reduction: str = Field(description="The query reduction")


def get_safe_file_slug(model_name: str) -> str:
    return model_name.replace("/", "-").replace(".", "-")


def load_existing_processed_ids(target_file: str) -> set:
    if not os.path.exists(target_file):
        return set()

    processed_ids = set()
    with open(target_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    if "query_id" in record:
                        processed_ids.add(str(record["query_id"]))
                except json.JSONDecodeError:
                    continue
    return processed_ids


def get_batch_messages(request_prompt: str, dataset_name: str, list_to_skip: set):
    batch_messages = []
    batch_metadata = []

    for query in ir_datasets.load(DATASET_NAME).queries_iter():
        qid = str(query.query_id)
        if qid in list_to_skip:
            logger.info(f"Skipping {qid}...")
            continue

        messages = get_chat_messages(request_prompt, query)
        batch_messages.append(messages)
        batch_metadata.append({
            "query_id": qid,
            "prompt": request_prompt,
            "messages": messages
        })

    return batch_messages, batch_metadata


def process_and_stream_chunks(llm: LLM, sampling_params: SamplingParams, model_name: str, prompt_id: str,
                              batch_messages: List[Any], batch_metadata: List[Dict[str, Any]], dataset_file: str,
                              telemetry_file: str) -> None:
    total_messages = len(batch_messages)
    for i in range(0, total_messages, CHUNK_SIZE):
        chunk_messages = batch_messages[i:i + CHUNK_SIZE]
        chunk_metadata = batch_metadata[i:i + CHUNK_SIZE]
        chunk_num = (i // CHUNK_SIZE) + 1
        total_chunks = (total_messages + CHUNK_SIZE - 1) // CHUNK_SIZE

        logger.info(f"Processing chunk {chunk_num} of {total_chunks}...")

        start_time = time.time()
        outputs = llm.chat(messages=chunk_messages, sampling_params=sampling_params, use_tqdm=True)
        duration = time.time() - start_time
        logger.info(f"Chunk {chunk_num} of {total_chunks} processed in {duration:.2f}s")

        with (open(dataset_file, 'a', encoding='utf-8') as f_data,
              open(telemetry_file, 'a', encoding='utf-8') as f_telemetry):
            for output, meta in zip(outputs, chunk_metadata):
                qid = meta["query_id"]
                response_text = output.outputs[0].text

                query_reduction = None
                try:
                    query_reduction = LongQueryReduction.model_validate_json(response_text).query_reduction
                except Exception:
                    logger.error(f"Failed to parse JSON from response: {response_text}")

                clean_record = {
                    "query_id": qid,
                    "prompt_id": prompt_id,
                    "query_reduction": query_reduction,
                    "model_name": model_name,
                }

                f_data.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                f_telemetry.write(json.dumps(collect_metadata(qid, output, response_text, query_reduction),
                                   ensure_ascii=False) + "\n")


def run_predictions_core(model_name, prompt_id):
    logger.info(f"Starting inference for Model: {model_name} | Prompt: {prompt_id}")

    model_slug = get_safe_file_slug(model_name)
    dataset_file = os.path.join(OUT_DIR, f"query-reductions-{model_slug}-{prompt_id}.jsonl")
    telemetry_file = os.path.join(LOG_DIR, f"vllm-telemetry-{model_slug}-{prompt_id}.log.jsonl")

    completed_ids = load_existing_processed_ids(dataset_file)
    batch_messages, batch_metadata = get_batch_messages(prompt_id, DATASET_NAME, completed_ids)

    if not batch_messages:
        logger.info(f"All jobs for Prompt '{prompt_id}' are already completed. Skipping.")
        return

    logger.info(f"Loading '{model_name}' (VRAM config: {GPU_UTIL * 100}%)...")
    llm = LLM(model=model_name, gpu_memory_utilization=GPU_UTIL, max_model_len=MAX_MODEL_LEN)
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        structured_outputs=StructuredOutputsParams(json=LongQueryReduction.model_json_schema())
    )
    process_and_stream_chunks(llm, sampling_params, model_name, prompt_id, batch_messages, batch_metadata, dataset_file, telemetry_file)


@click.command()
@click.option('--model_name', type=str, required=True, help="LLM model identifier from HuggingFace or path.")
def main(model_name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    for prompt_id in get_prompt_names():
        try:
            run_predictions_core(model_name, prompt_id)
        except Exception as e:
            logger.critical(f"Critical execution failure tracking pipeline {prompt_id}: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()
