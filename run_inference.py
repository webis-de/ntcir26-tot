#!/usr/bin/python3
import logging
import os
from typing import Optional

import click
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from config import OUT_DIR, GPU_UTIL, MAX_MODEL_LEN, TEMPERATURE, MAX_TOKENS, SEED, LOG_DIR
from long_query_reduction_prompts import get_prompt_names
from pipeline import LongQueryReduction, QueryReductionPipeline
from utils.vllm_utils import clean_gpu_context_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_predictions_core(model_name: str, prompt_id: str, llm: Optional[LLM] = None) -> Optional[LLM]:
    logger.info(f"Starting inference for Prompt: {prompt_id}")

    pipeline = QueryReductionPipeline(model_name, prompt_id)
    completed_ids = pipeline.load_processed_ids()
    batch_messages, batch_metadata = pipeline.prepare_batches(completed_ids)

    if not batch_messages or len(batch_messages) == 0:
        logger.info(f"All jobs for Prompt '{prompt_id}' are already completed. Skipping.")
        return llm

    if llm is None:
        logger.info(f"Loading '{model_name}' (VRAM config: {GPU_UTIL * 100}%)...")
        llm = LLM(model=model_name, gpu_memory_utilization=GPU_UTIL, max_model_len=MAX_MODEL_LEN)

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        structured_outputs=StructuredOutputsParams(json=LongQueryReduction.model_json_schema())
    )
    pipeline.process_and_stream(llm, sampling_params, batch_messages, batch_metadata)
    return llm


@click.command()
@click.option('--model', '-m', required=True, help="LLM model identifier from HuggingFace or path.")
def main(model_name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    logger.info(f"Starting inference for Model: {model_name}")
    llm = None

    for prompt_id in get_prompt_names():
        # try:
        llm = run_predictions_core(model_name, prompt_id, llm)
        # except Exception as e:
            # logger.critical(f"Critical execution failure tracking pipeline {prompt_id}: {str(e)}", exc_info=True)

        # clean_gpu_context_memory(llm, model_name)


if __name__ == '__main__':
    main()
