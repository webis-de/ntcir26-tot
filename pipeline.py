import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Set, Tuple, Dict, List, Any, Optional

import ir_datasets
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from config import OUT_DIR, LOG_DIR, DATASET_NAME, CHUNK_SIZE
from long_query_reduction_prompts import get_chat_messages
from utils.vllm_utils import collect_metadata


logger = logging.getLogger(__name__)


class LongQueryReduction(BaseModel):
    query_reduction: str = Field(description="The query reduction")


@dataclass(frozen=True)
class PipelinePaths:
    """Encapsulates system file paths for the inference job."""
    dataset_file: str
    telemetry_file: str


class QueryReductionPipeline:
    def __init__(self, model_name: str, prompt_id: str):
        self.model_name = model_name
        self.prompt_id = prompt_id

        model_slug = self._get_safe_file_slug(model_name)
        self.paths = PipelinePaths(
            dataset_file=os.path.join(OUT_DIR, f"query-reductions-{model_slug}-{prompt_id}.jsonl"),
            telemetry_file=os.path.join(LOG_DIR, f"vllm-telemetry-{model_slug}-{prompt_id}.log.jsonl")
        )

    @staticmethod
    def _get_safe_file_slug(model_name: str) -> str:
        return model_name.replace("/", "-").replace(".", "-")

    def load_processed_ids(self) -> Set[str]:
        if not os.path.exists(self.paths.dataset_file):
            return set()

        processed_ids = set()
        with open(self.paths.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "query_id" in record:
                            processed_ids.add(str(record["query_id"]))
                    except json.JSONDecodeError:
                        continue
        return processed_ids

    def prepare_batches(self, list_to_skip: Set[str]) -> Tuple[List[List[Dict[str, str]]], List[Dict[str, Any]]]:
        batch_messages = []
        batch_metadata = []
        skip_count = 0

        for query in ir_datasets.load(DATASET_NAME).queries_iter():
            qid = str(query.query_id)
            if qid in list_to_skip:
                skip_count += 1
                continue

            messages = get_chat_messages(self.prompt_id, query)
            batch_messages.append(messages)
            batch_metadata.append({
                "query_id": qid,
                "prompt": self.prompt_id,
                "messages": messages
            })

        if skip_count > 0:
            logger.info(f"Skipped {skip_count} already processed queries.")

        return batch_messages, batch_metadata

    @staticmethod
    def _parse_reduction(response_text: str) -> Optional[str]:
        try:
            data = json.loads(response_text)
            try:
                return LongQueryReduction.model_validate_json(data).query_reduction
            except Exception:
                return data.get("query_reduction")
        except Exception:
            logger.error(f"Failed to parse JSON schema from response snippet: {response_text[:100]}...")
            return None

    def process_and_stream(self, llm: LLM, sampling_params: SamplingParams,
                           batch_messages: List[Any], batch_metadata: List[Dict[str, Any]]) -> None:
        total_messages = len(batch_messages)

        with open(self.paths.dataset_file, 'a', encoding='utf-8') as f_data, \
                open(self.paths.telemetry_file, 'a', encoding='utf-8') as f_telemetry:

            for i in range(0, total_messages, CHUNK_SIZE):
                chunk_messages = batch_messages[i: i + CHUNK_SIZE]
                chunk_metadata = batch_metadata[i: i + CHUNK_SIZE]

                chunk_num = (i // CHUNK_SIZE) + 1
                total_chunks = (total_messages + CHUNK_SIZE - 1) // CHUNK_SIZE
                logger.info(f"Processing chunk {chunk_num} of {total_chunks}...")

                start_time = time.time()
                outputs = llm.chat(messages=chunk_messages, sampling_params=sampling_params, use_tqdm=True)
                logger.info(f"Chunk {chunk_num}/{total_chunks} completed in {time.time() - start_time:.2f}s")

                for output, meta in zip(outputs, chunk_metadata):
                    qid = meta["query_id"]
                    response_text = output.outputs[0].text
                    query_reduction = self._parse_reduction(response_text)

                    clean_record = {
                        "query_id": qid,
                        "prompt_id": self.prompt_id,
                        "query_reduction": query_reduction,
                        "model_name": self.model_name,
                    }

                    f_data.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                    f_telemetry.write(json.dumps(collect_metadata(qid, output, response_text, query_reduction),
                                                 ensure_ascii=False) + "\n")