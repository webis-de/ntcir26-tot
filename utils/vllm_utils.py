import logging
from typing import Optional

from vllm import LLM


logger = logging.getLogger(__name__)


def clean_gpu_context_memory(llm: Optional[LLM], model_name: str) -> None:
    if llm is not None:
        logger.info(f"Tearing down LLM engine context for {model_name}...")
        try:
            if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
                llm.llm_engine.shutdown()
        except Exception as e:
            logger.warning(f"Engine pool cleanup warning: {e}")
        # llm.engine.context_manager.free_all_cached_tensors()
        del llm
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def collect_metadata(query_id, output, generated_text, query_reduction):
    metrics = getattr(output, "metrics", None)

    metadata = {
        "query_id": query_id,
        "request_id": output.request_id,
        "prompt_text": output.prompt,
        "generated_text": generated_text,
        "parsed_query_reduction": query_reduction,
        "prompt_tokens": len(output.prompt_token_ids),
        "completion_tokens": len(output.outputs[0].token_ids),
        "arrival_time": getattr(metrics, "arrival_time", None),
        "first_token_time": getattr(metrics, "first_token_time", None),
        "finished_time": getattr(metrics, "finished_time", None),
    }

    if metadata["arrival_time"] and metadata["finished_time"]:
        metadata["duration_seconds"] = metadata["finished_time"] - metadata["arrival_time"]

    return metadata