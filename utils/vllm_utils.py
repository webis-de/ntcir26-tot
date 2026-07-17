

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