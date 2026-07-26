# Webis at NTCIR 2026: Tip-of-the-Tongue

## Dependencies

- Python3.14
```bash
pip3 install "git+https://github.com/NTCIR-ToT/ir_datasets.git"
pip3 install click numpy openai pyterrier tira vllm
```

## Running Predictions

1. Set your Hugging Face API key via environment variable.
```bash
export HF_TOKEN="your-api-key"
```

2. Run the script.

```bash
python3 run_inference.py -m "google/gemma-3-12b-it"
```


Used Models:
- google/gemma-3-12b-it
- meta-llama/Llama-3.1-8B-Instruct
- Qwen/Qwen2.5-14B-Instruct

last year:
- llama-3.1-8b-instant
- llama-3.3-70b-versatile
- openai/gpt-oss-120b

to try:
- Qwen/Qwen3-32B-AWQ
- hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
- shuyuej/Llama-3.3-70B-Instruct-GPTQ
- openai/gpt-oss-20b


## Retrieval

Index from [2025](https://github.com/TREC-ToT/bench/edit/main/trec25/README.md):

|[trec-tot-2025-pyterrier-index.zip](https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-pyterrier-index.zip) | PyTerrier | 11GB | a9a22ed35abb6cea842a7c5734987c82 


```bash
./run_retrieval.py --output runs --index trec-tot-2025-pyterrier-index --query-predictions 'predictions/vllm/*.jsonl'
```

