#!/bin/bash
#SBATCH --job-name=vllm-inference-ntcir26-tot
#SBATCH --partition=gammaweb
#SBATCH --nodes=1
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=64G
#SBATCH --time=0-4:00:00
#SBATCH --output=logs-vllm/master_%j.log

mkdir -p logs-vllm

source ~/miniconda3/bin/activate
conda activate ollama_env

pip3 uninstall -y ir_datasets
pip3 install -r requirements.txt

export HF_TOKEN=$(cat ~/.hf_token)

echo "Starting sequential evaluation suite..."

python3 run_inference.py \
  -m "google/gemma-3-12b-it" \
  -m "meta-llama/Llama-3.1-8B-Instruct" \
  -m "Qwen/Qwen2.5-14B-Instruct" \
  > logs-vllm/output_experiment.log 2>&1

echo "All Python evaluation experiments completed successfully."