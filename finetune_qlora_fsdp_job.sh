#!/bin/bash
#SBATCH --job-name=spider-improver-fsdp
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
# Every GPU process quantizes the model on the CPU and holds its 4-bit copy (~37GB for 70B)
# in CPU RAM until FSDP moves the shards to the GPUs: ~150GB for 4 GPUs, plus headroom while
# loading. The adapter merge at the end needs the bf16 model (~140GB). 16 x 16G = 256G.
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:24G
#SBATCH --output=spider-improver-fsdp_%j.out
#SBATCH --error=spider-improver-fsdp_%j.err
#SBATCH --account=es_rdm

set -euo pipefail

# Name of the conda environment to run the fine-tuning job in.
# Set up once on the LOGIN node (compute nodes have no/restricted internet):
#   cd "$HOME"
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
#   bash miniconda.sh -b -p "$HOME/miniconda3"
#   rm miniconda.sh
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"
#   conda create -y -n spider-improver python=3.11
ENV_NAME="spider-improver"
CONDA_ROOT="$HOME/miniconda3"

# Print job information
echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $PWD"
echo "=========================================="

# Print system information
echo "System information:"
echo "Hostname: $(hostname)"
echo "Operating System: $(uname -a)"
echo "CPU info: $(lscpu | grep 'Model name' | head -1)"
echo "Memory info: $(free -h | grep 'Mem:')"
echo ""

# Print GPU information
echo "GPU information:"
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo ""

# Move to the directory this job was submitted from, so relative paths
# (spider_data/, requirements.txt, .env, ...) resolve correctly. Note:
# $0 is NOT reliable here — sbatch copies the script into a scheduler
# spool dir and executes it from there, so SLURM_SUBMIT_DIR is the one
# that actually points back at the submission directory.
cd "$SLURM_SUBMIT_DIR"
echo "Running from: $PWD"

# Compute nodes have no direct internet access by default.
module load eth_proxy

# Activate the conda environment (must already exist — see setup commands
# above; compute nodes typically can't reach the internet to install conda
# itself, so that step has to happen on the login node beforehand).
if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "ERROR: no conda install found at $CONDA_ROOT." >&2
    echo "Run the one-time setup commands (see comment above ENV_NAME) on the login node first." >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: conda environment '$ENV_NAME' not found under $CONDA_ROOT." >&2
    echo "Create it on the login node first: conda create -y -n $ENV_NAME python=3.11" >&2
    exit 1
fi

conda activate "$ENV_NAME"
echo "Active environment: $ENV_NAME ($(which python))"

# Install/refresh the extra dependencies the fine-tuning script needs
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Run the fine-tuning job
echo ""
echo "=========================================="
echo "Starting fine-tuning job..."
echo "=========================================="
# The 70B checkpoint is ~140GB, more than a typical home quota: cache it on scratch.
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}"
echo "Hugging Face cache: $HF_HOME"

# Download the model before the GPU processes start: rank 0 would otherwise download
# ~140GB while the other ranks wait on it and risk a distributed timeout.
MODEL_ID=$(python -c "import finetune_qlora_fsdp as m; print(m.MODEL_ID)")
echo "Pre-downloading $MODEL_ID..."
python -c "
from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])
"

NUM_GPUS=$(nvidia-smi -L | wc -l)
# The GPUs run close to full; this lets PyTorch reuse fragmented free memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# torchrun defaults each process to 1 CPU thread; quantizing on the CPU needs more.
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))
echo "Launching on $NUM_GPUS GPUs, $OMP_NUM_THREADS CPU threads each"
accelerate launch --config_file fsdp_qlora.yaml --num_processes "$NUM_GPUS" finetune_qlora_fsdp.py

echo ""
echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
