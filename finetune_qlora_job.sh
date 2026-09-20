#!/bin/bash
#SBATCH --job-name=spider-improver
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --output=spider-improver_%j.out
#SBATCH --error=spider-improver_%j.err
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
python finetune_qlora.py

echo ""
echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
