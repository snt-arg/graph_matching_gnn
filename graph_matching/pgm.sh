#!/bin/bash -l
#SBATCH -c 1
#SBATCH --time=0-48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matteogiorgi196@gmail.com

# Stampa informazioni di debug
echo "=== SLURM JOB STARTED ==="
echo "Date: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job directory: $SLURM_SUBMIT_DIR"
echo "Current directory: $(pwd)"
echo "Loaded modules:"
module list 2>&1

# Spostati nella directory da cui hai inviato il job (fondamentale per path relativi)
cd $SLURM_SUBMIT_DIR || exit 1

# Attiva virtualenv
echo "Activating virtualenv..."
if [ -f ../.venv/bin/activate ]; then
  source ../.venv/bin/activate
else
  echo "ERROR: Virtualenv not found at ../../venv_aion/bin/activate"
  exit 1
fi

# Diagnostica Python
echo "Python path: $(which python3)"
python3 --version

# Controllo GPU
echo "Checking GPU availability..."
nvidia-smi || echo "nvidia-smi not available"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Esecuzione script Python
echo "Launching Python script..."
# python3 -u dataset_gen.py

# python3 -u graph_matching_train.py 
# python3 -u pgm_ws_equal.py 
# python3 -u pgm_ws_noise.py
# python3 -u pgm_room_equal.py
# python3 -u pgm_room_noise.py

# python3 -u optimization_gm.py 
# python3 -u optimization_ws.py 
# python3 -u optimization_room.py 
# python3 -u optimization_ws_room_inc_BCE_noMLP.py 
# python3 -u optimization_ws_room_inc_BCE.py 
# python3 -u optimization_ws_room_inc_WBCE.py 
# python3 -u optimization_ws_room_inc_WBCE_new.py

# python3 -u embedding_debug.py 
# python3 -u pgm_training_ws_room.py
# python3 -u pgm_training_ws_room_inc.py
# python3 -u pgm_training_ws_room_inc_BCE_noMLP.py 
# python3 -u pgm_training_ws_room_inc_BCE.py 
# python3 -u pgm_training_ws_room_inc_WBCE.py 
# python3 -u pgm_training_ws_room_inc_WBCE_curriculum.py
# python3 -u pgm_training_ws_room_inc_WBCE_hyperparam.py
# python3 -u pgm_training_ws_room_inc_WBCE_hyperparam_75.py
# python3 -u pgm_training_ws_room_inc_WBCE_gradual_unfreeze.py
# python3 -u pgm_training_ws_room_inc_WBCE_scratch_new.py


# Run a stage. `dataset` builds ALL datasets (no experiment needed); the other
# stages are parameterized by an experiment id (dataset/models subfolder under
# partial_graph_matching), shared by optimization.py and the training script.
#
# Usage:
#   sbatch pgm.sh dataset                        # generate the datasets (dataset_gen.py)
#   sbatch pgm.sh <experiment> [optimize|train|both] [extra args...]
# Examples:
#   sbatch pgm.sh dataset
#   sbatch pgm.sh fully_glob_95                 # HPO (default stage)
#   sbatch pgm.sh adj_no_glob_65 train          # train with the study's best params
#   sbatch pgm.sh adj_no_glob_65 train --no-resume
#   sbatch pgm.sh fully_glob_95 both            # HPO then training

TRAIN_SCRIPT="pgm_training_ws_room_inc_WBCE_scratch.py"

# `dataset` is a special stage that needs NO experiment, so handle it before
# requiring $1 as an experiment id.
if [ "$1" = "dataset" ]; then
  echo "Stage: dataset (dataset_gen.py)"
  python3 -u dataset_gen.py
  echo "=== SLURM JOB ENDED ==="
  exit $?
fi

EXPERIMENT="${1:?Usage: sbatch pgm.sh dataset  |  sbatch pgm.sh <experiment> [optimize|train|both] [extra args...]}"
STAGE="${2:-optimize}"
EXTRA_ARGS="${@:3}"   # anything after <experiment> and <stage> is forwarded to the script

echo "Experiment: $EXPERIMENT | Stage: $STAGE | Extra args: $EXTRA_ARGS"

case "$STAGE" in
  optimize)
    python3 -u optimization.py "$EXPERIMENT" $EXTRA_ARGS
    ;;
  train)
    python3 -u "$TRAIN_SCRIPT" "$EXPERIMENT" $EXTRA_ARGS
    ;;
  both)
    python3 -u optimization.py "$EXPERIMENT" && \
    python3 -u "$TRAIN_SCRIPT" "$EXPERIMENT" $EXTRA_ARGS
    ;;
  *)
    echo "ERROR: unknown stage '$STAGE' (expected: optimize | train | both)"
    exit 1
    ;;
esac


echo "=== SLURM JOB ENDED ==="
