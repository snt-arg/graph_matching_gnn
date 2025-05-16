#!/bin/bash -l
#SBATCH -c 1
#SBATCH --time=0-32:00:00
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

# Caricamento ambiente
module purge
module load env/legacy/2020b
module load lang/Python/3.8.6-GCCcore-10.2.0

# Attiva virtualenv
echo "Activating virtualenv..."
if [ -f ../../venv_aion/bin/activate ]; then
  source ../../venv_aion/bin/activate
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
python3 -u pgm_room_noise.py

# python3 -u optimization_gm.py 
# python3 -u optimization_ws.py 
# python3 -u optimization_room.py 

echo "=== SLURM JOB ENDED ==="
