#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint="gpu"
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=92500
#SBATCH --mail-type=END
#SBATCH --mail-user=g.nasta.work@gmail.com
#SBATCH -o logs/HLR_%j.out
#SBATCH -e logs/HLR_%j.err
echo $1
if [ -z "$2" ]
  then
    FROM_CHECKPOINT='False'
else
  FROM_CHECKPOINT=$2
fi
echo "Resume training from checkpoint: $FROM_CHECKPOINT"
module load anaconda/3/2020.02
module load cuda/11.2
module load pytorch/gpu-cuda-11.2/1.8.1


BATCH=8
# NUM_SPLITS=1
SPLIT=2
NUM_SPLITS=3
NPROC=2
# Other options
OPTIONS="--n_splits $NUM_SPLITS --split $SPLIT --nproc $NPROC --batch_size $BATCH"
# OPTIONS="--n_splits $NUM_SPLITS --nproc $NPROC --batch_size $BATCH"


# Computation ressources
GPU=1
if [ $GPU = 0 ]; then
OPTIONS="${OPTIONS} -cpu"
fi


# Run clinicaaddl
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py train \
  $1 --resume $FROM_CHECKPOINT $OPTIONS
