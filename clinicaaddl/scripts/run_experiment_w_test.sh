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
# SPLIT=2
if [ -z "$3" ]
  then
    NUM_SPLITS=5
else
  NUM_SPLITS=$3
fi

NPROC=2
# Other options
# OPTIONS="--n_splits $NUM_SPLITS --split $SPLIT --nproc $NPROC --batch_size $BATCH"
OPTIONS="--n_splits $NUM_SPLITS --nproc $NPROC --batch_size $BATCH"


# Computation ressources
GPU=1
if [ $GPU = 0 ]; then
OPTIONS="${OPTIONS} -cpu"
fi


# Run clinicaaddl
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py train \
  $1 --resume $FROM_CHECKPOINT $OPTIONS


CAPS_DIR="$HOME/MasterProject/DataAndExperiments/Data/CAPS"

if [[ $1 =~ "NNs_Bayesian" ] ]; then
    BAYESIAN = True 
else
    BAYESIAN = False 
fi
 
NBR_BAYESIAN_ITER=10


if [[ $1 =~ "Experiments_5-fold/" ]] ; then
FOLD_FOLDER="Experiments_5-fold"
fi

if [[ $1 =~ "Experiments_3-fold/" ]] ; then
FOLD_FOLDER="Experiments_3-fold"
fi

if [[ $1 =~ "Experiments/" ]] ; then
FOLD_FOLDER="Experiments"
fi

if [[ $1 =~ "Experiments-1.5T-3T/" ]] ; then

TSV_PATH="$HOME/MasterProject/DataAndExperiments/${FOLD_FOLDER}/Experiments-1.5-3T/labels/test"
POSTFIX="test_1.5-3T"
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TSV_PATH $1 $POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics balanced_accuracy loss last_checkpoint

fi


if [[ $1 =~ "Experiments-1.5T/" ]] ; then
                
TSV_PATH="$HOME/MasterProject/DataAndExperiments/${FOLD_FOLDER}/Experiments-1.5T/labels/test"
POSTFIX="test_1.5T"
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TSV_PATH $1 $POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics balanced_accuracy loss last_checkpoint

TEST_MS="3T"
TEST_POSTFIX="test_${TEST_MS}"
TEST_TSV_PATH="$HOME/MasterProject/DataAndExperiments/${FOLD_FOLDER}/Experiments-${TEST_MS}/labels/"
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TEST_TSV_PATH $1 $TEST_POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics balanced_accuracy loss last_checkpoint --baseline False
fi

if [[ $1 =~ "Experiments-3T/" ]] ; then
TSV_PATH="$HOME/MasterProject/DataAndExperiments/${FOLD_FOLDER}/Experiments-3T/labels/test"
POSTFIX="test_3T"
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TSV_PATH $1 $POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics balanced_accuracy loss last_checkpoint

TEST_MS="1.5T"
TEST_POSTFIX="test_${TEST_MS}"
TEST_TSV_PATH="$HOME/MasterProject/DataAndExperiments/${FOLD_FOLDER}/Experiments-${TEST_MS}/labels/"
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TEST_TSV_PATH $1 $TEST_POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics balanced_accuracy loss last_checkpoint --baseline False

fi



