#!/bin/bash
#SBATCH --partition=gpu_p1
#SBATCH --time=20:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1        # on réserve des coeurs physiques et non logiques
#SBATCH --ntasks=1
#SBATCH --workdir=/gpfswork/rech/zft/upd53tc/jobs/AD-DL/train/patch_level/single_cnn
#SBATCH --output=./exp12/pytorch_job_%j.out
#SBATCH --error=./exp12/pytorch_job_%j.err
#SBATCH --job-name=exp12_cnn
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=mauricio.diaz@inria.fr

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

# Experiment training autoencoder
eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

# Network structure
NETWORK="Conv4_FC3"
NETWORK_TYPE="single"
COHORT="ADNI"
DATE="reproducibility_results"

# Input arguments to clinicaaddl
CAPS_DIR="$SCRATCH/../commun/datasets/${COHORT}_rerun"
TSV_PATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="$SCRATCH/results/$DATE/"

# Computation ressources
NUM_PROCESSORS=32
GPU=1

# Dataset Management
PREPROCESSING='linear'
DIAGNOSES="AD CN"
SPLITS=5
SPLIT=$1

# Training arguments
EPOCHS=200
BATCH=32
BASELINE=1
ACCUMULATION=1
EVALUATION=20
LR=1e-5
WEIGHT_DECAY=1e-3
GREEDY_LEARNING=0
SIGMOID=0
NORMALIZATION=1
PATIENCE=20

# Pretraining
T_BOOL=1
T_PATH="patch3D_model-Conv4_FC3_preprocessing-linear_task-autoencoder_baseline-1_norm-1_splits-5"
T_PATH="$SCRATCH/results/$DATE/$T_PATH"
T_DIFF=0

# Other options
OPTIONS=""

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --use_gpu"
fi

if [ $NORMALIZATION = 1 ]; then
OPTIONS="${OPTIONS} --minmaxnormalization"
fi

if [ $T_BOOL = 1 ]; then
OPTIONS="$OPTIONS --transfer_learning_autoencoder --transfer_learning_path $T_PATH"
fi

if [ $BASELINE = 1 ]; then
echo "using only baseline data"
OPTIONS="$OPTIONS --baseline"
fi


NAME="patch3D_model-${NETWORK}_preprocessing-${PREPROCESSING}_task-autoencoder_baseline-${BASELINE}_norm-${NORMALIZATION}_selectionThreshold-07"

if [ $SPLITS > 0 ]; then
echo "Use of $SPLITS-fold cross validation, split $SPLIT"
NAME="${NAME}_splits-${SPLITS}"
fi

echo $NAME

# Run clinicaaddl
clinicadl train \
  patch \
  $CAPS_DIR \
  $TSV_PATH \
  $OUTPUT_DIR$NAME \
  $NETWORK \
  --network_type $NETWORK_TYPE \
  --batch_size $BATCH \
  --evaluation_steps $EVALUATION \
  --preprocessing $PREPROCESSING \
  --diagnoses $DIAGNOSES \
  --n_splits $SPLITS \
  --split $SPLIT \
  --accumulation_steps $ACCUMULATION \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --patience $PATIENCE \
  --selection_threshold 0.7 \
  $OPTIONS
