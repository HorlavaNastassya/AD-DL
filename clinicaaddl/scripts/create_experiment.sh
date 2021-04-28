#!/bin/bash
module load anaconda/3/2020.02
module load cuda/10.2
module load pytorch/gpu/1.6.0

# Experiment training CNN

# Importnant args
NETWORK="SEResNet18"
AUGMENTATION=True
EPOCHS=200
BATCH=10
LR=1e-4

# Input arguments to clinicaaddl
CAPS_DIR="$HOME//MasterProject/ADNI_data/CAPSPreprocessedT1linear"
TSV_PATH="$HOME/MasterProject/ADNI_data/DataPrep/labels/train"
OUTPUT_DIR="$HOME/MasterProject/NNs/"

# Dataset Management
PREPROCESSING='linear'
TASK='AD CN'
LOSS='WeightedCrossEntropy'
# LOSS='default'
BASELINE=True


# Training arguments
ACCUMULATION=1
EVALUATION=0
WEIGHT_DECAY=0
NORMALIZATION=1
PATIENCE=$EPOCHS
TOLERANCE=0
# Pretraining
T_BOOL=0
T_PATH=""
T_DIFF=0

# Other options
OPTIONS=""

if [ $AUGMENTATION = True ]; then
OPTIONS="$OPTIONS --data_augmentation RandomNoise RandomBiasField RandomGamma RandomRotation"
fi

if [ $T_BOOL = 1 ]; then
OPTIONS="$OPTIONS --pretrained_path $T_PATH -d $T_DIFF"
fi


TASK_NAME="${TASK// /_}"

echo $TASK_NAME

NAME="subject_model-${NETWORK}_preprocessing-${PREPROCESSING}_task-${TASK_NAME}_norm-${NORMALIZATION}_loss-${LOSS}_augm${AUGMENTATION}"


# echo $NAME

# Run clinicaaddl
python $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py prepare_train \
  image cnn \
  $CAPS_DIR \
  t1-${PREPROCESSING} \
  $TSV_PATH \
  $OUTPUT_DIR$NAME \
  $NETWORK \
  --baseline $BASELINE\
  --diagnoses $TASK \
  --loss $LOSS\
  --batch_size $BATCH \
  --evaluation_steps $EVALUATION \
  --accumulation_steps $ACCUMULATION \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --patience $PATIENCE \
  $OPTIONS
