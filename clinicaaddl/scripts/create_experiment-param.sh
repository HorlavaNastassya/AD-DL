#!/bin/bash
module load anaconda/3/2020.02
module load cuda/11.2
module load pytorch/gpu-cuda-11.2/1.8.1


# Experiment training CNN

# Importnant args

EPOCHS=100
BATCH=10
BAYESIAN=False

for LR in 1e-4
do
for MS in '1.5T-3T' '1.5T' '3T'
do
    for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3"

    do
        for LOSS in 'WeightedCrossEntropy' 'default'
        do
            for AUGMENTATION in True
            do
#                 echo -e "==========================================================================================================\n"
                # Input arguments to clinicaaddl
                CAPS_DIR="$HOME/MasterProject/DataAndExperiments/Data/CAPS"
                if [ $BAYESIAN = True ]; then
                NN_FOLDER="NNs_Bayesian"
                else
                NN_FOLDER="NNs"
                fi
                
                TSV_PATH="$HOME/MasterProject/DataAndExperiments/Experiments_5-fold/Experiments-${MS}/labels/train"
                OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments_5-fold/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"

                # Dataset Management
                PREPROCESSING='linear'
                TASK='AD CN'
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
#                 OPTIONS="$OPTIONS --data_augmentation RandomNoise RandomBiasField RandomGamma RandomRotation"
                OPTIONS="$OPTIONS --data_augmentation RandomBlur RandomNoise RandomBiasField RandomGamma"

                fi

                if [ $T_BOOL = 1 ]; then
                OPTIONS="$OPTIONS --pretrained_path $T_PATH -d $T_DIFF"
                fi


                TASK_NAME="${TASK// /_}"
                SAMPLER="random"

#                 echo $TASK_NAME

            NAME="subject_model-${NETWORK}_preprocessing-${PREPROCESSING}_task-${TASK_NAME}_norm-${NORMALIZATION}_loss-${LOSS}_augm${AUGMENTATION}-2"
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
                  --loss ${LOSS}\
                  --batch_size $BATCH \
                  --evaluation_steps $EVALUATION \
                  --accumulation_steps $ACCUMULATION \
                  --epochs $EPOCHS \
                  --learning_rate $LR \
                  --weight_decay $WEIGHT_DECAY \
                  --patience $PATIENCE \
                  --bayesian $BAYESIAN \
                  --sampler $SAMPLER \
                  $OPTIONS

            done
        done
    done
done
done
