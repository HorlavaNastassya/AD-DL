#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint="gpu"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=92500
#SBATCH --mail-type=END
#SBATCH --mail-user=g.nasta.work@gmail.com
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err

module load anaconda/3/2020.02
module load cuda/10.2
module load pytorch/gpu/1.6.0

# Experiment training CNN

# Importnant args

BAYESIAN=False
NBR_BAYESIAN_ITER=10

if [ -z "$1" ]
  then
    MS='1.5T-3T'
else
  MS=$1
fi

for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3"
do
                # Input arguments to clinicaaddl
    CAPS_DIR="$HOME/MasterProject/DataAndExperiments/Data/CAPS"
    if [ $BAYESIAN = True ]; then
        NN_FOLDER="NNs_Bayesian"
    else
        NN_FOLDER="NNs"
    fi
                
    OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"
                # Dataset Management
    for f in ${OUTPUT_DIR}/*; do
        if [ -d "$f" ] &&  [[ $f =~ "subject_model" ]]; then
                # Will not run if no directories are available
            echo -e "$f"
            TSV_PATH="$HOME/MasterProject/DataAndExperiments/Experiments/Experiments-${MS}/labels/train/train_splits-1/split-0/"
            POSTFIX="train"
            srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TSV_PATH $f $POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics last_checkpoint
            
            TSV_PATH="$HOME/MasterProject/DataAndExperiments/Experiments/Experiments-${MS}/labels/train/validation_splits-1/split-0/"
            POSTFIX="validation"
            srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TSV_PATH $f $POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics last_checkpoint
        fi
    done                
done
