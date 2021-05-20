#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint="gpu"
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=92500
#SBATCH --mail-type=END
#SBATCH --mail-user=g.nasta.work@gmail.com
#SBATCH -o logs/HLR_%j.out
#SBATCH -e logs/HLR_%j.err

# Importnant args

BAYESIAN=True
NBR_BAYESIAN_ITER=10

MS=$1
NETWORK=$2
echo $2


CAPS_DIR="$HOME/MasterProject/DataAndExperiments/Data/CAPS"
if [ $BAYESIAN = True ]; then
    NN_FOLDER="NNs_Bayesian"
else
    NN_FOLDER="NNs"
fi
                
TSV_PATH="$HOME/MasterProject/DataAndExperiments/Experiments/Experiments-${MS}/labels/test"
OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"
POSTFIX="test_${MS}"
                
for f in ${OUTPUT_DIR}/*; do
    if [ -d "$f" ] &&  [[ $f =~ "subject_model" ]] &&  [[ $f =~ "20210517" ||  $f =~ "20210518" ]] ; then
# Will not run if no directories are available
        echo -e "$f"
        srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TSV_PATH $f $POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics balanced_accuracy loss last_checkpoint
            
        if [[ $MS = "1.5T" ]]; then
        echo -e "$f"
        TEST_MS="3T"
        TEST_POSTFIX="test_${TEST_MS}"
        TEST_TSV_PATH="$HOME/MasterProject/DataAndExperiments/Experiments/Experiments-${TEST_MS}/labels/"
        srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TEST_TSV_PATH $f $TEST_POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics balanced_accuracy loss last_checkpoint --baseline False
        fi
            
        if [[ $MS = "3T" ]]; then
        echo -e "$f"
        TEST_MS="1.5T"
        TEST_POSTFIX="test_${TEST_MS}"
        TEST_TSV_PATH="$HOME/MasterProject/DataAndExperiments/Experiments/Experiments-${TEST_MS}/labels/"
        srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify $CAPS_DIR $TEST_TSV_PATH $f $TEST_POSTFIX --bayesian $BAYESIAN --nbr_bayesian_iter $NBR_BAYESIAN_ITER --selection_metrics balanced_accuracy loss last_checkpoint --baseline False
        fi

    fi
done    
