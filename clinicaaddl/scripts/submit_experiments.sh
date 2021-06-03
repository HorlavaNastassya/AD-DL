#!/bin/bash
echo "sumbit experiments"
for MS in '1.5T-3T' '1.5T' '3T' 
do
    for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3"
    do
        NN_FOLDER="NNs_Bayesian"
        OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"
        for f in ${OUTPUT_DIR}/*; do
            if [ -d "$f" ] &&  [[ $f =~ "subject_model" ]] &&  [[ $f =~ "WeightedCrossEntropy" ]] ; then
            echo $f
        # Input arguments to clinicaaddl
                sbatch run_experiment.sh $f False 1
            fi
        done            
    done
done