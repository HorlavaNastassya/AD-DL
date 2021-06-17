#!/bin/bash
echo "continue for 5 folds Bayesian NNs "
for MS in '1.5T-3T' '1.5T' '3T' 
do
    for NETWORK in "SEResNet50" "SEResNet50Expanded"  "ResNet18" "ResNet18Expanded" "Conv5_FC3"
    do
        NN_FOLDER="NNs_Bayesian"
        OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments_5-fold/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"
        for f in ${OUTPUT_DIR}/*; do
            if [ -d "$f" ] &&  [[ $f =~ "subject_model" ]] && [ ! -f "${f}/status.txt" ] ; then
                echo $f
                sbatch run_experiment_w_test.sh $f True 5
            fi
        done            
    done
    
done

echo "start for 5 folds Bayesian NNs "

for MS in '1.5T-3T' '1.5T' '3T' 
do
    for NETWORK in "SEResNet18" "SEResNet18Expanded"
    do
        NN_FOLDER="NNs_Bayesian"
        OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments_5-fold/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"
        for f in ${OUTPUT_DIR}/*; do
            if [ -d "$f" ] &&  [[ $f =~ "subject_model" ]] && [ ! -f "${f}/status.txt" ] ; then
                echo $f
                sbatch run_experiment_w_test.sh $f True 5
            fi
        done            
    done
    
done

echo "start for 5 folds NNs"
for MS in '1.5T-3T' '1.5T' '3T' 
do
    for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3" "SEResNet50" "SEResNet50Expanded"
    do
        NN_FOLDER="NNs"
        OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments_5-fold/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"
        for f in ${OUTPUT_DIR}/*; do
            if [ -d "$f" ] &&  [[ $f =~ "subject_model" ]] && [ ! -f "${f}/status.txt" ] ; then
                echo $f
                sbatch run_experiment_w_test.sh $f True 5
            fi
        done            
    done
done
