#!/bin/bash

# echo "resume for WeightedCrossEntropy and 3 folds"
# for MS in '1.5T-3T' '1.5T' '3T' 
# do
#     for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3"
#     do
#         NN_FOLDER="NNs_Bayesian"
#         OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"
#         for f in ${OUTPUT_DIR}/*; do
#             if [ -d "$f" ] &&  [[ $f =~ "subject_model" ]] &&  [[ $f =~ "WeightedCrossEntropy" ]] ; then
#             echo $f
#                 sbatch run_experiment_w_test.sh $f True 3
#             fi
#         done            
#     done
# done

sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/Conv5_FC3/subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/Conv5_FC3/subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/Conv5_FC3/subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/Conv5_FC3/subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_134522 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/Conv5_FC3/subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_134433 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/Conv5_FC3/subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_134429 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_134420 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_134417 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_134404 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_134407 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_134351 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_134355 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_134342 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_134339 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130201 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_130136 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130133 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130147 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130119 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/Conv5_FC3/subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_130 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/Conv5_FC3/subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130106 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_130056 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18Expanded/subject_model-SEResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130053 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_130043 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18Expanded/subject_model-ResNet18Expanded_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130039 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_130029 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130025 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_130015 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject//DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/ResNet18/subject_model-ResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmTrue_20210528_130011 False 3
sbatch run_experiment_w_test.sh /u/horlavanasta/MasterProject/DataAndExperiments/Experiments_3-fold/Experiments-1.5T-3T/NNs_Bayesian/SEResNet18/subject_model-SEResNet18_preprocessing-linear_task-AD_CN_norm-1_loss-default_augmFalse_20210528_130029 True 3


echo "retrain for WeightedCrossEntropy and Non Bayesian"
for MS in '1.5T-3T' '1.5T' '3T' 
do
    for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3"
    do
        NN_FOLDER="NNs"
        OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments/Experiments-${MS}/${NN_FOLDER}/${NETWORK}/"
        for f in ${OUTPUT_DIR}/*; do
            if [ -d "$f" ] &&  [[ $f =~ "subject_model" ]] &&  [[ $f =~ "WeightedCrossEntropy" ]] ; then
            echo $f
                sbatch run_experiment_w_test.sh $f True 1
            fi
        done            
    done
done