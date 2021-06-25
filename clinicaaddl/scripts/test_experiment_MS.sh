#!/bin/bash


for MS in '1.5T-3T' '1.5T' '3T'
do
for BAYESIAN in True False
do

    for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3"
    do
    # Input arguments to clinicaaddl
        sbatch test_network.sh $MS $NETWORK $BAYESIAN          
    done
done
done

