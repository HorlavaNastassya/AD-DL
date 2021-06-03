#!/bin/bash


# Experiment training CNN
if [ -z "$1" ]
  then
    MS='1.5T-3T'
else
    MS=$1
fi

for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3"

do
    # Input arguments to clinicaaddl
    sbatch test_network.sh $MS $NETWORK            
done

