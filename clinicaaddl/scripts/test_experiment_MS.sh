#!/bin/bash
module load anaconda/3/2020.02
module load cuda/10.2
module load pytorch/gpu/1.6.0

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

