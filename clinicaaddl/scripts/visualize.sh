#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint="gpu"
#SBATCH --time=08:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=92500
#SBATCH --mail-type=END
#SBATCH --mail-user=g.nasta.work@gmail.com
#SBATCH -o logs/HLR_%x_%j.out
#SBATCH -e logs/HLR_%x_%j.err


module load anaconda/3/2020.02
module load cuda/11.2
module load pytorch/gpu-cuda-11.2/1.8.1

srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/plot_results.py --bayesian True
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/plot_results.py --bayesian False

# BAYESIAN=FALSE

# history_modes=["loss", "loss","balanced_accuracy"]
# selection_metrics=["best_loss", "last_checkpoint", "best_balanced_accuracy"]

# srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/plot_combined.py --history_mode loss --selection_metric best_loss --bayesian $BAYESIAN

# srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/plot_combined.py --history_mode loss --selection_metric last_checkpoint --bayesian $BAYESIAN

# srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/plot_combined.py --history_mode balanced_accuracy --selection_metric best_balanced_accuracy --bayesian $BAYESIAN
