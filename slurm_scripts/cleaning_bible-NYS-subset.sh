#!/bin/bash

#SBATCH --job-name=cleaning_bible-NYS-subset
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32gb
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

apptainer run --cleanenv /home/aevans/apptainer/ood.sif /home/aevans/miniconda3/bin/python /home/aevans/nwp_bias/src/cleaning_bible-NYS-subset.py