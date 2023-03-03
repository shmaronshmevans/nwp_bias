#!/bin/bash

#SBATCH --job-name=land_buffer
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=10gb
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

singularity run /home/aevans/Kara_HW/builds/average_buffer.sif /home/aevans/miniconda3/bin/python /home/aevans/Kara_HW/src/average_buffer.py
