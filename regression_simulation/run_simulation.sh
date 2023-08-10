#!/bin/bash -l
#SBATCH
#SBATCH --time=10:00:00
#SBATCH --partition=shared
#SBATCH --mem=10GB
#SBATCH --nodes=1



temporary_results_dir="$1"
results_dir="$2"
n_samples="$3"
n_snps="$4"




python3 run_simulation.py $temporary_results_dir $results_dir $n_samples $n_snps
