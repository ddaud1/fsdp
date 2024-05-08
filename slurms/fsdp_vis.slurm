#!/bin/bash
# sbatch ./slurms/fsdp_vis.slurm 
#SBATCH --job-name=fsdp   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:3             # number of gpus per node
#SBATCH --constraint=gpu80
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jipeng.sun@princeton.edu

module purge
module load anaconda3/2021.11
conda activate PolarMeta

cd /scratch/gpfs/js2694/FSDP/fsdp/
viztracer --log_multiprocess fsdp_main_orig.py