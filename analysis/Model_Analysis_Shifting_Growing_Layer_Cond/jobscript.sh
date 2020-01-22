#!/usr/local_rwth/bin/zsh

### #SBATCH directives need to be in the first part of the jobscript

### Job name
#SBATCH --job-name=MRP_dke

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_%J_mrprun.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=1-00:00:00

### Request the amount of memory you need for your job.
### You can specify this in either MB (1024M) or GB (4G).

#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1

### Request a host with a Volta GPU
### If you need two GPUs, change the number accordingly                                                                                                                                                                                                                                                                                                                                                                                                                                   ### if needed: switch to your working directory (where you saved your program)                                                                                                                                                               export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### Load modules
module switch intel gcc
module load python/3.7.3
module load cuda/100
module load cudnn/7.4
pip install ../mrpenv/bin/activate

### your code goes here, the second part of the jobscript

python Continued_Training.py